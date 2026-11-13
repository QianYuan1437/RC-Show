"""Optimisation-based control: QP task-space control, model predictive control, iLQR.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..dynamics import Dynamics
from ..qp import QPSolution, solve_qp
from .base import Controller, Ref, State, pose_error, solve
from .gains import NULL_D, NULL_K, WN_JOINT, WN_TASK


def _accel_bounds(dyn: Dynamics, q, qd, t_v: float = 0.05, t_p: float = 0.15):
    """Joint acceleration box that keeps velocity and position limits reachable (velocity damper)."""
    vmax = dyn.vel_limit
    lo = torch.maximum((-vmax - qd) / t_v, 2 * (dyn.lower - q - qd * t_p) / t_p**2)
    hi = torch.minimum((vmax - qd) / t_v, 2 * (dyn.upper - q - qd * t_p) / t_p**2)
    mid = 0.5 * (lo + hi)
    return torch.minimum(lo, mid), torch.maximum(hi, mid)


class QPTaskSpace(Controller):
    """Task-space control as a QP over joint accelerations (as in whole-body controllers):

        min_qdd  1/2 |J qdd + Jdot qd - xdd*|^2 + w_n/2 |qdd - qdd_null|^2
        s.t.     tau_min <= M qdd + h <= tau_max           (motor torque limits)
                 qdd_min <= qdd <= qdd_max                  (joint position / velocity limits)

    xdd* is the OSC acceleration command. Without active constraints this *is* OSC; when a limit
    binds, the QP gives the best feasible acceleration instead of a clipped torque."""

    name, family = "qp", "optimal"
    needs = ("cartesian",)

    def __init__(self, dyn, wn: float = WN_TASK, w_null: float = 1e-2, iters: int = 30):
        super().__init__(dyn)
        self.kp, self.kd = wn**2, 2 * wn
        self.w_null = w_null
        self.iters = iters
        self.warm: QPSolution | None = None

    def reset(self, state, ref):
        B, n = state.q.shape
        z = torch.zeros(B, n, dtype=state.q.dtype, device=state.q.device)
        zc = torch.zeros(B, 2 * n, dtype=state.q.dtype, device=state.q.device)
        self.warm = QPSolution(z.clone(), zc.clone(), zc.clone(), z[:, 0].clone(), z[:, 0].clone())

    def __call__(self, state, ref):
        dyn = self.dyn
        q, qd = state.q, state.qd
        B, n = q.shape
        if self.warm is None:
            self.reset(state, ref)
        p, R, J = dyn.jacobian(q)
        M = dyn.mass_matrix(q)
        h = dyn.bias(q, qd)
        xd = (J @ qd[..., None]).squeeze(-1)
        a_star = ref.a + self.kp * pose_error(ref.p, ref.R, p, R) + self.kd * (ref.v - xd) - dyn.tcp_bias_acc(q, qd)
        q_null = ref.q_null if ref.q_null is not None else dyn.home.expand_as(q)
        a_null = NULL_K * (q_null - q) - NULL_D * qd
        eye = torch.eye(n, dtype=q.dtype, device=q.device)
        P = J.transpose(1, 2) @ J + self.w_null * eye
        c = -((J.transpose(1, 2) @ a_star[..., None]).squeeze(-1) + self.w_null * a_null)
        lo_a, hi_a = _accel_bounds(dyn, q, qd)
        A = torch.cat([M, eye.expand(B, n, n)], 1)
        tl = dyn.torque_limit
        l = torch.cat([-tl - h, lo_a], 1)
        u = torch.cat([tl - h, hi_a], 1)
        sol = solve_qp(P, c, A, l, u, iters=self.iters, rho=0.1, warm=self.warm)
        for f in ("x", "z", "y"):  # warm start for the next cycle, in place
            getattr(self.warm, f).copy_(getattr(sol, f))
        return (M @ sol.x[..., None]).squeeze(-1) + h


class JointMPC(Controller):
    """Linear MPC on top of feedback linearisation.

    With tau = M(q) u + h(q, qd) every joint is a double integrator q'' = u. Every `period`
    seconds an N-step QP plans the accelerations U = (u_0 .. u_{N-1}) (step dt_mpc):

        min  sum_k  w_q |q_k - q*_k|^2 + w_v |qd_k - qd*_k|^2 + w_u |u_k|^2
        s.t. |qd_k| <= v_max,  q_min <= q_k <= q_max,  tau_min <= M0 u_k + h0 <= tau_max

    (M0, h0 frozen at the current state). The reference is *previewed* over the horizon, so the
    controller slows down before a corner it could not take within the torque limits. Between
    re-plans the planned state is tracked with an inner computed-torque loop."""

    name, family = "mpc", "optimal"
    graph_safe = False  # re-plans at a lower rate, decided on the host

    def __init__(self, dyn, horizon: int = 20, dt_mpc: float = 0.02, period: float = 0.02,
                 w_q: float = 1e4, w_v: float = 10.0, w_u: float = 1e-2, iters: int = 60, torque_margin: float = 0.9):
        super().__init__(dyn)
        self.N, self.h_dt, self.period = horizon, dt_mpc, period
        self.w_q, self.w_v, self.w_u = w_q, w_v, w_u
        self.iters = iters
        self.margin = torque_margin
        self.kp, self.kd = WN_JOINT**2, 2 * WN_JOINT
        self._build()
        self.plan = None
        self.t_plan = None
        self.warm = None

    def _build(self):
        N, d, n = self.N, self.h_dt, self.dyn.n
        dt, dev = self.dyn.dtype, self.dyn.device
        # scalar double integrator: x_k = Ak x0 + sum_j Bkj u_j
        k = torch.arange(1, N + 1, dtype=dt, device=dev)
        j = torch.arange(N, dtype=dt, device=dev)
        steps = (k[:, None] - j[None, :] - 1).clamp(min=-1)  # steps after u_j acts
        valid = (j[None, :] < k[:, None]).to(dt)
        self.Sq = valid * (d**2 / 2 + steps.clamp(min=0) * d**2)  # position of step k from u_j
        self.Sv = valid * d
        self.k_time = k * d
        eye_n = torch.eye(n, dtype=dt, device=dev)
        # variables ordered [u_0 (n), u_1 (n), ...]: kron(S, I_n)
        self.Kq = torch.kron(self.Sq, eye_n)
        self.Kv = torch.kron(self.Sv, eye_n)
        P = 2 * (self.w_q * self.Kq.T @ self.Kq + self.w_v * self.Kv.T @ self.Kv
                 + self.w_u * torch.eye(N * n, dtype=dt, device=dev))
        self.P = P

    def reset(self, state, ref):
        self.plan, self.t_plan, self.warm = None, None, None

    def _solve(self, state, ref):
        dyn, N, n = self.dyn, self.N, self.dyn.n
        q, qd = state.q, state.qd
        B = q.shape[0]
        preview = ref.extra.get("preview")
        tk = state.t[:, None] + self.k_time[None]  # (B, N)
        if preview is not None:
            fut = preview(tk.reshape(-1))
            q_ref = fut.q.reshape(B, N, n)
            v_ref = fut.qd.reshape(B, N, n)
        else:
            q_ref = ref.q[:, None].expand(B, N, n)
            v_ref = torch.zeros_like(q_ref)
        # free response (u = 0)
        q_free = q[:, None] + qd[:, None] * self.k_time[None, :, None]
        v_free = qd[:, None].expand(B, N, n)
        eq = (q_ref - q_free).reshape(B, N * n)
        ev = (v_ref - v_free).reshape(B, N * n)
        c = -2 * (self.w_q * (eq @ self.Kq) + self.w_v * (ev @ self.Kv))
        M = dyn.mass_matrix(q)
        h = dyn.bias(q, qd)
        tl = self.margin * dyn.torque_limit
        A_tau = torch.zeros(B, N * n, N * n, dtype=q.dtype, device=q.device)
        for k in range(N):
            A_tau[:, k * n:(k + 1) * n, k * n:(k + 1) * n] = M
        A = torch.cat([A_tau, self.Kv.expand(B, -1, -1), self.Kq.expand(B, -1, -1)], 1)
        vmax = dyn.vel_limit.repeat(N)
        l = torch.cat([(-tl - h).repeat(1, N), -vmax - v_free.reshape(B, -1),
                       dyn.lower.repeat(N) - q_free.reshape(B, -1)], 1)
        u = torch.cat([(tl - h).repeat(1, N), vmax - v_free.reshape(B, -1),
                       dyn.upper.repeat(N) - q_free.reshape(B, -1)], 1)
        sol = solve_qp(self.P.expand(B, -1, -1), c, A, l, u, iters=self.iters, rho=1.0, warm=self.warm)
        self.warm = sol
        U = sol.x.view(B, N, n)
        return U

    def __call__(self, state, ref):
        dyn = self.dyn
        t = float(state.t[0])
        if self.plan is None or t - self.t_plan >= self.period - 1e-9:
            self.plan = self._solve(state, ref)
            self.t_plan = t
            self.x0 = (state.q.clone(), state.qd.clone())
        tau_s = t - self.t_plan
        k = min(int(tau_s / self.h_dt), self.N - 1)
        u = self.plan[:, k]
        # planned state at tau_s (piecewise-constant acceleration)
        q0, v0 = self.x0
        U = self.plan
        v_pl, q_pl = v0.clone(), q0.clone()
        remaining = tau_s
        for i in range(k + 1):
            step = min(self.h_dt, remaining)
            q_pl = q_pl + v_pl * step + 0.5 * U[:, i] * step**2
            v_pl = v_pl + U[:, i] * step
            remaining -= step
            if remaining <= 0:
                break
        q, qd = state.q, state.qd
        v = u + self.kp * (q_pl - q) + self.kd * (v_pl - qd)
        return (dyn.mass_matrix(q) @ v[..., None]).squeeze(-1) + dyn.bias(q, qd)


# ------------------------------------------------------------------------------------- iLQR

@dataclass
class ILQRResult:
    x: torch.Tensor  # (B, T+1, 2n) optimal state trajectory
    u: torch.Tensor  # (B, T, n) optimal torques
    K: torch.Tensor  # (B, T, n, 2n) TVLQR feedback gains
    dt: float
    cost: torch.Tensor  # (B,)
    history: list

    def reference(self, t) -> Ref:
        """Continuous joint reference of the plan: the knot accelerations held piecewise constant
        and integrated exactly (equals the knots at rest-to-rest ends), then hold the goal."""
        n = self.u.shape[-1]
        T = self.u.shape[1]
        if not hasattr(self, "_Q"):
            qd = self.x[:, :, n:]
            self._qdd = (qd[:, 1:] - qd[:, :-1]) / self.dt
            dQ = qd[:, :-1] * self.dt + 0.5 * self._qdd * self.dt**2
            self._Q = torch.cat([self.x[:, :1, :n], self.x[:, :1, :n] + dQ.cumsum(1)], 1)
        tf = (t / self.dt).clamp(min=0, max=T)
        i = tf.floor().long().clamp(max=T - 1)
        s = ((tf - i.to(tf.dtype)) * self.dt)[:, None]
        b = torch.arange(t.shape[0], device=t.device)
        moving = (tf < T)[:, None]
        qdd = self._qdd[b, i] * moving
        qd0 = self.x[b, i, n:]
        q = torch.where(moving, self._Q[b, i] + qd0 * s + 0.5 * qdd * s**2, self._Q[:, -1])
        return Ref(q=q, qd=torch.where(moving, qd0 + qdd * s, self.x[:, -1, n:]), qdd=qdd)


def forward_dynamics(dyn: Dynamics, q, qd, tau):
    """qdd = M^-1 (tau - h - D qd)  (MuJoCo plant's viscous damping included)."""
    M = dyn.mass_matrix(q)
    rhs = tau - dyn.bias(q, qd) - dyn.damping * qd
    return solve(M, rhs[..., None]).squeeze(-1)


def _step(dyn, x, u, dt):
    n = dyn.n
    q, qd = x[..., :n], x[..., n:]
    # semi-implicit Euler (like MuJoCo's integrator): update velocity first
    qdd = forward_dynamics(dyn, q, qd, u)
    qd2 = qd + dt * qdd
    return torch.cat([q + dt * qd2, qd2], -1)


def ilqr(dyn: Dynamics, x0, x_goal, T: int, dt: float = 0.01, iters: int = 60, w_u: float = 1e-3,
         w_goal: float = 1e4, w_vel_goal: float = 1e2, w_path: float = 0.0, w_vel: float = 0.05,
         w_vlim: float = 1e3, vlim_frac: float = 0.8, u_init=None) -> ILQRResult:
    """Torque-limited iLQR (Li & Todorov 2004; control limits by clamping, Tassa et al. 2014):

        min  sum_t  w_u/2 |u_t|^2 + w_path/2 |q_t - q_goal|^2 + w_vel/2 |qd_t|^2
                    + w_vlim/2 |max(|qd_t| - vlim_frac qd_max, 0)|^2
             + terminal  w_goal/2 |q_T - q_goal|^2 + w_vel_goal/2 |qd_T|^2
        s.t. x_{t+1} = f(x_t, u_t)  (full rigid-body dynamics),  |u| <= tau_max

    Torque alone is cheap to spend on the light wrist joints: without the velocity terms the
    "optimum" whips them at several times their speed limit.

    Linearisations A_t, B_t by forward-mode autodiff. Returns the trajectory and the TVLQR gains
    K_t, used by ``TVLQRTracking`` for closed-loop execution."""
    B, n = x0.shape[0], dyn.n
    tl = dyn.torque_limit
    vlim = vlim_frac * dyn.vel_limit
    u = torch.zeros(B, T, n, dtype=x0.dtype, device=x0.device) if u_init is None else u_init.clone()
    if u_init is None:  # start from gravity compensation
        u[:] = dyn.gravity_torque(x0[:, :n])[:, None]

    def rollout(u):
        xs = [x0]
        for t in range(T):
            xs.append(_step(dyn, xs[-1], u[:, t], dt))
        return torch.stack(xs, 1)

    def cost(x, u):
        c = 0.5 * w_u * (u**2).sum((1, 2)) * dt
        if w_path:
            c = c + 0.5 * w_path * ((x[:, :-1, :n] - x_goal[:, None, :n]) ** 2).sum((1, 2)) * dt
        qd = x[:, :-1, n:]
        c = c + 0.5 * w_vel * (qd**2).sum((1, 2)) * dt
        c = c + 0.5 * w_vlim * ((qd.abs() - vlim).clamp(min=0) ** 2).sum((1, 2)) * dt
        c = c + 0.5 * w_goal * ((x[:, -1, :n] - x_goal[:, :n]) ** 2).sum(1)
        c = c + 0.5 * w_vel_goal * (x[:, -1, n:] ** 2).sum(1)
        return c

    def lin(x, u):
        f = lambda xx, uu: _step(dyn, xx[None], uu[None], dt)[0]
        A, Bm = torch.vmap(torch.func.jacfwd(f, argnums=(0, 1)))(x.reshape(-1, 2 * n), u.reshape(-1, n))
        return A.view(B, T, 2 * n, 2 * n), Bm.view(B, T, 2 * n, n)

    x = rollout(u)
    J = cost(x, u)
    mu = torch.full((B,), 1e-3, dtype=x0.dtype, device=x0.device)
    history = [J.clone()]
    eye_n = torch.eye(n, dtype=x0.dtype, device=x0.device)
    for _ in range(iters):
        A, Bm = lin(x[:, :-1], u)
        # terminal cost derivatives
        Vx = torch.cat([w_goal * (x[:, -1, :n] - x_goal[:, :n]), w_vel_goal * x[:, -1, n:]], -1)
        Vxx = torch.diag_embed(torch.cat([torch.full((n,), w_goal), torch.full((n,), w_vel_goal)]).to(x0)).expand(B, -1, -1)
        ks = torch.zeros(B, T, n, dtype=x0.dtype, device=x0.device)
        Ks = torch.zeros(B, T, n, 2 * n, dtype=x0.dtype, device=x0.device)
        for t in reversed(range(T)):
            lx = torch.zeros(B, 2 * n, dtype=x0.dtype, device=x0.device)
            lxx = torch.zeros(B, 2 * n, 2 * n, dtype=x0.dtype, device=x0.device)
            if w_path:
                lx[:, :n] = w_path * dt * (x[:, t, :n] - x_goal[:, :n])
                lxx[:, :n, :n] = w_path * dt * eye_n
            qd_t = x[:, t, n:]
            over = (qd_t.abs() - vlim).clamp(min=0)
            lx[:, n:] = dt * (w_vel * qd_t + w_vlim * over * qd_t.sign())
            lxx[:, n:, n:] = torch.diag_embed(dt * (w_vel + w_vlim * (over > 0).to(x0.dtype)))
            lu = w_u * dt * u[:, t]
            luu = w_u * dt * eye_n
            At, Bt = A[:, t], Bm[:, t]
            Qx = lx + (At.transpose(1, 2) @ Vx[..., None]).squeeze(-1)
            Qu = lu + (Bt.transpose(1, 2) @ Vx[..., None]).squeeze(-1)
            Qxx = lxx + At.transpose(1, 2) @ Vxx @ At
            Quu = luu + Bt.transpose(1, 2) @ Vxx @ Bt + mu[:, None, None] * eye_n
            Qux = Bt.transpose(1, 2) @ Vxx @ At
            # control limits: free controls only (clamped ones get zero feedback)
            sat = ((u[:, t] >= tl - 1e-6) & (Qu < 0)) | ((u[:, t] <= -tl + 1e-6) & (Qu > 0))
            free = (~sat).to(x0.dtype)
            Fm = free[:, :, None] * free[:, None, :]
            Quu_f = Quu * Fm + torch.diag_embed(1 - free)
            k = -torch.linalg.solve(Quu_f, (Qu * free)[..., None]).squeeze(-1)
            K = -torch.linalg.solve(Quu_f, Qux * free[..., None])
            ks[:, t], Ks[:, t] = k, K
            Vx = Qx + (K.transpose(1, 2) @ Quu @ k[..., None]).squeeze(-1) + (K.transpose(1, 2) @ Qu[..., None]).squeeze(-1) \
                + (Qux.transpose(1, 2) @ k[..., None]).squeeze(-1)
            Vxx = Qxx + K.transpose(1, 2) @ Quu @ K + K.transpose(1, 2) @ Qux + Qux.transpose(1, 2) @ K
            Vxx = 0.5 * (Vxx + Vxx.transpose(1, 2))
        # line search
        best_J, best = J.clone(), (x, u)
        improved = torch.zeros(B, dtype=torch.bool, device=x0.device)
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.03):
            xs = [x0]
            un = torch.empty_like(u)
            for t in range(T):
                ut = u[:, t] + alpha * ks[:, t] + (Ks[:, t] @ (xs[-1] - x[:, t])[..., None]).squeeze(-1)
                ut = torch.maximum(torch.minimum(ut, tl), -tl)
                un[:, t] = ut
                xs.append(_step(dyn, xs[-1], ut, dt))
            xn = torch.stack(xs, 1)
            Jn = cost(xn, un)
            better = Jn < best_J
            best_J = torch.where(better, Jn, best_J)
            best = (torch.where(better[:, None, None], xn, best[0]), torch.where(better[:, None, None], un, best[1]))
            improved |= better
        mu = torch.where(improved, (mu / 3).clamp(min=1e-6), mu * 10)
        x, u = best
        J = best_J
        history.append(J.clone())
    # final gains around the converged trajectory
    A, Bm = lin(x[:, :-1], u)
    return ILQRResult(x=x, u=u, K=_tvlqr(A, Bm, n, dt), dt=dt, cost=J, history=history)


def _tvlqr(A, Bm, n, dt, q_pos: float = 1e4, q_vel: float = 10.0, r: float = 1e-2):
    """Time-varying LQR gains for tracking along a trajectory (Riccati recursion)."""
    Bsz, T = A.shape[:2]
    dtp, dev = A.dtype, A.device
    Q = torch.diag(torch.cat([torch.full((n,), q_pos), torch.full((n,), q_vel)])).to(dtp).to(dev) * dt
    R = r * dt * torch.eye(n, dtype=dtp, device=dev)
    P = (Q / dt).expand(Bsz, -1, -1)
    Ks = torch.zeros(Bsz, T, n, 2 * n, dtype=dtp, device=dev)
    for t in reversed(range(T)):
        At, Bt = A[:, t], Bm[:, t]
        S = R + Bt.transpose(1, 2) @ P @ Bt
        K = -torch.linalg.solve(S, Bt.transpose(1, 2) @ P @ At)
        Ks[:, t] = K
        Acl = At + Bt @ K
        P = Q + K.transpose(1, 2) @ R @ K + Acl.transpose(1, 2) @ P @ Acl
        P = 0.5 * (P + P.transpose(1, 2))
    return Ks


class TVLQRTracking(Controller):
    """Executes an iLQR solution: tau = u*(t) + K(t) (x - x*(t)), linear interpolation of the knots."""

    name, family = "ilqr", "optimal"
    needs = ()

    def __init__(self, dyn, sol: ILQRResult):
        super().__init__(dyn)
        self.sol = sol

    def __call__(self, state, ref):
        s = self.sol
        T = s.u.shape[1]
        n = self.dyn.n
        tf = (state.t / s.dt).clamp(min=0)
        i = tf.floor().long().clamp(max=T - 1)
        w = (tf - i.to(tf.dtype)).clamp(0, 1)[:, None]
        b = torch.arange(state.q.shape[0], device=state.q.device)
        x_star = (1 - w) * s.x[b, i] + w * s.x[b, (i + 1).clamp(max=T)]
        done = tf >= T
        u_star = torch.where(done[:, None], self.dyn.gravity_torque(state.q), s.u[b, i])
        x = torch.cat([state.q, state.qd], -1)
        K = s.K[b, i]
        fb = (K @ (x - x_star)[..., None]).squeeze(-1)
        # after the horizon: hold the final state with PD + gravity
        hold = 400.0 * (s.x[b, -1, :n] - state.q) - 40.0 * state.qd + self.dyn.gravity_torque(state.q)
        return torch.where(done[:, None], hold, u_star + fb)


__all__ = ["ILQRResult", "JointMPC", "QPTaskSpace", "Ref", "State", "TVLQRTracking", "forward_dynamics", "ilqr"]
