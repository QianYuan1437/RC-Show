"""Task-space (Cartesian) controllers and force control.

TCP error e = [p* - p; Log(R* R^T)], TCP twist xd = J qd, Lambda = (J M^-1 J^T)^-1.
For the redundant Panda a null-space posture task keeps the elbow near q_null; for the
6-DoF SR3-C the null space is empty and that term vanishes.
"""

from __future__ import annotations

import torch

from .base import Controller, Ref, State, damped_pinv, inv, pose_error, task_inertia, vec
from .gains import CART_K, CART_K_SOFT, NULL_D, NULL_K, WN_TASK
from .joint import JointVelocityMode


def _null_torque(dyn, state, ref, J, Minv, Lam):
    """Posture torque projected with the dynamically consistent null-space projector
    N^T = I - J^T Lambda J M^-1 (Khatib 1987): it cannot disturb the TCP."""
    if dyn.n <= 6:
        return torch.zeros_like(state.q)
    q_null = ref.q_null if ref.q_null is not None else dyn.home.expand_as(state.q)
    tau0 = NULL_K * (q_null - state.q) - NULL_D * state.qd
    eye = torch.eye(dyn.n, dtype=J.dtype, device=J.device)
    NT = eye - J.transpose(1, 2) @ Lam @ J @ Minv
    return (NT @ tau0[..., None]).squeeze(-1)


class ResolvedRate(JointVelocityMode):
    """*Cartesian velocity mode* (resolved-rate control, Whitney 1969): the TCP velocity command
    xd_cmd = xd* + Kx e is mapped to joint velocities with a damped pseudo-inverse,
    qd_cmd = J^+ xd_cmd + (I - J^+ J) k0 (q_null - q), then tracked by the inner joint-velocity loop."""

    name, family = "resolved_rate", "task"
    needs = ("cartesian",)

    def __init__(self, dyn, kx: float = 20.0, k_null: float = 2.0, lam: float = 0.02):
        super().__init__(dyn, kx)
        self.k_null, self.lam = k_null, lam

    def velocity_command(self, state, ref):
        p, R, J = self.dyn.jacobian(state.q)
        x_cmd = ref.v + self.kx * pose_error(ref.p, ref.R, p, R)
        Jp = damped_pinv(J, self.lam)
        qd = (Jp @ x_cmd[..., None]).squeeze(-1)
        if self.dyn.n > 6:
            q_null = ref.q_null if ref.q_null is not None else self.dyn.home.expand_as(state.q)
            eye = torch.eye(self.dyn.n, dtype=J.dtype, device=J.device)
            qd = qd + ((eye - Jp @ J) @ (self.k_null * (q_null - state.q))[..., None]).squeeze(-1)
        return torch.maximum(torch.minimum(qd, self.dyn.vel_limit), -self.dyn.vel_limit)


class CartesianImpedance(Controller):
    """Cartesian impedance control (Hogan 1985; Ott 2008):

        tau = J^T (K e + D (xd* - xd) + Lambda xdd*) + C(q, qd) qd + g(q) + N^T tau_null

    The TCP behaves like a 6-D spring-damper around the reference. The damping follows the
    *actual* task inertia: D = 2 zeta diag(sqrt(K_i Lambda_ii)) (the diagonal of the double-
    diagonalisation design), i.e. close to critical damping in every direction and posture."""

    name, family = "cart_impedance", "task"
    needs = ("cartesian",)

    def __init__(self, dyn, K=None, zeta: float = 1.0, feedforward: bool = True):
        super().__init__(dyn)
        self.K = vec(CART_K if K is None else K, dyn.home)
        self.zeta = zeta
        self.ff = feedforward

    def __call__(self, state, ref):
        dyn = self.dyn
        q, qd = state.q, state.qd
        p, R, J = dyn.jacobian(q)
        M = dyn.mass_matrix(q)
        Minv = inv(M)
        Lam = task_inertia(J, Minv)
        D = 2 * self.zeta * torch.sqrt(self.K * torch.diagonal(Lam, dim1=-2, dim2=-1).clamp_min(1e-6))
        e = pose_error(ref.p, ref.R, p, R)
        xd = (J @ qd[..., None]).squeeze(-1)
        F = self.K * e + D * (ref.v - xd)
        if self.ff:
            F = F + (Lam @ ref.a[..., None]).squeeze(-1)
        tau = (J.transpose(1, 2) @ F[..., None]).squeeze(-1) + dyn.bias(q, qd)
        return tau + _null_torque(dyn, state, ref, J, Minv, Lam)


class OperationalSpace(Controller):
    """Operational space control (Khatib 1987): decouple and linearise the TCP dynamics,

        F   = Lambda (xdd* + Kp e + Kd (xd* - xd) - Jdot qd)
        tau = J^T F + h(q, qd) + N^T tau_null

    so that the TCP error obeys e'' + Kd e' + Kp e = 0 in every direction (exact model)."""

    name, family = "osc", "task"
    needs = ("cartesian",)

    def __init__(self, dyn, wn: float = WN_TASK, zeta: float = 1.0):
        super().__init__(dyn)
        self.kp, self.kd = wn**2, 2 * zeta * wn

    def task_force(self, state, ref, J, Lam, p, R):
        e = pose_error(ref.p, ref.R, p, R)
        xd = (J @ state.qd[..., None]).squeeze(-1)
        acc = ref.a + self.kp * e + self.kd * (ref.v - xd) - self.dyn.tcp_bias_acc(state.q, state.qd)
        return (Lam @ acc[..., None]).squeeze(-1)

    def __call__(self, state, ref):
        dyn = self.dyn
        q, qd = state.q, state.qd
        p, R, J = dyn.jacobian(q)
        Minv = inv(dyn.mass_matrix(q))
        Lam = task_inertia(J, Minv)
        F = self.task_force(state, ref, J, Lam, p, R)
        tau = (J.transpose(1, 2) @ F[..., None]).squeeze(-1) + dyn.bias(q, qd)
        return tau + _null_torque(dyn, state, ref, J, Minv, Lam)


class Admittance(Controller):
    """Admittance control: the *measured* external wrench drives a virtual mass-spring-damper,

        Ma xdd_c + Da (xd_c - xd*) + Ka (x_c - x*) = F_ext       (translation; rotation analogous)

    whose state x_c becomes the reference of a stiff inner position loop (here OSC). Stable and
    precise on stiff industrial arms with an F/T sensor; the inverse of impedance control."""

    name, family = "admittance", "force"
    needs = ("cartesian",)

    def __init__(self, dyn, Ma=(4.0, 4.0, 4.0, 0.15, 0.15, 0.15), Ka=CART_K_SOFT, zeta: float = 1.0,
                 deadband: float = 0.5):
        super().__init__(dyn)
        self.Ma = vec(Ma, dyn.home)
        self.Ka = vec(Ka, dyn.home)
        self.Da = 2 * zeta * torch.sqrt(self.Ma * self.Ka)
        self.inner = OperationalSpace(dyn, wn=40.0)
        self.deadband = deadband
        self.x = None  # compliant offset [dp; dtheta] from the reference, and its rate
        self.xd = None

    def reset(self, state, ref):
        B = state.q.shape[0]
        self.x = torch.zeros(B, 6, dtype=state.q.dtype, device=state.q.device)
        self.xd = torch.zeros_like(self.x)

    def __call__(self, state, ref):
        if self.x is None:
            self.reset(state, ref)
        F = state.ft if state.ft is not None else torch.zeros_like(self.x)
        F = torch.sign(F) * (F.abs() - self.deadband).clamp(min=0)  # sensor noise dead band
        xdd = (F - self.Da * self.xd - self.Ka * self.x) / self.Ma
        self._keep("xd", self.xd + xdd * self.dt)
        self._keep("x", self.x + self.xd * self.dt)
        from ..traj import so3_exp

        cref = Ref(q=ref.q, qd=ref.qd, qdd=ref.qdd, p=ref.p + self.x[:, :3],
                   R=so3_exp(self.x[:, 3:]) @ ref.R, v=ref.v + self.xd, a=ref.a + xdd, q_null=ref.q_null)
        self.last_ref = cref
        return self.inner(state, cref)


class HybridForceMotion(OperationalSpace):
    """Hybrid force/motion control (Raibert & Craig 1981) in the operational-space form of Khatib:
    a selection matrix splits the task space into motion directions (OSC tracking) and one force
    direction n (the surface normal), where an explicit force loop acts:

        F = Lambda Omega (xdd* + Kp e + Kd de - Jdot qd)  +  n (f_cmd),
        f_cmd = -(f_d + Kpf (f_d - f) + Kif int(f_d - f)) - Kvf (n . xd)

    f = n . F_ext is the measured contact force (positive when pressing), f_d the desired one.
    Guarded move: until contact is detected (f > 1 N, latched) the normal direction is also
    position-controlled, so the probe touches down at the reference's speed instead of being
    pushed into the surface by the force loop."""

    name, family = "hybrid", "force"
    needs = ("cartesian", "force")

    def __init__(self, dyn, normal=(0.0, 0.0, 1.0), kpf: float = 0.3, kif: float = 6.0, kvf: float = 40.0,
                 wn: float = WN_TASK):
        super().__init__(dyn, wn=wn)
        self.n = vec(normal, dyn.home)
        self.kpf, self.kif, self.kvf = kpf, kif, kvf
        self.integ = None

    def reset(self, state, ref):
        self.integ = torch.zeros(state.q.shape[0], dtype=state.q.dtype, device=state.q.device)
        self.contact = torch.zeros_like(self.integ)

    def __call__(self, state, ref):
        if self.integ is None:
            self.reset(state, ref)
        dyn = self.dyn
        q, qd = state.q, state.qd
        p, R, J = dyn.jacobian(q)
        Minv = inv(dyn.mass_matrix(q))
        Lam = task_inertia(J, Minv)
        n6 = torch.cat([self.n, torch.zeros_like(self.n)])
        f_meas = (state.ft[:, :3] * self.n).sum(-1) if state.ft is not None else torch.zeros_like(self.integ)
        self.contact.copy_(torch.maximum(self.contact, (f_meas > 1.0).to(self.contact.dtype)))
        c = self.contact[:, None, None]
        eye6 = torch.eye(6, dtype=q.dtype, device=q.device)
        Omega = eye6 - c * torch.outer(n6, n6)  # full motion control until contact
        # motion part (only tangential translation + rotation)
        e = pose_error(ref.p, ref.R, p, R)
        xd = (J @ qd[..., None]).squeeze(-1)
        acc = ref.a + self.kp * e + self.kd * (ref.v - xd) - dyn.tcp_bias_acc(q, qd)
        F_motion = (Lam @ (Omega @ acc[..., None])).squeeze(-1)
        # force part along n (active after touch-down)
        f_d = ref.f[:, :3] @ (-self.n) if ref.f is not None else torch.zeros_like(self.integ)  # push into -n
        err = f_d - f_meas
        self._keep("integ", ((self.integ + err * self.dt) * self.contact).clamp(-20, 20))
        v_n = (xd[:, :3] * self.n).sum(-1)
        f_cmd = -(f_d + self.kpf * err + self.kif * self.integ) - self.kvf * v_n
        F = F_motion + (self.contact * f_cmd)[:, None] * n6
        tau = (J.transpose(1, 2) @ F[..., None]).squeeze(-1) + dyn.bias(q, qd)
        return tau + _null_torque(dyn, state, ref, J, Minv, Lam)


class ParallelForcePosition(CartesianImpedance):
    """Parallel force/position control (Chiaverini & Sciavicco 1993): a Cartesian impedance
    controller whose reference is shifted along the force direction by a PI on the force error.
    No selection matrix: where the environment does not resist, the position loop wins; where it
    does, the force integral wins (position error is traded for force accuracy)."""

    name, family = "parallel", "force"
    needs = ("cartesian", "force")

    def __init__(self, dyn, normal=(0.0, 0.0, 1.0), kpf: float = 0.0005, kif: float = 0.02):
        super().__init__(dyn, K=[1500.0, 1500.0, 1500.0, 150.0, 150.0, 150.0])
        self.n = vec(normal, dyn.home)
        self.kpf, self.kif = kpf, kif
        self.integ = None

    def reset(self, state, ref):
        self.integ = torch.zeros(state.q.shape[0], dtype=state.q.dtype, device=state.q.device)

    def __call__(self, state, ref):
        if self.integ is None:
            self.reset(state, ref)
        f_meas = (state.ft[:, :3] * self.n).sum(-1) if state.ft is not None else torch.zeros_like(self.integ)
        f_d = ref.f[:, :3] @ (-self.n) if ref.f is not None else torch.zeros_like(self.integ)
        err = f_d - f_meas
        self._keep("integ", (self.integ + err * self.dt).clamp(-50, 50))
        shift = -(self.kpf * err + self.kif * self.integ)  # move the reference into the surface
        r2 = Ref(q=ref.q, qd=ref.qd, qdd=ref.qdd, p=ref.p + shift[:, None] * self.n, R=ref.R, v=ref.v, a=ref.a,
                 q_null=ref.q_null)
        return super().__call__(state, r2)


__all__ = [
    "Admittance",
    "CartesianImpedance",
    "HybridForceMotion",
    "OperationalSpace",
    "ParallelForcePosition",
    "ResolvedRate",
    "State",
]
