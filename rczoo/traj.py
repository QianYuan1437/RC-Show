"""Trajectory generation: what a robot controller's "MoveJ / MoveL / MoveC" commands turn into.

All profiles are evaluated in closed form at arbitrary times (batched over axes), returning
position, velocity, acceleration and jerk.

* ``quintic``      fifth-order polynomial with boundary position / velocity / acceleration;
* ``Trapezoid``    LSPB / trapezoidal velocity (acceleration-limited, jerk is unbounded);
* ``DoubleS``      seven-segment jerk-limited "S-curve" (Biagiotti & Melchiorri, 2008, ch. 3.4);
* ``sync``         multi-axis synchronisation by time scaling: an axis profile planned with
                   limits (v, a, j) and duration T_k, run at t / lambda with lambda = T / T_k,
                   respects (v / lambda, a / lambda^2, j / lambda^3) -- so every axis ends together;
* ``MoveL / MoveC`` Cartesian straight line / circular arc with SLERP orientation;
* ``blend``        overlap ("superposition") blending of consecutive segments;
* ``toppra``       time-optimal path parameterisation by reachability analysis (Pham & Pham 2018)
                   under joint velocity, acceleration and torque limits;
* ``otg_step``     online time-optimal trajectory generation (acceleration-limited, Reflexxes
                   "Type II"), re-planned every control cycle from the current state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .dynamics import Dynamics, so3_log


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    th = w.norm(dim=-1, keepdim=True).unsqueeze(-1)
    K = torch.zeros(*w.shape[:-1], 3, 3, dtype=w.dtype, device=w.device)
    K[..., 0, 1], K[..., 0, 2] = -w[..., 2], w[..., 1]
    K[..., 1, 0], K[..., 1, 2] = w[..., 2], -w[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -w[..., 1], w[..., 0]
    small = th < 1e-8
    a = torch.where(small, 1 - th**2 / 6, torch.sin(th) / th.clamp_min(1e-12))
    b = torch.where(small, 0.5 - th**2 / 24, (1 - torch.cos(th)) / (th**2).clamp_min(1e-24))
    eye = torch.eye(3, dtype=w.dtype, device=w.device)
    return eye + a * K + b * (K @ K)


@dataclass
class Profile:
    """Samples of a 1-D (or per-axis) profile: each field has the shape of t (and axes)."""

    p: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor
    j: torch.Tensor


# ----------------------------------------------------------------------------- quintic

def quintic(q0, q1, T, t, v0=0.0, v1=0.0, a0=0.0, a1=0.0) -> Profile:
    """Quintic polynomial from (q0, v0, a0) to (q1, v1, a1) in time T, evaluated at t (clamped)."""
    q0, q1 = torch.as_tensor(q0), torch.as_tensor(q1)
    T = torch.as_tensor(T, dtype=q0.dtype)
    t = torch.as_tensor(t, dtype=q0.dtype).clamp(min=0)
    tt = torch.minimum(t, T)
    h = q1 - q0
    c0, c1, c2 = q0, torch.as_tensor(v0, dtype=q0.dtype), torch.as_tensor(a0, dtype=q0.dtype) / 2
    v1, a1 = torch.as_tensor(v1, dtype=q0.dtype), torch.as_tensor(a1, dtype=q0.dtype)
    a0_ = 2 * c2
    c3 = (20 * h - (8 * v1 + 12 * c1) * T - (3 * a0_ - a1) * T**2) / (2 * T**3)
    c4 = (-30 * h + (14 * v1 + 16 * c1) * T + (3 * a0_ - 2 * a1) * T**2) / (2 * T**4)
    c5 = (12 * h - 6 * (v1 + c1) * T + (a1 - a0_) * T**2) / (2 * T**5)
    p = c0 + c1 * tt + c2 * tt**2 + c3 * tt**3 + c4 * tt**4 + c5 * tt**5
    v = c1 + 2 * c2 * tt + 3 * c3 * tt**2 + 4 * c4 * tt**3 + 5 * c5 * tt**4
    a = 2 * c2 + 6 * c3 * tt + 12 * c4 * tt**2 + 20 * c5 * tt**3
    j = 6 * c3 + 24 * c4 * tt + 60 * c5 * tt**2
    done = t >= T
    # past the end: hold the final state (only meaningful for v1 = a1 = 0)
    p = torch.where(done, q1 + v1 * (t - T), p)
    v = torch.where(done, v1 + 0 * v, v)
    a = torch.where(done, torch.zeros_like(a), a)
    j = torch.where(done, torch.zeros_like(j), j)
    return Profile(p, v, a, j)


def quintic_min_time(dist, vmax, amax):
    """Shortest T for a rest-to-rest quintic: peak |v| = 15 d / (8 T), peak |a| = 10 d / (sqrt(3) T^2)."""
    d = torch.as_tensor(dist).abs()
    return torch.maximum(15 * d / (8 * torch.as_tensor(vmax)), torch.sqrt(10 * d / (math.sqrt(3) * torch.as_tensor(amax))))


# ----------------------------------------------------------------------------- trapezoid

class Trapezoid:
    """Rest-to-rest trapezoidal velocity profile per axis (axes in the last dim)."""

    def __init__(self, q0, q1, vmax, amax):
        self.q0 = torch.as_tensor(q0, dtype=torch.float64)
        self.q1 = torch.as_tensor(q1, dtype=torch.float64)
        h = self.q1 - self.q0
        self.sign = torch.where(h >= 0, 1.0, -1.0).to(h.dtype)
        self.h = h.abs()
        v = torch.as_tensor(vmax, dtype=h.dtype).expand_as(h)
        a = torch.as_tensor(amax, dtype=h.dtype).expand_as(h)
        tri = self.h < v**2 / a  # cannot reach vmax
        self.v = torch.where(tri, torch.sqrt(self.h * a), v)
        self.a = a
        self.Ta = self.v / a
        self.T = torch.where(tri, 2 * self.Ta, self.h / v + v / a)
        self.T = torch.where(self.h < 1e-12, torch.zeros_like(self.T), self.T)

    @property
    def duration(self) -> torch.Tensor:
        return self.T.max() if self.T.numel() else torch.tensor(0.0)

    def eval_own(self, t) -> Profile:
        t = torch.as_tensor(t, dtype=self.h.dtype)
        T, Ta, v, a = self.T, self.Ta, self.v, self.a
        t = torch.minimum(t.clamp(min=0), T)
        acc = t < Ta
        dec = t > T - Ta
        p = torch.where(acc, 0.5 * a * t**2, torch.where(dec, self.h - 0.5 * a * (T - t) ** 2, v * (t - Ta / 2)))
        vel = torch.where(acc, a * t, torch.where(dec, a * (T - t), v))
        ac = torch.where(acc, a, torch.where(dec, -a, torch.zeros_like(a)))
        ac = torch.where(T > 0, ac, torch.zeros_like(ac))
        vel = torch.where(T > 0, vel, torch.zeros_like(vel))
        p = torch.where(T > 0, p, self.h)
        s = self.sign
        return Profile(self.q0 + s * p, s * vel, s * ac, torch.zeros_like(p))


# ----------------------------------------------------------------------------- double S

class DoubleS:
    """Rest-to-rest jerk-limited (seven-segment) profile per axis."""

    def __init__(self, q0, q1, vmax, amax, jmax):
        self.q0 = torch.as_tensor(q0, dtype=torch.float64)
        self.q1 = torch.as_tensor(q1, dtype=torch.float64)
        h = self.q1 - self.q0
        self.sign = torch.where(h >= 0, 1.0, -1.0).to(h.dtype)
        self.h = H = h.abs()
        V = torch.as_tensor(vmax, dtype=H.dtype).expand_as(H)
        A = torch.as_tensor(amax, dtype=H.dtype).expand_as(H)
        J = torch.as_tensor(jmax, dtype=H.dtype).expand_as(H)
        # 1) assume vmax is reached
        reach_a = V * J >= A**2
        Tj = torch.where(reach_a, A / J, torch.sqrt(V / J))
        Ta = torch.where(reach_a, Tj + V / A, 2 * Tj)
        Tv = H / V - Ta
        # 2) vmax not reached: no cruise
        no_cruise = Tv <= 0
        reach_a2 = H >= 2 * A**3 / J**2
        Tj2 = torch.where(reach_a2, A / J, torch.pow(H / (2 * J), 1 / 3))
        Ta2 = torch.where(reach_a2, Tj2 / 2 + torch.sqrt((Tj2 / 2) ** 2 + H / A), 2 * Tj2)
        self.Tj = torch.where(no_cruise, Tj2, Tj)
        self.Ta = torch.where(no_cruise, Ta2, Ta)
        self.Tv = torch.where(no_cruise, torch.zeros_like(Tv), Tv)
        self.J = J
        self.alim = J * self.Tj
        self.vlim = (self.Ta - self.Tj) * self.alim
        self.T = 2 * self.Ta + self.Tv
        zero = H < 1e-12
        self.T = torch.where(zero, torch.zeros_like(self.T), self.T)

    @property
    def duration(self) -> torch.Tensor:
        return self.T.max()

    def _acc_phase(self, tau):
        J, Tj, Ta, alim, vlim = self.J, self.Tj, self.Ta, self.alim, self.vlim
        s1 = tau < Tj
        s3 = tau > Ta - Tj
        sig = Ta - tau
        p1, v1, a1, j1 = J * tau**3 / 6, J * tau**2 / 2, J * tau, J
        p2 = alim * (3 * tau**2 - 3 * Tj * tau + Tj**2) / 6
        v2, a2, j2 = alim * (tau - Tj / 2), alim, torch.zeros_like(J)
        p3 = vlim * Ta / 2 - vlim * sig + J * sig**3 / 6
        v3, a3, j3 = vlim - J * sig**2 / 2, J * sig, -J
        pick = lambda x1, x2, x3: torch.where(s1, x1, torch.where(s3, x3, x2))
        return pick(p1, p2, p3), pick(v1, v2, v3), pick(a1, a2, a3), pick(j1 + 0 * tau, j2 + 0 * tau, j3 + 0 * tau)

    def eval_own(self, t) -> Profile:
        t = torch.as_tensor(t, dtype=self.h.dtype)
        T, Ta, Tv = self.T, self.Ta, self.Tv
        t = torch.minimum(t.clamp(min=0), T)
        pa, va, aa, ja = self._acc_phase(torch.minimum(t, Ta))
        pd, vd, ad, jd = self._acc_phase((T - t).clamp(max=1e9))
        acc, cruise = t < Ta, (t >= Ta) & (t < Ta + Tv)
        pA = self.vlim * Ta / 2
        p = torch.where(acc, pa, torch.where(cruise, pA + self.vlim * (t - Ta), self.h - pd))
        v = torch.where(acc, va, torch.where(cruise, self.vlim + 0 * t, vd))
        a = torch.where(acc, aa, torch.where(cruise, 0 * t, -ad))
        j = torch.where(acc, ja, torch.where(cruise, 0 * t, jd))
        live = T > 0
        z = torch.zeros_like(p)
        s = self.sign
        return Profile(self.q0 + s * torch.where(live, p, self.h),
                       s * torch.where(live, v, z), s * torch.where(live, a, z), s * torch.where(live, j, z))


def sync(profile, t) -> Profile:
    """Evaluate a per-axis profile with every axis stretched to the slowest axis' duration."""
    T = profile.duration
    lam = torch.where(profile.T > 0, T / profile.T.clamp_min(1e-12), torch.ones_like(profile.T))
    t = torch.as_tensor(t, dtype=lam.dtype)
    pr = profile.eval_own(t / lam)
    return Profile(pr.p, pr.v / lam, pr.a / lam**2, pr.j / lam**3)


# ----------------------------------------------------------------------------- Cartesian

class MoveL:
    """Straight line + SLERP between two TCP poses with a shared scalar S-curve s(t) in [0, 1]."""

    def __init__(self, p0, R0, p1, R1, v_lin=0.25, a_lin=1.0, j_lin=8.0, v_ang=1.0, a_ang=4.0, j_ang=30.0):
        self.p0, self.R0 = torch.as_tensor(p0), torch.as_tensor(R0)
        self.dp = torch.as_tensor(p1) - self.p0
        self.w = so3_log(self.R0.transpose(-1, -2) @ torch.as_tensor(R1))  # body-frame rotation vector
        L, ang = float(self.dp.norm()), float(self.w.norm())
        # the scalar profile runs on s in [0, 1]; scale the limits to path length and angle
        lim = []
        for dist, v, a, j in ((L, v_lin, a_lin, j_lin), (ang, v_ang, a_ang, j_ang)):
            if dist > 1e-9:
                lim.append(DoubleS(0.0, 1.0, v / dist, a / dist, j / dist))
        self.prof = max(lim, key=lambda pr: float(pr.T)) if lim else DoubleS(0.0, 1.0, 1.0, 1.0, 1.0)
        self.T = float(self.prof.T)

    def __call__(self, t):
        """Pose, twist (world: [v; w]) and acceleration at times t (K,) -> (K,3), (K,3,3), (K,6), (K,6)."""
        s = self.prof.eval_own(torch.as_tensor(t, dtype=torch.float64))
        p = self.p0 + s.p[:, None] * self.dp
        R = self.R0 @ so3_exp(s.p[:, None] * self.w)
        w_world = (self.R0 @ self.w)
        vel = torch.cat([s.v[:, None] * self.dp, s.v[:, None] * w_world], -1)
        acc = torch.cat([s.a[:, None] * self.dp, s.a[:, None] * w_world], -1)
        return p, R, vel, acc


class MoveC:
    """Circular arc through p0 -> p_via -> p1, orientation SLERP R0 -> R1, S-curve in arc length."""

    def __init__(self, p0, p_via, p1, R0, R1, v_lin=0.25, a_lin=1.0, j_lin=8.0):
        a, b, c = (np.asarray(x, dtype=float) for x in (p0, p_via, p1))
        u, w = b - a, c - a
        n = np.cross(u, w)
        if np.linalg.norm(n) < 1e-9:
            raise ValueError("MoveC: the three points are collinear")
        # circumcentre
        center = a + (np.dot(w, w) * np.cross(n, u) + np.dot(u, u) * np.cross(w, n)) / (2 * np.dot(n, n))
        r = np.linalg.norm(a - center)
        e1 = (a - center) / r
        e3 = n / np.linalg.norm(n)
        e2 = np.cross(e3, e1)
        ang = lambda x: math.atan2(np.dot(x - center, e2), np.dot(x - center, e1)) % (2 * math.pi)
        self.theta = ang(c)
        self.center, self.r, self.e1, self.e2 = (torch.as_tensor(x) for x in (center, r, e1, e2))
        self.R0 = torch.as_tensor(R0)
        self.w = so3_log(self.R0.T @ torch.as_tensor(R1))
        L = r * self.theta
        self.prof = DoubleS(0.0, 1.0, v_lin / L, a_lin / L, j_lin / L)
        self.T = float(self.prof.T)

    def __call__(self, t):
        s = self.prof.eval_own(torch.as_tensor(t, dtype=torch.float64))
        th = s.p * self.theta
        c, sn = torch.cos(th)[:, None], torch.sin(th)[:, None]
        p = self.center + self.r * (c * self.e1 + sn * self.e2)
        dpdth = self.r * (-sn * self.e1 + c * self.e2)
        d2 = self.r * (-c * self.e1 - sn * self.e2)
        thd, thdd = s.v * self.theta, s.a * self.theta
        v = dpdth * thd[:, None]
        a = dpdth * thdd[:, None] + d2 * (thd**2)[:, None]
        R = self.R0 @ so3_exp(s.p[:, None] * self.w)
        ww = self.R0 @ self.w
        return p, R, torch.cat([v, s.v[:, None] * ww], -1), torch.cat([a, s.a[:, None] * ww], -1)


def blend_positions(seg_a, seg_b, overlap: float):
    """Superposition blending of two position trajectories x_a(t) (ends at the corner) and x_b(t)
    (starts at the corner): segment b starts `overlap` seconds before a ends and their *displacements*
    add up, so the path rounds the corner with continuous velocity and acceleration.
    Returns (f(t) -> (p, v, a), total duration)."""
    Ta, Tb = seg_a.T, seg_b.T
    overlap = min(overlap, Ta, Tb)
    start_b = Ta - overlap

    def f(t):
        t = torch.as_tensor(t, dtype=torch.float64)
        pa, _, va, aa = seg_a(t)
        pb, _, vb, ab = seg_b((t - start_b).clamp(min=0))
        on_b = (t >= start_b)[:, None]
        corner = seg_b(torch.zeros(1, dtype=torch.float64))[0]
        p = pa + torch.where(on_b, pb - corner, torch.zeros_like(pb))
        v = va[:, :3] + torch.where(on_b, vb[:, :3], torch.zeros_like(pb))
        a = aa[:, :3] + torch.where(on_b, ab[:, :3], torch.zeros_like(pb))
        return p, v, a

    return f, start_b + Tb


# ----------------------------------------------------------------------------- TOPP-RA

@dataclass
class ToppResult:
    s: np.ndarray  # grid (N+1,)
    sd: np.ndarray  # path velocity at the grid
    K: np.ndarray  # controllable sets of x = sd^2 (N+1, 2)
    mvc: np.ndarray  # maximum velocity curve (sd) from the stage constraints alone
    t: np.ndarray  # time at each grid point
    duration: float


def _lp2(Acon, bcon, objective):
    """max objective . (u, x) s.t. Acon (u, x) <= bcon, by vertex enumeration (2 variables)."""
    m = Acon.shape[0]
    i, j = np.triu_indices(m, 1)
    A1, A2 = Acon[i], Acon[j]
    det = A1[:, 0] * A2[:, 1] - A1[:, 1] * A2[:, 0]
    ok = np.abs(det) > 1e-12
    if not ok.any():
        return None
    i, j, det = i[ok], j[ok], det[ok]
    u = (bcon[i] * Acon[j, 1] - bcon[j] * Acon[i, 1]) / det
    x = (Acon[i, 0] * bcon[j] - Acon[j, 0] * bcon[i]) / det
    P = np.stack([u, x], 1)
    feas = (P @ Acon.T <= bcon + 1e-9).all(1)
    if not feas.any():
        return None
    vals = P[feas] @ objective
    k = int(np.argmax(vals))
    return P[feas][k]


def toppra(dyn: Dynamics, path_q, path_dq, path_ddq, s_grid, vmax, amax, tau_max=None) -> ToppResult:
    """Time-optimal parameterisation of a joint path q(s), s in [0, 1].

    Constraints per grid point i (x = sd^2, u = sdd):
        |q'(s) sd| <= vmax                       ->  q'^2 x <= vmax^2
        |q'(s) u + q''(s) x| <= amax
        |m(s) u + c(s) x + g(s)| <= tau_max      with m = M q', c = M q'' + C(q, q') q'
    Backward pass: controllable sets K_i; forward pass: greedy maximal u (Pham & Pham 2018).
    """
    N = len(s_grid) - 1
    ds = np.diff(s_grid)
    qp, qpp = np.asarray(path_dq), np.asarray(path_ddq)
    n = qp.shape[1]
    vmax = np.broadcast_to(np.asarray(vmax, float), (n,))
    amax = np.broadcast_to(np.asarray(amax, float), (n,))
    rows_a, rows_b, rows_c = [qp, -qp], [qpp, -qpp], [np.broadcast_to(-amax, qp.shape)] * 2
    if tau_max is not None:
        q = torch.as_tensor(np.asarray(path_q))
        m = dyn.rnea(q, torch.zeros_like(q), torch.as_tensor(qp), gravity=False).numpy()
        c = dyn.rnea(q, torch.as_tensor(qp), torch.as_tensor(qpp), gravity=False).numpy()  # M q'' + C(q,q') q'
        g = dyn.gravity_torque(q).numpy()
        tm = np.broadcast_to(np.asarray(tau_max, float), (n,))
        rows_a += [m, -m]
        rows_b += [c, -c]
        rows_c += [g - tm, -g - tm]
    # stage i constraint rows:  a u + b x + c <= 0
    Aa = np.concatenate(rows_a, 1)
    Ab = np.concatenate(rows_b, 1)
    Ac = np.concatenate(rows_c, 1)
    xmax_vel = np.min(np.where(qp**2 > 1e-12, vmax**2 / np.maximum(qp**2, 1e-12), np.inf), 1)

    def stage(i):
        A = np.stack([Aa[i], Ab[i]], 1)
        b = -Ac[i]
        A = np.concatenate([A, [[0, 1], [0, -1]]], 0)  # x <= xmax_vel, x >= 0
        b = np.concatenate([b, [min(xmax_vel[i], 1e6), 0]])
        return A, b

    # maximum velocity curve from each stage's constraints alone
    def top(i):
        r = _lp2(*stage(i), np.array([0.0, 1.0]))
        return 0.0 if r is None else r[1]

    mvc = np.array([top(i) for i in range(N + 1)])
    K = np.zeros((N + 1, 2))
    K[N] = [0, 0]  # come to rest at the end
    for i in range(N - 1, -1, -1):
        A, b = stage(i)
        # x + 2 ds u in K_{i+1}
        A2 = np.concatenate([A, [[2 * ds[i], 1], [-2 * ds[i], -1]]], 0)
        b2 = np.concatenate([b, [K[i + 1, 1], -K[i + 1, 0]]])
        hi = _lp2(A2, b2, np.array([0.0, 1.0]))
        lo = _lp2(A2, b2, np.array([0.0, -1.0]))
        if hi is None:
            raise RuntimeError(f"TOPP-RA: path not controllable at stage {i}")
        K[i] = [max(lo[1], 0), hi[1]]
    x = np.zeros(N + 1)
    x[0] = 0.0  # start at rest
    for i in range(N):
        A, b = stage(i)
        # 1-D LP in u with x fixed: A[:,0] u <= b - A[:,1] x, and x + 2 ds u in K_{i+1}
        rhs = b - A[:, 1] * x[i]
        ub, lb = np.inf, -np.inf
        for ai, ri in zip(A[:, 0], rhs):
            if ai > 1e-12:
                ub = min(ub, ri / ai)
            elif ai < -1e-12:
                lb = max(lb, ri / ai)
        ub = min(ub, (K[i + 1, 1] - x[i]) / (2 * ds[i]))
        lb = max(lb, (K[i + 1, 0] - x[i]) / (2 * ds[i]))
        u = max(ub, lb)
        x[i + 1] = max(x[i] + 2 * ds[i] * u, 0.0)
    sd = np.sqrt(np.maximum(x, 0))
    # time from ds / sd (trapezoidal in 1/sd, exact for constant sdd: dt = 2 ds / (sd_i + sd_{i+1}))
    dt = 2 * ds / np.maximum(sd[:-1] + sd[1:], 1e-9)
    t = np.concatenate([[0], np.cumsum(dt)])
    return ToppResult(np.asarray(s_grid), sd, K, np.sqrt(np.maximum(mvc, 0)), t, float(t[-1]))


# ----------------------------------------------------------------------------- online OTG

def otg_step(p, v, target, vmax, amax, dt):
    """One cycle of acceleration-limited time-optimal OTG per axis (Reflexxes Type II without
    phase synchronisation): from the current (p, v), apply the bang-coast-bang acceleration that
    reaches (target, 0) fastest, integrate one step, re-plan next cycle. Batched tensors."""
    d = target - p
    # stopping distance with full deceleration
    d_stop = v * v.abs() / (2 * amax)
    # direction of the time-optimal motion
    s = torch.where(d - d_stop >= 0, 1.0, -1.0).to(p.dtype)
    v_toward = s * v
    # braking now is needed if the remaining distance equals the stopping distance
    remaining = (d * s).clamp(min=0)
    brake = v_toward > 0
    need_brake = brake & (v_toward**2 / (2 * amax) >= remaining - 1e-12)
    a = torch.where(need_brake, -s * torch.minimum(amax, v_toward**2 / (2 * remaining.clamp_min(1e-12))),
                    torch.where(v_toward < vmax, s * amax, torch.where(v_toward > vmax, -s * amax, 0 * p)))
    # do not overshoot the cruise speed in one step
    v_new = v + a * dt
    v_new = torch.where((~need_brake) & (s * v_new > vmax) & (v_toward <= vmax), s * vmax, v_new)
    p_new = p + 0.5 * (v + v_new) * dt
    # snap at the target
    arrived = (d.abs() < 1e-6) & (v.abs() < amax * dt)
    p_new = torch.where(arrived, target, p_new)
    v_new = torch.where(arrived, torch.zeros_like(v_new), v_new)
    acc = (v_new - v) / dt
    return p_new, v_new, acc
