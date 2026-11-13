"""Joint-space controllers: what "position mode", "velocity mode" and "torque mode" of an arm
controller do internally, and the classic model-based / robust / adaptive laws on top of them.

Notation: e = q_d - q, de = qd_d - qd, h(q, qd) = C(q, qd) qd + g(q).
"""

from __future__ import annotations

import torch

from ..dynamics import Dynamics
from .base import Controller, Ref, State, vec
from .gains import GAINS, WN_JOINT


def _gains(dyn: Dynamics, key: str, like: torch.Tensor | None = None):
    v = GAINS[dyn.robot.name][key]
    return torch.as_tensor(v, dtype=dyn.dtype, device=dyn.device)


class JointPID(Controller):
    """Independent-joint PID, no model at all:  tau = Kp e + Ki int(e) + Kd de.
    The integral is what holds the arm up against gravity -- slowly, with overshoot."""

    name, family = "pid", "joint"

    def __init__(self, dyn, kp=None, ki=None, kd=None, i_limit_frac: float = 0.5):
        super().__init__(dyn)
        self.kp = _gains(dyn, "kp") if kp is None else vec(kp, dyn.home)
        self.ki = _gains(dyn, "ki") if ki is None else vec(ki, dyn.home)
        self.kd = _gains(dyn, "kd") if kd is None else vec(kd, dyn.home)
        self.i_lim = i_limit_frac * dyn.torque_limit
        self.integ = None

    def reset(self, state, ref):
        self.integ = torch.zeros_like(state.q)

    def __call__(self, state: State, ref: Ref):
        e, de = ref.q - state.q, ref.qd - state.qd
        if self.integ is None:
            self.reset(state, ref)
        # conditional integration (anti-windup): clamp the integral torque
        integ = self._keep("integ", torch.maximum(torch.minimum(self.integ + self.ki * e * self.dt, self.i_lim), -self.i_lim))
        return self.kp * e + self.kd * de + integ


class PDGravity(Controller):
    """PD + gravity compensation (Takegaki & Arimoto 1981): tau = Kp e + Kd de + g(q).
    The standard implementation of an arm's *position mode*; globally asymptotically stable for
    set-points, lags during fast tracking because inertia and Coriolis terms are not fed forward."""

    name, family = "pd_grav", "joint"

    def __init__(self, dyn, kp=None, kd=None):
        super().__init__(dyn)
        self.kp = _gains(dyn, "kp") if kp is None else vec(kp, dyn.home)
        self.kd = _gains(dyn, "kd") if kd is None else vec(kd, dyn.home)

    def __call__(self, state, ref):
        return self.kp * (ref.q - state.q) + self.kd * (ref.qd - state.qd) + self.dyn.gravity_torque(state.q)


class PDFeedforward(Controller):
    """PD+ (Paden & Panja 1988): tau = M(q) qdd_d + C(q, qd) qd_d + g(q) + Kp e + Kd de.
    Full model feedforward along the reference, fixed joint-space PD feedback."""

    name, family = "pd_ff", "joint"

    def __init__(self, dyn, kp=None, kd=None):
        super().__init__(dyn)
        self.kp = _gains(dyn, "kp") if kp is None else vec(kp, dyn.home)
        self.kd = _gains(dyn, "kd") if kd is None else vec(kd, dyn.home)

    def __call__(self, state, ref):
        q, qd = state.q, state.qd
        ff = self.dyn.slotine_li(q, qd, ref.qd, ref.qdd)  # M qdd_d + C(q, qd) qd_d + g in one pass
        return ff + self.kp * (ref.q - q) + self.kd * (ref.qd - qd)


class ComputedTorque(Controller):
    """Computed torque / inverse dynamics control: tau = M(q) (qdd_d + Kp e + Kd de) + h(q, qd).
    With an exact model the closed loop is n decoupled linear systems  e'' + Kd e' + Kp e = 0;
    Kp = wn^2, Kd = 2 zeta wn set the same bandwidth on every joint."""

    name, family = "ctc", "joint"

    def __init__(self, dyn, wn: float = WN_JOINT, zeta: float = 1.0, ki: float = 0.0):
        super().__init__(dyn)
        self.kp, self.kd, self.ki = wn**2, 2 * zeta * wn, ki
        self.integ = None

    def reset(self, state, ref):
        self.integ = torch.zeros_like(state.q)

    def __call__(self, state, ref):
        q, qd = state.q, state.qd
        e, de = ref.q - q, ref.qd - qd
        v = ref.qdd + self.kp * e + self.kd * de
        if self.ki:
            if self.integ is None:
                self.reset(state, ref)
            v = v + self.ki * self._keep("integ", self.integ + e * self.dt)
        M = self.dyn.mass_matrix(q)
        return (M @ v[..., None]).squeeze(-1) + self.dyn.bias(q, qd)


class SlidingMode(Controller):
    """Sliding-mode control (Slotine & Li 1991, ch. 9) with a boundary layer.

    qd_r = qd_d + Lambda e,  s = qd - qd_r  (s = 0 is the sliding surface: e decays with rate Lambda)
    tau = M qdd_r + C qd_r + g - Kd s - eta sat(s / phi)
    The switching term eta dominates any bounded model error (unknown payload, friction) once
    eta exceeds it; the boundary layer phi trades chattering for a small residual error."""

    name, family = "smc", "joint"

    def __init__(self, dyn, lam: float = 20.0, kd_scale: float = 1.0, eta_frac: float = 0.12, phi: float = 0.05):
        super().__init__(dyn)
        self.lam = lam
        self.kd = kd_scale * _gains(dyn, "kd") * 2.0
        self.eta = eta_frac * dyn.torque_limit
        self.phi = phi

    def __call__(self, state, ref):
        q, qd = state.q, state.qd
        e, de = ref.q - q, ref.qd - qd
        qd_r = ref.qd + self.lam * e
        qdd_r = ref.qdd + self.lam * de
        s = qd - qd_r
        tau = self.dyn.slotine_li(q, qd, qd_r, qdd_r)
        return tau - self.kd * s - self.eta * torch.clamp(s / self.phi, -1.0, 1.0)


class AdaptivePayload(Controller):
    """Adaptive control (Slotine & Li 1987) estimating the 10 inertial parameters of an unknown
    payload rigidly attached to the last link.

    The payload's dynamics are linear in pi = [m, m c, I_o] (mass, first moment, inertia about the
    last link's origin, all in the link frame):  M_pi qdd_r + C_pi qd_r + g_pi = Y(q, qd, qd_r, qdd_r) pi
    (``Dynamics.payload_regressor``, Christoffel C, so dM_pi/dt - 2 C_pi stays skew-symmetric).
    Control and update:

        tau  = M0 qdd_r + C0 qd_r + g0 + Y pi_hat - Kd s,     s = qd - qd_r
        dpi_hat/dt = -Gamma Y^T s
    V = 1/2 s^T M s + 1/2 pi~^T Gamma^-1 pi~ is non-increasing, hence s -> 0 and e -> 0."""

    name, family = "adaptive", "joint"

    def __init__(self, dyn, lam: float = 20.0, gamma=None):
        super().__init__(dyn)
        self.lam = lam
        self.kd = _gains(dyn, "kd") * 2.0
        g = torch.tensor([2.0, 0.2, 0.2, 0.2, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=dyn.dtype, device=dyn.device) \
            if gamma is None else vec(gamma, dyn.home)
        self.gamma = g * 5.0
        self.pi_hat = None

    def reset(self, state, ref):
        self.pi_hat = torch.zeros(state.q.shape[0], 10, dtype=state.q.dtype, device=state.q.device)

    def __call__(self, state, ref):
        if self.pi_hat is None:
            self.reset(state, ref)
        q, qd = state.q, state.qd
        e, de = ref.q - q, ref.qd - qd
        qd_r, qdd_r = ref.qd + self.lam * e, ref.qdd + self.lam * de
        s = qd - qd_r
        tau0 = self.dyn.slotine_li(q, qd, qd_r, qdd_r)
        Y = self.dyn.payload_regressor(q, qd, qd_r, qdd_r)
        tau = tau0 + (Y @ self.pi_hat[..., None]).squeeze(-1) - self.kd * s
        pi = self.pi_hat - self.dt * self.gamma * (Y.transpose(1, 2) @ s[..., None]).squeeze(-1)
        pi = torch.cat([pi[:, :1].clamp(min=0.0), pi[:, 1:]], 1)  # mass >= 0
        self._keep("pi_hat", pi)
        return tau

    @property
    def payload_mass(self) -> torch.Tensor:
        return self.pi_hat[:, 0]


class JointVelocityMode(Controller):
    """*Velocity mode*: an outer position loop produces a velocity command, an inner PI joint-
    velocity loop (what a servo drive runs) tracks it, with gravity feedforward:

        qd_cmd = clip(qd_d + Kx e, +-v_max)
        tau    = Kv (qd_cmd - qd) + Ki int(qd_cmd - qd) + g(q)"""

    name, family = "velocity", "joint"

    def __init__(self, dyn, kx: float = 20.0):
        super().__init__(dyn)
        self.kx = kx
        self.kv, self.kvi = _gains(dyn, "kv"), _gains(dyn, "kvi")
        self.integ = None

    def reset(self, state, ref):
        self.integ = torch.zeros_like(state.q)

    def velocity_command(self, state, ref):
        v = ref.qd + self.kx * (ref.q - state.q)
        return torch.maximum(torch.minimum(v, self.dyn.vel_limit), -self.dyn.vel_limit)

    def inner(self, state, qd_cmd):
        if self.integ is None:
            self.integ = torch.zeros_like(state.q)
        ev = qd_cmd - state.qd
        lim = 0.5 * self.dyn.torque_limit
        integ = self._keep("integ", torch.maximum(torch.minimum(self.integ + self.kvi * ev * self.dt, lim), -lim))
        return self.kv * ev + integ + self.dyn.gravity_torque(state.q)

    def __call__(self, state, ref):
        return self.inner(state, self.velocity_command(state, ref))


class JointImpedance(Controller):
    """Joint impedance mode (Franka's default): a compliant spring-damper per joint with Coriolis
    and gravity compensation  tau = K e + D de + C(q, qd) qd + g(q)  (+ optional M qdd_d).
    Low stiffness: the arm yields to contacts and pushes; tracking error grows with speed."""

    name, family = "joint_impedance", "joint"

    def __init__(self, dyn, k=None, d=None, feedforward: bool = True):
        super().__init__(dyn)
        self.k = _gains(dyn, "imp_k") if k is None else vec(k, dyn.home)
        self.d = _gains(dyn, "imp_d") if d is None else vec(d, dyn.home)
        self.ff = feedforward

    def __call__(self, state, ref):
        q, qd = state.q, state.qd
        tau = self.k * (ref.q - q) + self.d * (ref.qd - qd) + self.dyn.bias(q, qd)
        if self.ff:
            tau = tau + (self.dyn.mass_matrix(q) @ ref.qdd[..., None]).squeeze(-1)
        return tau


class GravityOnly(Controller):
    """Torque mode with gravity compensation only (the arm floats; baseline for interaction)."""

    name, family = "gravity", "interaction"
    needs = ()

    def __call__(self, state, ref):
        return self.dyn.gravity_torque(state.q)
