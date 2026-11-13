"""Physical human-robot interaction: hand guiding (drag teaching) and collision detection.
"""

from __future__ import annotations

import torch

from .base import Controller, Ref, State, vec


class DragTeaching(Controller):
    """Hand guiding / *drag teaching* in torque mode:

        tau = g(q) + f_c tanh(qd / v_s) + (Coulomb compensation)  - D qd  + tau_walls

    Gravity compensation makes the arm weightless; compensating the (identified) dry friction
    with a smooth sign makes it *light*; a small viscous damping keeps it from drifting. Virtual
    walls (joint-limit springs and a Cartesian box) keep the operator inside the safe workspace."""

    name, family = "drag", "interaction"
    needs = ()

    def __init__(self, dyn, friction_comp=None, damping: float = 0.8, v_smooth: float = 0.02,
                 wall_k: float = 200.0, box=None, box_k: float = 3000.0):
        super().__init__(dyn)
        self.fc = vec(0.0 if friction_comp is None else friction_comp, dyn.home).expand(dyn.n)
        self.damp = damping
        self.vs = v_smooth
        self.wall_k = wall_k
        self.box = None if box is None else vec(box, dyn.home)  # (2, 3): lower, upper corner
        self.box_k = box_k

    def __call__(self, state: State, ref: Ref):
        dyn = self.dyn
        q, qd = state.q, state.qd
        tau = dyn.gravity_torque(q) + self.fc * torch.tanh(qd / self.vs) - self.damp * qd
        # joint-limit walls, active in the last 5 degrees
        margin = 0.087
        lo, hi = dyn.lower + margin, dyn.upper - margin
        tau = tau + self.wall_k * ((lo - q).clamp(min=0) - (q - hi).clamp(min=0))
        if self.box is not None:
            p, _, J = dyn.jacobian(q)
            pen = (self.box[0] - p).clamp(min=0) - (p - self.box[1]).clamp(min=0)
            v = (J[:, :3] @ qd[..., None]).squeeze(-1)
            F = self.box_k * pen - 60.0 * v * (pen != 0)
            tau = tau + (J[:, :3].transpose(1, 2) @ F[..., None]).squeeze(-1)
        return tau


class MomentumObserver:
    """Generalised-momentum observer (De Luca & Mattone 2003; Haddadin et al. 2017):

        r(t) = K_O ( p(t) - p(0) - int_0^t (tau + C^T(q, qd) qd - g(q) - tau_f + r) ds ),   p = M(q) qd

    r is a first-order low-pass estimate of the external joint torque, dr/dt = K_O (tau_ext - r),
    without joint accelerations or inverting M. A collision is declared when |r_i| > threshold_i;
    the joint pattern isolates the link, and J^T-pseudo-inverse of r estimates a TCP force."""

    def __init__(self, dyn, k_o: float = 50.0, threshold=None, friction=None):
        self.dyn = dyn
        self.k = k_o
        self.th = vec(threshold if threshold is not None else 0.1 * dyn.torque_limit.cpu().numpy(), dyn.home)
        self.fc = None if friction is None else vec(friction, dyn.home)
        self.r = None
        self.integ = None
        self.p0 = None

    def reset(self, state: State):
        M = self.dyn.mass_matrix(state.q)
        self.p0 = (M @ state.qd[..., None]).squeeze(-1)
        self.r = torch.zeros_like(state.q)
        self.integ = torch.zeros_like(state.q)

    def update(self, state: State, tau: torch.Tensor, dt: float = 1e-3, active: torch.Tensor | None = None) -> torch.Tensor:
        """tau: torque applied during the last step. `active` (0/1) masks the first call."""
        if self.r is None:
            self.reset(state)
        dyn = self.dyn
        q, qd = state.q, state.qd
        M = dyn.mass_matrix(q)
        C = dyn.coriolis_matrix(q, qd)
        beta = dyn.gravity_torque(q) - (C.transpose(1, 2) @ qd[..., None]).squeeze(-1)
        # the plant's viscous damping is known (MJCF); dry friction only if identified
        tau_f = dyn.damping * qd
        if self.fc is not None:
            tau_f = tau_f + self.fc * torch.tanh(qd / 0.02)
        inc = (tau - beta - tau_f + self.r) * dt
        if active is not None:
            inc = inc * active
        self.integ.copy_(self.integ + inc)
        p = (M @ qd[..., None]).squeeze(-1)
        self.r.copy_(self.k * (p - self.p0 - self.integ))
        return self.r

    def detected(self) -> torch.Tensor:
        return (self.r.abs() > self.th).any(-1)


class CollisionReaction(Controller):
    """Wraps a nominal controller with a momentum observer; on detection switches to a reflex:
    'gravity' (stop and float), or 'reflex' (torque along the estimated external torque, i.e. move
    away from the collision, De Luca et al. 2006), then holds with gravity compensation + damping."""

    name, family = "collision", "interaction"

    def __init__(self, dyn, inner: Controller, strategy: str = "reflex", k_o: float = 50.0,
                 threshold=None, k_reflex: float = 1.5, damping: float = 5.0):
        super().__init__(dyn)
        self.inner = inner
        self.needs = inner.needs
        self.obs = MomentumObserver(dyn, k_o, threshold)
        self.strategy = strategy
        self.k_reflex = k_reflex
        self.damping = damping
        self.hit = None
        self.t_hit = None
        self.last_tau = None

    def reset(self, state, ref):
        self.inner.reset(state, ref)
        self.obs.reset(state)
        self.hit = torch.zeros(state.q.shape[0], dtype=torch.bool, device=state.q.device)
        self.t_hit = torch.full_like(state.t, float("inf"))
        self.last_tau = torch.zeros_like(state.q)
        self.started = torch.zeros(1, dtype=state.q.dtype, device=state.q.device)

    def __call__(self, state, ref):
        if self.hit is None:
            self.reset(state, ref)
        self.obs.update(state, self.last_tau, active=self.started)
        new = self.obs.detected() & ~self.hit
        self.t_hit.copy_(torch.where(new, state.t.to(self.t_hit.dtype), self.t_hit))
        self.hit.copy_(self.hit | new)
        self.started.fill_(1.0)
        tau_n = self.inner(state, ref)
        g = self.dyn.gravity_torque(state.q)
        if self.strategy == "gravity":
            react = g - self.damping * state.qd
        else:
            react = g + self.k_reflex * self.obs.r - self.damping * state.qd
        tau = torch.where(self.hit[:, None], react, tau_n)
        self._keep("last_tau", torch.maximum(torch.minimum(tau, self.dyn.torque_limit), -self.dyn.torque_limit))
        return tau

    def log_values(self):
        return {"r": self.obs.r, "hit": self.hit.to(self.obs.r.dtype)}
