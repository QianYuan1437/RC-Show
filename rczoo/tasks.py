"""Benchmark tasks: batched reference generators plus scripted external events.

Every task provides ``reference(t) -> Ref`` for a batch of worlds (t: (B,) seconds), an
``events(sim, t)`` hook (pushes, the "human hand" in hand guiding) and its own metrics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .control.base import Ref, cartesian_from_joint
from .dynamics import Dynamics
from .traj import quintic

PROBE_RADIUS = 0.012


def sample_pairs(dyn: Dynamics, B: int, seed: int = 0, step: float = 0.9, margin: float = 0.12):
    """Start / goal configurations inside the joint box with the TCP above the table (z > 0.15 m)
    and away from the base column; joints move by up to +-`step` rad."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    out0, out1 = [], []
    lo = dyn.lower + margin * (dyn.upper - dyn.lower)
    hi = dyn.upper - margin * (dyn.upper - dyn.lower)
    while sum(len(x) for x in out0) < B:
        n = 4 * B
        u = torch.rand(n, dyn.n, generator=g, dtype=dyn.dtype).to(dyn.device)
        q0 = lo + u * (hi - lo)
        d = (torch.rand(n, dyn.n, generator=g, dtype=dyn.dtype).to(dyn.device) * 2 - 1) * step
        q1 = torch.minimum(torch.maximum(q0 + d, lo), hi)
        ok = torch.ones(n, dtype=torch.bool, device=dyn.device)
        for q in (q0, q1, 0.5 * (q0 + q1)):
            p, _ = dyn.fk(q)
            ok &= (p[:, 2] > 0.15) & (p[:, :2].norm(dim=-1) > 0.25)
        out0.append(q0[ok])
        out1.append(q1[ok])
    return torch.cat(out0)[:B], torch.cat(out1)[:B]


@dataclass
class JointTrajectoryTask:
    """Rest-to-rest quintic joint trajectories of duration T, then hold. Joint and Cartesian
    references are consistent (the Cartesian one is FK of the joint one)."""

    dyn: Dynamics
    q0: torch.Tensor
    q1: torch.Tensor
    T: float = 1.5
    hold: float = 0.5
    name: str = "track"

    @property
    def duration(self) -> float:
        return self.T + self.hold

    def joint(self, t):
        pr = quintic(self.q0, self.q1, self.T, t[:, None])
        return pr.p, pr.v, pr.a

    def reference(self, t) -> Ref:
        q, qd, qdd = self.joint(t)
        ref = cartesian_from_joint(self.dyn, q, qd, qdd)
        ref.extra["preview"] = self._preview
        return ref

    def _preview(self, t_flat):
        B = self.q0.shape[0]
        K = t_flat.shape[0] // B
        q0 = self.q0.repeat_interleave(K, 0)
        q1 = self.q1.repeat_interleave(K, 0)
        pr = quintic(q0, q1, self.T, t_flat[:, None])
        return Ref(q=pr.p, qd=pr.v, qdd=pr.a)

    def start(self):
        return self.q0

    def events(self, sim, t):
        pass


@dataclass
class PushTask:
    """Hold a configuration while a horizontal force pulse hits the TCP (a bump by a person)."""

    dyn: Dynamics
    q0: torch.Tensor
    force: float = 30.0
    t_on: float = 0.3
    t_off: float = 0.5
    duration: float = 2.0
    seed: int = 0
    name: str = "push"

    def __post_init__(self):
        g = torch.Generator().manual_seed(self.seed)
        ang = torch.rand(self.q0.shape[0], generator=g, dtype=self.dyn.dtype) * 2 * math.pi
        d = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1).to(self.dyn.device)
        self.wrench = torch.cat([self.force * d, torch.zeros_like(d)], -1)

    def reference(self, t) -> Ref:
        z = torch.zeros_like(self.q0)
        ref = cartesian_from_joint(self.dyn, self.q0, z, z)
        ref.extra["preview"] = lambda tf: Ref(q=self.q0.repeat_interleave(tf.shape[0] // self.q0.shape[0], 0),
                                              qd=torch.zeros(tf.shape[0], self.dyn.n, dtype=self.dyn.dtype, device=self.dyn.device))
        return ref

    def start(self):
        return self.q0

    def events(self, sim, t):
        on = ((t >= self.t_on) & (t < self.t_off))[:, None]
        sim.set_push(torch.where(on, self.wrench, torch.zeros_like(self.wrench)))


@dataclass
class ContactTask:
    """Approach a horizontal surface, press with f_d, slide 15 cm along it, keep pressing.

    The reference TCP path lies *on* the surface (probe tangent to it); impedance-type
    controllers get it lowered by f_d / K_z so that their spring produces f_d."""

    dyn: Dynamics
    q0: torch.Tensor  # start configurations (TCP pointing down above the surface)
    surface_z: torch.Tensor  # (B,) height of the surface top
    f_d: float = 10.0
    t_approach: float = 1.0
    t_slide: float = 2.0
    hold: float = 1.0
    slide: float = 0.15
    name: str = "contact"
    lower_by: float = 0.0  # reference lowered below the surface (impedance-type controllers)

    def __post_init__(self):
        self.p0, self.R0 = self.dyn.fk(self.q0)
        self.f_vec = torch.zeros(self.q0.shape[0], 3, dtype=self.dyn.dtype, device=self.dyn.device)
        self.f_vec[:, 2] = -self.f_d
        # slide tangentially (perpendicular to the radius): the reach, hence the posture, barely
        # changes, which keeps the 6-DoF arm away from its elbow singularity
        g = torch.Generator().manual_seed(3)
        sgn = torch.where(torch.rand(self.q0.shape[0], generator=g) < 0.5, -1.0, 1.0).to(self.dyn.device, self.dyn.dtype)
        rad = self.p0[:, :2] / self.p0[:, :2].norm(dim=-1, keepdim=True)
        self.dir = torch.stack([-rad[:, 1] * sgn, rad[:, 0] * sgn, torch.zeros_like(sgn)], -1)

    @property
    def duration(self) -> float:
        return self.t_approach + self.t_slide + self.hold

    def path(self, t):
        z_c = self.surface_z + PROBE_RADIUS - self.lower_by
        # approach: quintic in z from p0 to the contact height
        a = quintic(self.p0[:, 2], z_c, self.t_approach, t)
        s = quintic(torch.zeros_like(t), torch.full_like(t, self.slide), self.t_slide, (t - self.t_approach).clamp(min=0))
        p = torch.stack([self.p0[:, 0], self.p0[:, 1], a.p], -1) + s.p[:, None] * self.dir
        v = torch.cat([s.v[:, None] * self.dir[:, :2], a.v[:, None]], -1)
        acc = torch.cat([s.a[:, None] * self.dir[:, :2], a.a[:, None]], -1)
        return p, v, acc

    def reference(self, t) -> Ref:
        p, v, a = self.path(t)
        zero = torch.zeros_like(p)
        # desired force ramps up over 0.3 s once the probe has touched down
        ramp = ((t - self.t_approach) / 0.3).clamp(0, 1)[:, None]
        f = ramp * self.f_vec
        return Ref(p=p, R=self.R0, v=torch.cat([v, zero], -1), a=torch.cat([a, zero], -1),
                   f=torch.cat([f, zero], -1), q_null=self.q0)

    def start(self):
        return self.q0

    def events(self, sim, t):
        pass


def downward_configs(dyn: Dynamics, B: int, seed: int = 0, tol: float = 1e-4):
    """Configurations with the tool pointing straight down, spread around the home TCP position
    (+-25 deg azimuth, -6..+4 cm radius, 5 cm lower), found by damped least squares IK from home;
    candidates that do not converge are discarded."""
    from .control.base import pose_error
    from .traj import so3_exp

    g = torch.Generator().manual_seed(seed)
    p_home, R_home = dyn.fk(dyn.home[None])
    r0 = float(p_home[0, :2].norm())
    phi0 = math.atan2(float(p_home[0, 1]), float(p_home[0, 0]))
    z0 = min(float(p_home[0, 2]), 0.35) - 0.05
    out = []
    while sum(len(x) for x in out) < B:
        K = 2 * B
        rnd = lambda: torch.rand(K, generator=g, dtype=dyn.dtype).to(dyn.device)
        r = r0 - 0.06 + 0.10 * rnd()
        phi = phi0 + (rnd() - 0.5) * 0.87
        p_t = torch.stack([r * torch.cos(phi), r * torch.sin(phi), torch.full_like(r, z0)], -1)
        # home orientation (tool down on both arms) turned with the azimuth: keeps the wrist
        # joints near their home values instead of winding the last joint towards its limit
        R_t = so3_exp(torch.stack([torch.zeros_like(phi), torch.zeros_like(phi), phi - phi0], -1)) @ R_home
        q = dyn.home.expand(K, dyn.n).clone()
        for _ in range(300):
            p, R, J = dyn.jacobian(q)
            e = pose_error(p_t, R_t, p, R)
            JJt = J @ J.transpose(1, 2) + 1e-4 * torch.eye(6, dtype=q.dtype, device=q.device)
            dq = (J.transpose(1, 2) @ torch.linalg.solve(JJt, e[..., None])).squeeze(-1)
            if dyn.n > 6:
                N = torch.eye(dyn.n, dtype=q.dtype, device=q.device) - J.transpose(1, 2) @ torch.linalg.solve(JJt, J)
                dq = dq + (N @ (0.2 * (dyn.home - q))[..., None]).squeeze(-1)
            q = dyn.clamp(q + dq.clamp(-0.2, 0.2))
        p, R, _ = dyn.jacobian(q)
        err = pose_error(p_t, R_t, p, R).norm(dim=-1)
        margin = ((q - dyn.lower).min(-1).values > 0.05) & ((dyn.upper - q).min(-1).values > 0.05)
        out.append(q[(err < tol) & margin])
    return torch.cat(out)[:B]


@dataclass
class GuidingTask:
    """Hand guiding: a 'hand' (stiff spring-damper, 600 N/m) drags the TCP along a 30 cm line in
    2 s. Metric: the force the person has to apply; a lighter arm needs less."""

    dyn: Dynamics
    q0: torch.Tensor
    T: float = 2.0
    hold: float = 0.5
    length: float = 0.3
    k_hand: float = 600.0
    d_hand: float = 40.0
    name: str = "guide"

    def __post_init__(self):
        self.p0, _ = self.dyn.fk(self.q0)
        g = torch.Generator().manual_seed(5)
        ang = torch.rand(self.q0.shape[0], generator=g, dtype=self.dyn.dtype).to(self.dyn.device) * 2 * math.pi
        self.dir = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], -1)
        self.hand_force = None

    @property
    def duration(self) -> float:
        return self.T + self.hold

    def hand(self, t):
        s = quintic(torch.zeros_like(t), torch.full_like(t, self.length), self.T, t)
        return self.p0 + s.p[:, None] * self.dir, s.v[:, None] * self.dir

    def reference(self, t) -> Ref:
        return Ref()

    def start(self):
        return self.q0

    def events(self, sim, t):
        self._t = t
        ph, vh = self.hand(t)
        p = sim.tcp_p
        _, _, J = self.dyn.jacobian(sim.q)
        v = (J[:, :3] @ sim.qd[..., None]).squeeze(-1)
        F = self.k_hand * (ph - p) + self.d_hand * (vh - v)
        self.hand_force = F
        sim.set_push(torch.cat([F, torch.zeros_like(F)], -1))

    def log_values(self):
        return {"hand_force": self.hand_force, "hand_p": self.hand(self._t)[0]} if self.hand_force is not None else {}


@dataclass
class CollisionTask(JointTrajectoryTask):
    """Joint trajectories during which, in a random half of the worlds, the TCP is hit by an
    unexpected force (a person walking into the arm) for 0.15 s at a random time."""

    force: float = 40.0
    seed: int = 0
    name: str = "collision"

    def __post_init__(self):
        B = self.q0.shape[0]
        g = torch.Generator().manual_seed(self.seed)
        dev, dt = self.dyn.device, self.dyn.dtype
        self.collides = (torch.arange(B) % 2 == 0).to(dev)
        self.t_on = (0.3 + 0.9 * torch.rand(B, generator=g, dtype=dt)).to(dev)
        ang = torch.rand(B, generator=g, dtype=dt) * 2 * math.pi
        el = (torch.rand(B, generator=g, dtype=dt) - 0.5) * 0.6
        d = torch.stack([torch.cos(ang) * torch.cos(el), torch.sin(ang) * torch.cos(el), torch.sin(el)], -1).to(dev)
        self.wrench = torch.cat([self.force * d, torch.zeros_like(d)], -1) * self.collides[:, None]

    def events(self, sim, t):
        on = ((t >= self.t_on) & (t < self.t_on + 0.15))[:, None]
        sim.set_push(torch.where(on, self.wrench, torch.zeros_like(self.wrench)))


@dataclass
class GoalTask:
    """A goal pose only (no trajectory): for controllers that plan the motion themselves."""

    dyn: Dynamics
    q0: torch.Tensor
    q1: torch.Tensor
    duration: float = 3.0
    name: str = "goal"

    def __post_init__(self):
        self.p1, self.R1 = self.dyn.fk(self.q1)

    def reference(self, t) -> Ref:
        z = torch.zeros_like(self.q1)
        return Ref(q=self.q1, qd=z, qdd=z, p=self.p1, R=self.R1, v=torch.zeros_like(z[:, :1]).expand(-1, 6),
                   a=torch.zeros_like(z[:, :1]).expand(-1, 6))

    def start(self):
        return self.q0

    def events(self, sim, t):
        pass
