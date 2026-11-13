"""Common controller interface.

A controller maps the measured state and a reference to joint torques at 1 kHz:

    tau = controller(state, ref)

* ``State``  t, q, qd and the F/T wrench ``ft`` (environment on robot, world frame, at the TCP);
* ``Ref``    joint reference (q, qd, qdd) and/or TCP reference (p, R, v = [v; w], a = [dv; dw]),
             an optional desired contact force ``f`` (what the robot should push with, world
             frame) and the null-space posture ``q_null``.

Every controller only uses its *nominal* model (``Dynamics`` with the MJCF parameters); the
simulated plant may carry an unknown payload or friction. Torques are clipped to the motor
limits by the plant (MuJoCo ctrlrange), and controllers report what they asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import torch

from ..dynamics import Dynamics, orientation_error


@dataclass
class State:
    t: torch.Tensor  # (B,)
    q: torch.Tensor  # (B, n)
    qd: torch.Tensor  # (B, n)
    ft: torch.Tensor | None = None  # (B, 6)


@dataclass
class Ref:
    q: torch.Tensor | None = None
    qd: torch.Tensor | None = None
    qdd: torch.Tensor | None = None
    p: torch.Tensor | None = None
    R: torch.Tensor | None = None
    v: torch.Tensor | None = None  # (B, 6) TCP twist [v; w], world
    a: torch.Tensor | None = None  # (B, 6) TCP acceleration
    f: torch.Tensor | None = None  # (B, 6) desired wrench the robot applies to the environment
    q_null: torch.Tensor | None = None
    extra: dict = field(default_factory=dict)

    def index(self, idx) -> Ref:
        kw = {}
        for f in fields(self):
            v = getattr(self, f.name)
            kw[f.name] = v[idx] if torch.is_tensor(v) else v
        return Ref(**kw)


def cartesian_from_joint(dyn: Dynamics, q, qd, qdd) -> Ref:
    """TCP reference that corresponds exactly to a joint reference (q, qd, qdd)."""
    p, R, J = dyn.jacobian(q)
    v = (J @ qd[..., None]).squeeze(-1)
    a = (J @ qdd[..., None]).squeeze(-1) + dyn.tcp_bias_acc(q, qd)
    return Ref(q=q, qd=qd, qdd=qdd, p=p, R=R, v=v, a=a, q_null=q)


def pose_error(p_des, R_des, p, R) -> torch.Tensor:
    """6-D error [p* - p; Log(R* R^T)] in the world frame."""
    return torch.cat([p_des - p, orientation_error(R_des, R)], -1)


def damped_pinv(J: torch.Tensor, lam: float = 1e-2) -> torch.Tensor:
    """J^T (J J^T + lam^2 I)^-1 (B, n, 6)."""
    m = J.shape[-2]
    eye = torch.eye(m, dtype=J.dtype, device=J.device)
    return J.transpose(-1, -2) @ inv(J @ J.transpose(-1, -2) + lam**2 * eye)


def task_inertia(J: torch.Tensor, Minv: torch.Tensor, lam: float = 1e-3) -> torch.Tensor:
    """Lambda = (J M^-1 J^T)^-1, regularised near singularities."""
    A = J @ Minv @ J.transpose(-1, -2)
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return inv(A + lam * eye)


def inv(A: torch.Tensor) -> torch.Tensor:
    """Matrix inverse without a host-side error check (CUDA-graph safe)."""
    return torch.linalg.inv_ex(A)[0]


def solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve_ex(A, B)[0]


def vec(x, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar / list gain to a (n,) or (6,) tensor on like's device."""
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


class Controller:
    """Base class. ``needs`` lists the reference fields a controller uses ('joint' / 'cartesian' /
    'force'); ``family`` groups controllers on the site."""

    name = "base"
    family = "joint"
    needs: tuple[str, ...] = ("joint",)
    # True if one control step can be captured in a CUDA graph (no host syncs, state updated
    # in place); the runner then replays the graph instead of launching hundreds of kernels.
    graph_safe = True

    def __init__(self, dyn: Dynamics):
        self.dyn = dyn

    def reset(self, state: State, ref: Ref) -> None:
        pass

    def __call__(self, state: State, ref: Ref) -> torch.Tensor:
        raise NotImplementedError

    @property
    def dt(self) -> float:
        return 1e-3

    def _keep(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Store controller state *in place* once allocated (CUDA-graph friendly)."""
        cur = getattr(self, name, None)
        if cur is None or cur.shape != value.shape:
            setattr(self, name, value.clone())
        else:
            cur.copy_(value)
        return getattr(self, name)
