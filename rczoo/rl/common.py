"""What the reaching policy and its deployments (``rczoo.control.learned``, the browser) share."""

from __future__ import annotations

import torch

from ..dynamics import Dynamics, orientation_error

POLICY_DT = 0.02
TARGET_RATE = 0.8  # fraction of the joint speed limit the target can move at


def obs_dim(n: int) -> int:
    return 4 * n + 9


def observation(dyn: Dynamics, q, qd, q_cmd, p, R, p_goal, R_goal, a_prev):
    """(B, 4n + 9), float32. Shared with the browser (see export)."""
    e_p = p_goal - p
    e_R = orientation_error(R_goal, R)
    return torch.cat([q, 0.1 * qd, 10.0 * (q_cmd - q), p, 5.0 * e_p, e_R, a_prev], -1).float()


def target_step(dyn: Dynamics) -> torch.Tensor:
    return TARGET_RATE * POLICY_DT * dyn.vel_limit
