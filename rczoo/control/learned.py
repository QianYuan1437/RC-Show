"""A learned (PPO) reaching policy on top of joint impedance control, as a controller.

The policy runs at 50 Hz and moves the joint target q_cmd; the 1 kHz inner loop is the joint PD
with nominal gravity compensation it was trained with (``rczoo.rl.reach_env``). The reference
gives only the goal pose (ref.p, ref.R): the policy plans and executes the motion itself.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..rl.common import POLICY_DT, observation, target_step
from .base import Controller
from .gains import GAINS


class LearnedReach(Controller):
    name, family = "rl", "learned"
    needs = ("cartesian",)

    def __init__(self, dyn, actor: torch.nn.Module | str | Path | None = None):
        super().__init__(dyn)
        if actor is None:
            actor = Path("runs") / f"reach_{dyn.robot.name}" / "actor.pt"
        if not isinstance(actor, torch.nn.Module):
            actor = torch.load(actor, weights_only=False)
        self.actor = actor.to(dyn.device).eval()
        g = GAINS[dyn.robot.name]
        self.kp = torch.as_tensor(g["kp"], dtype=dyn.dtype, device=dyn.device)
        self.kd = torch.as_tensor(g["kd"], dtype=dyn.dtype, device=dyn.device)
        self.scale = target_step(dyn)
        self.period = int(round(POLICY_DT / self.dt))
        self.q_cmd = self.a_prev = self.k = None

    def reset(self, state, ref):
        self.q_cmd = state.q.clone()
        self.a_prev = torch.zeros(state.q.shape, dtype=torch.float32, device=state.q.device)
        self.k = torch.zeros((), dtype=torch.long, device=state.q.device)

    def __call__(self, state, ref):
        if self.q_cmd is None:
            self.reset(state, ref)
        dyn = self.dyn
        q, qd = state.q, state.qd
        # the policy acts on every 20th cycle (graph-safe: computed always, applied by a mask)
        p, R = dyn.fk(q)
        obs = observation(dyn, q, qd, self.q_cmd, p, R, ref.p, ref.R, self.a_prev)
        a = self.actor(obs).clamp(-1, 1)
        act = (self.k % self.period == 0).to(dyn.dtype)
        q_cmd = torch.minimum(torch.maximum(self.q_cmd + act * self.scale * a.to(dyn.dtype), dyn.lower), dyn.upper)
        self._keep("q_cmd", q_cmd)
        self._keep("a_prev", torch.where(act.bool(), a, self.a_prev))
        self.k.add_(1)
        return self.kp * (self.q_cmd - q) - self.kd * qd + dyn.gravity_torque(q)
