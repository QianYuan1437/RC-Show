"""Learned pose reaching on top of joint impedance control, robust to unknown payloads.

This is how reinforcement learning is usually deployed on torque-controlled arms: the policy
does not output torques at 1 kHz but, at 50 Hz, *increments* of the joint-position target of a
1 kHz joint PD with (nominal) gravity compensation, i.e. Franka's joint impedance mode:

    q_cmd <- clip(q_cmd + s * a)                                      (policy, 50 Hz, a in [-1, 1]^n)
    tau    = clip(Kp (q_cmd - q) - Kd qd + g_nominal(q), +-tau_max)   (inner loop, 1 kHz)

The incremental target is an integrator: to hold an unknown payload the policy keeps q_cmd a
little "above" q, and the observed offset q_cmd - q (= the load torque / Kp) tells it how heavy
the load is. Domain randomisation: payload 0..2 kg, joint dry friction 0..3 % of the torque limit.

In MuJoCo Warp the inner loop is a position actuator per joint (gain Kp, bias -Kp q - Kd qd) plus
MuJoCo's gravity compensation of the robot's links (not the payload) routed through the
actuators, so the joint force range clamps the sum, as in the formula above.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from ..control.gains import GAINS
from ..dynamics import Dynamics, orientation_error
from ..model import PlantConfig, plant_spec
from ..robot import Robot
from .common import POLICY_DT, obs_dim, observation, target_step


@dataclass
class ReachConfig:
    episode_s: float = 3.0
    payload_max: float = 2.0
    friction_max: float = 0.03  # fraction of the torque limit
    step_max: float = 1.0  # rad, start -> goal configuration distance per joint


def rl_model(robot: Robot) -> mujoco.MjModel:
    """Plant with a payload body and the joint impedance inner loop as MuJoCo actuators."""
    spec = plant_spec(robot, PlantConfig(), payload_body=True)
    g = GAINS[robot.name]
    for b in spec.bodies:
        if b.name not in ("world", "payload"):
            b.gravcomp = 1.0
    for i, jn in enumerate(robot.joint_names):
        j = spec.joint(jn)
        j.actgravcomp = True
        j.actfrclimited = True
        j.actfrcrange = [-float(robot.torque_limit[i]), float(robot.torque_limit[i])]
    for a in spec.actuators:
        i = robot.joint_names.index(a.target)
        kp, kd = g["kp"][i], g["kd"][i]
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[0] = kp
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[0:3] = [0.0, -kp, -kd]
        a.ctrlrange = [float(robot.lower[i]), float(robot.upper[i])]
        a.ctrllimited = True
        a.forcelimited = False
    return spec.compile()


class ReachEnv:
    def __init__(self, robot: Robot, num_envs: int, cfg: ReachConfig | None = None, seed: int = 0,
                 randomize: bool = True):
        import mujoco_warp as mjw
        import warp as wp

        wp.config.quiet = True
        wp.init()
        torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream("cuda:0")))
        self.mjw, self.wp = mjw, wp
        self.robot = robot
        self.cfg = cfg or ReachConfig()
        self.N = num_envs
        self.randomize = randomize
        self.dyn = Dynamics(robot, "cuda")
        self.n = robot.ndof
        self.mjm = rl_model(robot)
        self.m = mjw.put_model(self.mjm)
        N = num_envs
        # per-world payload mass / inertia and joint friction
        self.pid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.m.body_mass = wp.array(np.tile(self.mjm.body_mass[None], (N, 1)).astype(np.float32), dtype=float)
        self.m.body_inertia = wp.array(np.tile(self.mjm.body_inertia[None], (N, 1, 1)).astype(np.float32), dtype=wp.vec3)
        self.m.dof_frictionloss = wp.array(np.tile(self.mjm.dof_frictionloss[None], (N, 1)).astype(np.float32), dtype=float)
        self.body_mass = wp.to_torch(self.m.body_mass)
        self.body_inertia = wp.to_torch(self.m.body_inertia)
        self.frictionloss = wp.to_torch(self.m.dof_frictionloss)
        self.d = mjw.make_data(self.mjm, nworld=N, nconmax=0, naconmax=0, njmax=16)
        tt = wp.to_torch
        self.qpos, self.qvel, self.ctrl = tt(self.d.qpos), tt(self.d.qvel), tt(self.d.ctrl)
        self.site_xpos, self.site_xmat = tt(self.d.site_xpos), tt(self.d.site_xmat)
        self.site = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.substeps = int(round(POLICY_DT / self.mjm.opt.timestep))
        self.max_steps = int(round(self.cfg.episode_s / POLICY_DT))
        self.g = torch.Generator(device="cuda").manual_seed(seed)
        dev, f32 = "cuda", torch.float32
        self.q_cmd = torch.zeros(N, self.n, device=dev, dtype=torch.float64)
        self.a_prev = torch.zeros(N, self.n, device=dev, dtype=f32)
        self.p_goal = torch.zeros(N, 3, device=dev, dtype=torch.float64)
        self.R_goal = torch.eye(3, device=dev, dtype=torch.float64).repeat(N, 1, 1)
        self.t = torch.zeros(N, dtype=torch.long, device=dev)
        self.payload = torch.zeros(N, device=dev)
        self.step_scale = target_step(self.dyn)
        self.lo = self.dyn.lower + 0.1 * (self.dyn.upper - self.dyn.lower)
        self.hi = self.dyn.upper - 0.1 * (self.dyn.upper - self.dyn.lower)
        self._graph = None

    # ---------------------------------------------------------------- sampling
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device="cuda", dtype=torch.float64)

    def sample_tasks(self, k: int):
        """Start / goal configurations: the goal pose is the FK of a configuration within
        +-step_max of the start, both with the TCP above the table and off the base column."""
        out0, out1 = [], []
        while sum(len(x) for x in out0) < k:
            m = 2 * k
            q0 = self.lo + self._rand(m, self.n) * (self.hi - self.lo)
            q1 = (q0 + (2 * self._rand(m, self.n) - 1) * self.cfg.step_max).clamp(self.lo, self.hi)
            ok = torch.ones(m, dtype=torch.bool, device="cuda")
            for q in (q0, q1):
                p, _ = self.dyn.fk(q)
                ok &= (p[:, 2] > 0.15) & (p[:, :2].norm(dim=-1) > 0.25)
            out0.append(q0[ok])
            out1.append(q1[ok])
        return torch.cat(out0)[:k], torch.cat(out1)[:k]

    # ---------------------------------------------------------------- env API
    def reset(self, idx: torch.Tensor | None = None, q0=None, q_goal=None, payload=None, friction=None):
        if idx is None:
            idx = torch.arange(self.N, device="cuda")
        k = idx.numel()
        if k == 0:
            return
        if q0 is None:
            q0, q_goal = self.sample_tasks(k)
        p_g, R_g = self.dyn.fk(q_goal)
        self.p_goal[idx], self.R_goal[idx] = p_g, R_g
        self.qpos[idx, : self.n] = q0.float()
        self.qvel[idx] = 0
        self.q_cmd[idx] = q0
        self.ctrl[idx] = q0.float()
        self.a_prev[idx] = 0
        self.t[idx] = 0
        if payload is None:
            payload = self._rand(k).float() * self.cfg.payload_max if self.randomize else torch.zeros(k, device="cuda")
        if friction is None:
            friction = (self._rand(k, 1) * self.cfg.friction_max * self.dyn.torque_limit).float() if self.randomize \
                else torch.zeros(k, self.n, device="cuda")
        m = payload.clamp(min=1e-6)
        self.payload[idx] = payload
        self.body_mass[idx, self.pid] = m
        self.body_inertia[idx, self.pid] = (m * (0.08**2) / 6)[:, None].expand(k, 3)
        self.frictionloss[idx, : self.n] = friction.expand(k, self.n)
        self.mjw.kinematics(self.m, self.d)

    @property
    def q(self):
        return self.qpos[:, : self.n].double()

    @property
    def qd(self):
        return self.qvel[:, : self.n].double()

    def tcp(self):
        return self.site_xpos[:, self.site].double(), self.site_xmat[:, self.site].reshape(-1, 3, 3).double()

    def observe(self):
        p, R = self.tcp()
        return observation(self.dyn, self.q, self.qd, self.q_cmd, p, R, self.p_goal, self.R_goal, self.a_prev)

    def _capture(self):
        with self.wp.ScopedCapture() as cap:
            for _ in range(self.substeps):
                self.mjw.step(self.m, self.d)
            self.mjw.kinematics(self.m, self.d)
        self._graph = cap.graph

    def step(self, a: torch.Tensor):
        a = a.clamp(-1, 1)
        q_cmd = torch.minimum(torch.maximum(self.q_cmd + self.step_scale * a.double(), self.dyn.lower), self.dyn.upper)
        self.q_cmd.copy_(q_cmd)
        self.ctrl[:, : self.n] = q_cmd.float()
        if self._graph is None:
            self._capture()
        self.wp.capture_launch(self._graph)
        self.t += 1
        p, R = self.tcp()
        e_p = (self.p_goal - p).norm(dim=-1)
        e_R = orientation_error(self.R_goal, R).norm(dim=-1)
        qd = self.qd
        near_limit = ((self.dyn.lower + 0.05 - self.q).clamp(min=0) + (self.q - self.dyn.upper + 0.05).clamp(min=0)).sum(-1)
        # linear terms pull from far away, the exponentials reward precision; 10 cm ~ 0.4 rad
        r = (-4.0 * e_p - 1.0 * e_R
             + torch.exp(-e_p / 0.05) + 2.0 * torch.exp(-e_p / 0.005)
             + torch.exp(-e_R / 0.1) + 2.0 * torch.exp(-e_R / 0.02)
             - 0.01 * ((a - self.a_prev) ** 2).sum(-1)
             - 0.002 * (qd**2).sum(-1).float() - 5.0 * near_limit.float())
        r = r.float() * 0.1
        self.a_prev.copy_(a)
        timeout = self.t >= self.max_steps
        bad = ~torch.isfinite(self.qvel).all(-1)
        info = {"e_p": e_p, "e_R": e_R}
        return r.float().nan_to_num(0.0), bad, timeout, info


__all__ = ["POLICY_DT", "ReachConfig", "ReachEnv", "obs_dim", "observation", "rl_model", "target_step"]
