"""Simulators with one interface: thousands of worlds on MuJoCo Warp, or one world on CPU MuJoCo.

Both run physics and control at 1 kHz: ``step(tau)`` writes the joint torques, advances one
physics step and returns. State is read as torch tensors (batch first):

    q, qd        joint positions / velocities                         (B, n)
    ft           external wrench at the TCP, world frame, measured by the probe's F/T sensor:
                 the force/torque the *environment applies to the robot* (B, 6)
    tcp_p/R      TCP pose from MuJoCo                                  (B, 3), (B, 3, 3)
    mocap_pos    pose of the contact surface (if any)                  (B, 1, 3)
    xfrc         external force applied to the TCP body (pushes)       (B, 6), world frame

The F/T reading comes from the step's forward pass, i.e. it lags the state by one sample,
like a real sensor read in the same control cycle.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from .model import PlantConfig, plant_model, tcp_body_name
from .robot import Robot


class _Common:
    robot: Robot
    mjm: mujoco.MjModel

    def _ids(self):
        m = self.mjm
        self.n = self.robot.ndof
        self.site_tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.site_ft = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ft")
        self.tcp_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, tcp_body_name(self.robot))
        adr = lambda name: int(m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, name)])
        self.adr_f, self.adr_t = adr("ft_force"), adr("ft_torque")

    def _wrench_world(self, sensordata: torch.Tensor, site_xmat: torch.Tensor) -> torch.Tensor:
        # MuJoCo's force/torque sensors report, in the site frame, the wrench the parent body
        # exerts on the child (the probe). In quasi-statics that is minus the environment's
        # wrench on the probe, so the sign is flipped (probe inertia, 20 g, is neglected).
        # Checked in tests/test_sim.py by pressing on a surface.
        R = site_xmat[:, self.site_ft].reshape(-1, 3, 3)
        f = -(R @ sensordata[:, self.adr_f:self.adr_f + 3, None]).squeeze(-1)
        t = -(R @ sensordata[:, self.adr_t:self.adr_t + 3, None]).squeeze(-1)
        return torch.cat([f, t], -1)


class BatchSim(_Common):
    """`num_envs` copies of the plant on MuJoCo Warp; zero-copy torch views of the state."""

    def __init__(self, robot: Robot, num_envs: int, cfg: PlantConfig | None = None,
                 mjm: mujoco.MjModel | None = None):
        import mujoco_warp as mjw
        import warp as wp

        wp.config.quiet = True
        wp.init()
        # torch runs on warp's stream: tensor ops and graph launches are ordered without syncs.
        torch.cuda.set_stream(wp.stream_to_torch(wp.get_stream("cuda:0")))
        self._mjw, self._wp = mjw, wp
        self.robot = robot
        self.cfg = cfg or PlantConfig()
        self.mjm = mjm if mjm is not None else plant_model(robot, self.cfg)
        self.num_envs = num_envs
        self._ids()
        self.m = mjw.put_model(self.mjm)
        self.d = mjw.make_data(self.mjm, nworld=num_envs, nconmax=8, naconmax=8 * num_envs, njmax=40)
        tt = wp.to_torch
        self.qpos, self.qvel, self.ctrl = tt(self.d.qpos), tt(self.d.qvel), tt(self.d.ctrl)
        self.sensordata = tt(self.d.sensordata)
        self.site_xpos, self.site_xmat = tt(self.d.site_xpos), tt(self.d.site_xmat)
        self.xfrc_applied = tt(self.d.xfrc_applied)
        self.mocap_pos = tt(self.d.mocap_pos) if self.mjm.nmocap else None
        self.mocap_quat = tt(self.d.mocap_quat) if self.mjm.nmocap else None
        self.time = torch.zeros(num_envs, device="cuda", dtype=torch.float64)
        self._graph = None

    # ------------------------------------------------------------------ state
    @property
    def q(self) -> torch.Tensor:
        return self.qpos[:, : self.n].double()

    @property
    def qd(self) -> torch.Tensor:
        return self.qvel[:, : self.n].double()

    @property
    def ft(self) -> torch.Tensor:
        return self._wrench_world(self.sensordata.double(), self.site_xmat.double())

    @property
    def tcp_p(self) -> torch.Tensor:
        return self.site_xpos[:, self.site_tcp].double()

    @property
    def tcp_R(self) -> torch.Tensor:
        return self.site_xmat[:, self.site_tcp].reshape(-1, 3, 3).double()

    def reset(self, q: torch.Tensor, qd: torch.Tensor | None = None, idx: torch.Tensor | None = None):
        if idx is None:
            idx = torch.arange(self.num_envs, device="cuda")
        self.qpos[idx, : self.n] = q.to(self.qpos.dtype)
        self.qvel[idx, : self.n] = 0 if qd is None else qd.to(self.qvel.dtype)
        self.ctrl[idx] = 0
        self.xfrc_applied[idx] = 0
        self.time[idx] = 0
        self._mjw.forward(self.m, self.d)

    def set_push(self, wrench: torch.Tensor):
        """External wrench (B, 6) on the TCP body, world frame, applied at the body's com."""
        self.xfrc_applied[:, self.tcp_body] = wrench.to(self.xfrc_applied.dtype)

    # ------------------------------------------------------------------ stepping
    def _capture(self):
        with self._wp.ScopedCapture() as cap:
            self._mjw.step(self.m, self.d)
        self._graph = cap.graph

    def step(self, tau: torch.Tensor):
        self.ctrl.copy_(tau.to(self.ctrl.dtype))
        if self._graph is None:
            self._capture()
        self._wp.capture_launch(self._graph)
        self.time += self.mjm.opt.timestep


class CpuSim(_Common):
    """One world on CPU MuJoCo with the same tensor interface (B = 1). Used by tests/fixtures."""

    def __init__(self, robot: Robot, cfg: PlantConfig | None = None, mjm: mujoco.MjModel | None = None):
        self.robot = robot
        self.cfg = cfg or PlantConfig()
        self.mjm = mjm if mjm is not None else plant_model(robot, self.cfg)
        self.d = mujoco.MjData(self.mjm)
        self.num_envs = 1
        self._ids()

    def _t(self, a) -> torch.Tensor:
        return torch.as_tensor(np.array(a, dtype=np.float64))[None]

    @property
    def q(self):
        return self._t(self.d.qpos[: self.n])

    @property
    def qd(self):
        return self._t(self.d.qvel[: self.n])

    @property
    def ft(self):
        return self._wrench_world(self._t(self.d.sensordata), self._t(self.d.site_xmat))

    @property
    def tcp_p(self):
        return self._t(self.d.site_xpos[self.site_tcp])

    @property
    def tcp_R(self):
        return self._t(self.d.site_xmat[self.site_tcp].reshape(3, 3))

    @property
    def time(self):
        return torch.tensor([self.d.time])

    @property
    def mocap_pos(self):
        return self._t(self.d.mocap_pos) if self.mjm.nmocap else None

    def set_mocap(self, pos, quat=None):
        self.d.mocap_pos[0] = pos
        if quat is not None:
            self.d.mocap_quat[0] = quat

    def reset(self, q, qd=None, idx=None):
        mujoco.mj_resetData(self.mjm, self.d)
        self.d.qpos[: self.n] = np.asarray(q, dtype=float).reshape(-1)
        if qd is not None:
            self.d.qvel[: self.n] = np.asarray(qd, dtype=float).reshape(-1)
        mujoco.mj_forward(self.mjm, self.d)

    def set_push(self, wrench):
        self.d.xfrc_applied[self.tcp_body] = np.asarray(wrench, dtype=float).reshape(-1)

    def step(self, tau):
        self.d.ctrl[:] = np.asarray(tau.detach().cpu() if torch.is_tensor(tau) else tau, dtype=float).reshape(-1)
        mujoco.mj_step(self.mjm, self.d)
