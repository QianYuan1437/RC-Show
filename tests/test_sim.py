"""The simulated plant: payload placement, F/T sensor sign, and the nominal model with payload."""

import mujoco
import numpy as np
import torch

from rczoo.dynamics import Dynamics, with_params
from rczoo.model import PlantConfig, plant_model
from rczoo.robot import load_robot
from rczoo.sim import CpuSim


def _q(robot):
    return torch.as_tensor(robot.home)[None] + 0.2


def test_payload_com_is_where_the_config_says():
    for name in ("panda", "sr3"):
        robot = load_robot(name)
        dyn = Dynamics(robot)
        m = plant_model(robot, PlantConfig(payload=1.0))
        d = mujoco.MjData(m)
        q = _q(robot)
        d.qpos[: robot.ndof] = q[0].numpy()
        mujoco.mj_forward(m, d)
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "payload")
        p, R = dyn.fk(q)
        expected = p[0] + R[0] @ torch.tensor([0.0, 0.0, 0.05], dtype=torch.float64)
        np.testing.assert_allclose(d.xipos[pid], expected.numpy(), atol=1e-9)


def test_identified_payload_model_matches_the_plant():
    for name in ("panda", "sr3"):
        robot = load_robot(name)
        dyn = Dynamics(robot)
        di = with_params(dyn, dyn.payload_params(1.0, inertia=torch.eye(3, dtype=torch.float64) * 0.08**2 / 6))
        m = plant_model(robot, PlantConfig(payload=1.0))
        d = mujoco.MjData(m)
        q = _q(robot)
        qd = torch.full_like(q, 0.4)
        d.qpos[: robot.ndof] = q[0].numpy()
        d.qvel[: robot.ndof] = qd[0].numpy()
        mujoco.mj_forward(m, d)
        # the probe body (20 g, not in the nominal model) is the only remaining difference
        np.testing.assert_allclose(di.bias(q, qd)[0].numpy(), d.qfrc_bias[: robot.ndof], atol=0.05)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, M)
        np.testing.assert_allclose(di.mass_matrix(q)[0].numpy(), M[: robot.ndof, : robot.ndof], atol=5e-3)


def test_ft_sensor_reads_the_force_of_the_environment_on_the_robot():
    """Gravity compensation + a downward push into a surface: the sensor must read +z."""
    robot = load_robot("panda")
    dyn = Dynamics(robot)
    q0 = torch.as_tensor(robot.home)[None]
    p0, _ = dyn.fk(q0)
    sim = CpuSim(robot, PlantConfig(surface=True))
    sim.reset(q0[0].numpy())  # (resets the mocap surface too: place it afterwards)
    sim.set_mocap([p0[0, 0].item(), p0[0, 1].item(), p0[0, 2].item() - 0.012 - 0.01])  # top just at the probe
    _, _, J = dyn.jacobian(q0)
    push = J[0, :3].T @ torch.tensor([0.0, 0.0, -10.0], dtype=torch.float64)  # 10 N down at the TCP
    for _ in range(1500):
        tau = dyn.gravity_torque(sim.q)[0] + push - 5.0 * sim.qd[0]
        sim.step(tau)
    f = sim.ft[0]
    assert 8.0 < f[2] < 12.0, f
    assert abs(f[0]) < 1.0 and abs(f[1]) < 1.0
