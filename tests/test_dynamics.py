import mujoco
import numpy as np
import pytest
import torch

from rczoo.dynamics import Dynamics
from rczoo.robot import load_robot

ROBOTS = ["panda", "sr3"]


def _mj(robot, q, qd, qdd=None):
    m = robot.mj_model()
    d = mujoco.MjData(m)
    d.qpos[:] = q
    d.qvel[:] = qd
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, sid)
    jdp, jdr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    mujoco.mj_jacDot(m, d, jdp, jdr, d.site_xpos[sid], m.site_bodyid[sid])
    out = {"M": M, "bias": d.qfrc_bias.copy(), "J": np.vstack([jp, jr]),
           "Jdqd": np.concatenate([jdp @ qd, jdr @ qd]),
           "p": d.site_xpos[sid].copy(), "R": d.site_xmat[sid].reshape(3, 3).copy()}
    if qdd is not None:
        d.qacc[:] = qdd
        mujoco.mj_inverse(m, d)
        out["tau"] = d.qfrc_inverse + d.qfrc_passive  # rigid-body + armature part only (inverse = M qacc + c - passive)
    return out


@pytest.mark.parametrize("name", ROBOTS)
def test_dynamics_match_mujoco(name):
    robot = load_robot(name)
    dyn = Dynamics(robot)
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.uniform(robot.lower, robot.upper)
        qd = rng.normal(size=robot.ndof)
        qdd = rng.normal(size=robot.ndof) * 3
        ref = _mj(robot, q, qd, qdd)
        tq = lambda a: torch.as_tensor(a)[None]
        p, R, J = dyn.jacobian(tq(q))
        np.testing.assert_allclose(p[0].numpy(), ref["p"], atol=1e-10)
        np.testing.assert_allclose(R[0].numpy(), ref["R"], atol=1e-10)
        np.testing.assert_allclose(J[0].numpy(), ref["J"], atol=1e-10)
        np.testing.assert_allclose(dyn.mass_matrix(tq(q))[0].numpy(), ref["M"], atol=1e-10)
        np.testing.assert_allclose(dyn.bias(tq(q), tq(qd))[0].numpy(), ref["bias"], atol=1e-9)
        np.testing.assert_allclose(dyn.rnea(tq(q), tq(qd), tq(qdd))[0].numpy(), ref["tau"], atol=1e-9)
        np.testing.assert_allclose(dyn.tcp_bias_acc(tq(q), tq(qd))[0].numpy(), ref["Jdqd"], atol=1e-9)


@pytest.mark.parametrize("name", ROBOTS)
def test_coriolis_properties(name):
    robot = load_robot(name)
    dyn = Dynamics(robot)
    g = torch.Generator().manual_seed(1)
    q = dyn.sample_q(4, generator=g)
    qd = torch.randn(4, dyn.n, dtype=torch.float64, generator=g)
    C = dyn.coriolis_matrix_autograd(q, qd)
    # C qd + g = h
    h = dyn.bias(q, qd)
    np.testing.assert_allclose((C @ qd[..., None]).squeeze(-1) + dyn.gravity_torque(q), h, atol=1e-9)
    # dM/dt - 2C skew-symmetric, dM/dt by finite differences along qd
    eps = 1e-6
    Mdot = (dyn.mass_matrix(q + eps * qd) - dyn.mass_matrix(q - eps * qd)) / (2 * eps)
    S = Mdot - 2 * C
    np.testing.assert_allclose(S + S.transpose(1, 2), torch.zeros_like(S), atol=1e-6)


def test_payload_matches_mujoco_body():
    robot = load_robot("panda")
    dyn = Dynamics(robot)
    spec = mujoco.MjSpec.from_file(str(robot.xml_path))
    hand = spec.body("hand")
    b = hand.add_body(name="payload", pos=[0, 0, 0.1034 + 0.05])
    b.mass = 1.5
    b.inertia = [1e-3, 1e-3, 1e-3]
    b.ipos = [0, 0, 0]
    m = spec.compile()
    d = mujoco.MjData(m)
    q = robot.home + 0.3
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    P = dyn.payload_params(1.5, com=(0, 0, 0.05))
    # payload inertia about its com in the TCP frame = 1e-3 I; rotate into link frame is a no-op for I
    P = P.__class__(P.mass, P.com, P.inertia)
    P2 = dyn.params.with_payload(1.5, dyn.tcp_p + dyn.tcp_R @ torch.tensor([0, 0, 0.05], dtype=torch.float64),
                                 1e-3 * torch.eye(3, dtype=torch.float64))
    g_mj = d.qfrc_bias.copy()
    g_ours = dyn.gravity_torque(torch.as_tensor(q)[None], params=P2)[0].numpy()
    np.testing.assert_allclose(g_ours, g_mj, atol=1e-9)


@pytest.mark.parametrize("name", ROBOTS)
def test_christoffel_rnea(name):
    robot = load_robot(name)
    dyn = Dynamics(robot)
    g = torch.Generator().manual_seed(2)
    q = dyn.sample_q(4, generator=g)
    qd, w, a = (torch.randn(4, dyn.n, dtype=torch.float64, generator=g) for _ in range(3))
    C = dyn.coriolis_matrix_autograd(q, qd)
    np.testing.assert_allclose(dyn.coriolis_matrix(q, qd), C, atol=1e-10)
    ref = (dyn.mass_matrix(q) @ a[..., None] + C @ w[..., None]).squeeze(-1) + dyn.gravity_torque(q)
    np.testing.assert_allclose(dyn.slotine_li(q, qd, w, a), ref, atol=1e-9)


@pytest.mark.parametrize("name", ROBOTS)
def test_payload_regressor(name):
    robot = load_robot(name)
    dyn = Dynamics(robot)
    g = torch.Generator().manual_seed(3)
    q = dyn.sample_q(3, generator=g)
    qd, w, a = (torch.randn(3, dyn.n, dtype=torch.float64, generator=g) for _ in range(3))
    m, c_l = 1.3, torch.tensor([0.01, -0.02, 0.15], dtype=torch.float64)
    I_c = torch.diag(torch.tensor([2e-3, 3e-3, 1e-3], dtype=torch.float64))
    P = dyn.params.with_payload(m, c_l, I_c)
    diff = dyn.slotine_li(q, qd, w, a, params=P) - dyn.slotine_li(q, qd, w, a)
    I_o = I_c + m * (c_l @ c_l * torch.eye(3, dtype=torch.float64) - torch.outer(c_l, c_l))
    pi = torch.tensor([m, *(m * c_l), I_o[0, 0], I_o[1, 1], I_o[2, 2], I_o[0, 1], I_o[0, 2], I_o[1, 2]],
                      dtype=torch.float64)
    Y = dyn.payload_regressor(q, qd, w, a)
    np.testing.assert_allclose((Y @ pi[:, None]).squeeze(-1), diff, atol=1e-9)
