
import numpy as np
import pytest
import torch

from rczoo.dynamics import Dynamics
from rczoo.robot import load_robot
from rczoo.traj import DoubleS, MoveC, MoveL, Trapezoid, otg_step, quintic, so3_exp, sync, toppra

f64 = torch.float64


def _fd_check(prof_fn, T, tol, accel=True):
    t = torch.linspace(0, T, 4001, dtype=f64)
    pr = prof_fn(t)
    dt = float(t[1] - t[0])
    pairs = ((pr.p, pr.v), (pr.v, pr.a)) if accel else ((pr.p, pr.v),)
    for x, dx in pairs:
        num = (x[2:] - x[:-2]) / (2 * dt)
        assert torch.allclose(num, dx[1:-1], atol=tol * (1 + float(dx.abs().max()))), float((num - dx[1:-1]).abs().max())


def test_quintic_boundary_and_derivatives():
    pr = quintic(torch.tensor([0.0, 1.0], dtype=f64), torch.tensor([1.0, -2.0], dtype=f64), 2.0,
                 torch.tensor([[0.0], [2.0]], dtype=f64))
    assert torch.allclose(pr.p, torch.tensor([[0.0, 1.0], [1.0, -2.0]], dtype=f64))
    assert torch.allclose(pr.v, torch.zeros(2, 2, dtype=f64), atol=1e-12)
    _fd_check(lambda t: quintic(torch.tensor(0.3, dtype=f64), torch.tensor(1.2, dtype=f64), 1.5, t,
                                v0=0.2, v1=-0.1, a0=0.5, a1=0.0), 1.5, 2e-3)


@pytest.mark.parametrize("cls,args", [(Trapezoid, (0.8, 2.0)), (DoubleS, (0.8, 2.0, 10.0))])
@pytest.mark.parametrize("dist", [0.01, 0.3, 2.0])
def test_profiles_respect_limits(cls, args, dist):
    prof = cls(torch.tensor([0.1], dtype=f64), torch.tensor([0.1 - dist], dtype=f64), *args)
    T = float(prof.duration)
    t = torch.linspace(0, T * 1.05, 20001, dtype=f64)[:, None]
    pr = prof.eval_own(t)
    assert abs(float(pr.p[-1, 0]) - (0.1 - dist)) < 1e-9
    assert float(pr.v.abs().max()) <= args[0] + 1e-9
    assert float(pr.a.abs().max()) <= args[1] + 1e-9
    if cls is DoubleS:
        assert float(pr.j.abs().max()) <= args[2] + 1e-9
    # a trapezoid's acceleration is discontinuous: only check p -> v there
    _fd_check(lambda tt: prof.eval_own(tt[:, None]), T, 2e-3, accel=cls is DoubleS)


def test_double_s_is_time_optimal_vs_trapezoid():
    tr = Trapezoid(torch.tensor([0.0], dtype=f64), torch.tensor([1.0], dtype=f64), 1.0, 2.0)
    ds = DoubleS(torch.tensor([0.0], dtype=f64), torch.tensor([1.0], dtype=f64), 1.0, 2.0, 1e6)
    assert abs(float(tr.duration) - float(ds.duration)) < 1e-3  # huge jerk limit -> trapezoid
    ds2 = DoubleS(torch.tensor([0.0], dtype=f64), torch.tensor([1.0], dtype=f64), 1.0, 2.0, 5.0)
    assert float(ds2.duration) > float(tr.duration)


def test_sync_ends_together():
    q0 = torch.tensor([0.0, 0.0, 0.0], dtype=f64)
    q1 = torch.tensor([1.0, -0.2, 0.05], dtype=f64)
    prof = DoubleS(q0, q1, 1.0, 3.0, 20.0)
    T = float(prof.duration)
    pr = sync(prof, torch.tensor([[0.5 * T], [T]], dtype=f64))
    assert torch.allclose(pr.p[1], q1)
    frac = (pr.p[0] - q0) / (q1 - q0)
    assert torch.allclose(frac, frac[0].expand(3), atol=1e-12)  # all axes at the same phase


def test_movel_and_movec():
    R0 = torch.eye(3, dtype=f64)
    R1 = so3_exp(torch.tensor([0.0, 0.0, 0.8], dtype=f64))
    ml = MoveL(torch.tensor([0.5, 0.0, 0.4], dtype=f64), R0, torch.tensor([0.5, 0.3, 0.3], dtype=f64), R1)
    p, R, v, _ = ml(torch.tensor([0.0, ml.T], dtype=f64))
    assert torch.allclose(p[1], torch.tensor([0.5, 0.3, 0.3], dtype=f64)) and torch.allclose(R[1], R1, atol=1e-12)
    assert float(v.abs().max()) < 1e-9
    a_, b_, c_ = np.array([0.5, 0, 0.4]), np.array([0.6, 0.1, 0.4]), np.array([0.5, 0.2, 0.4])
    mc = MoveC(a_, b_, c_, R0, R0)
    t = torch.linspace(0, mc.T, 2001, dtype=f64)
    p, _, v, _ = mc(t)
    assert np.allclose(p[-1].numpy(), c_, atol=1e-9)
    assert (p - torch.as_tensor(b_)).norm(dim=1).min() < 1e-3  # passes through the via point
    r = (p - mc.center).norm(dim=1)
    assert torch.allclose(r, r[0].expand_as(r), atol=1e-9)
    assert float(v[:, :3].norm(dim=1).max()) <= 0.25 + 1e-6


@pytest.mark.parametrize("name", ["panda", "sr3"])
def test_toppra_respects_limits(name):
    robot = load_robot(name)
    dyn = Dynamics(robot)
    q0, q1 = torch.as_tensor(robot.home), torch.as_tensor(robot.home) + 0.8
    q1 = dyn.clamp(q1[None])[0]
    s = np.linspace(0, 1, 201)
    # a curved path: straight line plus a bump on joint 1
    bump = np.sin(np.pi * s)[:, None] * np.eye(robot.ndof)[0] * 0.5
    path = q0.numpy() + s[:, None] * (q1 - q0).numpy() + bump
    dq = np.gradient(path, s, axis=0)
    ddq = np.gradient(dq, s, axis=0)
    vmax, amax = robot.vel_limit, 5.0
    res = toppra(dyn, path, dq, ddq, s, vmax, amax, tau_max=robot.torque_limit * 0.8)
    sd = res.sd
    vel = dq * sd[:, None]
    assert (np.abs(vel) <= vmax * (1 + 1e-6) + 1e-9).all()
    sdd = np.diff(sd**2) / (2 * np.diff(s))
    acc = dq[:-1] * sdd[:, None] + ddq[:-1] * (sd[:-1] ** 2)[:, None]
    assert (np.abs(acc) <= amax * 1.02 + 1e-6).all()
    assert res.duration > 0 and sd[0] == 0 and sd[-1] < 1e-6


def test_otg_reaches_target():
    p = torch.tensor([0.0, 1.0, -0.5], dtype=f64)
    v = torch.tensor([0.5, -1.0, 0.0], dtype=f64)
    tgt = torch.tensor([1.0, 1.0, -0.4], dtype=f64)
    vmax, amax, dt = torch.tensor(1.0, dtype=f64), torch.tensor(2.0, dtype=f64), 1e-3
    for _ in range(4000):
        p, v, a = otg_step(p, v, tgt, vmax, amax, dt)
        assert float(a.abs().max()) <= 2.0 + 1e-6 and float(v.abs().max()) <= 1.0 + 1e-6 + 0
    assert torch.allclose(p, tgt, atol=1e-5) and torch.allclose(v, torch.zeros(3, dtype=f64), atol=1e-6)
