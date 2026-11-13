"""Closed-loop episode runner: 1 kHz control on any simulator, with decimated logging.

On the GPU one control cycle (scripted events, reference, controller, clipping, statistics) is
captured once into a CUDA graph and replayed; the physics step is a separate Warp graph on the
same stream. This removes the launch overhead of the many small tensor ops (a ~20x speed-up).
"""

from __future__ import annotations

import time

import torch

from .control.base import Controller, State
from .dynamics import orientation_error

LOG_KEYS = ("t", "q", "qd", "q_ref", "tau", "p", "p_ref", "ft", "e_rot")  # plus task/controller log_values()


@torch.no_grad()
def run(sim, ctrl: Controller, task, log_every: int = 10, hooks=None, graph: bool | None = None) -> dict:
    """Run one episode on all worlds. Returns logs (B, K, ...) and statistics."""
    q0 = task.start()
    B = q0.shape[0]
    dev = q0.device
    sim.reset(q0)
    dt = sim.mjm.opt.timestep
    steps = int(round(task.duration / dt))
    use_graph = (graph if graph is not None else (dev.type == "cuda" and getattr(ctrl, "graph_safe", True))) and not hooks
    t_buf = torch.zeros(B, dtype=torch.float64, device=dev)
    tl = ctrl.dyn.torque_limit
    acc = {k: torch.zeros(B, dtype=torch.float64, device=dev) for k in ("sat", "energy", "tau_sq")}
    out_tau = torch.zeros(B, ctrl.dyn.n, dtype=torch.float64, device=dev)
    logbuf: dict[str, torch.Tensor] = {}

    def cycle():
        task.events(sim, t_buf)
        state = State(t=t_buf, q=sim.q, qd=sim.qd, ft=sim.ft)
        ref = task.reference(t_buf)
        tau = ctrl(state, ref)
        tau_c = torch.maximum(torch.minimum(tau, tl), -tl)
        acc["sat"].add_((tau.abs() > tl * 1.0001).any(-1).to(torch.float64))
        acc["energy"].add_((tau_c * state.qd).abs().sum(-1) * dt)
        acc["tau_sq"].add_((tau_c**2).sum(-1) * dt)
        out_tau.copy_(tau_c)
        p, R = ctrl.dyn.fk(state.q)
        vals = {
            "t": t_buf, "q": state.q, "qd": state.qd,
            "q_ref": ref.q if ref.q is not None else torch.zeros_like(state.q),
            "tau": tau_c, "p": p,
            "p_ref": ref.p if ref.p is not None else torch.zeros_like(p),
            "ft": state.ft if state.ft is not None else torch.zeros(B, 6, dtype=p.dtype, device=dev),
            "e_rot": orientation_error(ref.R, R).norm(dim=-1) if ref.R is not None else torch.zeros_like(t_buf),
        }
        for src in (task, ctrl):  # optional extra signals (hand force, observer residual, ...)
            if hasattr(src, "log_values"):
                vals.update(src.log_values())
        for k, v in vals.items():
            if k not in logbuf:
                logbuf[k] = v.detach().clone()
            else:
                logbuf[k].copy_(v)
        if hooks:
            for h in hooks:
                h(state, ref, tau_c)

    state0 = State(t=t_buf, q=sim.q, qd=sim.qd, ft=sim.ft)
    ctrl.reset(state0, task.reference(t_buf))
    g = None
    if use_graph:
        # warm-up (allocates lazily created state), then restore the initial state and capture
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                cycle()
        torch.cuda.current_stream().wait_stream(side)
        sim.reset(q0)
        ctrl.reset(state0, task.reference(t_buf))
        for v in acc.values():
            v.zero_()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            cycle()
        sim.reset(q0)
    logs = {k: [] for k in logbuf} if logbuf else None
    t0 = time.perf_counter()
    for k in range(steps):
        t_buf.fill_(k * dt)
        if g is not None:
            g.replay()
        else:
            cycle()
        if k % log_every == 0:
            if logs is None:
                logs = {key: [] for key in logbuf}
            for key in logs:
                logs[key].append(logbuf[key].clone())
        sim.step(out_tau)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    out = {k: torch.stack(v, 1) for k, v in logs.items()}
    out.update(saturation_frac=acc["sat"] / steps, energy=acc["energy"], tau_sq=acc["tau_sq"], wall=wall,
               steps=steps, us_per_step=wall / steps * 1e6, graph=g is not None)
    return out
