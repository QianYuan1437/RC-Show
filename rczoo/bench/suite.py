"""The RC Zoo benchmark: every controller on both arms in batched MuJoCo Warp (1 kHz torque
control, the plant differs from the controllers' nominal model only where a condition says so).

    python -m rczoo.bench.suite --out site/docs/public/data [--robots panda,sr3] [--B 256]

Tracks
  A  tracking   quintic joint trajectories (1.5 s + 0.5 s hold); nominal plant, unknown 1.5 kg
                payload, joint dry friction (2 % of the torque limit)
  B  push       hold a pose while a 30 N, 0.2 s pulse hits the TCP
  C  contact    approach a surface, press with 10 N, slide 15 cm; three surface stiffnesses
  D  guiding    a "hand" drags the TCP 30 cm; how hard does the person have to pull?
  E  collision  momentum-observer collision detection during motion (half the worlds are hit)
  F  ilqr       energy-optimal (iLQR + TVLQR) vs quintic + computed torque, same duration

Outputs ``benchmark.json`` (metrics) and ``examples.json`` (time series of a few worlds).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from ..control import CollisionReaction, TVLQRTracking, ilqr, make
from ..control.gains import CART_K, CART_K_SOFT
from ..dynamics import Dynamics
from ..model import PlantConfig
from ..robot import load_robot
from ..runner import run
from ..sim import BatchSim
from ..tasks import (
    PROBE_RADIUS,
    CollisionTask,
    ContactTask,
    GuidingTask,
    JointTrajectoryTask,
    PushTask,
    downward_configs,
    sample_pairs,
)

TRACKING = ["pid", "pd_grav", "velocity", "joint_impedance", "pd_ff", "ctc", "smc", "adaptive",
            "resolved_rate", "cart_impedance", "osc", "qp", "mpc"]
PUSH = ["pid", "pd_grav", "joint_impedance", "ctc", "smc", "cart_impedance", "osc", "qp"]
CONTACT = ["cart_impedance", "admittance", "hybrid", "parallel"]
SLOW = {"mpc": 64}  # controllers that re-plan on the host run on fewer worlds


def _f(x, nd: int = 4):
    """Round for JSON (tensors / floats, nested)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().tolist()
    if isinstance(x, float):
        return float(f"{x:.{nd}g}") if math.isfinite(x) else None
    if isinstance(x, list):
        return [_f(v, nd) for v in x]
    if isinstance(x, dict):
        return {k: _f(v, nd) for k, v in x.items()}
    return x


def _med(x):
    return float(x.median())


def _example(out, keys, worlds=(0,), every=2):
    """Time series of a few worlds for the site's plots (50 Hz)."""
    ex = {"t": out["t"][0, ::every]}
    for k, fn in keys.items():
        v = fn(out)
        ex[k] = v[list(worlds), ::every]
    return ex


def _tcp_err(out):
    return (out["p"] - out["p_ref"]).norm(dim=-1)


def friction_values(dyn) -> list[float]:
    return (0.02 * dyn.torque_limit).tolist()


# ------------------------------------------------------------------------------- A: tracking

def track_tracking(robot, dyn, B, seed=0):
    q0, q1 = sample_pairs(dyn, B, seed=seed)
    conds = {
        "nominal": PlantConfig(),
        "payload": PlantConfig(payload=1.5),
        "friction": PlantConfig(frictionloss=friction_values(dyn)),
    }
    metrics, examples = {}, {}
    for cname, cfg in conds.items():
        metrics[cname], examples[cname] = {}, {}
        sims = {}
        for nm in TRACKING:
            b = min(B, SLOW.get(nm, B))
            if b not in sims:
                sims[b] = BatchSim(robot, b, cfg)
            task = JointTrajectoryTask(dyn, q0[:b], q1[:b], T=1.5, hold=0.5)
            out = run(sims[b], make(nm, dyn), task)
            ep = _tcp_err(out)
            eq = (out["q"] - out["q_ref"]).norm(dim=-1)
            bad = ~torch.isfinite(ep).all(1)
            ep = torch.nan_to_num(ep, nan=1.0)
            metrics[cname][nm] = {
                "B": b,
                "tcp_max_mm": _med(ep.max(1).values) * 1e3,
                "tcp_max_p90_mm": float(ep.max(1).values.quantile(0.9)) * 1e3,
                "tcp_rms_mm": float(ep.pow(2).mean().sqrt()) * 1e3,
                "final_mm": _med(ep[:, -1]) * 1e3,
                "q_rms_deg": float(torch.nan_to_num(eq).pow(2).mean().sqrt()) * 180 / math.pi,
                "saturation": float(out["saturation_frac"].mean()),
                "energy_J": float(out["energy"].mean()),
                "tau_sq": float(out["tau_sq"].mean()),
                "diverged": int(bad.sum()),
                "us_per_step": out["us_per_step"],
            }
            print(f"  [{cname}] {nm:15s} tcp_max={metrics[cname][nm]['tcp_max_mm']:7.2f}mm "
                  f"final={metrics[cname][nm]['final_mm']:6.3f}mm sat={metrics[cname][nm]['saturation']:.3f} "
                  f"{out['us_per_step']:.0f}us/step", flush=True)
            examples[cname][nm] = _example(out, {
                "e_tcp_mm": lambda o: _tcp_err(o) * 1e3,
                "tau_rel": lambda o: o["tau"] / dyn.torque_limit,
                "q": lambda o: o["q"], "q_ref": lambda o: o["q_ref"],
            })
    return metrics, examples


# ----------------------------------------------------------------------------------- B: push

def track_push(robot, dyn, B, seed=1):
    q0, _ = sample_pairs(dyn, B, seed=seed)
    sim = BatchSim(robot, B, PlantConfig())
    metrics, examples = {}, {}
    for nm in PUSH:
        task = PushTask(dyn, q0, force=30.0, t_on=0.3, t_off=0.5, duration=2.0)
        out = run(sim, make(nm, dyn), task)
        ep = _tcp_err(out)
        t = out["t"][0]
        after = t >= task.t_off
        # settling: last time the deviation exceeds 1 mm (relative to the end of the pulse)
        above = (ep > 1e-3) & after
        idx = torch.where(above.any(1), (above * torch.arange(len(t), device=t.device)).argmax(1), 0)
        settle = (t[idx] - task.t_off).clamp(min=0)
        metrics[nm] = {
            "max_dev_mm": _med(ep.max(1).values) * 1e3,
            "settle_s": _med(settle),
            "unsettled": int((ep[:, -1] > 1e-3).sum()),
            "final_mm": _med(ep[:, -1]) * 1e3,
            "stiffness_N_per_mm": 30.0 / max(_med(ep.max(1).values) * 1e3, 1e-6),
        }
        print(f"  push {nm:15s} max={metrics[nm]['max_dev_mm']:6.2f}mm settle={metrics[nm]['settle_s']:.3f}s", flush=True)
        examples[nm] = _example(out, {"dev_mm": lambda o: _tcp_err(o) * 1e3}, worlds=(0, 1, 2))
    return metrics, examples


# -------------------------------------------------------------------------------- C: contact

SURFACES = {  # MuJoCo solref time constant: softer surface = larger time constant
    "soft": (0.1, 1.0),
    "medium": (0.05, 1.0),
    "stiff": (0.02, 1.0),
}


def contact_lower_by(nm: str, f_d: float) -> float:
    """How far below the surface the reference goes: impedance-type controllers produce force only
    by spring deflection (their designer assumes a rigid surface), the explicit force controllers
    need a slight downward bias only to make sure the probe touches down."""
    return {"cart_impedance": f_d / CART_K[2], "admittance": f_d / CART_K_SOFT[2], "hybrid": 0.003}.get(nm, 0.0)


def track_contact(robot, dyn, B, seed=0, f_d=10.0):
    q0 = downward_configs(dyn, B, seed=seed)
    p0, _ = dyn.fk(q0)
    surface_z = p0[:, 2] - PROBE_RADIUS - 0.05  # 5 cm below the probe
    metrics, examples = {}, {}
    for sname, solref in SURFACES.items():
        sim = BatchSim(robot, B, PlantConfig(surface=True, surface_solref=solref))
        metrics[sname], examples[sname] = {}, {}
        for nm in CONTACT:
            sim.mocap_pos[:, 0, 0] = p0[:, 0].float()
            sim.mocap_pos[:, 0, 1] = p0[:, 1].float()
            sim.mocap_pos[:, 0, 2] = (surface_z - 0.01).float()  # box half height 1 cm
            task = ContactTask(dyn, q0, surface_z, f_d=f_d, lower_by=contact_lower_by(nm, f_d))
            out = run(sim, make(nm, dyn), task)
            t = out["t"][0]
            fz = out["ft"][..., 2]
            f_ref = (((t - task.t_approach) / 0.3).clamp(0, 1) * f_d)[None]
            win = t >= task.t_approach + 0.3  # pressing and sliding
            slide = (t >= task.t_approach + 0.3) & (t < task.t_approach + task.t_slide)
            exy = (out["p"][..., :2] - out["p_ref"][..., :2]).norm(dim=-1)
            pen = surface_z[:, None] + PROBE_RADIUS - out["p"][..., 2]
            fe = fz - f_ref
            metrics[sname][nm] = {
                "peak_N": _med(fz.max(1).values),
                "mean_N": float(fz[:, win].mean()),
                "rms_err_N": float(fe[:, win].pow(2).mean().sqrt()),
                "rms_err_slide_N": float(fe[:, slide].pow(2).mean().sqrt()),
                "xy_err_mm": float(exy[:, win].mean()) * 1e3,
                "penetration_mm": float(pen[:, win].mean()) * 1e3,
                "lost_contact": float((fz[:, win] < 0.5).float().mean()),
                "k_env_N_per_mm": float(fz[:, win].mean() / max(float(pen[:, win].mean()) * 1e3, 1e-6)),
            }
            m = metrics[sname][nm]
            print(f"  contact[{sname}] {nm:15s} mean={m['mean_N']:6.2f}N rms_err={m['rms_err_N']:5.2f}N peak={m['peak_N']:6.1f}N "
                  f"xy={m['xy_err_mm']:5.2f}mm pen={m['penetration_mm']:.2f}mm", flush=True)
            examples[sname][nm] = _example(out, {
                "fz": lambda o: o["ft"][..., 2],
                "f_ref": lambda o: f_ref.expand_as(o["ft"][..., 2]),
                "z_mm": lambda o: (o["p"][..., 2] - surface_z[:, None] - PROBE_RADIUS) * 1e3,
                "xy_err_mm": lambda o: (o["p"][..., :2] - o["p_ref"][..., :2]).norm(dim=-1) * 1e3,
            })
    return metrics, examples


# -------------------------------------------------------------------------------- D: guiding

def track_guiding(robot, dyn, B, seed=2):
    q0, _ = sample_pairs(dyn, B, seed=seed)
    fric = friction_values(dyn)
    sim = BatchSim(robot, B, PlantConfig(frictionloss=fric))
    variants = {
        "joint_impedance": ("joint_impedance", {}),  # stays stiff: a baseline, not a teaching mode
        "gravity": ("gravity", {}),
        "drag": ("drag", {"friction_comp": [0.8 * f for f in fric]}),
        "drag_nofc": ("drag", {}),
    }
    metrics, examples = {}, {}
    for key, (nm, kw) in variants.items():
        task = GuidingTask(dyn, q0)
        ctrl = make(nm, dyn, **kw)
        if nm == "joint_impedance":
            from ..control.base import Ref
            z = torch.zeros_like(q0)
            task.reference = lambda t, z=z: Ref(q=q0, qd=z, qdd=z)
        out = run(sim, ctrl, task)
        F = out["hand_force"].norm(dim=-1)
        t = out["t"][0]
        mov = t < task.T
        lag = (out["hand_p"] - out["p"]).norm(dim=-1)
        metrics[key] = {
            "mean_force_N": _med(F[:, mov].mean(1)),
            "peak_force_N": _med(F.max(1).values),
            "final_lag_mm": _med(lag[:, -1]) * 1e3,
            "drift_mm": _med((out["p"][:, -1] - out["p"][:, int(task.T / 0.01)]).norm(dim=-1)) * 1e3,
        }
        print(f"  guide {key:16s} mean={metrics[key]['mean_force_N']:6.2f}N peak={metrics[key]['peak_force_N']:6.2f}N", flush=True)
        examples[key] = _example(out, {"force_N": lambda o: o["hand_force"].norm(dim=-1)}, worlds=(0, 1, 2))
    return metrics, examples


# ------------------------------------------------------------------------------ E: collision

def track_collision(robot, dyn, B, seed=3):
    q0, q1 = sample_pairs(dyn, B, seed=seed)
    metrics, examples = {}, {}
    from ..dynamics import with_params
    # payload: 1 kg unknown to the controller and the observer; payload_id: the same load
    # identified and entered into the model (what Franka / xCore ask for when a tool is mounted)
    side = 0.08  # the plant's payload: a solid 8 cm cube, 5 cm below the TCP
    dyn_id = with_params(dyn, dyn.payload_params(1.0, inertia=torch.eye(3, dtype=dyn.dtype) * side**2 / 6))
    conds = {"nominal": (PlantConfig(), dyn), "payload": (PlantConfig(payload=1.0), dyn),
             "payload_id": (PlantConfig(payload=1.0), dyn_id)}
    for cname, (cfg, model) in conds.items():
        sim = BatchSim(robot, B, cfg)
        metrics[cname], examples[cname] = {}, {}
        for th_frac in (0.03, 0.05, 0.1, 0.2):
            task = CollisionTask(dyn, q0, q1, T=1.5, hold=0.5)
            ctrl = CollisionReaction(model, make("ctc", model), strategy="reflex",
                                     threshold=(th_frac * dyn.torque_limit).tolist())
            out = run(sim, ctrl, task)
            hit = out["hit"][:, -1] > 0.5
            t = out["t"][0]
            first = torch.where(out["hit"] > 0.5, t[None].expand_as(out["hit"]), torch.full_like(out["hit"], 1e9)).min(1).values
            col = task.collides
            delay = first[col & hit] - task.t_on[col & hit]
            early = col & hit & (first < task.t_on)  # fired before the impact: a false alarm too
            metrics[cname][f"{th_frac:g}"] = {
                "threshold_frac": th_frac,
                "detection_rate": float((hit & col & ~early).float().sum() / col.sum()),
                "false_alarm_rate": float(((hit & ~col) | early).float().sum() / B),
                "delay_ms": float(delay[delay >= 0].median()) * 1e3 if (delay >= 0).any() else None,
            }
            m = metrics[cname][f"{th_frac:g}"]
            print(f"  collision[{cname}] th={th_frac:4.2f} det={m['detection_rate']:.2f} fa={m['false_alarm_rate']:.2f} "
                  f"delay={m['delay_ms']}", flush=True)
            if th_frac == 0.05:
                rel = out["r"].abs() / (th_frac * dyn.torque_limit)
                examples[cname] = _example(out, {
                    "r_rel_max": lambda o: rel.max(-1).values,
                    "hit": lambda o: o["hit"],
                    "push": lambda o: ((o["t"] >= task.t_on[:, None]) & (o["t"] < task.t_on[:, None] + 0.15)
                                       & task.collides[:, None]).to(o["t"].dtype),
                }, worlds=(0, 1))
    return metrics, examples


# ----------------------------------------------------------------------------------- F: iLQR

def track_ilqr(robot, dyn, B=16, seed=4, T_s=1.0, dt=0.01):
    q0, q1 = sample_pairs(dyn, B, seed=seed, step=0.7)
    x0 = torch.cat([q0, torch.zeros_like(q0)], -1)
    xg = torch.cat([q1, torch.zeros_like(q1)], -1)
    T = int(round(T_s / dt))
    t0 = time.perf_counter()
    sol = ilqr(dyn, x0, xg, T, dt=dt, iters=60)
    solve_s = time.perf_counter() - t0
    sim = BatchSim(robot, B, PlantConfig())
    res, examples = {}, {}

    for key in ("quintic_ctc", "ilqr_ctc", "ilqr_tvlqr"):
        task = JointTrajectoryTask(dyn, q0, q1, T=T_s, hold=0.5)
        if key != "quintic_ctc":
            task.reference = sol.reference  # the optimised trajectory replaces the quintic
        ctrl = TVLQRTracking(dyn, sol) if key == "ilqr_tvlqr" else make("ctc", dyn)
        out = run(sim, ctrl, task)
        mov = out["t"][0] < T_s
        tau = out["tau"]
        e_final = (out["q"][:, -1] - q1).norm(dim=-1)
        res[key] = {
            "tau_sq": float(out["tau_sq"].mean()),
            "tau_sq_motion": float((tau[:, mov] ** 2).sum(-1).mean(1).mean() * T_s),
            "peak_tau_rel": _med((tau.abs() / dyn.torque_limit).amax((1, 2))),
            "energy_J": float(out["energy"].mean()),
            "final_err_deg": _med(e_final) * 180 / math.pi,
        }
        print(f"  ilqr {key:12s} {res[key]}", flush=True)
        examples[key] = _example(out, {"tau_rel": lambda o: o["tau"] / dyn.torque_limit,
                                       "q": lambda o: o["q"]}, every=1)
    res["plan"] = {"solve_s": solve_s, "iters": len(sol.history) - 1, "B": B, "dt": dt,
                   "tau_sq": float((sol.u**2).sum(-1).sum(1).mean() * dt),
                   "cost_history": [float(h.mean()) for h in sol.history]}
    res["saving"] = 1 - res["ilqr_ctc"]["tau_sq_motion"] / res["quintic_ctc"]["tau_sq_motion"]
    return res, examples


# ------------------------------------------------------------------------------ G: learning

LEARNING = ["pd_grav", "ctc", "smc", "adaptive", "rl"]


def track_learning(robot, dyn, B, seed=5):
    """Reach a goal pose with an unknown payload: model-based controllers follow a 1.5 s quintic
    to the goal configuration, the RL policy only gets the goal pose."""
    from ..tasks import GoalTask
    q0, q1 = sample_pairs(dyn, B, seed=seed, step=1.0)
    metrics, examples = {}, {}
    for cname, cfg in {"0kg": PlantConfig(), "1kg": PlantConfig(payload=1.0), "2kg": PlantConfig(payload=2.0),
                       "friction": PlantConfig(frictionloss=friction_values(dyn))}.items():
        sim = BatchSim(robot, B, cfg)
        metrics[cname], examples[cname] = {}, {}
        for nm in LEARNING:
            task = GoalTask(dyn, q0, q1) if nm == "rl" else JointTrajectoryTask(dyn, q0, q1, T=1.5, hold=1.5)
            out = run(sim, make(nm, dyn), task)
            p1, R1 = dyn.fk(q1)
            ep = (out["p"] - p1[:, None]).norm(dim=-1)
            from ..dynamics import orientation_error
            eR = orientation_error(R1[:, None].expand(-1, ep.shape[1], -1, -1).reshape(-1, 3, 3),
                                   dyn.fk(out["q"].reshape(-1, dyn.n))[1]).norm(dim=-1).view(B, -1)
            t = out["t"][0]
            far = ep > 5e-3
            idx = (far * torch.arange(len(t), device=t.device)).argmax(1)
            reach_t = torch.where(far.any(1), t[(idx + 1).clamp(max=len(t) - 1)], t[0])
            ok = (ep[:, -1] < 5e-3) & (eR[:, -1] < 2 * math.pi / 180)
            metrics[cname][nm] = {
                "final_mm": _med(ep[:, -1]) * 1e3,
                "final_p90_mm": float(ep[:, -1].quantile(0.9)) * 1e3,
                "final_rot_deg": _med(eR[:, -1]) * 180 / math.pi,
                "reach_s": _med(reach_t),
                "success": float(ok.float().mean()),
                "energy_J": float(out["energy"].mean()),
            }
            m = metrics[cname][nm]
            print(f"  learning[{cname}] {nm:9s} final={m['final_mm']:6.2f}mm p90={m['final_p90_mm']:6.2f} "
                  f"rot={m['final_rot_deg']:.2f}deg reach={m['reach_s']:.2f}s success={m['success']:.2f}", flush=True)
            examples[cname][nm] = _example(out, {"e_mm": lambda o: (o["p"] - p1[:, None]).norm(dim=-1) * 1e3},
                                           worlds=(0, 1, 2))
    return metrics, examples


TRACKS = {"tracking": track_tracking, "push": track_push, "contact": track_contact,
          "guiding": track_guiding, "collision": track_collision, "ilqr": track_ilqr,
          "learning": track_learning}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("site/docs/public/data"))
    ap.add_argument("--robots", default="panda,sr3")
    ap.add_argument("--tracks", default=",".join(t for t in TRACKS if t != "learning"))  # learning: after RL training
    ap.add_argument("--B", type=int, default=256)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bench_path, ex_path = args.out / "benchmark.json", args.out / "examples.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    examples = json.loads(ex_path.read_text()) if ex_path.exists() else {}
    for rn in args.robots.split(","):
        robot = load_robot(rn)
        dyn = Dynamics(robot, "cuda")
        bench.setdefault(rn, {})
        examples.setdefault(rn, {})
        for tr in args.tracks.split(","):
            print(f"== {rn} / {tr}", flush=True)
            t0 = time.perf_counter()
            fn = TRACKS[tr]
            m, ex = fn(robot, dyn, args.B) if tr != "ilqr" else fn(robot, dyn)
            bench[rn][tr] = _f(m)
            examples[rn][tr] = _f(ex, 4)
            print(f"   ({time.perf_counter() - t0:.0f}s)", flush=True)
            bench_path.write_text(json.dumps(bench, separators=(",", ":")))
            ex_path.write_text(json.dumps(examples, separators=(",", ":")))
    bench["meta"] = {"dt": 0.001, "B": args.B, "torch": torch.__version__,
                     "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    bench_path.write_text(json.dumps(bench, separators=(",", ":")))


if __name__ == "__main__":
    main()
