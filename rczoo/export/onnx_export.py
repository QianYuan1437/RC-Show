"""Export the reaching policies to ONNX for the browser (onnxruntime-web).

    python -m rczoo.export.onnx_export --out site/docs/public/models

Writes ``<out>/<robot>/reach.onnx`` and ``reach.json`` (observation layout, action scaling,
inner-loop gains and a few reference observation -> action pairs for the browser's self-test),
and the training curve to ``<out>/../data/training/reach_<robot>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ..control.gains import GAINS
from ..dynamics import Dynamics
from ..rl.common import POLICY_DT, TARGET_RATE, obs_dim, observation, target_step
from ..robot import load_robot


def export(name: str, runs: Path, out: Path):
    actor = torch.load(runs / f"reach_{name}" / "actor.pt", weights_only=False).eval().cpu()
    robot = load_robot(name)
    dyn = Dynamics(robot, "cpu")
    D = obs_dim(robot.ndof)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "reach.onnx"
    torch.onnx.export(actor, (torch.zeros(1, D),), str(path), input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=17, dynamo=False)
    # reference pairs (random but plausible states)
    g = torch.Generator().manual_seed(0)
    q = dyn.sample_q(4, margin=0.15, generator=g)
    q1 = dyn.sample_q(4, margin=0.15, generator=g)
    qd = torch.randn(4, robot.ndof, generator=g, dtype=torch.float64) * 0.3
    q_cmd = q + torch.randn(4, robot.ndof, generator=g, dtype=torch.float64) * 0.02
    p, R = dyn.fk(q)
    pg, Rg = dyn.fk(q1)
    a_prev = torch.rand(4, robot.ndof, generator=g) * 2 - 1
    obs = observation(dyn, q, qd, q_cmd, p, R, pg, Rg, a_prev)
    with torch.no_grad():
        act = actor(obs)
    sess = ort.InferenceSession(str(path))
    err = float(np.abs(sess.run(None, {"obs": obs.numpy()})[0] - act.numpy()).max())
    print(f"{name}: reach.onnx max |onnx - torch| = {err:.2e} ({path.stat().st_size / 1e3:.0f} kB)")
    meta = {
        "obs_dim": D, "act_dim": robot.ndof, "policy_dt": POLICY_DT, "target_rate": TARGET_RATE,
        "target_step": target_step(dyn).tolist(),
        "kp": GAINS[name]["kp"], "kd": GAINS[name]["kd"],
        "obs_layout": "q, 0.1 qd, 10 (q_cmd - q), p_tcp, 5 (p_goal - p), Log(R_goal R^T), a_prev",
        "fixtures": {"q": q.tolist(), "qd": qd.tolist(), "q_cmd": q_cmd.tolist(), "p_goal": pg.tolist(),
                     "R_goal": Rg.tolist(), "a_prev": a_prev.tolist(), "obs": obs.tolist(), "action": act.tolist()},
    }
    (d / "reach.json").write_text(json.dumps(meta))
    export_log(name, runs, out.parent / "data" / "training")


def export_log(name: str, runs: Path, out: Path):
    """Training curve for the site: data/training/reach_<robot>.json (column arrays)."""
    recs = [json.loads(line) for line in (runs / f"reach_{name}" / "log.jsonl").read_text().splitlines() if line]
    keys = ["iter", "transitions", "reward", "final_pos_mm", "final_rot_deg", "time_min"]
    out.mkdir(parents=True, exist_ok=True)
    (out / f"reach_{name}.json").write_text(json.dumps({k: [round(r[k], 4) for r in recs] for k in keys}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("site/docs/public/models"))
    ap.add_argument("--robots", default="panda,sr3")
    a = ap.parse_args()
    for rn in a.robots.split(","):
        export(rn, a.runs, a.out)
