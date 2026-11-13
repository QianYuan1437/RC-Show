"""Train the reaching policy (PPO, MuJoCo Warp, thousands of worlds).

    python -m rczoo.rl.train --robot panda --iters 1500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..robot import load_robot
from .ppo import PPO, ExportedActor, PPOConfig
from .reach_env import ReachEnv, obs_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="panda")
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out or Path("runs") / f"reach_{args.robot}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    robot = load_robot(args.robot)
    env = ReachEnv(robot, args.envs, seed=args.seed)
    D = obs_dim(robot.ndof)
    pcfg = PPOConfig(num_steps=32, gamma=0.99, entropy_coef=0.003, value_coef=1.0, hidden=(256, 256, 128))
    ppo = PPO(D, robot.ndof, args.envs, pcfg)
    env.reset()
    env.t = torch.randint(0, env.max_steps, (args.envs,), device="cuda")  # desynchronise episodes
    obs = env.observe()
    log = open(out / "log.jsonl", "w")  # noqa: SIM115 - kept open for the whole run
    t0 = time.time()
    for it in range(args.iters + 1):
        sum_r = torch.zeros((), device="cuda")
        fin_p, fin_R, n_fin = (torch.zeros((), device="cuda") for _ in range(3))
        for _ in range(pcfg.num_steps):
            a = ppo.act_rollout(obs)
            r, bad, timeout, info = env.step(a)
            done = bad | timeout
            obs_next = env.observe().nan_to_num(0.0)
            tv = pcfg.gamma * ppo.value(obs_next) * (timeout & ~bad).float()
            ppo.record(r, done, tv)
            sum_r += r.sum()
            fin_p += (info["e_p"] * timeout).sum()
            fin_R += (info["e_R"] * timeout).sum()
            n_fin += timeout.sum()
            env.reset(done.nonzero().squeeze(-1))
            obs = env.observe()
        stats = ppo.update(obs)
        if it % 10 == 0:
            rec = {"iter": it, "reward": (sum_r / (pcfg.num_steps * args.envs)).item(),
                   "final_pos_mm": (fin_p / n_fin.clamp_min(1)).item() * 1e3,
                   "final_rot_deg": (fin_R / n_fin.clamp_min(1)).item() * 57.2958, **stats,
                   "time_min": (time.time() - t0) / 60, "transitions": (it + 1) * pcfg.num_steps * args.envs}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
        if it % 100 == 0:
            torch.save({"state_dict": ppo.ac.state_dict(), "iter": it, "obs_dim": D}, out / "model.pt")
    torch.save({"state_dict": ppo.ac.state_dict(), "iter": args.iters, "obs_dim": D}, out / "model.pt")
    torch.save(ExportedActor(ppo.ac).cpu(), out / "actor.pt")


if __name__ == "__main__":
    main()
