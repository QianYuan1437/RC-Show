"""Minimal PPO (Schulman et al. 2017) for massively parallel GPU environments.

The structure follows rsl_rl, which PolicyIK and most Isaac/MJX locomotion work use: Gaussian
policy with a state-independent log-std, GAE, clipped surrogate, clipped value loss, and an
adaptive learning rate that keeps the per-update KL near a target. Observations are normalised
with running statistics that get frozen into the exported actor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PPOConfig:
    num_steps: int = 24  # rollout length per env per iteration
    epochs: int = 5
    minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 2.0
    lr: float = 1e-3
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    kl_target: float = 0.01
    max_grad_norm: float = 1.0
    hidden: tuple[int, ...] = (256, 256)
    init_log_std: float = -0.5
    max_log_std: float = 0.0  # actions are clipped to [-1, 1]; a larger std is pure noise
    min_log_std: float = -20.0  # a floor keeps the KL-adaptive learning rate from collapsing


class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip
        self.frozen = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.frozen:
            return
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean += delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta**2 * self.count * bc / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def mlp(d_in, d_out, hidden, act=nn.ELU):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, d_out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.actor = mlp(obs_dim, act_dim, cfg.hidden)
        self.critic = mlp(obs_dim, 1, cfg.hidden)
        self.log_std = nn.Parameter(torch.full((act_dim,), cfg.init_log_std))
        self.max_log_std = cfg.max_log_std
        self.min_log_std = cfg.min_log_std
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs_n):
        mu = self.actor(obs_n)
        std = self.log_std.clamp(self.min_log_std, self.max_log_std).exp()
        return torch.distributions.Normal(mu, std.expand_as(mu))

    def act(self, obs):
        obs_n = self.norm(obs)
        d = self.dist(obs_n)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.critic(obs_n).squeeze(-1), d.mean, d.stddev

    def act_deterministic(self, obs):
        return self.actor(self.norm(obs))


class ExportedActor(nn.Module):
    """obs -> deterministic action with the observation normaliser baked in (for ONNX)."""

    def __init__(self, ac: ActorCritic):
        super().__init__()
        self.register_buffer("mean", ac.norm.mean.detach().clone())
        self.register_buffer("std", torch.sqrt(ac.norm.var + 1e-8).detach().clone())
        self.clip = ac.norm.clip
        import copy
        self.actor = copy.deepcopy(ac.actor)

    def forward(self, obs):
        x = ((obs - self.mean) / self.std).clamp(-self.clip, self.clip)
        return self.actor(x)


class PPO:
    def __init__(self, obs_dim: int, act_dim: int, num_envs: int, cfg: PPOConfig,
                 device: str = "cuda"):
        self.cfg = cfg
        self.ac = ActorCritic(obs_dim, act_dim, cfg).to(device)
        self.opt = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self.lr = cfg.lr
        T, N = cfg.num_steps, num_envs
        z = lambda *s: torch.zeros(*s, device=device)
        self.obs = z(T, N, obs_dim)
        self.act = z(T, N, act_dim)
        self.logp = z(T, N)
        self.val = z(T, N)
        self.rew = z(T, N)
        self.done = z(T, N)
        self.mu = z(T, N, act_dim)
        self.sigma = z(T, N, act_dim)
        self.t = 0

    @torch.no_grad()
    def act_rollout(self, obs):
        self.ac.norm.update(obs)
        a, logp, v, mu, sigma = self.ac.act(obs)
        i = self.t
        self.obs[i], self.act[i], self.logp[i], self.val[i] = obs, a, logp, v
        self.mu[i], self.sigma[i] = mu, sigma
        return a

    @torch.no_grad()
    def record(self, rew, done, timeout_value=None):
        """``timeout_value``: bootstrap value already multiplied by gamma for truncated envs."""
        r = rew.clone()
        if timeout_value is not None:
            r += timeout_value
        self.rew[self.t], self.done[self.t] = r, done.float()
        self.t += 1

    @torch.no_grad()
    def value(self, obs):
        return self.ac.critic(self.ac.norm(obs)).squeeze(-1)

    def update(self, last_obs):
        cfg = self.cfg
        with torch.no_grad():
            last_v = self.value(last_obs)
            adv = torch.zeros_like(self.rew)
            gae = torch.zeros_like(last_v)
            for t in reversed(range(cfg.num_steps)):
                nv = last_v if t == cfg.num_steps - 1 else self.val[t + 1]
                nonterm = 1.0 - self.done[t]
                delta = self.rew[t] + cfg.gamma * nv * nonterm - self.val[t]
                gae = delta + cfg.gamma * cfg.lam * nonterm * gae
                adv[t] = gae
            ret = adv + self.val
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        obs, act, logp_old, val_old = flat(self.obs), flat(self.act), flat(self.logp), flat(self.val)
        mu_old, sig_old, adv, ret = flat(self.mu), flat(self.sigma), flat(adv), flat(ret)
        B = obs.shape[0]
        mb = B // cfg.minibatches
        stats = {"kl": 0.0, "pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
        n = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(B, device=obs.device)
            for k in range(cfg.minibatches):
                idx = perm[k * mb:(k + 1) * mb]
                o = self.ac.norm(obs[idx])
                d = self.ac.dist(o)
                logp = d.log_prob(act[idx]).sum(-1)
                ent = d.entropy().sum(-1).mean()
                with torch.no_grad():  # analytic KL(old || new) for the adaptive LR
                    s0, s1 = sig_old[idx], d.stddev
                    kl = (torch.log(s1 / s0) + (s0**2 + (mu_old[idx] - d.mean) ** 2) / (2 * s1**2)
                          - 0.5).sum(-1).mean()
                    if kl > 2 * cfg.kl_target:
                        self.lr = max(cfg.lr_min, self.lr / 1.5)
                    elif 0 < kl < cfg.kl_target / 2:
                        self.lr = min(cfg.lr_max, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                a = adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = torch.exp(logp - logp_old[idx])
                pi_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                v = self.ac.critic(o).squeeze(-1)
                v_clip = val_old[idx] + (v - val_old[idx]).clamp(-cfg.clip, cfg.clip)
                v_loss = torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                loss = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), cfg.max_grad_norm)
                self.opt.step()
                stats["kl"] += kl.item()
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                n += 1
        self.t = 0
        out = {k: v / n for k, v in stats.items()}
        out["lr"] = self.lr
        out["std"] = self.ac.log_std.clamp(self.ac.min_log_std, self.ac.max_log_std).exp().mean().item()
        return out
