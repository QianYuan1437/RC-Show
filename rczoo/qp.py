"""Batched convex QP solver (ADMM, the OSQP iteration) in PyTorch.

    minimise   1/2 x^T P x + c^T x      subject to   l <= A x <= u

Every problem in the batch has the same sizes; P, A may differ per problem. Each solve
factorises K = P + sigma I + rho A^T A once and runs a fixed number of iterations,
which makes the cost predictable inside a 1 kHz loop. Warm starts (x, z, y) are supported.
Infinite bounds are allowed (l = -inf, u = inf).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QPSolution:
    x: torch.Tensor
    z: torch.Tensor
    y: torch.Tensor
    primal_res: torch.Tensor
    dual_res: torch.Tensor


def solve_qp(P, c, A, l, u, iters: int = 40, rho: float = 1.0, sigma: float = 1e-6, alpha: float = 1.6,
             warm: QPSolution | None = None) -> QPSolution:
    n = c.shape[1]
    eye = torch.eye(n, dtype=c.dtype, device=c.device)
    # equality rows (l == u) get a larger penalty, as in OSQP
    rho_vec = torch.where((u - l).abs() < 1e-9, 1e3 * rho, torch.full_like(l, rho))
    At = A.transpose(1, 2)
    K = P + sigma * eye + At @ (rho_vec[..., None] * A)
    # K is small (n <= ~150) and fixed during the iterations: invert once (batched LU without
    # host-side checks, CUDA-graph safe), then every iteration is a matrix-vector product.
    Kinv = torch.linalg.inv_ex(K)[0]
    x = torch.zeros_like(c) if warm is None else warm.x
    z = torch.zeros_like(l) if warm is None else warm.z
    y = torch.zeros_like(l) if warm is None else warm.y
    for _ in range(iters):
        rhs = sigma * x - c + (At @ (rho_vec * z - y)[..., None]).squeeze(-1)
        x_t = (Kinv @ rhs[..., None]).squeeze(-1)
        z_t = (A @ x_t[..., None]).squeeze(-1)
        x = alpha * x_t + (1 - alpha) * x
        z_relax = alpha * z_t + (1 - alpha) * z
        z_new = torch.minimum(torch.maximum(z_relax + y / rho_vec, l), u)
        y = y + rho_vec * (z_relax - z_new)
        z = z_new
    Ax = (A @ x[..., None]).squeeze(-1)
    r_p = (Ax - z).abs().amax(-1)
    r_d = ((P @ x[..., None]).squeeze(-1) + c + (At @ y[..., None]).squeeze(-1)).abs().amax(-1)
    return QPSolution(x, z, y, r_p, r_d)
