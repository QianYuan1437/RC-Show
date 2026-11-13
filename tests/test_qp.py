import numpy as np
import torch
from scipy.optimize import minimize

from rczoo.qp import solve_qp


def test_qp_matches_scipy():
    g = torch.Generator().manual_seed(0)
    B, n, m = 6, 5, 8
    L = torch.randn(B, n, n, generator=g, dtype=torch.float64)
    P = L @ L.transpose(1, 2) + 0.5 * torch.eye(n, dtype=torch.float64)
    c = torch.randn(B, n, generator=g, dtype=torch.float64)
    A = torch.randn(B, m, n, generator=g, dtype=torch.float64)
    l = -torch.rand(B, m, generator=g, dtype=torch.float64) - 0.1
    u = torch.rand(B, m, generator=g, dtype=torch.float64) + 0.1
    sol = solve_qp(P, c, A, l, u, iters=600, rho=1.0)
    for b in range(B):
        Pb, cb, Ab, lb, ub = (x[b].numpy() for x in (P, c, A, l, u))
        cons = [{"type": "ineq", "fun": lambda x, Ab=Ab, ub=ub: ub - Ab @ x},
                {"type": "ineq", "fun": lambda x, Ab=Ab, lb=lb: Ab @ x - lb}]
        ref = minimize(lambda x: 0.5 * x @ Pb @ x + cb @ x, np.zeros(n), constraints=cons, method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 500}).x
        np.testing.assert_allclose(sol.x[b].numpy(), ref, atol=1e-4)
