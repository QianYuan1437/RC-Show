"""Batched rigid-body kinematics and dynamics of a serial arm in PyTorch.

Everything is written in the *world* frame, which keeps the recursions short:

    forward pass   o_i, R_i (FK), axis z_i = R_i a_i,
                   w_i  = w_{i-1}  + z_i qd_i
                   dw_i = dw_{i-1} + z_i qdd_i + w_{i-1} x z_i qd_i
                   a_i  = a_{i-1}  + dw_{i-1} x (o_i - o_{i-1}) + w_{i-1} x (w_{i-1} x (o_i - o_{i-1}))
    backward pass  F_i = m_i a_ci,  N_i = I_i dw_i + w_i x I_i w_i
                   f_i = F_i + f_{i+1}
                   n_i = N_i + r_ci x F_i + n_{i+1} + (o_{i+1} - o_i) x f_{i+1}
                   tau_i = z_i . n_i + armature_i qdd_i

(recursive Newton-Euler, RNEA; gravity enters as a base acceleration a_0 = -g). From RNEA:

    M(q)        = [RNEA(q, 0, e_1), ..., RNEA(q, 0, e_n)]   (gravity off)
    g(q)        = RNEA(q, 0, 0)
    h(q, qd)    = RNEA(q, qd, 0) = C(q, qd) qd + g(q)
    C(q, qd)    from the Christoffel symbols of M (autograd), so that dM/dt - 2C is skew-symmetric.

Joint viscous damping and dry friction are *not* part of these terms (as in MuJoCo, where they
are passive forces); controllers that compensate friction add them explicitly.

Batch dimension first: q (B, n). ``params`` optionally overrides the link parameters (e.g. a
payload added to the last link) — see ``LinkParams``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .robot import Robot

GRAVITY = 9.81


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([
        torch.stack([z, -v[..., 2], v[..., 1]], -1),
        torch.stack([v[..., 2], z, -v[..., 0]], -1),
        torch.stack([-v[..., 1], v[..., 0], z], -1),
    ], -2)


def axis_angle(axis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotation about a unit axis (3,) by angles q (B,) -> (B, 3, 3) (Rodrigues)."""
    K = skew(axis).expand(q.shape[0], 3, 3)
    s, c = torch.sin(q)[:, None, None], torch.cos(q)[:, None, None]
    eye = torch.eye(3, dtype=q.dtype, device=q.device)
    return eye + s * K + (1 - c) * (K @ K)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrices (..., 3, 3) -> rotation vectors (..., 3). Robust near 0 and pi, and
    branch-free (no host synchronisation, so it can be captured in a CUDA graph)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    c = ((tr - 1) / 2).clamp(-1.0, 1.0)
    w = 0.5 * torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                           R[..., 1, 0] - R[..., 0, 1]], -1)
    s = w.norm(dim=-1)
    theta = torch.atan2(s, c)
    scale = torch.where(s > 1e-9, theta / s.clamp_min(1e-12), 1 + theta**2 / 6)
    out = w * scale.unsqueeze(-1)
    # near pi the skew part vanishes: take the axis from the symmetric part's largest column
    S = 0.5 * (R + R.transpose(-1, -2))
    diag = torch.diagonal(S, dim1=-2, dim2=-1)
    k = diag.argmax(-1, keepdim=True)
    col = torch.gather(S, -1, k.unsqueeze(-2).expand(*S.shape[:-1], 1)).squeeze(-1)
    axis = col / col.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sgn = torch.sign((axis * w).sum(-1))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    out_pi = axis * (sgn * theta).unsqueeze(-1)
    return torch.where((c < -0.999).unsqueeze(-1), out_pi, out)


def orientation_error(R_des: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """e_R = Log(R_des R^T) in the world frame (the convention of every task-space controller)."""
    return so3_log(R_des @ R.transpose(-1, -2))


@dataclass(frozen=True)
class LinkParams:
    """Rigid-body parameters per link in the link frame (torch tensors)."""

    mass: torch.Tensor  # (n,)
    com: torch.Tensor  # (n, 3)
    inertia: torch.Tensor  # (n, 3, 3) about the com

    def with_payload(self, mass: float, com: torch.Tensor | np.ndarray,
                     inertia: torch.Tensor | np.ndarray | None = None) -> LinkParams:
        """Rigidly attach a payload (mass, com in the last link frame, inertia about its com)."""
        com = torch.as_tensor(com, dtype=self.mass.dtype, device=self.mass.device)
        I_p = torch.zeros(3, 3, dtype=self.mass.dtype, device=self.mass.device) if inertia is None else \
            torch.as_tensor(inertia, dtype=self.mass.dtype, device=self.mass.device)
        m0, c0, I0 = self.mass[-1], self.com[-1], self.inertia[-1]
        m = m0 + mass
        c = (m0 * c0 + mass * com) / m
        eye = torch.eye(3, dtype=m.dtype, device=m.device)

        def shift(I, mm, r):
            return I + mm * (r @ r * eye - torch.outer(r, r))

        I_new = shift(I0, m0, c0 - c) + shift(I_p, mass, com - c)
        mass_t, com_t, inertia_t = self.mass.clone(), self.com.clone(), self.inertia.clone()
        mass_t[-1], com_t[-1], inertia_t[-1] = m, c, I_new
        return LinkParams(mass_t, com_t, inertia_t)


class Dynamics:
    """Kinematics and dynamics of one robot, batched over configurations."""

    def __init__(self, robot: Robot, device: str | torch.device = "cpu", dtype=torch.float64):
        self.robot = robot
        self.n = robot.ndof
        self.device = torch.device(device)
        self.dtype = dtype
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)
        self.base_p = t(robot.base_pos)
        self.base_R = t(quat_to_mat(robot.base_quat))
        self.body_p = t(robot.body_pos)
        self.body_R = t(np.stack([quat_to_mat(q) for q in robot.body_quat]))
        self.axis = t(robot.joint_axis)
        self.tcp_p = t(robot.tcp_pos)
        self.tcp_R = t(quat_to_mat(robot.tcp_quat))
        self.armature = t(robot.armature)
        self.damping = t(robot.damping)
        self.lower, self.upper = t(robot.lower), t(robot.upper)
        self.home = t(robot.home)
        self.torque_limit = t(robot.torque_limit)
        self.vel_limit = t(robot.vel_limit)
        self.gravity = t([0.0, 0.0, -GRAVITY])
        self.params = LinkParams(t(robot.link_mass), t(robot.link_com), t(robot.link_inertia))
        # symmetric unit matrices for the 6 inertia parameters (xx, yy, zz, xy, xz, yz)
        sb = np.zeros((6, 3, 3))
        for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
            sb[k, i, j] = sb[k, j, i] = 1.0
        self._sym_basis = t(sb)

    def to(self, device=None, dtype=None) -> Dynamics:
        return Dynamics(self.robot, device or self.device, dtype or self.dtype)

    # ------------------------------------------------------------------ kinematics

    def link_frames(self, q: torch.Tensor):
        """World origins o (B, n, 3), rotations R (B, n, 3, 3) and joint axes z (B, n, 3)."""
        B = q.shape[0]
        R = self.base_R.expand(B, 3, 3)
        p = self.base_p.expand(B, 3)
        os, Rs, zs = [], [], []
        for i in range(self.n):
            p = p + R @ self.body_p[i]
            R = R @ self.body_R[i]
            zs.append(R @ self.axis[i])
            R = R @ axis_angle(self.axis[i], q[:, i])
            os.append(p)
            Rs.append(R)
        return torch.stack(os, 1), torch.stack(Rs, 1), torch.stack(zs, 1)

    def fk(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """TCP position (B, 3) and rotation (B, 3, 3)."""
        o, R, _ = self.link_frames(q)
        return o[:, -1] + R[:, -1] @ self.tcp_p, R[:, -1] @ self.tcp_R

    def jacobian(self, q: torch.Tensor):
        """TCP pose and world-frame geometric Jacobian J = [Jv; Jw] (B, 6, n)."""
        o, R, z = self.link_frames(q)
        p = o[:, -1] + R[:, -1] @ self.tcp_p
        Jv = torch.cross(z, p[:, None] - o, dim=-1)
        J = torch.cat([Jv, z], -1).transpose(1, 2)
        return p, R[:, -1] @ self.tcp_R, J

    # ------------------------------------------------------------------ dynamics

    def _forward(self, q, qd, qdd, gravity: bool):
        o, R, z = self.link_frames(q)
        B = q.shape[0]
        zero = torch.zeros(B, 3, dtype=q.dtype, device=q.device)
        w, dw = zero, zero
        a = -self.gravity.expand(B, 3) if gravity else zero
        o_prev = self.base_p.expand(B, 3)
        ws, dws, accs = [], [], []
        for i in range(self.n):
            d = o[:, i] - o_prev
            a = a + torch.cross(dw, d, dim=-1) + torch.cross(w, torch.cross(w, d, dim=-1), dim=-1)
            zi = z[:, i]
            dw = dw + zi * qdd[:, i:i + 1] + torch.cross(w, zi * qd[:, i:i + 1], dim=-1)
            w = w + zi * qd[:, i:i + 1]
            ws.append(w)
            dws.append(dw)
            accs.append(a)
            o_prev = o[:, i]
        return o, R, z, torch.stack(ws, 1), torch.stack(dws, 1), torch.stack(accs, 1)

    def rnea(self, q: torch.Tensor, qd: torch.Tensor, qdd: torch.Tensor, gravity: bool = True,
             params: LinkParams | None = None, armature: bool = True) -> torch.Tensor:
        """Inverse dynamics tau = M(q) qdd + C(q, qd) qd + g(q)  (B, n)."""
        P = params or self.params
        o, R, z, w, dw, a = self._forward(q, qd, qdd, gravity)
        r = (R @ P.com[None, :, :, None]).squeeze(-1)  # (B, n, 3) com offset in world
        a_c = a + torch.cross(dw, r, dim=-1) + torch.cross(w, torch.cross(w, r, dim=-1), dim=-1)
        I = R @ P.inertia[None] @ R.transpose(-1, -2)
        F = P.mass[None, :, None] * a_c
        N = (I @ dw[..., None]).squeeze(-1) + torch.cross(w, (I @ w[..., None]).squeeze(-1), dim=-1)
        tau = []
        f_next = torch.zeros_like(F[:, 0])
        n_next = torch.zeros_like(F[:, 0])
        for i in reversed(range(self.n)):
            arm = o[:, i + 1] - o[:, i] if i + 1 < self.n else torch.zeros_like(f_next)
            n_i = N[:, i] + torch.cross(r[:, i], F[:, i], dim=-1) + n_next + torch.cross(arm, f_next, dim=-1)
            f_i = F[:, i] + f_next
            tau.append((z[:, i] * n_i).sum(-1))
            f_next, n_next = f_i, n_i
        tau = torch.stack(tau[::-1], 1)
        if armature:
            tau = tau + self.armature * qdd
        return tau

    def mass_matrix(self, q: torch.Tensor, params: LinkParams | None = None) -> torch.Tensor:
        """M(q) (B, n, n), including the rotor armature."""
        B, n = q.shape
        eye = torch.eye(n, dtype=q.dtype, device=q.device)
        qq = q.repeat_interleave(n, 0)
        cols = self.rnea(qq, torch.zeros_like(qq), eye.repeat(B, 1), gravity=False, params=params)
        return cols.view(B, n, n).transpose(1, 2)

    def gravity_torque(self, q: torch.Tensor, params: LinkParams | None = None) -> torch.Tensor:
        z = torch.zeros_like(q)
        return self.rnea(q, z, z, gravity=True, params=params)

    def bias(self, q: torch.Tensor, qd: torch.Tensor, params: LinkParams | None = None) -> torch.Tensor:
        """h(q, qd) = C(q, qd) qd + g(q) (= MuJoCo's qfrc_bias)."""
        return self.rnea(q, qd, torch.zeros_like(q), gravity=True, params=params)

    # ------------------------------------------------------------------ spatial algebra
    #
    # Plucker coordinates at the world origin, angular part first: a revolute joint's motion
    # subspace is S_i = [z_i; o_i x z_i]. The recursion below (Echeandia & Wensing 2021) computes
    #     M(q) qdd_r + C(q, qd) qd_r + g(q)
    # with C the *Christoffel* Coriolis matrix (dM/dt - 2C skew-symmetric), in one pass:
    #     v_i  = v_{i-1}  + S_i qd_i          vr_i = vr_{i-1} + S_i qd_r,i
    #     a_i  = a_{i-1}  + S_i qdd_r,i + v_i x S_i qd_r,i        (a_0 = -gravity)
    #     f_i  = I_i a_i + B(I_i, v_i) vr_i,   B(I, v) w = 1/2 (v x* I w - I (v x w) + w x* I v)
    #     tau_i = S_i^T sum_{k >= i} f_k
    # With qd_r = qd it reduces to the ordinary Newton-Euler equations.

    def _spatial_inertia(self, o, R, params: LinkParams | None = None):
        """World-origin spatial inertias (B, n, 6, 6) of all links."""
        P = params or self.params
        c = o + (R @ P.com[None, :, :, None]).squeeze(-1)
        Ic = R @ P.inertia[None] @ R.transpose(-1, -2)
        m = P.mass[None, :, None, None]
        cx = skew(c)
        eye = torch.eye(3, dtype=o.dtype, device=o.device)
        top = torch.cat([Ic + m * cx @ cx.transpose(-1, -2), m * cx], -1)
        bot = torch.cat([m * cx.transpose(-1, -2), m * eye.expand_as(Ic)], -1)
        return torch.cat([top, bot], -2)

    @staticmethod
    def _crm(v):
        w, u = v[..., :3], v[..., 3:]
        Z = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
        return torch.cat([torch.cat([skew(w), Z], -1), torch.cat([skew(u), skew(w)], -1)], -2)

    @classmethod
    def _bmul(cls, I6, v, w):
        """B(I, v) w = 1/2 (v x* (I w) - I (v x w) + w x* (I v)),  x* = -crm^T."""
        mv = lambda M, x: (M @ x[..., None]).squeeze(-1)
        crf = lambda x: -cls._crm(x).transpose(-1, -2)
        return 0.5 * (mv(crf(v), mv(I6, w)) - mv(I6, mv(cls._crm(v), w)) + mv(crf(w), mv(I6, v)))

    def _spatial_pass(self, q, qd, qd_r, qdd_r, gravity: bool):
        o, R, z = self.link_frames(q)
        B = q.shape[0]
        S = torch.cat([z, torch.cross(o, z, dim=-1)], -1)  # (B, n, 6)
        v = torch.zeros(B, 6, dtype=q.dtype, device=q.device)
        vr = torch.zeros_like(v)
        a = torch.zeros_like(v)
        if gravity:
            a = a.clone()
            a[:, 3:] = -self.gravity
        vs, vrs, accs = [], [], []
        for i in range(self.n):
            Si = S[:, i]
            v = v + Si * qd[:, i:i + 1]
            vr = vr + Si * qd_r[:, i:i + 1]
            a = a + Si * qdd_r[:, i:i + 1] + (self._crm(v) @ (Si * qd_r[:, i:i + 1])[..., None]).squeeze(-1)
            vs.append(v)
            vrs.append(vr)
            accs.append(a)
        return o, R, S, torch.stack(vs, 1), torch.stack(vrs, 1), torch.stack(accs, 1)

    def slotine_li(self, q, qd, qd_r, qdd_r, gravity: bool = True, params: LinkParams | None = None,
                   armature: bool = True) -> torch.Tensor:
        """M(q) qdd_r + C(q, qd) qd_r + g(q) with the Christoffel C, in one recursive pass."""
        o, R, S, v, vr, a = self._spatial_pass(q, qd, qd_r, qdd_r, gravity)
        I6 = self._spatial_inertia(o, R, params)
        f = (I6 @ a[..., None]).squeeze(-1) + self._bmul(I6, v, vr)
        f = torch.flip(torch.cumsum(torch.flip(f, [1]), 1), [1])  # sum over descendants
        tau = (S * f).sum(-1)
        if armature:
            tau = tau + self.armature * qdd_r
        return tau

    def coriolis_product(self, q, qd, w, params: LinkParams | None = None) -> torch.Tensor:
        """C(q, qd) w (Christoffel C)."""
        z = torch.zeros_like(q)
        return self.slotine_li(q, qd, w, z, gravity=False, params=params, armature=False)

    def coriolis_matrix(self, q: torch.Tensor, qd: torch.Tensor,
                        params: LinkParams | None = None) -> torch.Tensor:
        """C(q, qd) (B, n, n) from the Christoffel symbols of M, column by column."""
        B, n = q.shape
        eye = torch.eye(n, dtype=q.dtype, device=q.device)
        cols = self.coriolis_product(q.repeat_interleave(n, 0), qd.repeat_interleave(n, 0), eye.repeat(B, 1), params)
        return cols.view(B, n, n).transpose(1, 2)

    def coriolis_matrix_autograd(self, q: torch.Tensor, qd: torch.Tensor,
                                 params: LinkParams | None = None) -> torch.Tensor:
        """Reference implementation: c_ijk = (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) / 2 by autodiff."""
        def M_single(qs):
            return self.mass_matrix(qs[None], params)[0]

        dM = torch.vmap(torch.func.jacfwd(M_single))(q)  # dM[b, i, j, k] = dM_ij/dq_k
        c = 0.5 * (dM + dM.transpose(2, 3) - dM.permute(0, 3, 1, 2))
        return (c * qd[:, None, None, :]).sum(-1)

    def payload_regressor(self, q, qd, qd_r, qdd_r) -> torch.Tensor:
        """Y (B, n, 10) with Y pi = (M qdd_r + C qd_r + g) of a rigid payload on the last link,
        pi = [m, h_x, h_y, h_z, I_xx, I_yy, I_zz, I_xy, I_xz, I_yz]: mass, first moment and
        inertia about the last link's origin, in the link frame. Linear in pi by construction."""
        o, R, S, v, vr, a = self._spatial_pass(q, qd, qd_r, qdd_r, gravity=True)
        on, Rn = o[:, -1], R[:, -1]
        vn, vrn, an = v[:, -1], vr[:, -1], a[:, -1]
        B = q.shape[0]
        dt, dev = q.dtype, q.device
        eye = torch.eye(3, dtype=dt, device=dev)
        ox = skew(on)
        basis = []
        # mass
        I6 = torch.zeros(B, 6, 6, dtype=dt, device=dev)
        I6[:, :3, :3] = ox @ ox.transpose(1, 2)
        I6[:, :3, 3:] = ox
        I6[:, 3:, :3] = ox.transpose(1, 2)
        I6[:, 3:, 3:] = eye
        basis.append(I6)
        # first moment (link frame axis k): h_w = R e_k
        for k in range(3):
            hx = skew(Rn[:, :, k])
            I6 = torch.zeros(B, 6, 6, dtype=dt, device=dev)
            I6[:, :3, :3] = ox @ hx.transpose(1, 2) + hx @ ox.transpose(1, 2)
            I6[:, :3, 3:] = hx
            I6[:, 3:, :3] = hx.transpose(1, 2)
            basis.append(I6)
        for E in self._sym_basis:
            I6 = torch.zeros(B, 6, 6, dtype=dt, device=dev)
            I6[:, :3, :3] = Rn @ E @ Rn.transpose(1, 2)
            basis.append(I6)
        I6 = torch.stack(basis, 1)  # (B, 10, 6, 6)
        f = (I6 @ an[:, None, :, None]).squeeze(-1) + self._bmul(I6, vn[:, None].expand(B, 10, 6), vrn[:, None].expand(B, 10, 6))
        return torch.einsum("bis,bps->bip", S, f)  # every joint carries the payload

    def tcp_bias_acc(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        """dJ/dt qd (B, 6): TCP spatial acceleration (classical linear; angular) at qdd = 0."""
        _, R, _, w, dw, a = self._forward(q, qd, torch.zeros_like(q), gravity=False)
        r = (R[:, -1] @ self.tcp_p[:, None]).squeeze(-1)
        wn, dwn = w[:, -1], dw[:, -1]
        lin = a[:, -1] + torch.cross(dwn, r, dim=-1) + torch.cross(wn, torch.cross(wn, r, dim=-1), dim=-1)
        return torch.cat([lin, dwn], -1)

    def tcp_velocity(self, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
        _, _, J = self.jacobian(q)
        return (J @ qd[..., None]).squeeze(-1)

    # ------------------------------------------------------------------ helpers

    def clamp(self, q: torch.Tensor) -> torch.Tensor:
        return torch.minimum(torch.maximum(q, self.lower), self.upper)

    def sample_q(self, n: int, margin: float = 0.0, generator: torch.Generator | None = None) -> torch.Tensor:
        u = torch.rand(n, self.n, dtype=self.dtype, device=self.device, generator=generator)
        lo = self.lower + margin * (self.upper - self.lower)
        hi = self.upper - margin * (self.upper - self.lower)
        return lo + u * (hi - lo)

    def payload_params(self, mass: float, com=(0.0, 0.0, 0.05), inertia=None) -> LinkParams:
        """Link parameters with a payload at `com` in the TCP frame (point mass unless `inertia`,
        about the payload's com in the TCP frame, is given)."""
        com_link = self.tcp_p + self.tcp_R @ torch.as_tensor(com, dtype=self.dtype, device=self.device)
        if inertia is not None:
            inertia = self.tcp_R @ torch.as_tensor(inertia, dtype=self.dtype, device=self.device) @ self.tcp_R.T
        return self.params.with_payload(mass, com_link, inertia)


def with_params(dyn: Dynamics, params: LinkParams) -> Dynamics:
    """A copy of `dyn` whose nominal link parameters are `params`."""
    out = Dynamics.__new__(Dynamics)
    out.__dict__.update(dyn.__dict__)
    out.params = params
    return out


__all__ = ["Dynamics", "LinkParams", "orientation_error", "skew", "so3_log", "with_params"]
