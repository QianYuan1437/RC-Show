"""Robot descriptions shared by every controller.

The MJCF file is the single source of truth. ``Robot`` extracts, once:

* the serial chain (fixed transform between consecutive joints, joint axes, limits, TCP);
* the rigid-body parameters of every *link* — a link is the body driven by a joint plus all
  fixed descendants (hand, fingers, probe, tool) merged into one rigid body — expressed in the
  link frame: mass, centre of mass and rotational inertia about the centre of mass;
* motor torque limits, rotor armature and viscous joint damping.

``rczoo.dynamics`` evaluates kinematics and dynamics from these numbers; the tests check them
against MuJoCo (mj_fullM, qfrc_bias, mj_jacSite, mj_rne) to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).parent / "assets"

ROBOTS = {
    "panda": {
        "xml": ASSET_DIR / "panda" / "panda.xml",
        "label": "Franka Emika Panda",
        # joint velocity limits (rad/s, Franka datasheet)
        "vel_limit": [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61],
    },
    "sr3": {
        "xml": ASSET_DIR / "sr3" / "sr3.xml",
        "label": "ROKAE xMate SR3-C",
        # URDF velocity limits (rad/s)
        "vel_limit": [3.1416] * 6,
    },
}


@dataclass(frozen=True)
class Robot:
    name: str
    label: str
    xml_path: Path
    ndof: int
    joint_names: tuple[str, ...]
    lower: np.ndarray  # (n,) joint limits (rad)
    upper: np.ndarray
    home: np.ndarray  # (n,)
    vel_limit: np.ndarray  # (n,) rad/s
    torque_limit: np.ndarray  # (n,) N m
    armature: np.ndarray  # (n,) reflected rotor inertia added to M's diagonal
    damping: np.ndarray  # (n,) viscous joint damping (N m s / rad)
    # chain: joint i frame = joint (i-1) frame * T(body_pos[i], body_quat[i]) * Rot(axis[i], q_i)
    base_pos: np.ndarray  # (3,)
    base_quat: np.ndarray  # (4,) wxyz
    body_pos: np.ndarray  # (n, 3)
    body_quat: np.ndarray  # (n, 4)
    joint_axis: np.ndarray  # (n, 3) in the joint frame
    tcp_pos: np.ndarray  # (3,) TCP site in the last joint frame
    tcp_quat: np.ndarray  # (4,)
    # rigid-body parameters per link, in the link (joint) frame
    link_mass: np.ndarray  # (n,)
    link_com: np.ndarray  # (n, 3)
    link_inertia: np.ndarray  # (n, 3, 3) about the centre of mass
    joint_body_ids: tuple[int, ...]

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    def mj_model(self) -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_path(str(self.xml_path))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _quat_rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, v, q)
    return out


def _mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


@cache
def load_robot(name: str = "panda") -> Robot:
    cfg = ROBOTS[name]
    m = mujoco.MjModel.from_xml_path(str(cfg["xml"]))
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    n = len(hinge)
    bodies = [int(m.jnt_bodyid[j]) for j in hinge]
    for j in hinge:
        assert np.allclose(m.jnt_pos[j], 0.0), "joint anchors must coincide with body frames"

    def chain_to(body: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Fixed transform stop-frame -> body-frame (composition of body pos/quat)."""
        pos, quat = np.zeros(3), np.array([1.0, 0, 0, 0])
        b = body
        while b != stop:
            pos = m.body_pos[b] + _quat_rot(m.body_quat[b], pos)
            quat = _quat_mul(m.body_quat[b], quat)
            b = int(m.body_parentid[b])
        return pos, quat

    base_pos, base_quat = chain_to(int(m.body_parentid[bodies[0]]), 0)
    body_pos, body_quat, prev = [], [], int(m.body_parentid[bodies[0]])
    for b in bodies:
        p, q = chain_to(b, prev)
        body_pos.append(p)
        body_quat.append(q)
        prev = b

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    p_site, q_site = chain_to(int(m.site_bodyid[sid]), bodies[-1])
    tcp_pos = p_site + _quat_rot(q_site, m.site_pos[sid])
    tcp_quat = _quat_mul(q_site, m.site_quat[sid])

    # Composite rigid-body parameters per link, evaluated at q = 0 in world coordinates and
    # mapped back into each link frame.
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    mujoco.mj_kinematics(m, d)

    def driving_link(b: int) -> int:
        while b != 0:
            if b in bodies:
                return bodies.index(b)
            b = int(m.body_parentid[b])
        return -1

    masses, coms, inertias = np.zeros(n), np.zeros((n, 3)), np.zeros((n, 3, 3))
    for k in range(n):
        R_l, o_l = d.xmat[bodies[k]].reshape(3, 3), d.xpos[bodies[k]]
        members = [b for b in range(1, m.nbody) if driving_link(b) == k and m.body_mass[b] > 0]
        mass = sum(m.body_mass[b] for b in members)
        c_w = sum(m.body_mass[b] * d.xipos[b] for b in members) / mass
        I_w = np.zeros((3, 3))
        for b in members:
            R_b = d.ximat[b].reshape(3, 3)
            r = d.xipos[b] - c_w
            I_w += R_b @ np.diag(m.body_inertia[b]) @ R_b.T + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
        masses[k] = mass
        coms[k] = R_l.T @ (c_w - o_l)
        inertias[k] = R_l.T @ I_w @ R_l

    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    home = m.key_qpos[key][:n].copy() if key >= 0 else np.zeros(n)
    act_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
    return Robot(
        name=name,
        label=cfg["label"],
        xml_path=cfg["xml"],
        ndof=n,
        joint_names=tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge),
        lower=m.jnt_range[hinge, 0].copy(),
        upper=m.jnt_range[hinge, 1].copy(),
        home=home,
        vel_limit=np.array(cfg["vel_limit"], dtype=float),
        torque_limit=np.array([m.actuator_ctrlrange[act_joint[j], 1] for j in hinge]),
        armature=np.array([m.dof_armature[m.jnt_dofadr[j]] for j in hinge]),
        damping=np.array([m.dof_damping[m.jnt_dofadr[j]] for j in hinge]),
        base_pos=base_pos,
        base_quat=base_quat,
        body_pos=np.array(body_pos),
        body_quat=np.array(body_quat),
        joint_axis=m.jnt_axis[hinge].copy(),
        tcp_pos=tcp_pos,
        tcp_quat=tcp_quat,
        link_mass=masses,
        link_com=coms,
        link_inertia=inertias,
        joint_body_ids=tuple(bodies),
    )
