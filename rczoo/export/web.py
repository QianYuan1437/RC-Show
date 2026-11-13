"""Export a robot for the browser.

    python -m rczoo.export.web --out site/docs/public/models

Per robot ``<out>/<name>/``:
  link<k>.glb      visual meshes of link k (vertex coloured, link frame), k = 0..n
  robot.json       chain (same numbers as ``rczoo.robot.Robot``), limits, gains, plant defaults
  plant.xml        geometry-free MuJoCo model for MuJoCo WASM: motors, payload body, contact
                   probe + F/T sensor, a mocap surface (parked below the floor)
  fixtures.json    reference values computed in Python (FK, dynamics, controller torques,
                   trajectory profiles) that the TypeScript ports are unit-tested against
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import trimesh

from ..control.gains import CART_K, CART_K_SOFT, GAINS, NULL_D, NULL_K, WN_JOINT, WN_TASK
from ..model import PlantConfig, plant_spec
from ..robot import ROBOTS, load_robot


def _driving_link(m, body: int, joint_bodies: list[int]) -> int:
    b = body
    while b != 0:
        if b in joint_bodies:
            return joint_bodies.index(b) + 1
        b = int(m.body_parentid[b])
    return 0


def _geom_mesh(m, g) -> trimesh.Trimesh | None:
    t = m.geom_type[g]
    size = m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        return trimesh.Trimesh(m.mesh_vert[va:va + vn].astype(np.float64), m.mesh_face[fa:fa + fn].astype(np.int64),
                               process=False)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=float(size[0]))
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=float(size[0]), height=2 * float(size[1]), sections=24)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)
    return None


def export_meshes(robot, out: Path):
    m = robot.mj_model()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    jb = list(robot.joint_body_ids)
    parts: dict[int, list[trimesh.Trimesh]] = {}
    for g in range(m.ngeom):
        if m.geom_group[g] != 2:
            continue
        mesh = _geom_mesh(m, g)
        if mesh is None:
            continue
        link = _driving_link(m, int(m.geom_bodyid[g]), jb)
        ref = 0 if link == 0 else jb[link - 1]
        Tw_l, Tw_g = np.eye(4), np.eye(4)
        Tw_l[:3, :3], Tw_l[:3, 3] = d.xmat[ref].reshape(3, 3), d.xpos[ref]
        Tw_g[:3, :3], Tw_g[:3, 3] = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        mesh.apply_transform(np.linalg.inv(Tw_l) @ Tw_g)
        rgba = m.geom_rgba[g] if m.geom_matid[g] < 0 else m.mat_rgba[m.geom_matid[g]]
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh, vertex_colors=np.tile((np.asarray(rgba) * 255).astype(np.uint8), (len(mesh.vertices), 1)))
        parts.setdefault(link, []).append(mesh)
    links = []
    for k in range(robot.ndof + 1):
        scene = trimesh.Scene()
        for i, mesh in enumerate(parts.get(k, [])):
            scene.add_geometry(mesh, node_name=f"g{i}")
        scene.export(out / f"link{k}.glb")
        links.append({"index": k, "file": f"link{k}.glb", "joint": k - 1})
    return links


def plant_xml(robot) -> str:
    """Geometry-free plant (visual geoms and meshes removed; inertias are explicit)."""
    spec = plant_spec(robot, PlantConfig(surface=True), payload_body=True)
    m0 = spec.compile()
    for g in list(spec.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH or g.group == 2:
            spec.delete(g)
    for mesh in list(spec.meshes):
        spec.delete(mesh)
    for tex in list(spec.textures):
        spec.delete(tex)
    m1 = spec.compile()
    moving = m0.body_weldid != 0  # bodies welded to the world (the base) do not matter
    assert np.allclose(m0.body_mass[moving], m1.body_mass[moving])
    assert np.allclose(m0.body_inertia[moving], m1.body_inertia[moving])
    return spec.to_xml()


def robot_json(robot, links) -> dict:
    cfg = PlantConfig()
    return {
        "name": robot.name,
        "label": ROBOTS[robot.name]["label"],
        "ndof": robot.ndof,
        "joint_names": list(robot.joint_names),
        "lower": robot.lower.tolist(), "upper": robot.upper.tolist(), "home": robot.home.tolist(),
        "vel_limit": robot.vel_limit.tolist(), "torque_limit": robot.torque_limit.tolist(),
        "armature": robot.armature.tolist(), "damping": robot.damping.tolist(),
        "base_pos": robot.base_pos.tolist(), "base_quat": robot.base_quat.tolist(),
        "body_pos": robot.body_pos.tolist(), "body_quat": robot.body_quat.tolist(),
        "joint_axis": robot.joint_axis.tolist(),
        "tcp_pos": robot.tcp_pos.tolist(), "tcp_quat": robot.tcp_quat.tolist(),
        "links": links,
        "gains": {**GAINS[robot.name], "wn_joint": WN_JOINT, "wn_task": WN_TASK, "cart_k": CART_K,
                  "cart_k_soft": CART_K_SOFT, "null_k": NULL_K, "null_d": NULL_D},
        "plant": {"timestep": cfg.timestep, "payload_offset": list(cfg.payload_offset),
                  "surface_size": list(cfg.surface_size), "probe_radius": 0.012},
        "convention": "quaternions are wxyz; link k>=1 frame = base * prod_{i<k}(body_i * Rot(axis_i, q_i)); "
                      "TCP = link_n frame * (tcp_pos, tcp_quat). Z is up.",
    }


def fixtures(robot) -> dict:
    """Reference numbers for the TypeScript ports (nominal model, float64)."""
    from ..control import make
    from ..control.base import State, cartesian_from_joint
    from ..dynamics import Dynamics
    from ..traj import DoubleS, Trapezoid, quintic

    dyn = Dynamics(robot, "cpu")
    rng = np.random.default_rng(0)
    lo, hi = robot.lower + 0.15 * (robot.upper - robot.lower), robot.upper - 0.15 * (robot.upper - robot.lower)
    q = torch.as_tensor(rng.uniform(lo, hi, (4, robot.ndof)))
    qd = torch.as_tensor(rng.normal(0, 0.5, (4, robot.ndof)))
    q_ref = q + torch.as_tensor(rng.normal(0, 0.05, (4, robot.ndof)))
    qd_ref = torch.as_tensor(rng.normal(0, 0.3, (4, robot.ndof)))
    qdd_ref = torch.as_tensor(rng.normal(0, 1.0, (4, robot.ndof)))
    p, R, J = dyn.jacobian(q)
    out = {"q": q.tolist(), "qd": qd.tolist(), "q_ref": q_ref.tolist(), "qd_ref": qd_ref.tolist(),
           "qdd_ref": qdd_ref.tolist(), "p": p.tolist(), "R": R.tolist(), "J": J.tolist(),
           "M": dyn.mass_matrix(q).tolist(), "bias": dyn.bias(q, qd).tolist(), "g": dyn.gravity_torque(q).tolist(),
           "jdot_qd": dyn.tcp_bias_acc(q, qd).tolist(), "tau": {}}
    ref = cartesian_from_joint(dyn, q_ref, qd_ref, qdd_ref)
    ref.q_null = torch.as_tensor(robot.home).expand_as(q)
    state = State(t=torch.zeros(4, dtype=torch.float64), q=q, qd=qd, ft=torch.zeros(4, 6, dtype=torch.float64))
    for nm in ["pd_grav", "pd_ff", "ctc", "smc", "joint_impedance", "cart_impedance", "osc"]:
        c = make(nm, dyn)
        c.reset(state, ref)
        out["tau"][nm] = c(state, ref).tolist()
    out["ref"] = {"p": ref.p.tolist(), "R": ref.R.tolist(), "v": ref.v.tolist(), "a": ref.a.tolist()}
    # trajectory profiles (scalar, rest to rest)
    t = torch.linspace(0, 2.0, 21, dtype=torch.float64)
    qp = quintic(torch.tensor(0.0, dtype=torch.float64), torch.tensor(1.0, dtype=torch.float64), 2.0, t)
    tr = Trapezoid(0.0, 1.0, 1.0, 2.0).eval_own(t)
    trs = Trapezoid(0.0, 0.2, 1.0, 2.0).eval_own(t)  # triangular: vmax not reached
    ds = DoubleS(0.0, 1.0, 1.0, 2.0, 10.0)
    dss = DoubleS(0.0, 0.05, 1.0, 2.0, 10.0)  # neither vmax nor amax reached
    out["traj"] = {"t": t.tolist(), "quintic": [qp.p.tolist(), qp.v.tolist(), qp.a.tolist()],
                   "trapezoid": [tr.p.tolist(), tr.v.tolist(), tr.a.tolist()],
                   "trapezoid_short": [trs.p.tolist(), trs.v.tolist(), trs.a.tolist()],
                   "double_s": {"T": float(ds.T), **{k: getattr(ds.eval_own(t), k).tolist() for k in "pvaj"}},
                   "double_s_short": {"T": float(dss.T), **{k: getattr(dss.eval_own(t), k).tolist() for k in "pvaj"}}}
    return out


def export(name: str, out: Path):
    robot = load_robot(name)
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    links = export_meshes(robot, d)
    (d / "robot.json").write_text(json.dumps(robot_json(robot, links), indent=1))
    (d / "plant.xml").write_text(plant_xml(robot))
    (d / "fixtures.json").write_text(json.dumps(fixtures(robot)))
    print(f"exported {name} -> {d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("site/docs/public/models"))
    ap.add_argument("--robots", default="panda,sr3")
    a = ap.parse_args()
    for rn in a.robots.split(","):
        export(rn, a.out)
