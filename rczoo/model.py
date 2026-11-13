"""Plant variants built from the robot MJCF with MjSpec.

The *nominal* model (``rczoo.dynamics`` with the MJCF's link parameters) is what controllers
know. The *plant* that is simulated may differ from it, which is how the benchmarks probe
robustness:

* ``payload``      an unmodelled mass rigidly attached below the TCP;
* ``frictionloss`` Coulomb (dry) joint friction, N m per joint;
* ``surface``      a contact surface (a mocap box, so it can be moved per world and at run time)
                   that only the TCP probe can touch, with a chosen stiffness/damping.

The browser loads one exported XML per robot that contains all three (payload mass and friction
are changed at run time through the model arrays, the surface is moved through mocap_pos).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .robot import Robot

TIMESTEP = 0.001  # 1 kHz physics and torque control


@dataclass
class PlantConfig:
    payload: float = 0.0  # kg
    payload_offset: tuple[float, float, float] = (0.0, 0.0, 0.05)  # in the TCP frame
    frictionloss: float | list[float] = 0.0
    surface: bool = False
    surface_pos: tuple[float, float, float] = (0.6, 0.0, -1.0)  # parked below the floor
    surface_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    surface_size: tuple[float, float, float] = (0.25, 0.25, 0.01)  # half sizes (m)
    # MuJoCo soft-contact parameters of the surface (also set on the probe, so that the geom
    # mixing does not alter them). With solimp (0.5, 0.6, 0.01) the contact is a nonlinear spring;
    # around 10 N the measured stiffness is ~1.4 / 3.7 / 21 kN/m for solref time constants
    # 0.1 / 0.05 / 0.02 s (benchmark "soft" / "medium" / "stiff"): a compliant probe tip.
    surface_solref: tuple[float, float] = (0.05, 1.0)
    surface_solimp: tuple[float, float, float] = (0.5, 0.6, 0.01)
    surface_condim: int = 1
    timestep: float = TIMESTEP
    extra: dict = field(default_factory=dict)


def tcp_body_name(robot: Robot) -> str:
    m = robot.mj_model()
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[sid]))


def plant_spec(robot: Robot, cfg: PlantConfig | None = None, payload_body: bool = False) -> mujoco.MjSpec:
    """MjSpec of the simulated plant. ``payload_body`` forces a payload body even for 0 kg (browser)."""
    cfg = cfg or PlantConfig()
    spec = mujoco.MjSpec.from_file(str(robot.xml_path))
    spec.option.timestep = cfg.timestep
    if cfg.payload > 0 or payload_body:
        tcp_site = spec.site("tcp")
        parent = spec.body(tcp_body_name(robot))
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(tcp_site.quat, dtype=float))
        pos = np.asarray(tcp_site.pos) + R.reshape(3, 3) @ np.asarray(cfg.payload_offset)
        b = parent.add_body(name="payload", pos=pos.tolist(), quat=list(tcp_site.quat))
        b.mass = max(cfg.payload, 1e-6)
        r = 0.04  # solid cube of side 8 cm
        b.inertia = [max(cfg.payload, 1e-6) * (2 * r) ** 2 / 6] * 3
        b.ipos = [0.0, 0.0, 0.0]  # centre of mass at the body origin (add_body(pos=...) also sets ipos)
        b.iquat = [1.0, 0.0, 0.0, 0.0]
        b.explicitinertial = True
        b.add_geom(name="payload_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[r, r, r],
                   contype=0, conaffinity=0, group=1, rgba=[0.85, 0.55, 0.2, 0.9], mass=0)
    fl = np.broadcast_to(np.asarray(cfg.frictionloss, dtype=float), (robot.ndof,))
    for i, jn in enumerate(robot.joint_names):
        spec.joint(jn).frictionloss = float(fl[i])
    if cfg.surface:
        spec.option.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        # Frictionless surface (condim 1): with friction, a position-controlled TCP sliding on a
        # surface sticks and slips, and MuJoCo's soft friction constraint turns the stick-slip
        # into normal-force jumps of +-5 N that would swamp the comparison of force controllers.
        # (The site discusses this; set surface_condim=3 to see it.)
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        sb = spec.worldbody.add_body(name="surface", mocap=True, pos=list(cfg.surface_pos),
                                     quat=list(cfg.surface_quat))
        sb.add_geom(name="surface", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(cfg.surface_size),
                    contype=1, conaffinity=2, condim=cfg.surface_condim, friction=[0.2, 0.005, 0.0001],
                    solref=list(cfg.surface_solref), solimp=[*cfg.surface_solimp, 0.5, 2],
                    rgba=[0.55, 0.62, 0.72, 0.6], group=1)
        probe = spec.geom("probe")
        probe.solref = list(cfg.surface_solref)
        probe.solimp = [*cfg.surface_solimp, 0.5, 2]
        probe.condim = cfg.surface_condim
    return spec


def plant_model(robot: Robot, cfg: PlantConfig | None = None) -> mujoco.MjModel:
    return plant_spec(robot, cfg).compile()
