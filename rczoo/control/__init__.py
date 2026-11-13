"""Controllers. ``make(name, dyn)`` builds one by its id (the ids are shared with the site)."""

from .base import Controller, Ref, State, cartesian_from_joint, pose_error
from .interaction import CollisionReaction, DragTeaching, MomentumObserver
from .joint import (
    AdaptivePayload,
    ComputedTorque,
    GravityOnly,
    JointImpedance,
    JointPID,
    JointVelocityMode,
    PDFeedforward,
    PDGravity,
    SlidingMode,
)
from .learned import LearnedReach
from .optimal import JointMPC, QPTaskSpace, TVLQRTracking, ilqr
from .task import (
    Admittance,
    CartesianImpedance,
    HybridForceMotion,
    OperationalSpace,
    ParallelForcePosition,
    ResolvedRate,
)

REGISTRY = {c.name: c for c in [
    JointPID, PDGravity, PDFeedforward, ComputedTorque, SlidingMode, AdaptivePayload, JointVelocityMode,
    JointImpedance, GravityOnly, ResolvedRate, CartesianImpedance, OperationalSpace, Admittance,
    HybridForceMotion, ParallelForcePosition, QPTaskSpace, JointMPC, DragTeaching, LearnedReach,
]}


def make(name: str, dyn, **kw) -> Controller:
    return REGISTRY[name](dyn, **kw)


__all__ = [
    "REGISTRY",
    "CollisionReaction",
    "Controller",
    "MomentumObserver",
    "Ref",
    "State",
    "TVLQRTracking",
    "cartesian_from_joint",
    "ilqr",
    "make",
    "pose_error",
]
