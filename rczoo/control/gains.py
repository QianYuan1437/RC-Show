"""Default gains per robot, shared by benchmarks, tests and the exported browser configs.

Joint-space torque gains follow Franka's example controllers for the Panda and are scaled by the
motor torque limits for the SR3-C. Model-based controllers (computed torque, OSC, MPC) are tuned
by a closed-loop natural frequency instead, which is robot independent.
"""

from __future__ import annotations

GAINS = {
    "panda": {
        "kp": [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0],
        "kd": [50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0],
        "ki": [300.0, 300.0, 300.0, 300.0, 120.0, 80.0, 25.0],
        # joint impedance mode (compliant): a quarter of the stiffness
        "imp_k": [150.0, 150.0, 150.0, 150.0, 60.0, 40.0, 15.0],
        "imp_d": [25.0, 25.0, 25.0, 25.0, 12.0, 10.0, 6.0],
        # inner joint-velocity loop (velocity mode)
        "kv": [60.0, 60.0, 60.0, 60.0, 25.0, 15.0, 6.0],
        "kvi": [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 60.0],
    },
    "sr3": {
        "kp": [500.0, 500.0, 250.0, 120.0, 120.0, 60.0],
        "kd": [40.0, 40.0, 20.0, 10.0, 10.0, 6.0],
        "ki": [250.0, 250.0, 120.0, 60.0, 60.0, 30.0],
        "imp_k": [125.0, 125.0, 60.0, 30.0, 30.0, 15.0],
        "imp_d": [20.0, 20.0, 10.0, 5.0, 5.0, 3.0],
        "kv": [50.0, 50.0, 25.0, 12.0, 12.0, 6.0],
        "kvi": [500.0, 500.0, 250.0, 120.0, 120.0, 60.0],
    },
}

# robot-independent settings
WN_JOINT = 25.0  # rad/s, computed torque / PD+ outer loop
WN_TASK = 25.0  # rad/s, OSC
CART_K = [2000.0, 2000.0, 2000.0, 150.0, 150.0, 150.0]  # Cartesian impedance (N/m, N m/rad)
CART_K_SOFT = [300.0, 300.0, 300.0, 30.0, 30.0, 30.0]  # compliant interaction
NULL_K = 20.0  # null-space posture stiffness (N m/rad), redundant arms only
NULL_D = 4.0
