"""Unitree G1 (29 DoF) articulation, following BeyondMimic's actuator model.

The URDF is the one the LAFAN1 motions were retargeted to (g1_29dof_rev_1_0). Joint drives are
implicit PD with a 10 Hz natural frequency and damping ratio 2 on the reflected motor armature, and
the action scale is 0.25 * effort_limit / stiffness, as in BeyondMimic.
"""

from __future__ import annotations

import os

from .g1_contract import (
    ARMATURE_4010,
    ARMATURE_5020,
    ARMATURE_7520_14,
    ARMATURE_7520_22,
    DAMPING_RATIO,
    DEFAULT_JOINT_POS,
    G1_JOINT_NAMES,
    JOINT_MOTORS,
    NATURAL_FREQ,
    TRACKED_BODIES,
    _kd,
    _kp,
    action_scale,
    default_joint_pos_list,
    motor_of,
)

try:
    import isaaclab.sim as sim_utils
except ImportError:  # pragma: no cover - Apple Silicon path
    sim_utils = None
try:
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
except ImportError:  # pragma: no cover - Apple Silicon path
    ImplicitActuatorCfg = None
    ArticulationCfg = None

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
URDF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof_rev_1_0.urdf")
MJCF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof.xml")
USD_DIR = os.path.join(DATA_DIR, "g1_usd")
USD_PATH = os.path.join(USD_DIR, "g1_29dof", "g1_29dof.usda")


def _actuators() -> dict[str, ImplicitActuatorCfg]:
    if ImplicitActuatorCfg is None:
        raise RuntimeError("Isaac Lab is required to build the Isaac articulation config")
    acts = {}
    for name in G1_JOINT_NAMES:
        effort, vel, arm = motor_of(name)
        acts[name.replace("_joint", "")] = ImplicitActuatorCfg(
            joint_names_expr=[name],
            effort_limit_sim=effort,
            velocity_limit_sim=vel,
            stiffness=_kp(arm),
            damping=_kd(arm),
            armature=arm,
        )
    return acts


if ArticulationCfg is None:
    G1_CFG = None
else:
    G1_CFG = ArticulationCfg(
        # USD conversion is performed offline by tools/mjcf_to_usd.py.
        spawn=sim_utils.UsdFileCfg(
            usd_path=USD_PATH,
            variants={"Physics": "mujoco"},
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76), joint_pos=DEFAULT_JOINT_POS, joint_vel={".*": 0.0}),
        soft_joint_pos_limit_factor=0.9,
        actuators=_actuators(),
    )
