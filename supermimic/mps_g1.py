"""Host-driven MPS simulator for the origin G1 MJCF.

This backend keeps the origin actuator contract but uses MuJoCo Warp's MLX
engine. It is a physics backend, not a JAX-jittable replacement for JAXMJXG1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .g1_contract import (
    DAMPING_RATIO,
    G1_JOINT_NAMES,
    NATURAL_FREQ,
    action_scale,
    default_joint_pos_list,
    motor_of,
)

CONTROL_DT = 0.02
ROOT_HEIGHT = 0.76


def _integrator_name(mujoco, integrator):
    labels = (
        ("mjINT_EULER", "Euler"),
        ("mjINT_RK4", "RK4"),
        ("mjINT_IMPLICIT", "implicit"),
        ("mjINT_IMPLICITFAST", "implicitfast"),
    )
    for enum_name, label in labels:
        if hasattr(mujoco.mjtIntegrator, enum_name) and integrator == getattr(mujoco.mjtIntegrator, enum_name):
            return label
    return str(int(integrator))


@dataclass(frozen=True)
class MPSModelInfo:
    original_integrator: str
    timestep: float
    foot_sphere_ids: tuple[int, ...]
    disabled_cylinder_ids: tuple[int, ...]
    restored_armature_dofs: int
    collision_mode: str = "origin_foot_only"
    floor_geom_name: str = "mjx_ground"
    explicit_contact_pairs: int = 0


def prepare_mps_model(xml_path: str | Path):
    """Load and adapt the original G1 model for MPS-supported features."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if (model.nq, model.nv, model.nu) != (36, 35, len(G1_JOINT_NAMES)):
        raise ValueError(f"expected origin G1 dimensions (36, 35, 29), got {(model.nq, model.nv, model.nu)}")
    if (
        model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
        or model.jnt_qposadr[0] != 0
        or model.jnt_dofadr[0] != 0
    ):
        raise ValueError("expected the floating base in the first seven qpos and six dof slots")

    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    )
    if actuator_names != tuple(G1_JOINT_NAMES):
        raise ValueError("MJCF actuator order does not match the origin G1 contract")

    for actuator_id, joint_name in enumerate(G1_JOINT_NAMES):
        effort, _, armature = motor_of(joint_name)
        kp = armature * NATURAL_FREQ**2
        kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
        expected_force = np.array([-effort, effort], dtype=np.float64)
        if not np.isclose(model.actuator_gainprm[actuator_id, 0], kp, rtol=2e-4, atol=1e-5):
            raise ValueError(f"unexpected position-actuator stiffness for {joint_name}")
        if not np.isclose(model.actuator_biasprm[actuator_id, 1], -kp, rtol=2e-4, atol=1e-5):
            raise ValueError(f"unexpected position-actuator bias for {joint_name}")
        if not np.isclose(model.actuator_biasprm[actuator_id, 2], -kd, rtol=2e-4, atol=1e-5):
            raise ValueError(f"unexpected position-actuator damping for {joint_name}")
        if not np.allclose(model.actuator_forcerange[actuator_id], expected_force, rtol=1e-6, atol=1e-5):
            raise ValueError(f"unexpected force limits for {joint_name}")
        if not model.actuator_forcelimited[actuator_id]:
            raise ValueError(f"expected force-limited actuator for {joint_name}")

        joint_id = int(model.actuator_trnid[actuator_id, 0])
        loaded_joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if loaded_joint_name != joint_name:
            raise ValueError(f"actuator {joint_name} is not attached to its matching joint")
        if model.jnt_qposadr[joint_id] != 7 + actuator_id or model.jnt_dofadr[joint_id] != 6 + actuator_id:
            raise ValueError(f"joint state layout does not match the origin order at {joint_name}")
        model.dof_armature[model.jnt_dofadr[joint_id]] = armature

    # Armature updates affect compiled inverse-weight constants as well.
    mujoco.mj_setConst(model, mujoco.MjData(model))

    original_integrator = _integrator_name(mujoco, model.opt.integrator)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mjx_ground")
    if floor_id < 0 or model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("origin G1 MJCF must contain its mjx_ground plane")

    geom_type = np.asarray(model.geom_type)
    original_contype = np.asarray(model.geom_contype).copy()
    original_conaffinity = np.asarray(model.geom_conaffinity).copy()
    active = (original_contype != 0) | (original_conaffinity != 0)
    foot_spheres = np.flatnonzero(active & (geom_type == mujoco.mjtGeom.mjGEOM_SPHERE))
    cylinders = np.flatnonzero(active & (geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER))
    if len(foot_spheres) != 8 or len(cylinders) != 4:
        raise ValueError(
            "expected 8 active G1 foot spheres and 4 active cylinders; "
            f"found {len(foot_spheres)} and {len(cylinders)}"
        )

    # MPS supports plane-sphere but not plane-cylinder; disable self-collision too.
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    model.geom_conaffinity[foot_spheres] = 1
    model.geom_contype[floor_id] = 1

    info = MPSModelInfo(
        original_integrator=original_integrator,
        timestep=float(model.opt.timestep),
        foot_sphere_ids=tuple(int(i) for i in foot_spheres),
        disabled_cylinder_ids=tuple(int(i) for i in cylinders),
        restored_armature_dofs=len(G1_JOINT_NAMES),
    )
    return model, info


def prepare_mps_vector_env_model(xml_path: str | Path):
    """Load an explicit-pair full-body scene or adapt the legacy origin model."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    if model.npair == 0:
        return prepare_mps_model(xml_path)

    if (model.nq, model.nv, model.nu) != (36, 35, len(G1_JOINT_NAMES)):
        raise ValueError(
            f"expected G1 dimensions (36, 35, 29), got {(model.nq, model.nv, model.nu)}"
        )
    if (
        model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
        or model.jnt_qposadr[0] != 0
        or model.jnt_dofadr[0] != 0
    ):
        raise ValueError("expected the floating base in the first seven qpos and six dof slots")
    actuator_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)
    )
    if actuator_names != tuple(G1_JOINT_NAMES):
        raise ValueError("MJCF actuator order does not match the origin G1 contract")

    for actuator_id, joint_name in enumerate(G1_JOINT_NAMES):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) != joint_name:
            raise ValueError(f"actuator {joint_name} is not attached to its matching joint")
        if model.jnt_qposadr[joint_id] != 7 + actuator_id or model.jnt_dofadr[joint_id] != 6 + actuator_id:
            raise ValueError(f"joint state layout does not match the origin order at {joint_name}")
        expected_armature = motor_of(joint_name)[2]
        if not np.isclose(model.dof_armature[6 + actuator_id], expected_armature, rtol=1e-5, atol=1e-8):
            raise ValueError(f"unexpected armature for {joint_name}")

    integrator = _integrator_name(mujoco, model.opt.integrator)
    if integrator != "implicitfast":
        raise ValueError(f"full-body proxy scene requires implicitfast, got {integrator}")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0 or model.geom_type[floor_id] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise ValueError("explicit-pair G1 scene must contain a plane geom named 'floor'")
    other_geom_ids = np.arange(model.ngeom) != floor_id
    if np.any(model.geom_contype[other_geom_ids]) or np.any(model.geom_conaffinity[other_geom_ids]):
        raise ValueError("explicit-pair G1 collision proxies must disable broadphase masks")
    model.geom_contype[floor_id] = 0
    model.geom_conaffinity[floor_id] = 0

    required_pairs = (
        "head_floor", "torso_floor", "pelvis_floor", "left_hand_floor", "right_hand_floor",
        "left_shoulder_floor", "right_shoulder_floor", "left_elbow_floor", "right_elbow_floor",
    )
    missing_pairs = [
        name for name in required_pairs
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, name) < 0
    ]
    if missing_pairs:
        raise ValueError(f"full-body G1 scene is missing required contact pairs: {missing_pairs}")

    return model, MPSModelInfo(
        original_integrator=integrator,
        timestep=float(model.opt.timestep),
        foot_sphere_ids=(),
        disabled_cylinder_ids=(),
        restored_armature_dofs=0,
        collision_mode="explicit_pairs",
        floor_geom_name="floor",
        explicit_contact_pairs=int(model.npair),
    )


def prepare_origin_reference_model(xml_path: str | Path):
    """Prepare the same armatures/collisions while retaining origin integration."""
    import mujoco

    model, info = prepare_mps_model(xml_path)
    if info.original_integrator != "implicitfast":
        raise ValueError(f"expected origin implicitfast integrator, got {info.original_integrator}")
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    return model, info


class MPSG1:
    """Batched origin-G1 simulator with a 20 ms position-control step."""

    def __init__(self, xml_path: str | Path, num_envs: int, substeps: int = 4):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if substeps < 1:
            raise ValueError("substeps must be positive")

        self.model, self.info = prepare_mps_model(xml_path)
        if not np.isclose(self.model.opt.timestep * substeps, CONTROL_DT, rtol=0.0, atol=1e-8):
            raise ValueError(
                f"substeps={substeps} at timestep={self.model.opt.timestep} "
                f"does not produce the origin control interval {CONTROL_DT}s"
            )
        try:
            from mujoco_warp.mps.robot import BatchedEngine
        except ImportError as exc:
            raise RuntimeError("MPSG1 requires the mujoco_warp fork with its MLX extra") from exc

        self.num_envs = num_envs
        self.substeps = substeps
        self.engine = BatchedEngine(self.model, nworld=num_envs)
        self.default_joint_pos = np.asarray(default_joint_pos_list(), dtype=np.float32)
        self.action_scales = np.asarray([action_scale(name) for name in G1_JOINT_NAMES], dtype=np.float32)
        self.default_qpos = np.broadcast_to(
            np.asarray(self.model.qpos0, dtype=np.float32), (num_envs, self.model.nq)
        ).copy()
        self.default_qpos[:, 2] = ROOT_HEIGHT
        self.default_qpos[:, 7:] = self.default_joint_pos
        self.default_qvel = np.zeros((num_envs, self.model.nv), dtype=np.float32)
        self.default_ctrl = np.broadcast_to(self.default_joint_pos, (num_envs, self.model.nu)).copy()
        self.reset()

    def set_state(self, qpos, qvel, ctrl=None) -> None:
        qpos = np.asarray(qpos, dtype=np.float32)
        qvel = np.asarray(qvel, dtype=np.float32)
        expected_qpos = (self.num_envs, self.model.nq)
        expected_qvel = (self.num_envs, self.model.nv)
        if qpos.shape != expected_qpos or qvel.shape != expected_qvel:
            raise ValueError(f"expected state shapes {expected_qpos} and {expected_qvel}")
        if ctrl is not None:
            ctrl = np.asarray(ctrl, dtype=np.float32)
            if ctrl.shape != (self.num_envs, self.model.nu):
                raise ValueError(f"expected ctrl shape {(self.num_envs, self.model.nu)}")
        self.engine.set_state(qpos, qvel, ctrl)

    def reset(self, mask=None, qpos=None, qvel=None, ctrl=None):
        qpos = self.default_qpos if qpos is None else np.asarray(qpos, dtype=np.float32)
        qvel = self.default_qvel if qvel is None else np.asarray(qvel, dtype=np.float32)
        ctrl = self.default_ctrl if ctrl is None else np.asarray(ctrl, dtype=np.float32)
        if mask is None:
            self.set_state(qpos, qvel, ctrl)
        else:
            self.engine.reset_worlds(mask, qpos=qpos, qvel=qvel, ctrl=ctrl)
        return self.engine.get_state()

    def step_action(self, action, joint_offset=None):
        action = np.asarray(action, dtype=np.float32)
        expected = (self.num_envs, len(G1_JOINT_NAMES))
        if action.shape != expected:
            raise ValueError(f"expected action shape {expected}, got {action.shape}")
        if joint_offset is None:
            joint_offset = np.zeros_like(action)
        else:
            joint_offset = np.asarray(joint_offset, dtype=np.float32)
            if joint_offset.shape != expected:
                raise ValueError(f"expected joint_offset shape {expected}")
        targets = self.default_joint_pos + joint_offset + self.action_scales * np.clip(action, -20.0, 20.0)
        return self.step_targets(targets)

    def step_targets(self, targets):
        targets = np.asarray(targets, dtype=np.float32)
        expected = (self.num_envs, self.model.nu)
        if targets.shape != expected:
            raise ValueError(f"expected target shape {expected}, got {targets.shape}")
        self.engine.step(ctrl=targets)
        for _ in range(self.substeps - 1):
            self.engine.step()
        return self.engine.get_state()
