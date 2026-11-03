"""View a native MLX/MPS SuperMimic checkpoint in the MuJoCo window."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mujoco
import mujoco.viewer
import numpy as np
from mlx.utils import tree_flatten

from supermimic.g1_contract import default_joint_pos_list
from supermimic.mlx_checkpoint import load_models, resolve_checkpoint_files
from supermimic.mlx_geometry import G1Kinematics, mat_to_quat
from supermimic.mlx_mdp import CRITIC_EXTRA, INV_DIM, OBS_DIM, MDP
from supermimic.mlx_motion import MotionLib
from supermimic.mlx_mps_env import CONTROL_DT, MPSVectorEnv
from supermimic.mlx_networks import Critic, Policy, WorldModel


DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "checkpoints" / "origin-quarter-fp16-20it"
)


def _checkpoint_files(path: Path) -> tuple[Path, Path]:
    return resolve_checkpoint_files(path)


def _load_models(path: Path):
    checkpoint_path, metadata_path = _checkpoint_files(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "inference-only-model":
        raise ValueError(f"unsupported checkpoint kind: {metadata.get('kind')!r}")
    config = metadata.get("args", {})
    if not config:
        raise ValueError("checkpoint metadata does not contain training arguments")

    matmul_dtype = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }.get(config.get("mlp_matmul_dtype", "float32"))
    if matmul_dtype is None:
        raise ValueError(f"unsupported MLP matmul dtype: {config['mlp_matmul_dtype']!r}")

    goal_dim = 73 * int(config["goal_frames"])
    critic_dim = INV_DIM + CRITIC_EXTRA + goal_dim
    wm = WorldModel(hidden=(int(config["wm_width"]),) * 3, matmul_dtype=matmul_dtype)
    policy = Policy(
        OBS_DIM,
        goal_dim,
        hist=int(config["hist"]),
        matmul_dtype=matmul_dtype,
    )
    critic = Critic(critic_dim, matmul_dtype=matmul_dtype)
    load_models(checkpoint_path, wm, policy, critic)

    probe = policy(
        mx.zeros((1, int(config["hist"]), OBS_DIM), dtype=mx.float32),
        mx.zeros((1, goal_dim), dtype=mx.float32),
    )
    params = (wm.net.parameters(), policy.parameters(), critic.parameters())
    mx.eval(probe, *[value for _, value in tree_flatten(params)])
    if not np.isfinite(np.asarray(probe)).all():
        raise ValueError("checkpoint produced non-finite actions during load validation")

    return checkpoint_path, metadata, config, wm, policy, critic


def _validate_runtime_physics(config, env: MPSVectorEnv, xml_path: Path) -> None:
    expected = config.get("physics")
    if not expected:
        return
    expected_xml = Path(expected["xml"]).resolve()
    if expected_xml != xml_path.resolve():
        raise ValueError(f"checkpoint expects physics XML {expected_xml}, got {xml_path.resolve()}")
    actual = {
        "collision_mode": env.info.collision_mode,
        "integrator": env.info.original_integrator,
        "timestep_seconds": env.info.timestep,
        "substeps": env.substeps,
        "explicit_contact_pairs": env.info.explicit_contact_pairs,
        "contact_row_cap": env.contact_row_cap,
    }
    mismatches = {
        key: (expected[key], actual[key])
        for key in actual
        if key in expected
        and not (
            np.isclose(expected[key], actual[key], rtol=0.0, atol=1e-10)
            if key == "timestep_seconds"
            else expected[key] == actual[key]
        )
    }
    if mismatches:
        raise ValueError(f"checkpoint physics configuration mismatch: {mismatches}")


def _clip_index(motion: MotionLib, selector: str) -> int:
    if selector.isdigit():
        index = int(selector)
        if 0 <= index < len(motion.names):
            return index
    if selector in motion.names:
        return motion.names.index(selector)
    raise ValueError(f"unknown clip {selector!r}; available clips: {', '.join(motion.names)}")


def _reset_reference(env: MPSVectorEnv, motion: MotionLib, frame: int) -> None:
    ids = np.array([0], dtype=np.int32)
    env.friction_np[0] = 0.75
    env.com_np[0] = 0.0
    env._apply_world_randomization(ids)
    env.xi = mx.array([[0.75] + [0.0] * 32], dtype=mx.float32)

    frame_array = mx.array([frame], dtype=mx.int32)
    p, r, v, w, q, qd = motion.ref_state(frame_array)
    quat = mat_to_quat(r)
    root_vel_local = (mx.swapaxes(r, -1, -2) @ w[..., None]).squeeze(-1)
    qpos = mx.concatenate((p, quat, q), axis=-1)
    qvel = mx.concatenate((v, root_vel_local, qd), axis=-1)
    mx.eval(qpos, qvel, q)
    env.engine.reset_worlds(mx.array([True]), qpos=qpos, qvel=qvel, ctrl=q)

    env.a_prev = mx.zeros((1, 29), dtype=mx.float32)
    env.ep_ret = mx.zeros((1,), dtype=mx.float32)
    env.ep_disc_ret = mx.zeros((1,), dtype=mx.float32)
    env.f[0] = frame
    env.ep_step[0] = 0
    env.start_bin[0] = motion.frame_bin_np[frame]


def _viewer_model_with_ghost(xml_path: Path):
    spec = mujoco.MjSpec.from_file(str(xml_path))
    reference = mujoco.MjSpec.from_file(str(xml_path))
    frame = spec.worldbody.add_frame(name="reference_ghost")
    spec.attach(reference, prefix="ref_", frame=frame)
    model = spec.compile()

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if body_name.startswith("ref_") or geom_name.startswith("ref_"):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            model.geom_rgba[geom_id] = (0.18, 0.78, 1.0, 0.3)

    for material_id in range(model.nmat):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id) or ""
        if name.startswith("ref_"):
            model.mat_rgba[material_id] = (0.18, 0.78, 1.0, 0.3)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ref_floor")
    if floor_id >= 0:
        model.geom_rgba[floor_id, 3] = 0.0
        material_id = int(model.geom_matid[floor_id])
        if material_id >= 0:
            model.mat_rgba[material_id, 3] = 0.0

    return model


def _sync_viewer(env: MPSVectorEnv, viewer_model, viewer_data, motion, reference_frame: int) -> None:
    sim_nq, sim_nv = env.model.nq, env.model.nv
    if viewer_model.nq != sim_nq * 2 or viewer_model.nv != sim_nv * 2:
        raise ValueError("ghost viewer model must contain exactly two copies of the simulated robot")

    viewer_data.qpos[:sim_nq] = np.asarray(env.engine.qpos[0], dtype=np.float64)
    viewer_data.qvel[:sim_nv] = np.asarray(env.engine.qvel[0], dtype=np.float64)

    frame = mx.array([reference_frame], dtype=mx.int32)
    p, r, v, w, q, qd = motion.ref_state(frame)
    quat = mat_to_quat(r)
    root_vel_local = (mx.swapaxes(r, -1, -2) @ w[..., None]).squeeze(-1)
    reference_qpos = mx.concatenate((p, quat, q), axis=-1)
    reference_qvel = mx.concatenate((v, root_vel_local, qd), axis=-1)
    mx.eval(reference_qpos, reference_qvel)
    viewer_data.qpos[sim_nq:] = np.asarray(reference_qpos[0], dtype=np.float64)
    viewer_data.qvel[sim_nv:] = np.asarray(reference_qvel[0], dtype=np.float64)
    mujoco.mj_forward(viewer_model, viewer_data)


def _policy_step(env, motion, policy, history, frame: int, hist_len: int, goal_frames: int):
    frame_array = mx.array([frame], dtype=mx.int32)
    obs = env.mdp.obs(env.state(), env.a_prev, env.xi, frame_array)
    mx.eval(obs)
    if history is None:
        history = mx.broadcast_to(obs[:, None, :], (1, hist_len, OBS_DIM))
    else:
        history = mx.concatenate((history[:, 1:, :], obs[:, None, :]), axis=1)

    goal = motion.goal(frame_array, goal_frames)
    action = policy(history, goal)
    mx.eval(action)
    if not np.isfinite(np.asarray(action)).all():
        raise FloatingPointError(f"policy produced non-finite action at frame {frame}")

    target = env.default_q + env.xi[:, 1:30] + env.action_scales * mx.clip(action, -20.0, 20.0)
    for substep in range(env.substeps):
        env.engine.step(ctrl=target if substep == 0 else None)
    env.a_prev = action
    env.f[0] = frame + 1
    return history


def _view(config, policy, clip_selector: str, show_ghost: bool = True) -> None:
    xml_path = Path(config["xml"])
    motion_dir = Path(config["motion_dir"])
    kin = G1Kinematics(str(xml_path))
    motion = MotionLib.from_dir(str(motion_dir), kin)
    goal_frames = int(config["goal_frames"])
    motion.precompute_goals(goal_frames)
    clip = _clip_index(motion, clip_selector)
    start = int(motion.clip_start_np[clip])
    end = int(motion.clip_end_np[clip])

    mdp = MDP(kin, motion, mx.array(default_joint_pos_list(), dtype=mx.float32))
    env = MPSVectorEnv(
        xml_path,
        1,
        motion,
        mdp,
        gamma=float(config["gamma"]),
        max_episode_steps=int(config["max_episode_steps"]),
        substeps=int(config["substeps"]),
        seed=int(config["seed"]),
    )
    _validate_runtime_physics(config, env, xml_path)
    _reset_reference(env, motion, start)
    viewer_model = _viewer_model_with_ghost(xml_path) if show_ghost else env.model
    viewer_data = mujoco.MjData(viewer_model)
    if show_ghost:
        _sync_viewer(env, viewer_model, viewer_data, motion, start)
    else:
        viewer_data.qpos[:] = np.asarray(env.engine.qpos[0], dtype=np.float64)
        viewer_data.qvel[:] = np.asarray(env.engine.qvel[0], dtype=np.float64)
        mujoco.mj_forward(viewer_model, viewer_data)

    print(
        f"clip {clip}: {motion.names[clip]} | frames {start}:{end} | "
        f"steps {end - start} | MPS physics | "
        f"{'cyan ghost: reference + solid policy' if show_ghost else 'ghost off'} | "
        "looping; close the window to stop",
        flush=True,
    )
    history = None
    with mujoco.viewer.launch_passive(viewer_model, viewer_data) as viewer:
        viewer.cam.lookat[:] = viewer_data.qpos[:3] + np.array([0.0, 0.0, 0.25])
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -12.0
        while viewer.is_running():
            tick = time.perf_counter()
            frame = int(env.f[0])
            if frame >= end:
                _reset_reference(env, motion, start)
                history = None
                frame = start
            history = _policy_step(
                env,
                motion,
                policy,
                history,
                frame,
                int(config["hist"]),
                goal_frames,
            )
            if show_ghost:
                _sync_viewer(env, viewer_model, viewer_data, motion, int(env.f[0]))
            else:
                viewer_data.qpos[:] = np.asarray(env.engine.qpos[0], dtype=np.float64)
                viewer_data.qvel[:] = np.asarray(env.engine.qvel[0], dtype=np.float64)
                mujoco.mj_forward(viewer_model, viewer_data)
            viewer.sync()
            delay = CONTROL_DT - (time.perf_counter() - tick)
            if delay > 0:
                time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--clip", default="0", help="clip index or exact motion name")
    parser.add_argument("--no-ghost", action="store_true", help="hide the translucent reference robot")
    parser.add_argument("--check-only", action="store_true", help="load weights and validate one policy output")
    args = parser.parse_args()

    if not mx.metal.is_available():
        raise RuntimeError("MLX Metal is unavailable; use the Warp-MPS training environment")
    mx.set_default_device(mx.gpu)
    checkpoint_path, metadata, config, wm, policy, critic = _load_models(args.checkpoint)
    iteration = int(metadata.get("iteration", -1))
    print(
        f"Loaded {checkpoint_path} | iteration {iteration} | "
        f"MLP matmul {config.get('mlp_matmul_dtype', 'float32')} | device {mx.default_device()}",
        flush=True,
    )
    if args.check_only:
        print("Checkpoint load and finite-action validation passed.", flush=True)
        return

    _view(config, policy, args.clip, show_ghost=not args.no_ghost)


if __name__ == "__main__":
    main()
