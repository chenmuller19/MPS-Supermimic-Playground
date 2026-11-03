"""Probe actor BPTT gradients from an inference-only MLX checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from supermimic.g1_contract import default_joint_pos_list
from supermimic.mlx_checkpoint import load_models, resolve_checkpoint_files
from supermimic.mlx_geometry import G1Kinematics
from supermimic.mlx_mdp import CRITIC_EXTRA, INV_DIM, OBS_DIM, MDP, normalize_xi
from supermimic.mlx_motion import MotionLib
from supermimic.mlx_networks import Critic, Policy, WorldModel
from supermimic.mlx_train import _nonfinite_gradient_paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="run directory, snapshot directory, or model npz")
    parser.add_argument("--motion-dir", type=Path, default=REPO / "data" / "origin_lafan1_g1")
    parser.add_argument("--xml", type=Path, default=REPO / "data" / "menagerie_unitree_g1" / "scene_mjx.xml")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--clip-index", type=int, default=-1, help="-1 samples starts across all clips")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-critic", action="store_true", help="alias for --objective reward")
    parser.add_argument("--objective", choices=("full", "reward", "critic"), default="full")
    parser.add_argument("--stop-critic-value-grad", action="store_true")
    parser.add_argument("--stop-critic-feature-grad", action="store_true")
    parser.add_argument("--critic-feature-group", choices=("all", "invariant", "joint", "body", "root", "relative", "dynamics"), default="all")
    parser.add_argument("--detach-final-body-position", action="store_true")
    parser.add_argument("--stop-critic-survival-grad", action="store_true")
    parser.add_argument("--critic-scale", type=float, default=1.0)
    parser.add_argument("--matmul-dtype", choices=("float32", "float16", "bfloat16"), default=None)
    args = parser.parse_args()
    if args.no_critic:
        args.objective = "reward"
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    return args


def _sample_frames(motion, batch_size: int, history: int, horizon: int, clip_index: int, seed: int) -> np.ndarray:
    if clip_index < -1 or clip_index >= len(motion.names):
        raise ValueError(f"clip-index must be -1 or in [0, {len(motion.names) - 1}]")
    clips = range(len(motion.names)) if clip_index == -1 else (clip_index,)
    candidates = []
    for clip in clips:
        lo = int(motion.clip_start_np[clip]) + history - 1
        hi = int(motion.clip_end_np[clip]) - horizon
        if hi >= lo:
            candidates.append(np.arange(lo, hi + 1, dtype=np.int32))
    if not candidates:
        raise ValueError("no clip has enough frames for the requested history and horizon")
    candidates = np.concatenate(candidates)
    rng = np.random.default_rng(seed)
    return rng.choice(candidates, size=batch_size, replace=len(candidates) < batch_size).astype(np.int32)


def _first_nonfinite_time(rows) -> int | None:
    for step, row in enumerate(rows):
        values = row if isinstance(row, (tuple, list)) else (row,)
        for value in values:
            mx.eval(value)
            if not np.isfinite(np.asarray(value)).all():
                return step
    return None


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    mx.set_default_device(mx.gpu)

    checkpoint_path, metadata_path = resolve_checkpoint_files(args.checkpoint)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    saved_args = metadata.get("args", {})
    horizon = args.horizon or int(saved_args.get("horizon", 32))
    goal_frames = int(saved_args.get("goal_frames", 10))
    hist = int(saved_args.get("hist", 10))
    width = int(saved_args.get("wm_width", 2048))
    dtype_name = args.matmul_dtype or saved_args.get("mlp_matmul_dtype", "float32")
    dtype = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}[dtype_name]

    kin = G1Kinematics(str(args.xml))
    motion = MotionLib.from_dir(str(args.motion_dir), kin)
    mdp = MDP(kin, motion, mx.array(default_joint_pos_list(), dtype=mx.float32))
    wm = WorldModel(hidden=(width,) * 3, matmul_dtype=dtype)
    goal_dim = 73 * goal_frames
    policy = Policy(OBS_DIM, goal_dim, hist=hist, matmul_dtype=dtype)
    critic = Critic(INV_DIM + CRITIC_EXTRA + goal_dim, matmul_dtype=dtype)
    load_models(checkpoint_path, wm, policy, critic)

    frames_np = _sample_frames(
        motion, args.batch_size, hist, horizon, args.clip_index, args.seed
    )
    frames = mx.array(frames_np, dtype=mx.int32)
    frame_seq_np = frames_np[None, :] + np.arange(horizon + 1, dtype=np.int32)[:, None]
    frame_seq = mx.array(frame_seq_np, dtype=mx.int32)
    unique_frames, goal_indices = np.unique(frame_seq_np, return_inverse=True)
    goal_table = motion.goal(mx.array(unique_frames, dtype=mx.int32), goal_frames)
    goal_indices = mx.array(goal_indices.reshape(frame_seq_np.shape), dtype=mx.int32)
    xi_np = np.zeros((args.batch_size, 33), dtype=np.float32)
    xi_np[:, 0] = 0.75
    xi_np[:, 30:33] = 0.0
    xi = mx.array(xi_np)
    xi_n = normalize_xi(xi)
    zeros_action = mx.zeros((args.batch_size, 29), dtype=mx.float32)
    hist_frames = frames_np[None, :] - np.arange(hist - 1, -1, -1, dtype=np.int32)[:, None]
    hist_frames_mx = mx.array(hist_frames, dtype=mx.int32)
    hist_state = motion.ref_state(hist_frames_mx)
    hist_xi = mx.broadcast_to(xi[None], (hist, args.batch_size, 33))
    hist_action = mx.zeros((hist, args.batch_size, 29), dtype=mx.float32)
    obs = mdp.obs(hist_state, hist_action, hist_xi, hist_frames_mx)
    obs_history = mx.swapaxes(mdp.add_noise(obs), 0, 1)
    state0 = motion.ref_state(frames)
    body_position_rows, reward_rows, done_rows, state_rows = [], [], [], []
    critic_input_rows = []

    checkpointed_policy = None
    checkpointed_wm = None
    if saved_args.get("checkpoint_actor_rollout", True):
        checkpointed_policy = mx.checkpoint(
            lambda params, observations, goals: (policy.update(params), policy.decode(observations, goals))[1]
        )
        checkpointed_wm = mx.checkpoint(
            lambda params, state, action, dynamics: (wm.net.update(params), wm.step(state, action, dynamics))[1]
        )

    def actor_forward(actor, observations, goals):
        if checkpointed_policy is None:
            return actor.decode(observations, goals)
        return checkpointed_policy(actor.trainable_parameters(), observations, goals)

    def actor_wm_step(state, action, dynamics):
        if checkpointed_wm is None:
            return wm.step(state, action, dynamics)
        return checkpointed_wm(wm.net.trainable_parameters(), state, action, dynamics)

    def objective(actor):
        unique_goals = actor.encode_goal(goal_table)
        goal_seq = unique_goals[goal_indices]
        current, previous_action = state0, zeros_action
        current_obs = obs_history
        for t in range(horizon):
            action = actor_forward(actor, current_obs, goal_seq[t])
            current = actor_wm_step(current, action, xi_n)
            current_frame = frame_seq[t + 1]
            current_obs1 = mdp.add_noise(mdp.obs(current, action, xi, current_frame))
            current_obs = mx.concatenate((current_obs[:, 1:], current_obs1[:, None]), axis=1)
            reward, body_positions = mdp.reward(
                current, current_frame, action, previous_action, return_body_positions=True
            )
            done = mdp.soft_termination(current, current_frame)
            state_rows.append(current)
            body_position_rows.append(body_positions)
            reward_rows.append(reward)
            done_rows.append(done)
            previous_action = action

        states = tuple(mx.stack([row[i] for row in state_rows], axis=0) for i in range(6))
        rewards = mx.stack(reward_rows, axis=0)
        done = mx.stack(done_rows, axis=0)
        alive = mx.cumprod(1.0 - done, axis=0)
        c_t = mx.concatenate((mx.ones((1, args.batch_size), dtype=mx.float32), alive[:-1]), axis=0)
        discount = mx.power(mx.array(0.99, dtype=mx.float32), mx.arange(horizon, dtype=mx.float32))[:, None]
        if args.objective == "critic":
            result = mx.zeros((args.batch_size,), dtype=mx.float32)
        else:
            result = mx.sum(discount * c_t * rewards, axis=0)
        if args.objective in ("full", "critic"):
            final_body_positions = body_position_rows[-1]
            if args.detach_final_body_position:
                final_body_positions = mx.stop_gradient(final_body_positions)
            critic_features = mdp.critic_features(
                tuple(value[-1] for value in states),
                frame_seq[-1],
                xi_n,
                body_positions=final_body_positions,
            )
            final_goal = goal_table[goal_indices[-1]]
            feature_group_slices = {
                "invariant": (0, INV_DIM),
                "joint": (INV_DIM, INV_DIM + 29),
                "body": (INV_DIM + 29, INV_DIM + 29 + 42),
                "root": (INV_DIM + 29 + 42, INV_DIM + 29 + 42 + 3),
                "relative": (INV_DIM + 29 + 42 + 3, INV_DIM + 29 + 42 + 9),
                "dynamics": (INV_DIM + 29 + 42 + 9, INV_DIM + CRITIC_EXTRA),
            }
            if args.critic_feature_group != "all":
                start, end = feature_group_slices[args.critic_feature_group]
                feature_mask_np = np.zeros((INV_DIM + CRITIC_EXTRA,), dtype=np.float32)
                feature_mask_np[start:end] = 1.0
                feature_mask = mx.array(feature_mask_np, dtype=critic_features.dtype)
                critic_features = mx.stop_gradient(critic_features) + (
                    critic_features - mx.stop_gradient(critic_features)
                ) * feature_mask
            critic_input = mx.concatenate((critic_features, final_goal), axis=-1)
            critic_input_rows.append(critic_input)
            if args.stop_critic_feature_grad:
                critic_input = mx.stop_gradient(critic_input)
            critic_value = critic(critic_input)
            if args.stop_critic_value_grad:
                critic_value = mx.stop_gradient(critic_value)
            survival = mx.stop_gradient(alive[-1]) if args.stop_critic_survival_grad else alive[-1]
            result = result + args.critic_scale * (0.99**horizon) * survival * critic_value
        return -mx.mean(result)

    loss, gradients = nn.value_and_grad(policy, objective)(policy)
    gradient_leaves = tree_flatten(gradients)
    mx.eval(loss, *[value for _, value in gradient_leaves])
    bad_parameters = _nonfinite_gradient_paths(gradients)
    loss_value = float(np.asarray(loss))
    gradient_norm = float(np.sqrt(sum(np.square(np.asarray(value, dtype=np.float64)).sum() for _, value in gradient_leaves)))
    forward_bad_step = _first_nonfinite_time(state_rows)
    bad_reward_step = _first_nonfinite_time(reward_rows)
    bad_done_step = _first_nonfinite_time(done_rows)
    critic_input_grad = None
    critic_bad_rows = []
    critic_bad_row_stats = []
    critic_layer_trace = []
    critic_prefix_gradient_trace = []
    critic_parameter_stats = []
    if critic_input_rows:
        critic_input = critic_input_rows[-1]
        critic_input_grad = mx.grad(lambda value: mx.sum(critic(value)))(critic_input)
        critic_output = critic(critic_input)
        mx.eval(critic_input_grad, critic_output)
        input_np = np.asarray(critic_input)
        input_grad_np = np.asarray(critic_input_grad)
        output_np = np.asarray(critic_output)
        critic_bad_rows = np.flatnonzero(~np.isfinite(input_grad_np).all(axis=-1)).tolist()
        critic_bad_row_stats = [
            {
                "row": int(row),
                "input_finite": bool(np.isfinite(input_np[row]).all()),
                "input_abs_max": float(np.nanmax(np.abs(input_np[row]))),
                "output": float(output_np[row]),
                "gradient_bad_values": int((~np.isfinite(input_grad_np[row])).sum()),
            }
            for row in critic_bad_rows[:8]
        ]
        if critic_bad_rows:
            trace = critic.norm(critic_input)
            for index, layer in enumerate(critic.net.layers.layers):
                trace = layer(trace)
                mx.eval(trace)
                row_trace = np.asarray(trace)[critic_bad_rows[0]]
                critic_layer_trace.append({
                    "layer": index,
                    "finite": bool(np.isfinite(row_trace).all()),
                    "abs_max": float(np.nanmax(np.abs(row_trace))),
                    "min": float(np.nanmin(row_trace)),
                    "max": float(np.nanmax(row_trace)),
                })
            net_input = critic.norm(critic_input)
            layers = critic.net.layers.layers
            for end in range(len(layers)):
                def prefix_loss(value, end=end):
                    for layer in layers[: end + 1]:
                        value = layer(value)
                    return mx.sum(value)

                prefix_grad = mx.grad(prefix_loss)(net_input)
                mx.eval(prefix_grad)
                prefix_np = np.asarray(prefix_grad)
                critic_prefix_gradient_trace.append({
                    "through_layer": end,
                    "bad_values": int((~np.isfinite(prefix_np)).sum()),
                    "bad_rows": np.flatnonzero(~np.isfinite(prefix_np).all(axis=-1)).tolist()[:8],
                })
            critic_parameter_stats = [
                {
                    "name": name,
                    "finite": bool(np.isfinite(np.asarray(value)).all()),
                    "abs_max": float(np.max(np.abs(np.asarray(value, dtype=np.float64)))),
                }
                for name, value in tree_flatten(critic.net.parameters())
            ]

    print(json.dumps({
        "checkpoint": str(checkpoint_path),
        "iteration": metadata.get("iteration"),
        "motion_clips_loaded": len(motion.names),
        "sampled_clips": sorted(set(motion.clip_id_np[frames_np].tolist())),
        "batch_size": args.batch_size,
        "horizon": horizon,
        "matmul_dtype": dtype_name,
        "objective": args.objective,
        "stop_critic_value_grad": args.stop_critic_value_grad,
        "stop_critic_feature_grad": args.stop_critic_feature_grad,
        "critic_feature_group": args.critic_feature_group,
        "detach_final_body_position": args.detach_final_body_position,
        "stop_critic_survival_grad": args.stop_critic_survival_grad,
        "critic_scale": args.critic_scale,
        "loss_finite": bool(np.isfinite(loss_value)),
        "loss": loss_value,
        "gradient_norm": gradient_norm,
        "gradient_norm_finite": bool(np.isfinite(gradient_norm)),
        "bad_parameters": bad_parameters,
        "critic_input_gradient_finite": (
            None if critic_input_grad is None else bool(np.isfinite(np.asarray(critic_input_grad)).all())
        ),
        "critic_input_gradient_norm": (
            None if critic_input_grad is None else float(np.linalg.norm(np.asarray(critic_input_grad, dtype=np.float64)))
        ),
        "critic_input_gradient_bad_values": (
            [] if critic_input_grad is None else _nonfinite_gradient_paths({"critic_input": critic_input_grad})
        ),
        "critic_input_bad_rows": critic_bad_rows[:32],
        "critic_bad_row_stats": critic_bad_row_stats,
        "critic_bad_row_layer_trace": critic_layer_trace,
        "critic_prefix_gradient_trace": critic_prefix_gradient_trace,
        "critic_parameter_stats": critic_parameter_stats,
        "first_nonfinite_state_step": forward_bad_step,
        "first_nonfinite_reward_step": bad_reward_step,
        "first_nonfinite_done_step": bad_done_step,
    }, indent=2))
    if not np.isfinite(gradient_norm) or bad_parameters:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
