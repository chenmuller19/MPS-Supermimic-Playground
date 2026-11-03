"""MLX SuperMimic learner with a batched MuJoCo MPS G1 simulator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from supermimic.g1_contract import default_joint_pos_list
from supermimic.mlx_checkpoint import save_checkpoint_snapshot
from supermimic.mlx_geometry import G1Kinematics
from supermimic.mlx_mdp import CRITIC_EXTRA, INV_DIM, OBS_DIM, flatten_state, normalize_xi, unflatten_state
from supermimic.mlx_motion import MotionLib
from supermimic.mlx_mdp import MDP
from supermimic.mlx_mps_env import MPSVectorEnv
from supermimic.mlx_networks import Critic, Policy, WorldModel
from supermimic.mlx_replay import ReplayBuffer


PROFILES = {
    "local": {
        "num_envs": 128,
        "iterations": 320_000,
        "buffer_steps": 1000,
        "wm_window": 8,
        "wm_batches": 10,
        "wm_batch": 256,
        "pi_batch": 256,
        "wm_width": 2048,
        "horizon": 32,
        "goal_frames": 10,
        "hist": 10,
        "lr": 2e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "utd": 0.4,
        "action_noise": 0.1,
        "critic_epochs": 5,
        "critic_minibatches": 4,
        "warmup_steps": 64,
        "wm_only_iters": 500,
        "lr_decay_iters": 250_000,
        "ckpt_every": 2000,
        "log_every": 50,
    },
    "origin": {
        "num_envs": 4096,
        "iterations": 320_000,
        "buffer_steps": 1000,
        "wm_window": 8,
        "wm_batches": 10,
        "wm_batch": 2048,
        "pi_batch": 4096,
        "wm_width": 2048,
        "horizon": 32,
        "goal_frames": 10,
        "hist": 10,
        "lr": 2e-4,
        "gamma": 0.99,
        "lam": 0.95,
        "utd": 0.4,
        "action_noise": 0.1,
        "critic_epochs": 5,
        "critic_minibatches": 4,
        "warmup_steps": 64,
        "wm_only_iters": 500,
        "lr_decay_iters": 250_000,
        "ckpt_every": 2000,
        "log_every": 50,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="local")
    for name, kind in (
        ("num_envs", int), ("iterations", int), ("buffer_steps", int), ("wm_window", int),
        ("wm_batches", int), ("wm_batch", int), ("pi_batch", int), ("wm_width", int),
        ("horizon", int), ("goal_frames", int), ("hist", int), ("lr", float), ("gamma", float),
        ("lam", float), ("utd", float), ("action_noise", float), ("critic_epochs", int),
        ("critic_minibatches", int), ("warmup_steps", int), ("wm_only_iters", int),
        ("lr_decay_iters", int), ("ckpt_every", int), ("log_every", int),
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=kind, default=None)
    parser.add_argument("--substeps", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--no-critic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--motion-dir", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--phase-timing", action="store_true")
    parser.add_argument("--phase-timing-warmup-iters", type=int, default=1)
    parser.add_argument(
        "--debug-nonfinite-grad",
        action="store_true",
        help="Report non-finite gradient parameter paths and first invalid rollout step.",
    )
    parser.add_argument(
        "--mlp-matmul-dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Linear-layer matmul precision; master parameters and non-matmul math stay float32.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint-actor-rollout",
        dest="checkpoint_actor_rollout",
        action="store_true",
        help="Rematerialize actor/world-model/reward activations during imagined-rollout backpropagation.",
    )
    checkpoint_group.add_argument(
        "--no-checkpoint-actor-rollout",
        dest="checkpoint_actor_rollout",
        action="store_false",
        help="Disable activation rematerialization and use more memory.",
    )
    parser.set_defaults(checkpoint_actor_rollout=True)
    args = parser.parse_args()
    for name, value in PROFILES[args.profile].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.motion_dir = (args.motion_dir or REPO / "data" / "origin_lafan1_g1").resolve()
    args.xml = (args.xml or REPO / "data" / "menagerie_unitree_g1" / "scene_mjx.xml").resolve()
    args.log_dir = (args.log_dir or REPO / "logs" / f"mlx_mps_{args.profile}").resolve()
    if args.utd <= 0 or args.num_envs < 1:
        parser.error("utd and num-envs must be positive")
    if args.lr <= 0 or args.action_noise < 0:
        parser.error("lr must be positive and action-noise cannot be negative")
    if not (0 < args.gamma < 1) or not (0 <= args.lam <= 1):
        parser.error("gamma must be between 0 and 1, and lambda must be in [0, 1]")
    if min(args.iterations, args.wm_only_iters, args.warmup_steps, args.phase_timing_warmup_iters) < 0:
        parser.error("iteration counts and warmup lengths cannot be negative")
    if min(args.buffer_steps, args.wm_window, args.wm_batches, args.wm_batch, args.pi_batch, args.wm_width) < 1:
        parser.error("buffer, world-model, and policy batch sizes must be positive")
    if args.buffer_steps < args.wm_window:
        parser.error("buffer-steps must be at least wm-window")
    if args.horizon < 1 or args.hist < 1 or args.goal_frames < 1 or args.max_episode_steps < 1:
        parser.error("horizon, hist, goal-frames, and max-episode-steps must be positive")
    if args.substeps is not None and args.substeps < 1:
        parser.error("substeps must be positive when specified")
    if args.critic_epochs < 1 or not (1 <= args.critic_minibatches <= args.horizon * args.pi_batch):
        parser.error("critic epochs and minibatches must fit the imagined policy batch")
    if args.lr_decay_iters < 1 or args.ckpt_every < 1 or args.log_every < 1:
        parser.error("lr-decay-iters, ckpt-every, and log-every must be positive")
    if args.warmup_steps + max(1, int(1.0 / args.utd)) < args.wm_window:
        parser.error("warmup plus the first collection must fill one world-model window")
    return args


def _tree_leaves(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _tree_leaves(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tree_leaves(value)
    else:
        yield tree


def _nonfinite_gradient_paths(grads, limit: int = 8) -> list[str]:
    bad = []
    for path, value in tree_flatten(grads):
        if not hasattr(value, "shape"):
            continue
        mx.eval(value)
        array = np.asarray(value)
        count = int(np.count_nonzero(~np.isfinite(array)))
        if count:
            bad.append(f"{path}: {count}/{array.size}")
            if len(bad) >= limit:
                break
    return bad


def _first_nonfinite_step(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        steps = [_first_nonfinite_step(item) for item in value]
        steps = [step for step in steps if step is not None]
        return min(steps) if steps else None
    mx.eval(value)
    array = np.asarray(value)
    if array.ndim == 0:
        return 0 if not np.isfinite(array) else None
    invalid = ~np.isfinite(array).reshape(array.shape[0], -1).all(axis=1)
    indices = np.flatnonzero(invalid)
    return int(indices[0]) if len(indices) else None


def _tree_map(fn, tree):
    if isinstance(tree, dict):
        return {key: _tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_tree_map(fn, value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, value) for value in tree)
    return fn(tree)


def _global_norm(tree) -> mx.array:
    sums = [mx.sum(value.astype(mx.float32) ** 2) for value in _tree_leaves(tree) if hasattr(value, "shape")]
    return mx.sqrt(mx.sum(mx.stack(sums)))


def _apply_gradients(optimizer, model, grads, max_norm: float, max_value: float | None = None, skip_nonfinite: bool = True):
    norm = _global_norm(grads)
    mx.eval(norm)
    norm_value = float(np.asarray(norm))
    if skip_nonfinite and not math.isfinite(norm_value):
        return norm_value, False
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))

    def clip(grad):
        value = grad * scale
        return mx.clip(value, -max_value, max_value) if max_value is not None else value

    clipped = _tree_map(clip, grads)
    optimizer.update(model, clipped)
    arrays = [value for value in _tree_leaves((model.parameters(), optimizer.state)) if hasattr(value, "shape")]
    mx.eval(*arrays)
    return norm_value, True


def lr_at(iteration: int, args) -> float:
    if iteration >= args.lr_decay_iters:
        return args.lr * 0.03
    return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * iteration / args.lr_decay_iters)))


def _stack_state_list(states, axis: int):
    return tuple(mx.stack([state[i] for state in states], axis=axis) for i in range(6))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    mx.set_default_device(mx.gpu)
    if args.log_dir.exists() and any(args.log_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty; choose a new --log-dir: {args.log_dir}")
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if not args.xml.is_file():
        raise FileNotFoundError(f"G1 MJCF not found: {args.xml}")
    if not args.motion_dir.is_dir():
        raise FileNotFoundError(f"motion directory not found: {args.motion_dir}")

    kin = G1Kinematics(str(args.xml))
    motion = MotionLib.from_dir(str(args.motion_dir), kin)
    cached_goal_table = motion.precompute_goals(args.goal_frames)
    if cached_goal_table is None:
        print("Full motion goal table exceeds 256 MiB; will cache only sampled unique frames.", flush=True)
    mdp = MDP(kin, motion, mx.array(default_joint_pos_list(), dtype=mx.float32))
    env = MPSVectorEnv(
        args.xml, args.num_envs, motion, mdp, gamma=args.gamma,
        max_episode_steps=args.max_episode_steps, substeps=args.substeps, seed=args.seed,
    )
    args.substeps = env.substeps
    args.physics = {
        "xml": str(args.xml),
        "collision_mode": env.info.collision_mode,
        "integrator": env.info.original_integrator,
        "timestep_seconds": env.info.timestep,
        "substeps": env.substeps,
        "explicit_contact_pairs": env.info.explicit_contact_pairs,
        "contact_row_cap": env.contact_row_cap,
    }
    with (args.log_dir / "args.json").open("w", encoding="utf-8") as stream:
        json.dump(vars(args), stream, indent=2, default=str)
    run_info = {
        "motion_clips": motion.names,
        "motion_frames_50hz": motion.num_frames,
        "control_dt_seconds": 0.02,
        "physics_xml": str(args.xml),
        "physics_integrator": env.info.original_integrator,
        "physics_timestep_seconds": env.info.timestep,
        "physics_substeps": args.substeps,
        "collision_mode": env.info.collision_mode,
        "explicit_contact_pairs": env.info.explicit_contact_pairs,
        "contact_row_cap": env.contact_row_cap,
        "collection_mode": "synchronous",
        "converted_cylinder_capsules": len(env.cylinder_ids),
        "foot_spheres": len(env.info.foot_sphere_ids),
        "checkpoint_kind": "inference-only-model; not an exact training resume",
        "mlp_matmul_dtype": args.mlp_matmul_dtype,
    }
    with (args.log_dir / "run_info.json").open("w", encoding="utf-8") as stream:
        json.dump(run_info, stream, indent=2)
    print(
        f"MLX learner | {args.num_envs} MPS worlds | {len(motion.names)} motion clips "
        f"({motion.num_frames} frames at 50 Hz) | {env.info.original_integrator} "
        f"{args.substeps}x{env.info.timestep * 1000:.1f} ms",
        flush=True,
    )
    print(
        f"MPS collisions: {env.info.collision_mode} | "
        f"{env.info.explicit_contact_pairs} explicit pairs | "
        f"contact row cap {env.contact_row_cap}",
        flush=True,
    )
    print("Collection mode: synchronous (MLX streams are thread-local)", flush=True)
    print(f"MLP matmul dtype: {args.mlp_matmul_dtype}; master parameters remain float32", flush=True)
    if len(motion.names) == 1:
        print(f"WARNING: only one motion clip is available: {motion.names[0]}", flush=True)
    goal_dim = 73 * args.goal_frames
    critic_dim = INV_DIM + CRITIC_EXTRA + goal_dim
    matmul_dtype = {
        "float32": mx.float32,
        "float16": mx.float16,
        "bfloat16": mx.bfloat16,
    }[args.mlp_matmul_dtype]
    wm = WorldModel(hidden=(args.wm_width,) * 3, matmul_dtype=matmul_dtype)
    policy = Policy(OBS_DIM, goal_dim, hist=args.hist, matmul_dtype=matmul_dtype)
    critic = Critic(critic_dim, matmul_dtype=matmul_dtype)
    checkpointed_policy = None
    checkpointed_wm_step = None
    checkpointed_rollout_signals = None
    if args.checkpoint_actor_rollout:
        def policy_checkpoint_fn(params, obs_history, encoded_goal):
            policy.update(params)
            return policy.decode(obs_history, encoded_goal)

        def wm_checkpoint_fn(params, state, action, xi_n):
            wm.net.update(params)
            return wm.step(state, action, xi_n)

        def rollout_signals_fn(state, frame, action, previous_action):
            frame = mx.stop_gradient(frame)
            reward = mdp.reward(
                state,
                frame,
                action,
                previous_action,
                return_body_positions=not args.no_critic,
            )
            done = mdp.soft_termination(state, frame)
            return reward, done

        checkpointed_policy = mx.checkpoint(policy_checkpoint_fn)
        checkpointed_wm_step = mx.checkpoint(wm_checkpoint_fn)
        checkpointed_rollout_signals = mx.checkpoint(rollout_signals_fn)
        print("MLX actor rollout activation checkpointing enabled", flush=True)
    opt_wm = optim.Adam(args.lr)
    opt_pi = optim.Adam(args.lr, betas=(0.9, 0.95))
    opt_v = optim.Adam(args.lr)
    replay = ReplayBuffer(args.buffer_steps, args.num_envs)
    obs_hist = mx.zeros((args.num_envs, args.hist, OBS_DIM), dtype=mx.float32)
    err_floor = mx.array(
        [3e-3] * 3 + [5e-3] * 3 + [2e-2] * 3 + [5e-2] * 3 + [3e-3] * 29 + [5e-2] * 29,
        dtype=mx.float32,
    )

    def ingest_transition(transition, obs, goal) -> None:
        replay.add(transition)
        s0 = unflatten_state(transition["s"])
        s1 = unflatten_state(transition["s1"])
        wm.in_norm.update(wm.inputs(s0, transition["a"], normalize_xi(transition["xi"])))
        wm.out_norm.update(wm.targets(s0, s1))

    def collect(n_steps: int, stats: dict) -> None:
        nonlocal obs_hist
        for _ in range(n_steps):
            state = env.state()
            frame = mx.array(env.f, dtype=mx.int32)
            fresh = mx.array(env.ep_step == 0)
            obs = mdp.add_noise(mdp.obs(state, env.a_prev, env.xi, frame))
            shifted = mx.concatenate((obs_hist[:, 1:], obs[:, None, :]), axis=1)
            replicated = mx.broadcast_to(obs[:, None, :], obs_hist.shape)
            obs_hist = mx.where(fresh[:, None, None], replicated, shifted)
            goal = motion.goal(frame, args.goal_frames)
            policy.obs_norm.update(obs)
            policy.goal_norm.update(goal)
            action = policy(obs_hist, goal)
            action = mx.clip(action + args.action_noise * mx.random.normal(shape=action.shape), -20.0, 20.0)
            transition = env.step(action)
            ingest_transition(transition, obs, goal)
            stats["sim_rew"].append(float(np.asarray(mx.mean(transition["r"])) / 0.1))
            if "episode_len" in transition:
                stats["ep_len"].append(float(np.mean(transition["episode_len"])))
                stats["ep_ret"].append(float(np.mean(transition["episode_ret_norm"])))
                stats["term"].append(float(np.sum(transition["term"])))
                stats["done"].append(float(np.sum(transition["done"])))

    def wm_batch_loss(net, batch):
        state = unflatten_state(batch["s"][:, 0])
        predictions = []
        for t in range(args.wm_window):
            state = wm.step(state, batch["a"][:, t], normalize_xi(batch["xi"][:, t]), net=net)
            predictions.append(state)
        predicted = _stack_state_list(predictions, axis=1)
        target = unflatten_state(batch["s1"])
        errors = wm.state_error(predicted, target)
        scaled = (errors / mx.maximum(wm.out_norm.std(), err_floor)) ** 2
        per_frame = mx.mean(scaled, axis=-1)
        return mx.sum(per_frame * batch["mask"]) / batch["s"].shape[0]

    def wm_update():
        batch = replay.sample_windows(args.wm_batch, args.wm_window)
        loss, grads = nn.value_and_grad(wm.net, wm_batch_loss)(wm.net, batch)
        _apply_gradients(opt_wm, wm.net, grads, 1.0, skip_nonfinite=False)
        # The reported eight-step error is evaluated on the updated model.
        state = unflatten_state(batch["s"][:, 0])
        predictions = []
        for t in range(args.wm_window):
            state = wm.step(state, batch["a"][:, t], normalize_xi(batch["xi"][:, t]))
            predictions.append(state)
        predicted = _stack_state_list(predictions, axis=1)
        errors = wm.state_error(predicted, unflatten_state(batch["s1"]))
        per_frame = mx.mean((errors / mx.maximum(wm.out_norm.std(), err_floor)) ** 2, axis=-1)
        last_mask = batch["mask"][:, -1]
        e8 = mx.sum(per_frame[:, -1] * last_mask) / mx.maximum(mx.sum(last_mask), 1.0)
        mx.eval(loss, e8)
        return float(np.asarray(loss)) / args.wm_window, float(np.asarray(e8))

    def critic_input(state, frame, xi_n, goal, body_positions=None):
        features = mdp.critic_features(state, frame, xi_n, body_positions=body_positions)
        return mx.concatenate((features, goal), axis=-1)

    def actor_forward(actor, obs_history, encoded_goal):
        if checkpointed_policy is None:
            return actor.decode(obs_history, encoded_goal)
        return checkpointed_policy(actor.trainable_parameters(), obs_history, encoded_goal)

    def policy_update():
        policy_start = time.perf_counter()
        start, history = replay.sample_starts(args.pi_batch, args.hist, horizon=args.horizon)
        state0 = unflatten_state(start["s"])
        xi, xi_n = start["xi"], normalize_xi(start["xi"])
        frame, a_prev = start["f"], start["a_prev"]
        obs_history = mdp.add_noise(mdp.obs(unflatten_state(history["s"]), history["a_prev"], history["xi"], history["f"]))
        frame_seq_np = np.asarray(frame, dtype=np.int32)[None, :] + np.arange(
            args.horizon + 1, dtype=np.int32
        )[:, None]
        unique_frames, frame_goal_indices = np.unique(frame_seq_np, return_inverse=True)
        frame_seq = mx.array(frame_seq_np, dtype=mx.int32)
        frame_goal_indices = mx.array(
            frame_goal_indices.reshape(frame_seq_np.shape), dtype=mx.int32
        )
        unique_goal_features = motion.goal(
            mx.array(unique_frames, dtype=mx.int32), args.goal_frames
        )
        mx.eval(frame_seq, frame_goal_indices, unique_goal_features)
        prepare_ms = 0.0
        if args.phase_timing:
            mx.eval(*[value for value in _tree_leaves((start, history, obs_history)) if hasattr(value, "shape")])
            prepare_ms = (time.perf_counter() - policy_start) * 1000.0
        rollout_cache = {}

        def actor_objective(actor):
            unique_goal_latents = actor.encode_goal(unique_goal_features)
            goal_latent_seq = unique_goal_latents[frame_goal_indices]
            current, previous_action = state0, a_prev
            current_obs = obs_history
            states, actions, rewards, terminations = [], [], [], []
            body_positions = []
            for t in range(args.horizon):
                action = actor_forward(actor, current_obs, goal_latent_seq[t])
                if checkpointed_wm_step is None:
                    current = wm.step(current, action, xi_n)
                else:
                    current = checkpointed_wm_step(
                        wm.net.trainable_parameters(), current, action, xi_n
                    )
                current_frame = frame_seq[t + 1]
                current_obs1 = mdp.add_noise(mdp.obs(current, action, xi, current_frame))
                current_obs = mx.concatenate((current_obs[:, 1:], current_obs1[:, None, :]), axis=1)
                if checkpointed_rollout_signals is None:
                    reward_output = mdp.reward(
                        current,
                        current_frame,
                        action,
                        previous_action,
                        return_body_positions=not args.no_critic,
                    )
                    done = mdp.soft_termination(current, current_frame)
                else:
                    reward_output, done = checkpointed_rollout_signals(
                        current, current_frame, action, previous_action
                    )
                if args.no_critic:
                    reward = reward_output
                else:
                    reward, body_pos = reward_output
                    body_positions.append(body_pos)
                states.append(current)
                if args.debug_nonfinite_grad:
                    actions.append(action)
                rewards.append(reward)
                terminations.append(done)
                previous_action = action

            state_rollout = _stack_state_list(states, axis=0)
            reward = mx.stack(rewards, axis=0)
            done = mx.stack(terminations, axis=0)
            if args.no_critic:
                body_positions = None
            else:
                body_positions = mx.stack(body_positions, axis=0)
            alive = mx.cumprod(1.0 - done, axis=0)
            c_t = mx.concatenate((mx.ones((1, args.pi_batch), dtype=mx.float32), alive[:-1]), axis=0)
            rollout_cache.update({
                "states": state_rollout,
                "reward": reward,
                "done": done,
                "alive": alive,
                "c_t": c_t,
                "body_positions": body_positions,
            })
            if args.debug_nonfinite_grad:
                rollout_cache["actions"] = mx.stack(actions, axis=0)
            discount = mx.power(args.gamma, mx.arange(args.horizon, dtype=mx.float32))[:, None]
            objective = mx.sum(discount * c_t * reward, axis=0)
            if not args.no_critic:
                final_goal = (
                    motion.goal(frame_seq[-1], args.goal_frames)
                    if cached_goal_table is not None
                    else unique_goal_features[frame_goal_indices[-1]]
                )
                final_input = critic_input(
                    states[-1], frame_seq[-1], xi_n, final_goal, body_positions[-1]
                )
                objective = objective + (args.gamma**args.horizon) * alive[-1] * critic(final_input)
            if args.debug_nonfinite_grad:
                rollout_cache["objective"] = objective
            return -mx.mean(objective)

        if args.phase_timing:
            mx.reset_peak_memory()
        actor_start = time.perf_counter()
        loss, grads = nn.value_and_grad(policy, actor_objective)(policy)
        grad_norm, applied = _apply_gradients(opt_pi, policy, grads, 1.0, max_value=0.1, skip_nonfinite=True)
        if args.debug_nonfinite_grad and not applied:
            invalid_paths = _nonfinite_gradient_paths(grads)
            diagnostics = {
                key: _first_nonfinite_step(rollout_cache.get(key))
                for key in ("actions", "states", "reward", "done")
            }
            objective = rollout_cache.get("objective")
            if objective is not None:
                mx.eval(objective)
                objective_finite = bool(np.isfinite(np.asarray(objective)).all())
            else:
                objective_finite = None
            print(
                f"nonfinite_policy_gradient at iteration {iteration + 1}: "
                f"norm={grad_norm}, bad_parameters={invalid_paths}, rollout={diagnostics}, "
                f"objective_finite={objective_finite}",
                flush=True,
            )
        actor_ms = (time.perf_counter() - actor_start) * 1000.0
        if args.phase_timing:
            actor_peak_mib = mx.get_peak_memory() / (1024 * 1024)
            actor_active_mib = mx.get_active_memory() / (1024 * 1024)
            actor_cache_mib = mx.get_cache_memory() / (1024 * 1024)
        else:
            actor_peak_mib = actor_active_mib = actor_cache_mib = 0.0

        states = rollout_cache["states"]
        body_positions = rollout_cache["body_positions"]
        reward = mx.stop_gradient(rollout_cache["reward"])
        done = mx.stop_gradient(rollout_cache["done"])
        alive = mx.stop_gradient(rollout_cache["alive"])
        c_t = mx.stop_gradient(rollout_cache["c_t"])
        imag_reward = mx.mean(mx.sum(c_t * reward, axis=0) / args.horizon / 0.1)
        survival = mx.mean(alive[-1])

        v_loss = mx.array(0.0, dtype=mx.float32)
        critic_data_ms = 0.0
        critic_update_ms = 0.0
        if not args.no_critic:
            critic_data_start = time.perf_counter()
            goal_seq = (
                motion.goal(frame_seq, args.goal_frames)
                if cached_goal_table is not None
                else unique_goal_features[frame_goal_indices]
            )
            state_seq = tuple(
                mx.concatenate((state0[i][None], states[i]), axis=0)
                for i in range(6)
            )
            xi_seq = mx.broadcast_to(xi_n[None], (args.horizon + 1, args.pi_batch, 33))
            flat_state = tuple(
                value.reshape((args.horizon + 1) * args.pi_batch, *value.shape[2:])
                for value in state_seq
            )
            initial_body_positions = mdp.kin.forward(*state0)[0]
            all_body_positions = mx.concatenate(
                (initial_body_positions[None], body_positions), axis=0
            )
            flat_body_positions = all_body_positions.reshape(
                (args.horizon + 1) * args.pi_batch, *all_body_positions.shape[2:]
            )
            x = critic_input(
                flat_state,
                frame_seq.reshape(-1),
                xi_seq.reshape(-1, 33),
                goal_seq.reshape(-1, goal_dim),
                flat_body_positions,
            )
            x = mx.stop_gradient(x.reshape((args.horizon + 1, args.pi_batch, -1)))
            critic.norm.update(x)
            values = mx.stop_gradient(critic(x))
            target = values[-1]
            target_rows = []
            for t in range(args.horizon - 1, -1, -1):
                target = reward[t] + args.gamma * (1.0 - done[t]) * (
                    (1.0 - args.lam) * values[t + 1] + args.lam * target
                )
                target_rows.append(target)
            targets = mx.stop_gradient(mx.stack(target_rows[::-1], axis=0))
            x_flat = x[:-1].reshape(args.horizon * args.pi_batch, -1)
            y_flat = targets.reshape(-1)
            if args.phase_timing:
                mx.eval(x_flat, y_flat)
                critic_data_ms = (time.perf_counter() - critic_data_start) * 1000.0
            n = args.horizon * args.pi_batch
            minibatch = n // args.critic_minibatches
            critic_update_start = time.perf_counter()
            for _ in range(args.critic_epochs):
                permutation = np.random.permutation(n)
                for index in range(args.critic_minibatches):
                    select = mx.array(permutation[index * minibatch : (index + 1) * minibatch], dtype=mx.int32)
                    x_mb, y_mb = x_flat[select], y_flat[select]

                    def value_loss(model, x_batch, y_batch):
                        return mx.mean((model(x_batch) - y_batch) ** 2)

                    v_loss, v_grads = nn.value_and_grad(critic, value_loss)(critic, x_mb, y_mb)
                    _apply_gradients(opt_v, critic, v_grads, 1.0)
            mx.eval(v_loss)
            critic_update_ms = (time.perf_counter() - critic_update_start) * 1000.0

        mx.eval(loss, imag_reward, survival)
        loss_value = float(np.asarray(loss))
        value_loss_value = float(np.asarray(v_loss))
        imagined_reward_value = float(np.asarray(imag_reward))
        survival_value = float(np.asarray(survival))
        if args.phase_timing:
            total_ms = (time.perf_counter() - policy_start) * 1000.0
            other_ms = max(total_ms - prepare_ms - actor_ms - critic_data_ms - critic_update_ms, 0.0)
            print(
                f"policy_stage it {iteration + 1} | sample_prepare {prepare_ms:.2f} ms "
                f"| actor_bptt_update {actor_ms:.2f} ms "
                f"actor_mem_peak {actor_peak_mib:.1f} active {actor_active_mib:.1f} "
                f"cache {actor_cache_mib:.1f} MiB "
                f"| critic_features_gae {critic_data_ms:.2f} ms "
                f"| critic_updates {critic_update_ms:.2f} ms "
                f"| other {other_ms:.2f} ms | total {total_ms:.2f} ms",
                flush=True,
            )
        return (
            loss_value,
            value_loss_value,
            imagined_reward_value,
            grad_norm,
            survival_value,
            applied,
        )

    def save_checkpoint(iteration: int, wall_seconds: float):
        return save_checkpoint_snapshot(args.log_dir, iteration, wall_seconds, args, wm, policy, critic)

    csv_path = args.log_dir / "train_log.csv"
    fields = (
        "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,"
        "imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr,pi_grad_applied\n"
    )
    new_log = not csv_path.exists() or csv_path.stat().st_size == 0
    sim_steps = 0
    stats = {key: [] for key in ("sim_rew", "ep_len", "ep_ret", "term", "done")}
    phase_totals = {key: 0.0 for key in ("collect", "wm", "policy_critic", "iteration")}
    phase_count = 0
    if args.warmup_steps:
        collect(args.warmup_steps, stats)
        sim_steps += args.warmup_steps * args.num_envs
    start_time, log_time = time.time(), time.time()
    step_acc = 0.0
    agg = {key: [] for key in ("wm", "wm8", "pi", "v", "ir", "gn", "surv", "applied")}
    with csv_path.open("a", encoding="utf-8") as log_file:
        if new_log:
            log_file.write(fields)
        for iteration in range(args.iterations):
            iteration_start = time.perf_counter()
            lr = lr_at(iteration, args)
            opt_wm.learning_rate = lr
            opt_pi.learning_rate = lr
            opt_v.learning_rate = lr
            step_acc += 1.0 / args.utd
            collect_steps = int(step_acc)
            step_acc -= collect_steps
            phase_start = time.perf_counter()
            collect(collect_steps, stats)
            collect_ms = (time.perf_counter() - phase_start) * 1000.0
            sim_steps += collect_steps * args.num_envs
            phase_start = time.perf_counter()
            for _ in range(args.wm_batches):
                wm_loss, wm_error8 = wm_update()
            wm_ms = (time.perf_counter() - phase_start) * 1000.0
            agg["wm"].append(wm_loss)
            agg["wm8"].append(wm_error8)
            policy_ms = 0.0
            if iteration >= args.wm_only_iters:
                phase_start = time.perf_counter()
                pi_loss, value_loss, imagined_reward, grad_norm, survival, applied = policy_update()
                policy_ms = (time.perf_counter() - phase_start) * 1000.0
                agg["pi"].append(pi_loss)
                agg["v"].append(value_loss)
                agg["ir"].append(imagined_reward)
                agg["gn"].append(grad_norm)
                agg["surv"].append(survival)
                agg["applied"].append(float(applied))
            iteration_ms = (time.perf_counter() - iteration_start) * 1000.0
            if args.phase_timing:
                print(
                    f"phase it {iteration + 1} | collect {collect_ms:.2f} ms ({collect_steps} steps) "
                    f"| wm {wm_ms:.2f} ms ({args.wm_batches} updates) "
                    f"| policy+critic {policy_ms:.2f} ms | total {iteration_ms:.2f} ms",
                    flush=True,
                )
                if iteration >= args.phase_timing_warmup_iters:
                    phase_totals["collect"] += collect_ms
                    phase_totals["wm"] += wm_ms
                    phase_totals["policy_critic"] += policy_ms
                    phase_totals["iteration"] += iteration_ms
                    phase_count += 1
            if (iteration + 1) % args.log_every == 0:
                mean = lambda values: float(np.mean(values)) if values else 0.0
                elapsed = max(time.time() - log_time, 1e-8)
                ips = args.log_every / elapsed
                wall_h = (time.time() - start_time) / 3600
                term_fraction = sum(stats["term"]) / max(sum(stats["done"]), 1.0)
                row = [
                    iteration + 1, wall_h, sim_steps, mean(agg["wm"]), mean(agg["wm8"]), mean(agg["pi"]),
                    mean(agg["v"]), mean(agg["ir"]), mean(agg["gn"]), mean(agg["surv"]), mean(stats["sim_rew"]),
                    mean(stats["ep_len"]), mean(stats["ep_ret"]), term_fraction, ips, lr, mean(agg["applied"]),
                ]
                log_file.write(",".join(f"{value:.6g}" if isinstance(value, float) else str(value) for value in row) + "\n")
                log_file.flush()
                log_time = time.time()
                print(
                    f"it {iteration + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} "
                    f"| pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} "
                    f"| sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} "
                    f"term {term_fraction:.2f} | {ips:.2f} it/s",
                    flush=True,
                )
                stats = {key: [] for key in stats}
                agg = {key: [] for key in agg}

            if (iteration + 1) % args.ckpt_every == 0 or iteration + 1 == args.iterations:
                save_checkpoint(iteration + 1, time.time() - start_time)

    if args.phase_timing and phase_count:
        means = {key: round(value / phase_count, 2) for key, value in phase_totals.items()}
        print(f"phase_timing_steady_mean_ms={means}", flush=True)

    if args.iterations == 0:
        save_checkpoint(0, 0.0)


if __name__ == "__main__":
    main()
