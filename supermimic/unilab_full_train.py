"""Full SuperMimic world-model/BPTT training loop on a UniLab G1 task."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

from .buffer import ReplayBuffer
from .kinematics import G1Kinematics
from .mdp import MDP, OBS_DIM, STATE_DIM, XI_DIM, CRITIC_EXTRA, INV_DIM, flatten_state, normalize_xi, unflatten_state
from .networks import Critic, Policy, RunningNorm, WorldModel
from .unilab_adapter import UniLabMotionAdapter


def masked_world_model_loss(error: torch.Tensor, mask: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Match SuperMimic's sum-over-time, mean-over-batch objective."""
    per_step = error.square().mean(-1)
    return (per_step * mask).sum() / batch_size


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    return torch.device(name)


def make_env(args, *, eval_mode: bool = False):
    root = Path(args.unilab_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from unilab.base.config_adapter import EnvConfigAdapter, create_env
    from unilab.base.registry import ensure_registries
    ensure_registries()
    with initialize_config_dir(config_dir=str(root / "src/unilab/conf/sac"), version_base=None):
        cfg = compose(config_name="config", overrides=[
            "task=g1_motion_tracking/mujoco",
            f"algo.num_envs={args.num_envs}",
            f"algo.max_iterations={args.iterations}",
            "training.collector_tensor_device=cpu",
        ])
    with open_dict(cfg.env.commands.motion.params):
        cfg.env.commands.motion.params.anchor_body_name = "pelvis"
    cfg.env.max_episode_seconds = 1.0e9
    if eval_mode:
        cfg.env.commands.motion.params.sampling_mode = "mixed"
        cfg.env.commands.motion.params.sampling_start_ratio = 1.0
        with open_dict(cfg.env.commands.motion.params):
            cfg.env.commands.motion.params.adaptive_alpha = 0.0
            cfg.env.commands.motion.params.adaptive_lambda = 0.0
            cfg.env.commands.motion.params.adaptive_uniform_ratio = 1.0
        cfg.env.commands.motion.params.pose_range = {}
        cfg.env.commands.motion.params.velocity_range = {}
        cfg.env.commands.motion.params.joint_position_range = [0.0, 0.0]
        cfg.env.commands.motion.params.joint_default_position_range = [0.0, 0.0]
        with open_dict(cfg.env.terminations):
            cfg.env.terminations.pop("ee_body_pos", None)
        with open_dict(cfg.env):
            cfg.env.events = {}
    else:
        cfg.env.commands.motion.params.sampling_mode = "adaptive"
        with open_dict(cfg.env.commands.motion.params):
            cfg.env.commands.motion.params.adaptive_alpha = 0.0
            cfg.env.commands.motion.params.adaptive_uniform_ratio = 1.0
        with open_dict(cfg.env.events):
            cfg.env.events.pop("push_robot", None)
            cfg.env.events.pop("base_com", None)
            cfg.env.events.pop("encoder_bias", None)
            cfg.env.events.pop("foot_friction", None)
        with open_dict(cfg.env.terminations):
            cfg.env.terminations.pop("ee_body_pos", None)
    override = EnvConfigAdapter(cfg, root_dir=root).build_task_env_cfg_override()
    scene = override.get("scene")
    if isinstance(scene, dict) and scene.get("model_file"):
        scene["model_file"] = str((root / scene["model_file"]).resolve())
    env = create_env(cfg, num_envs=args.num_envs, sim_backend=args.sim, env_cfg_override=override)
    env.set_autoreset(False)
    return env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--unilab-root", default="../UniLab-main")
    p.add_argument("--sim", default="mujoco")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=320000)
    p.add_argument("--utd", type=float, default=0.4)
    p.add_argument("--collect-steps", type=int, default=None, help="override collection steps derived from UTD")
    p.add_argument("--warmup-steps", type=int, default=64)
    p.add_argument("--buffer-steps", type=int, default=1000)
    p.add_argument("--wm-batch", type=int, default=2048)
    p.add_argument("--wm-window", type=int, default=8)
    p.add_argument("--pi-batch", type=int, default=4096)
    p.add_argument("--goal-frames", type=int, default=10)
    p.add_argument("--hist", type=int, default=10)
    p.add_argument("--horizon", type=int, default=32)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--wm-layers", type=int, default=3)
    p.add_argument("--z-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-decay-iters", type=int, default=250000)
    p.add_argument("--ckpt-every", type=int, default=2000)
    p.add_argument("--wm-only-iters", type=int, default=500)
    p.add_argument("--amp", choices=("none", "bf16"), default="none")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--action-noise", type=float, default=0.1)
    p.add_argument("--wm-updates", type=int, default=10)
    p.add_argument("--critic-epochs", type=int, default=5)
    p.add_argument("--critic-minibatches", type=int, default=4)
    p.add_argument("--log-dir", default="logs/unilab_full")
    p.add_argument("--resume", default=None, help="resume model/optimizer state from a full checkpoint")
    args = p.parse_args()
    if args.utd <= 0:
        raise ValueError("--utd must be positive")
    dev = device_for(args.device)
    from . import networks as network_module
    if args.amp == "bf16":
        if dev.type not in {"mps", "cuda"}:
            raise ValueError("bf16 autocast is supported here only for MPS/CUDA learners")
        network_module.AMP_DTYPE = torch.bfloat16
    env = make_env(args)
    out_dir = Path(args.log_dir); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        state = env.init_state()
        adapter = UniLabMotionAdapter(env)
        xml_path = str(Path(args.unilab_root).resolve() / "src/unilab/assets/robots/g1/scene_flat.xml")
        kin = G1Kinematics(str(dev), xml_path=xml_path)
        from .rotations import quat_to_mat
        class Reference:
            root_p = adapter.ref_body_pos_w[:, adapter.root_body_index].to(dev)
            root_R = quat_to_mat(adapter.ref_body_quat_w[:, adapter.root_body_index]).to(dev)
            body_p = adapter.ref_body_pos_w.to(dev)
            body_R = quat_to_mat(adapter.ref_body_quat_w).to(dev)
            body_v = adapter.ref_body_lin_vel_w.to(dev)
            body_w = adapter.ref_body_ang_vel_w.to(dev)
            q = adapter.ref_joint_pos.to(dev)
            qd = adapter.ref_joint_vel.to(dev)
        for name in ("root_p", "root_R", "body_p", "body_R", "body_v", "body_w", "q", "qd"):
            setattr(Reference, name, getattr(Reference, name).to(dev))
        motion = Reference()
        default_q = adapter.default_q[0].to(dev)
        mdp = MDP(kin, motion, default_q)
        goal_dim, history, horizon = 73 * args.goal_frames, args.hist, args.horizon
        wm = WorldModel(hidden=(args.hidden,) * args.wm_layers).to(dev)
        pi = Policy(OBS_DIM, goal_dim, hist=history, z_dim=args.z_dim).to(dev)
        critic = Critic(INV_DIM + CRITIC_EXTRA + goal_dim).to(dev)
        opt_wm = torch.optim.Adam(wm.parameters(), lr=args.lr)
        opt_pi = torch.optim.Adam(pi.parameters(), lr=args.lr, betas=(0.9, 0.95))
        opt_v = torch.optim.Adam(critic.parameters(), lr=args.lr)
        replay = ReplayBuffer(args.buffer_steps, args.num_envs, str(dev))
        a_prev = torch.zeros(args.num_envs, 29, device=dev)
        obs_hist = torch.zeros(args.num_envs, history, OBS_DIM, device=dev)
        critic_features_dim = INV_DIM + CRITIC_EXTRA + goal_dim
        critic_norm = RunningNorm(critic_features_dim).to(dev)
        sim_steps = 0
        start_iteration = 0
        ep_steps = state.info["steps"].to(dev).clone()
        if args.resume:
            checkpoint = torch.load(args.resume, map_location=dev)
            expected_contract = {"state_dim": STATE_DIM, "xi_dim": XI_DIM, "obs_dim": OBS_DIM, "action_dim": 29}
            if checkpoint.get("contract") != expected_contract:
                raise ValueError("checkpoint SuperMimic contract dimensions do not match this code")
            wm.load_state_dict(checkpoint["wm"]); pi.load_state_dict(checkpoint["pi"]); critic.load_state_dict(checkpoint["critic"])
            if "critic_norm" in checkpoint:
                critic_norm.load_state_dict(checkpoint["critic_norm"])
            elif "critic_norm_mean" in checkpoint:
                critic_norm.mean.copy_(checkpoint["critic_norm_mean"]); critic_norm.var.copy_(checkpoint["critic_norm_var"]); critic_norm.count.copy_(checkpoint["critic_norm_count"])
            opt_wm.load_state_dict(checkpoint["opt_wm"]); opt_pi.load_state_dict(checkpoint["opt_pi"]); opt_v.load_state_dict(checkpoint["opt_v"])
            start_iteration = int(checkpoint.get("iteration", 0)); sim_steps = int(checkpoint.get("sim_steps", 0))
            if "replay" in checkpoint:
                replay.load_state_dict(checkpoint["replay"])
            elif "replay" not in checkpoint and "iteration" in checkpoint:
                raise ValueError("checkpoint lacks replay state and cannot resume the same training trajectory")
            if "torch_rng" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng"].cpu().to(dtype=torch.uint8))
            if dev.type == "mps" and "mps_rng" in checkpoint and hasattr(torch.mps, "set_rng_state"):
                torch.mps.set_rng_state(checkpoint["mps_rng"].cpu().to(dtype=torch.uint8))

        error_floor = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3),
                                 torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                                 torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)

        def lr_at(iteration: int) -> float:
            if iteration >= args.lr_decay_iters:
                return args.lr * 0.03
            progress = iteration / max(args.lr_decay_iters, 1)
            return args.lr * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * progress)))

        def refresh_motion():
            return None

        @torch.no_grad()
        def collect(steps):
            nonlocal a_prev, obs_hist, sim_steps, ep_steps
            for _ in range(steps):
                refresh_motion()
                S0 = unflatten_state(adapter.flat_state().to(dev)); f_cpu = adapter.frame; f = f_cpu.to(dev); ce = adapter.clip_end.to(dev); xi = adapter.xi().to(dev)
                goal = adapter.future_goal(f_cpu, args.goal_frames, adapter.clip_end).to(dev)
                o = mdp.add_noise(mdp.obs(S0, a_prev, xi, f))
                fresh = ep_steps == 0
                obs_hist = torch.where(fresh[:, None, None], o[:, None].expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o[:, None]), 1))
                pi.obs_norm.update(o); pi.goal_norm.update(goal)
                action = (pi(obs_hist, goal) + args.action_noise * torch.randn_like(a_prev)).clamp(-20, 20)
                nxt = env.step(adapter.to_unilab_action(action).to(env.device))
                clip_end = f + 1 >= ce
                done = (nxt.terminated | nxt.truncated).to(dev) | clip_end
                S1 = unflatten_state(adapter.flat_state().to(dev))
                wm.in_norm.update(wm.inputs(S0, action, normalize_xi(xi)))
                wm.out_norm.update(wm.targets(S0, S1))
                replay.add({"s": flatten_state(S0), "s1": flatten_state(S1), "a": action, "a_prev": a_prev,
                            "xi": xi, "r": nxt.reward.to(dev), "f": f, "clip_end": ce,
                            "ep_step": ep_steps.to(dtype=torch.int32), "done": done})
                a_prev = action
                ep_steps = nxt.info["steps"].to(dev).clone()
                sim_steps += args.num_envs
                if bool(done.any()):
                    ids = done.nonzero().flatten().to(env.device)
                    env.reset(ids)
                    ep_steps = ep_steps.clone(); ep_steps[done] = 0
                    a_prev = a_prev.clone(); a_prev[done] = 0
                    obs_hist = obs_hist.clone(); obs_hist[done] = 0

        if not args.resume:
            collect(args.warmup_steps)
        losses = []
        step_acc = 0.0
        wm_update_acc = 0.0
        for iteration in range(start_iteration + 1, start_iteration + args.iterations + 1):
            lr = lr_at(iteration - 1)
            for optimizer in (opt_wm, opt_pi, opt_v):
                for group in optimizer.param_groups:
                    group["lr"] = lr
            if dev.type == "mps":
                torch.mps.synchronize()
            step_acc += 1.0 / args.utd
            collect_steps = int(step_acc) if args.collect_steps is None else args.collect_steps
            if args.collect_steps is None:
                step_acc -= collect_steps
            collect_started = time.perf_counter()
            collect(collect_steps)
            if dev.type == "mps":
                torch.mps.synchronize()
            collect_seconds = time.perf_counter() - collect_started
            update_started = time.perf_counter()
            if replay.size < max(args.wm_batch, args.pi_batch) or replay.size < args.wm_window + 1:
                continue
            wm_update_acc += float(args.wm_updates)
            wm_updates_now = int(wm_update_acc)
            wm_update_acc -= wm_updates_now
            for _ in range(wm_updates_now):
                w = replay.sample_windows(args.wm_batch, args.wm_window)
                S = unflatten_state(w["s"][:, 0]); preds=[]
                errors = []
                for t in range(args.wm_window):
                    S = wm.step(S, w["a"][:, t], normalize_xi(w["xi"][:, t]))
                    target_s = unflatten_state(w["s1"][:, t])
                    errors.append(wm.state_error(S, target_s))
                err = torch.stack(errors, 1)
                std = torch.maximum(wm.out_norm.std(), error_floor)
                normalized_error = err / std
                loss = masked_world_model_loss(normalized_error, w["mask"], args.wm_batch)
                opt_wm.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(wm.parameters(), 1); opt_wm.step()
            if iteration - 1 < args.wm_only_iters:
                continue
            starts, hist = replay.sample_starts(args.pi_batch, history, horizon)
            S = unflatten_state(starts["s"]); xi = starts["xi"]; f = starts["f"]
            # The sampled history is reconstructed from physical states, preserving episode boundaries.
            refresh_motion()
            oh = mdp.add_noise(mdp.obs(unflatten_state(hist["s"]), hist["a_prev"], hist["xi"], hist["f"]))
            g = adapter.future_goal(f, args.goal_frames, starts["clip_end"])
            set_grad = lambda module, flag: [q.requires_grad_(flag) for q in module.parameters()]
            set_grad(wm, False); set_grad(critic, False)
            returns = torch.zeros(args.pi_batch, device=dev); disc=1.0; alive = torch.ones(args.pi_batch, device=dev); prev=starts["a_prev"]
            g0 = adapter.future_goal(f, args.goal_frames, starts["clip_end"])
            g = g0
            S0 = S
            Ss=[]; As=[]; Aps=[]; Fs=[]; Gs=[]; Rs=[]; Ds=[]
            for _ in range(horizon):
                Aps.append(prev)
                a = pi(oh, g); S = wm.step(S, a, normalize_xi(xi)); f = f + 1
                g = adapter.future_goal(f, args.goal_frames, starts["clip_end"])
                imagined_obs = mdp.add_noise(mdp.obs(S, a, xi, f))
                oh = torch.cat((oh[:, 1:], imagined_obs[:, None]), 1)
                r, d = mdp.reward(S, f, a, prev), mdp.soft_termination(S, f)
                returns = returns + disc * alive * r
                Rs.append(r)
                alive = alive * (1 - d)
                disc *= args.gamma; prev=a; Ss.append(S); As.append(a); Fs.append(f); Gs.append(g); Ds.append(d)
            terminal_features = torch.cat((mdp.critic_features(S, f, normalize_xi(xi)), g), -1)
            terminal_value = critic.net(critic_norm(terminal_features)).squeeze(-1)
            returns = returns + (args.gamma ** horizon) * alive * terminal_value
            pi_loss = -returns.mean(); opt_pi.zero_grad(set_to_none=True); pi_loss.backward()
            pi_grad_norm = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
            nn.utils.clip_grad_value_(pi.parameters(), 0.1)
            if torch.isfinite(pi_grad_norm):
                opt_pi.step()
            set_grad(wm, True); set_grad(critic, True)

            with torch.no_grad():
                states_c = tuple(torch.cat((S0[i].unsqueeze(0), torch.stack([state[i] for state in Ss])), 0).flatten(0, 1) for i in range(6))
                frames_c = torch.cat((starts["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                goals_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = normalize_xi(xi).unsqueeze(0).expand(horizon + 1, -1, -1).flatten(0, 1)
                features = torch.cat((mdp.critic_features(states_c, frames_c, xi_c), goals_c), -1).view(horizon + 1, args.pi_batch, -1)
            critic_norm.update(features)
            with torch.no_grad():
                values = critic.net(critic_norm(features)).squeeze(-1)
                advantage = torch.zeros_like(values[0])
                targets = []
                for t in range(horizon - 1, -1, -1):
                    delta = Rs[t].detach() + args.gamma * (1 - Ds[t].detach()) * values[t + 1] - values[t]
                    advantage = delta + args.gamma * args.lam * (1 - Ds[t].detach()) * advantage
                    targets.append(advantage + values[t])
                targets = torch.stack(targets[::-1])
            inputs = critic_norm(features[:-1].detach()).reshape(-1, features.shape[-1])
            target_values = targets.reshape(-1)
            batch_size = inputs.shape[0]
            minibatches = min(args.critic_minibatches, batch_size)
            value_loss = torch.zeros((), device=dev)
            for _ in range(args.critic_epochs):
                for indices in torch.tensor_split(torch.randperm(batch_size, device=dev), minibatches):
                    value_loss = nn.functional.mse_loss(critic.net(inputs[indices]).squeeze(-1), target_values[indices])
                    opt_v.zero_grad(set_to_none=True); value_loss.backward(); nn.utils.clip_grad_norm_(critic.parameters(),1); opt_v.step()
            losses.append(float(loss.detach().cpu()))
            if dev.type == "mps":
                torch.mps.synchronize()
            update_seconds = time.perf_counter() - update_started
            if iteration == 1 or iteration % 10 == 0:
                print(f"iter {iteration:4d} | replay {replay.size:5d} | wm {losses[-1]:.5f} | imag {float(returns.detach().mean()):.5f} | collect {collect_seconds:.3f}s | update {update_seconds:.3f}s | device {dev}", flush=True)
        ckpt = {"wm": wm.state_dict(), "pi": pi.state_dict(), "critic": critic.state_dict(), "critic_norm": critic_norm.state_dict(), "opt_wm": opt_wm.state_dict(), "opt_pi": opt_pi.state_dict(), "opt_v": opt_v.state_dict(), "iteration": start_iteration + args.iterations, "sim_steps": sim_steps, "args": vars(args), "replay": replay.state_dict(), "torch_rng": torch.get_rng_state(), "contract": {"state_dim": STATE_DIM, "xi_dim": XI_DIM, "obs_dim": OBS_DIM, "action_dim": 29}}
        if dev.type == "mps" and hasattr(torch.mps, "get_rng_state"):
            ckpt["mps_rng"] = torch.mps.get_rng_state()
        torch.save(ckpt, out_dir / "unilab_supermimic_full.pt")
        (out_dir / "summary.json").write_text(json.dumps({"device":str(dev), "state_dim":STATE_DIM, "obs_dim":OBS_DIM, "xi_dim":XI_DIM, "sim_steps":sim_steps, "last_world_loss":losses[-1] if losses else None}, indent=2)+"\n")
    finally:
        env.close()

if __name__ == "__main__":
    main()
