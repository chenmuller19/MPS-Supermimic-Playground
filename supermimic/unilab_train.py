"""SuperMimic learner running on a UniLab Manager-Based environment.

UniLab owns simulation, reset, motion commands, tracking rewards, and backend lifecycle. This
module only consumes the public TorchEnv contract and implements the SuperMimic-style loop:
collect real transitions -> fit a differentiable observation/reward world model -> imagine a
horizon with the policy -> update policy/value. The simulator remains the selected UniLab backend;
Torch MPS is used only for learner tensors when requested.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, inp: int, out: int, hidden: tuple[int, ...] | int = (512, 512)):
        super().__init__()
        if isinstance(hidden, int):
            hidden = (hidden, hidden)
        layers: list[nn.Module] = []
        width = inp
        for h in hidden:
            layers.extend((nn.Linear(width, h), nn.SiLU()))
            width = h
        layers.append(nn.Linear(width, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Replay:
    """Fixed-shape device-resident replay storage for the learner."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device):
        self.obs = torch.empty((capacity, obs_dim), device=device)
        self.action = torch.empty((capacity, action_dim), device=device)
        self.reward = torch.empty((capacity,), device=device)
        self.next_obs = torch.empty_like(self.obs)
        self.done = torch.empty((capacity,), dtype=torch.bool, device=device)
        self.capacity, self.ptr, self.size = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        n = obs.shape[0]
        if n > self.capacity:
            raise ValueError("num_envs cannot exceed replay capacity")
        end = self.ptr + n
        if end <= self.capacity:
            sl = slice(self.ptr, end)
            self.obs[sl], self.action[sl], self.reward[sl] = obs, action, reward
            self.next_obs[sl], self.done[sl] = next_obs, done
        else:
            first = self.capacity - self.ptr
            self.add(obs[:first], action[:first], reward[:first], next_obs[:first], done[:first])
            self.add(obs[first:], action[first:], reward[first:], next_obs[first:], done[first:])
            return
        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch: int) -> tuple[torch.Tensor, ...]:
        if self.size < 2:
            raise RuntimeError("replay buffer needs at least two transitions")
        ids = torch.randint(self.size, (batch,), device=self.obs.device)
        return self.obs[ids], self.action[ids], self.reward[ids], self.next_obs[ids], self.done[ids]


def _resolve_learner_device(name: str) -> torch.device:
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but this PyTorch build has no available MPS")
    return torch.device(name)


def _make_unilab_env(args: argparse.Namespace):
    root = Path(args.unilab_root).resolve()
    sys.path.insert(0, str(root / "src"))
    try:
        from hydra import compose, initialize_config_dir
        from unilab.base.config_adapter import EnvConfigAdapter, create_env
        from unilab.base.registry import ensure_registries
    except ImportError as exc:
        raise RuntimeError(
            "UniLab dependencies are missing. Run `make setup` in UniLab-main, then use that environment."
        ) from exc
    ensure_registries()
    # UniLab's CLI composes algorithm-local roots (conf/sac), not the parent conf directory.
    # SAC owns the complete G1 motion-tracking base and its MuJoCo host-bridge owner.
    config_dir = root / "src" / "unilab" / "conf" / "sac"
    overrides = [
        "task=g1_motion_tracking/mujoco",
        f"algo.num_envs={args.num_envs}",
        "training.collector_tensor_device=cpu",
        f"algo.max_iterations={max(1, args.iterations)}",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    adapter = EnvConfigAdapter(cfg, root_dir=root)
    override = adapter.build_task_env_cfg_override()
    scene = override.get("scene")
    if isinstance(scene, dict) and scene.get("model_file"):
        scene["model_file"] = str((root / scene["model_file"]).resolve())
    env = create_env(cfg, num_envs=args.num_envs, sim_backend=args.sim, env_cfg_override=override)
    if hasattr(env, "set_autoreset"):
        env.set_autoreset(False)
    return env


def train(args: argparse.Namespace) -> dict[str, object]:
    learner = _resolve_learner_device(args.device)
    env = _make_unilab_env(args)
    try:
        state = env.init_state()
        obs = state.obs[args.obs_group].to(learner).float().contiguous()
        action_dim = int(env.action_space.shape[0])
        obs_dim = obs.shape[-1]
        policy = MLP(obs_dim, action_dim, args.hidden).to(learner)
        world = MLP(obs_dim + action_dim, obs_dim + 1, args.hidden).to(learner)
        value = MLP(obs_dim, 1, (256, 256)).to(learner)
        opt_pi = torch.optim.Adam(policy.parameters(), lr=args.lr)
        opt_wm = torch.optim.Adam(world.parameters(), lr=args.lr)
        opt_v = torch.optim.Adam(value.parameters(), lr=args.lr)
        replay = Replay(args.buffer_size, obs_dim, action_dim, learner)
        stats: list[float] = []
        start = time.perf_counter()

        for iteration in range(1, args.iterations + 1):
            for _ in range(args.collect_steps):
                with torch.no_grad():
                    action = torch.tanh(policy(obs)) * args.action_limit
                env_action = action.detach().to(env.device, dtype=torch.float32).contiguous()
                nxt_state = env.step(env_action)
                next_obs = nxt_state.obs[args.obs_group].to(learner).float().contiguous()
                reward = nxt_state.reward.to(learner).float()
                done = (nxt_state.terminated | nxt_state.truncated).to(learner)
                replay.add(obs, action, reward, next_obs, done)
                obs = next_obs
                done_ids = torch.nonzero(done, as_tuple=False).flatten().to(env.device)
                if done_ids.numel():
                    reset_obs, _ = env.reset(done_ids)
                    obs = obs.clone()
                    obs[done_ids.to(learner)] = reset_obs[args.obs_group].to(learner).float()

            if replay.size >= args.batch_size:
                for _ in range(args.world_updates):
                    o, a, r, no, d = replay.sample(args.batch_size)
                    pred = world(torch.cat((o, a), dim=-1))
                    wm_loss = nn.functional.smooth_l1_loss(pred[:, :-1].contiguous(), (no - o).contiguous())
                    wm_loss = wm_loss + nn.functional.smooth_l1_loss(pred[:, -1], r)
                    opt_wm.zero_grad(set_to_none=True)
                    wm_loss.backward()
                    opt_wm.step()

                # Freeze the learned dynamics during policy BPTT, preserving gradients through its input.
                for p in world.parameters():
                    p.requires_grad_(False)
                start_obs, *_ = replay.sample(args.batch_size)
                imagined = start_obs
                returns = torch.zeros((args.batch_size,), device=learner)
                discount = 1.0
                for _ in range(args.horizon):
                    act = torch.tanh(policy(imagined)) * args.action_limit
                    transition = world(torch.cat((imagined, act), dim=-1))
                    imagined = imagined + transition[:, :-1]
                    returns = returns + discount * transition[:, -1]
                    discount *= args.gamma
                target = returns.detach()
                v_loss = nn.functional.smooth_l1_loss(value(start_obs).squeeze(-1), target)
                pi_loss = -returns.mean()
                opt_pi.zero_grad(set_to_none=True)
                pi_loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt_pi.step()
                opt_v.zero_grad(set_to_none=True)
                v_loss.backward()
                opt_v.step()
                for p in world.parameters():
                    p.requires_grad_(True)
                stats.append(float(wm_loss.detach().cpu()))

            if iteration == 1 or iteration % args.log_every == 0:
                print(
                    f"iter {iteration:5d} | replay {replay.size:6d} | "
                    f"wm {stats[-1] if stats else float('nan'):.5f} | device {learner}",
                    flush=True,
                )

        elapsed = time.perf_counter() - start
        result = {"device": str(learner), "sim": args.sim, "iterations": args.iterations,
                  "obs_dim": obs_dim, "action_dim": action_dim, "seconds": elapsed,
                  "last_world_loss": stats[-1] if stats else None}
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        torch.save({"policy": policy.state_dict(), "world": world.state_dict(), "value": value.state_dict(),
                    "obs_dim": obs_dim, "action_dim": action_dim, "args": vars(args)},
                   Path(args.log_dir) / "unilab_supermimic.pt")
        Path(args.log_dir, "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description="SuperMimic world-model learner on UniLab")
    p.add_argument("--unilab-root", default="../UniLab-main")
    p.add_argument("--sim", default="mujoco", choices=("mujoco", "motrix"))
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    p.add_argument("--obs-group", default="obs")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--collect-steps", type=int, default=4)
    p.add_argument("--buffer-size", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--world-updates", type=int, default=2)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--action-limit", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--log-dir", default="logs/unilab_supermimic")
    print(json.dumps(train(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
