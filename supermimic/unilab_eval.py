"""Evaluate a UniLab-backed SuperMimic checkpoint through the public env contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .unilab_train import MLP, _make_unilab_env, _resolve_learner_device


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    learner = _resolve_learner_device(args.device)
    env = _make_unilab_env(args)
    try:
        ckpt = torch.load(args.checkpoint, map_location=learner, weights_only=False)
        state = env.init_state()
        obs = state.obs[args.obs_group].to(learner).float()
        hidden = int(ckpt["args"].get("hidden", 512))
        policy = MLP(ckpt["obs_dim"], ckpt["action_dim"], (hidden, hidden)).to(learner)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()
        returns = torch.zeros(args.num_envs, device=learner)
        lengths = torch.zeros(args.num_envs, device=learner)
        failed = torch.zeros(args.num_envs, dtype=torch.bool, device=learner)
        for _ in range(args.steps):
            with torch.no_grad():
                action = torch.tanh(policy(obs)) * ckpt["args"].get("action_limit", 1.0)
            result = env.step(action.to(env.device, dtype=torch.float32).contiguous())
            returns += result.reward.to(learner)
            lengths += 1
            done = (result.terminated | result.truncated).to(learner)
            failed |= result.terminated.to(learner)
            ids = torch.nonzero(done, as_tuple=False).flatten().to(env.device)
            obs = result.obs[args.obs_group].to(learner).float()
            if ids.numel():
                reset_obs, _ = env.reset(ids)
                obs[ids.to(learner)] = reset_obs[args.obs_group].to(learner).float()
        result = {"device": str(learner), "sim": args.sim, "steps": args.steps,
                  "mean_return": float(returns.mean().cpu()), "mean_length": float(lengths.mean().cpu()),
                  "terminated_envs": int(failed.sum().cpu())}
        print(json.dumps(result, indent=2))
        return result
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a UniLab SuperMimic checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--unilab-root", default="../UniLab-main")
    p.add_argument("--sim", default="mujoco", choices=("mujoco", "motrix"))
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    p.add_argument("--obs-group", default="obs")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
