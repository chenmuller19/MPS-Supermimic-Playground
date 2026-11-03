"""Apple Silicon training entry point.

This is the macOS-compatible reference learner: it trains a differentiable motion-tracking
policy from G1 retarget CSVs without Isaac Lab. MuJoCo/Isaac remain optional evaluation backends;
the learner uses Torch MPS and keeps large tensors contiguous and preallocated.

It is intentionally a policy pretraining and smoke-training path, not a claim of physics parity
with the Isaac/Newton experiment. Use UniLab's MuJoCo backend for full CPU-authoritative rollout.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path

import torch
from torch import nn

try:
    from .apple_backend import make_runtime, synchronize
except ImportError:  # Support `python supermimic/apple_train.py` from a source checkout.
    from apple_backend import make_runtime, synchronize


class MotionPolicy(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 512, action_dim: int = 29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_csv(path: str) -> torch.Tensor:
    rows: list[list[float]] = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if row and not row[0].startswith("#"):
                rows.append([float(v) for v in row])
    data = torch.tensor(rows, dtype=torch.float32)
    if data.ndim != 2 or data.shape[1] < 36:
        raise ValueError(f"{path} must contain root pose plus 29 joints (at least 36 columns)")
    return data[:, 7:36].contiguous()


def _synthetic(frames: int, action_dim: int = 29) -> torch.Tensor:
    t = torch.linspace(0, 12 * torch.pi, frames)
    phase = torch.arange(action_dim, dtype=torch.float32) * 0.17
    return (0.45 * torch.sin(t[:, None] * (0.4 + (phase % 0.7)) + phase)).float()


def _make_batch(q: torch.Tensor, batch: int, history: int, horizon: int, device: torch.device):
    max_start = q.shape[0] - history - horizon - 1
    if max_start < 1:
        raise ValueError("motion clip is too short for the selected history/horizon")
    starts = torch.randint(max_start, (batch,))
    hist = torch.stack([q[starts + i] for i in range(history)], dim=1)
    goal = torch.stack([q[starts + history + i] for i in range(horizon)], dim=1)
    target = goal[:, 0] - q[starts + history - 1]
    x = torch.cat((hist.flatten(1), goal.flatten(1)), dim=-1)
    return x.to(device), target.to(device)


def train(args: argparse.Namespace) -> dict[str, float | str | int]:
    runtime = make_runtime(args.device, amp=args.amp, compile_model=args.compile, memory_fraction=args.memory_fraction)
    device = runtime.device
    torch.manual_seed(args.seed)
    if args.motion:
        files = sorted(glob.glob(args.motion))
        if not files:
            raise FileNotFoundError(f"no motion CSV matched {args.motion!r}")
        q = torch.cat([_load_csv(f) for f in files], dim=0)
    else:
        q = _synthetic(args.synthetic_frames)
    q = q.contiguous()
    input_dim = (args.history + args.goal_horizon) * q.shape[1]
    policy = MotionPolicy(input_dim, args.hidden, q.shape[1]).to(device)
    if runtime.compile:
        try:
            policy = torch.compile(policy, mode="reduce-overhead")
        except Exception as exc:
            print(f"[apple] torch.compile unavailable, using eager mode: {exc}")
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    os.makedirs(args.log_dir, exist_ok=True)
    t0 = time.perf_counter()
    last_loss = float("nan")
    for step in range(1, args.steps + 1):
        x, target = _make_batch(q, args.batch_size, args.history, args.goal_horizon, device)
        if runtime.amp_dtype:
            with torch.autocast(device.type, dtype=runtime.amp_dtype):
                pred = policy(x)
                loss = torch.nn.functional.smooth_l1_loss(pred.float(), target)
        else:
            pred = policy(x)
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
        if step == 1 or step % args.log_every == 0:
            synchronize(device)
            print(f"step {step:6d} | loss {last_loss:.6f} | device {device}", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - t0
    ckpt = {
        "model": policy.state_dict(), "input_dim": input_dim, "action_dim": q.shape[1],
        "history": args.history, "goal_horizon": args.goal_horizon, "device": str(device),
    }
    torch.save(ckpt, Path(args.log_dir) / "apple_policy.pt")
    summary = {"device": str(device), "steps": args.steps, "loss": last_loss, "seconds": elapsed,
               "frames": int(q.shape[0]), "checkpoint": str(Path(args.log_dir) / "apple_policy.pt")}
    Path(args.log_dir, "apple_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="SuperMimic Apple Silicon policy pretraining")
    p.add_argument("--motion", help="glob for LAFAN1 G1 CSV files")
    p.add_argument("--device", default="auto", choices=("auto", "mps", "cpu"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--history", type=int, default=10)
    p.add_argument("--goal-horizon", type=int, default=10)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--synthetic-frames", type=int, default=4096)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--log-dir", default="logs/apple_pretrain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="use float16 autocast on MPS")
    p.add_argument("--compile", action="store_true", help="try torch.compile on MPS")
    p.add_argument("--memory-fraction", type=float, default=0.8,
                   help="MPS adaptive low-watermark ratio (pressure threshold, not a hard cap)")
    args = p.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
