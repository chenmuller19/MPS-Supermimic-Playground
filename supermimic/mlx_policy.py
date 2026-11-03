"""Optional MLX policy inference backend.

MLX is intentionally not imported by the training path. This module is for Apple Silicon
inference/benchmarking after a Torch policy checkpoint has been exported to NPZ.
"""

from __future__ import annotations

from pathlib import Path


def export_policy_npz(checkpoint: str, output: str) -> None:
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {k: v.detach().float().numpy() for k, v in ckpt["model"].items()}
    state["meta_input_dim"] = ckpt["input_dim"]
    state["meta_action_dim"] = ckpt["action_dim"]
    state["meta_history"] = ckpt["history"]
    state["meta_goal_horizon"] = ckpt["goal_horizon"]
    import numpy as np

    np.savez(output, **state)


class MlxPolicy:
    def __init__(self, weights: dict[str, object]):
        try:
            import mlx.core as mx
            import mlx.nn as nn
        except ImportError as exc:
            raise RuntimeError("MLX is optional; install it with `pip install mlx` on Apple Silicon") from exc
        self.mx, self.nn = mx, nn
        self.layers = []
        index = 0
        while f"net.{index}.weight" in weights:
            self.layers.append((mx.array(weights[f"net.{index}.weight"]), mx.array(weights[f"net.{index}.bias"])))
            index += 2

    @classmethod
    def from_npz(cls, path: str | Path) -> "MlxPolicy":
        import numpy as np

        with np.load(path) as data:
            return cls({k: data[k] for k in data.files})

    def __call__(self, x):
        y = x
        for i, (weight, bias) in enumerate(self.layers):
            y = self.mx.add(self.mx.matmul(y, self.mx.transpose(weight)), bias)
            if i < len(self.layers) - 1:
                y = self.mx.silu(y)
        return y
