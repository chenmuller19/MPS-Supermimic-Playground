"""Apple Silicon runtime helpers for the Torch/MPS SuperMimic path.

The simulator/data side stays CPU-authoritative on macOS. Learner tensors are placed on
MPS when available, which is the supported PyTorch accelerator on Apple Silicon.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AppleRuntime:
    device: torch.device
    amp_dtype: torch.dtype | None
    compile: bool
    memory_fraction: float


def resolve_device(requested: str = "auto") -> torch.device:
    value = requested.strip().lower()
    if value == "auto":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, mps, cpu")


def configure_memory(fraction: float) -> None:
    """Set MPS adaptive commit/reclamation pressure, not a hard memory cap."""
    if not 0.05 <= fraction <= 1.0:
        raise ValueError("memory_fraction must be in [0.05, 1.0]")
    # Keep Torch's high watermark (default 1.7) intact; tighten its low watermark to
    # encourage earlier command-buffer commits and reclamation under unified-memory pressure.
    if "PYTORCH_MPS_LOW_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{fraction:.3f}"


def make_runtime(device: str, *, amp: bool, compile_model: bool, memory_fraction: float) -> AppleRuntime:
    configure_memory(memory_fraction)
    resolved = resolve_device(device)
    # MPS autocast is optional and less stable than fp32 for rotations/rewards.
    amp_dtype = torch.float16 if amp and resolved.type == "mps" else None
    return AppleRuntime(resolved, amp_dtype, compile_model and resolved.type == "mps", memory_fraction)


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
