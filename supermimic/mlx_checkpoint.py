"""Native MLX model-only checkpoints for inference and evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import uuid

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


MODEL_FILENAME = "last_mlx_models.npz"
METADATA_FILENAME = "last_mlx_models.json"


def _module_arrays(prefix: str, module) -> dict[str, mx.array]:
    return {f"{prefix}|{name}": value for name, value in tree_flatten(module.parameters())}


def _norm_arrays(prefix: str, norm) -> dict[str, mx.array]:
    return {
        f"{prefix}|mean": norm.mean,
        f"{prefix}|var": norm.var,
        f"{prefix}|count": mx.array([norm.count], dtype=mx.float32),
        f"{prefix}|eps": mx.array([norm.eps], dtype=mx.float32),
        f"{prefix}|max_count": mx.array([norm.max_count], dtype=mx.float32),
    }


def save_models(path: str | Path, wm, policy, critic) -> None:
    """Save weights and input/output normalizers; optimizer and replay are not included."""
    arrays = {}
    arrays.update(_module_arrays("wm", wm.net))
    arrays.update(_module_arrays("policy", policy))
    arrays.update(_module_arrays("critic", critic))
    arrays.update(_norm_arrays("wm_in_norm", wm.in_norm))
    arrays.update(_norm_arrays("wm_out_norm", wm.out_norm))
    arrays.update(_norm_arrays("policy_obs_norm", policy.obs_norm))
    arrays.update(_norm_arrays("policy_goal_norm", policy.goal_norm))
    arrays.update(_norm_arrays("critic_norm", critic.norm))
    mx.eval(*arrays.values())
    mx.savez(str(path), **arrays)


def save_checkpoint_snapshot(run_dir, iteration: int, wall_seconds: float, args, wm, policy, critic) -> Path:
    """Publish an immutable iteration snapshot and atomically advance ``latest``."""
    run_dir = Path(run_dir)
    snapshots_dir = run_dir / "checkpoints"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    snapshot_name = f"iter_{int(iteration):08d}_{uuid.uuid4().hex}"
    snapshot_path = snapshots_dir / snapshot_name
    latest = run_dir / "latest"
    if latest.exists() and not latest.is_symlink():
        raise FileExistsError(f"refusing to replace non-symlink latest path: {latest}")

    with tempfile.TemporaryDirectory(prefix=".checkpoint-", dir=snapshots_dir) as temporary:
        staging = Path(temporary)
        model_path = staging / MODEL_FILENAME
        save_models(model_path, wm, policy, critic)
        if not model_path.is_file():
            raise RuntimeError(f"checkpoint writer did not create {model_path}")
        with model_path.open("rb") as stream:
            os.fsync(stream.fileno())

        metadata = {
            "kind": "inference-only-model",
            "iteration": int(iteration),
            "wall_seconds": float(wall_seconds),
            "args": vars(args),
            "limitations": [
                "optimizer states, replay, simulator state, RNG, and UTD accumulator are not stored"
            ],
        }
        metadata_path = staging / METADATA_FILENAME
        with metadata_path.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())

        if snapshot_path.exists():
            raise FileExistsError(f"checkpoint snapshot already exists: {snapshot_path}")
        os.rename(staging, snapshot_path)

    temporary_link = run_dir / f".latest-{uuid.uuid4().hex}"
    try:
        temporary_link.symlink_to(os.path.relpath(snapshot_path, run_dir), target_is_directory=True)
        os.replace(temporary_link, latest)
    finally:
        temporary_link.unlink(missing_ok=True)
    return snapshot_path


def resolve_checkpoint_files(path: str | Path) -> tuple[Path, Path]:
    """Resolve legacy flat checkpoints, immutable snapshots, or a run directory."""
    path = Path(path)
    if path.is_dir():
        checkpoint = path / MODEL_FILENAME
        metadata = path / METADATA_FILENAME
        if not (checkpoint.is_file() and metadata.is_file()):
            latest = path / "latest"
            if latest.exists() or latest.is_symlink():
                resolved = latest.resolve(strict=True)
                snapshots_dir = (path / "checkpoints").resolve()
                if resolved.parent != snapshots_dir or not resolved.is_dir():
                    raise ValueError(f"latest checkpoint must point into {snapshots_dir}")
                path = resolved
                checkpoint = path / MODEL_FILENAME
                metadata = path / METADATA_FILENAME
    else:
        checkpoint = path
        metadata = checkpoint.with_name(METADATA_FILENAME)

    if not checkpoint.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"expected {MODEL_FILENAME} and {METADATA_FILENAME} at {path}")
    return checkpoint, metadata


def _load_module(data: dict, prefix: str, module) -> None:
    params = [(key[len(prefix) + 1 :], value) for key, value in data.items() if key.startswith(prefix + "|")]
    if not params:
        raise ValueError(f"checkpoint has no parameter tree for {prefix}")
    module.update(tree_unflatten(params))


def _load_norm(data: dict, prefix: str, norm) -> None:
    try:
        norm.load_state_dict(
            {
                "mean": data[f"{prefix}|mean"],
                "var": data[f"{prefix}|var"],
                "count": float(data[f"{prefix}|count"][0]),
                "eps": float(data[f"{prefix}|eps"][0]),
                "max_count": float(data[f"{prefix}|max_count"][0]),
            }
        )
    except KeyError as exc:
        raise ValueError(f"checkpoint is missing normalizer data for {prefix}") from exc


def load_models(path: str | Path, wm, policy, critic) -> None:
    """Load a model-only checkpoint; this does not resume the training process."""
    data = mx.load(str(path))
    for prefix, module in (("wm", wm.net), ("policy", policy), ("critic", critic)):
        _load_module(data, prefix, module)
    for prefix, norm in (
        ("wm_in_norm", wm.in_norm),
        ("wm_out_norm", wm.out_norm),
        ("policy_obs_norm", policy.obs_norm),
        ("policy_goal_norm", policy.goal_norm),
        ("critic_norm", critic.norm),
    ):
        _load_norm(data, prefix, norm)
