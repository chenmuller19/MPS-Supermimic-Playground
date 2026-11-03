#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  printf 'MPS-Supermimic requires macOS on Apple Silicon.\n' >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  printf 'Install uv first: brew install uv\n' >&2
  exit 1
fi

cd "$ROOT"
uv python install 3.12
uv sync --locked --extra test
uv run python -c 'import mlx.core as mx; import mujoco; import mujoco_warp.mps.robot; assert mx.metal.is_available(), "MLX Metal is unavailable"; print("MLX Metal, MuJoCo, and mujoco_warp.mps are ready")'
