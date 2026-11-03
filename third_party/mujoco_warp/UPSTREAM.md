# Vendored MuJoCo-Warp MPS Backend

This directory vendors the Python package `mujoco_warp` from the workspace checkout of [lucidrains/mujoco_warp](https://github.com/lucidrains/mujoco_warp), at base commit `1b65d3b5d5b721eff34a799cdc98a7bd7626a03b`. It is a fork in the MuJoCo-Warp lineage, not an official Apple or NVIDIA distribution. The fork and the included upstream license are Apache-2.0.

The package includes its source, MPS implementation, tests, and small test fixtures. The dependency metadata is retained in `pyproject.toml` so `uv` can install the vendored package without a submodule or network Git clone. Upstream documentation is preserved as `README-upstream.md`.

The MPS working-tree changes carried into this project are summarized in the repository [README.md](../../README.md). Do not replace this directory with an unmodified PyPI package: doing so would discard the collision and `implicitfast` patches.
