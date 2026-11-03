# Third-Party Notices

Licenses are component-specific. The root [LICENSE](LICENSE) covers the SuperMimic project code; it does not replace the licenses listed below.

| Component | Use in this project | License / attribution |
|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Physics model and viewer APIs | Apache-2.0; see upstream repository |
| [MuJoCo-Warp](https://github.com/google-deepmind/mujoco_warp) | Model conversion and Warp-based engine lineage | Apache-2.0 |
| [lucidrains/mujoco_warp](https://github.com/lucidrains/mujoco_warp) | Vendored fork containing `mujoco_warp.mps`; base commit `1b65d3b5d5b721eff34a799cdc98a7bd7626a03b` | Apache-2.0; copy of the license is in `third_party/mujoco_warp/LICENSE` |
| [NVIDIA Warp](https://github.com/NVIDIA/warp) | Warp dependency and comparison/reference paths | Apache-2.0; see upstream repository |
| [Apple MLX](https://github.com/ml-explore/mlx) | MLX arrays, differentiation, and Metal execution | MIT; see upstream repository |
| [Unitree G1 description in MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) | Included G1 MJCF and mesh assets, source snapshot `f3b6013629c7b94f8ee93fd977b161fa8350f9bc` | BSD-3-Clause; retain `data/menagerie_unitree_g1/LICENSE` |
| [LAFAN1 G1 retargeting dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) | Optional motion input, downloaded by the user | CC BY-NC-ND 4.0 (Ubisoft La Forge), per upstream project; no CSV files are distributed here |

## Backend Attribution

The MPS engine is implemented with MLX and Apple's Metal APIs. `warp-lang` remains a dependency of the vendored MuJoCo-Warp fork and supports its model/reference paths; training does not require an NVIDIA GPU or use NVIDIA Warp's CUDA execution path for the MLX MPS simulation.

The project-specific changes layered on top of the vendored fork are summarized in the root [README.md](README.md). Preserve this file and all component license files when redistributing the repository.
