"""Benchmark Warp MPS on an explicit MPS-compatible Menagerie G1 derivative."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from supermimic.g1_contract import (
    DAMPING_RATIO,
    G1_JOINT_NAMES,
    JOINT_MOTORS,
    NATURAL_FREQ,
    action_scale,
    default_joint_pos_list,
)


DEFAULT_XML = Path(__file__).resolve().parents[1] / "data/menagerie_unitree_g1/scene_mjx.xml"
CONTROL_DT = 0.02
ROOT_HEIGHT = 0.75
def prepare_mps_compatible_model(xml_path: Path):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(xml_path))
    original_pair_names = {pair.name for pair in spec.pairs}
    spec.njmax = 1024
    spec.nconmax = 512
    model = spec.compile()
    if model.npair != len(original_pair_names):
        raise ValueError(f"expected {len(original_pair_names)} contact pairs, got {model.npair}")

    if model.nu != len(G1_JOINT_NAMES):
        raise ValueError(f"expected {len(G1_JOINT_NAMES)} G1 actuators, got {model.nu}")
    actuator_names = tuple(model.actuator(i).name for i in range(model.nu))
    if actuator_names != tuple(G1_JOINT_NAMES):
        raise ValueError("Menagerie actuator order does not match the shared G1 contract")

    for actuator_id, joint_name in enumerate(G1_JOINT_NAMES):
        key = joint_name.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
        effort, _, armature = JOINT_MOTORS[key]
        kp = armature * NATURAL_FREQ**2
        kd = 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        dof_id = int(model.jnt_dofadr[joint_id])
        if not np.isclose(model.dof_armature[dof_id], armature, rtol=2e-5, atol=1e-7):
            raise ValueError(f"Menagerie armature does not match the G1 contract at {joint_name}")
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_forcerange[actuator_id] = (-effort, effort)
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_ctrllimited[actuator_id] = 0

    original_integrator = int(model.opt.integrator)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    return model, original_integrator


def make_initial_state(model, worlds: int):
    qpos = np.broadcast_to(model.qpos0, (worlds, model.nq)).astype(np.float32).copy()
    qpos[:, 2] = ROOT_HEIGHT
    qpos[:, 7:] = np.asarray(default_joint_pos_list(), dtype=np.float32)
    qvel = np.zeros((worlds, model.nv), dtype=np.float32)
    ctrl = np.broadcast_to(qpos[:, 7:], (worlds, model.nu)).copy()
    return qpos, qvel, ctrl


def make_actions(worlds: int, steps: int, model, substeps: int):
    action_scale_values = np.asarray([action_scale(name) for name in G1_JOINT_NAMES], dtype=np.float32)
    t = np.arange(steps, dtype=np.float32)[:, None, None] * substeps * model.opt.timestep
    env_phase = np.arange(worlds, dtype=np.float32)[None, :, None] * 0.013
    joint_phase = np.arange(model.nu, dtype=np.float32)[None, None, :] * 0.17
    actions = 0.25 * np.sin(2.0 * np.pi * t + env_phase + joint_phase)
    defaults = np.asarray(default_joint_pos_list(), dtype=np.float32)
    targets = defaults[None, None, :] + action_scale_values[None, None, :] * actions
    return targets.astype(np.float32)


def _advance_mps(engine, targets, substeps: int):
    engine.step(ctrl=targets)
    for _ in range(substeps - 1):
        engine.step()
    qpos, qvel = engine.get_state()
    return np.asarray(qpos).copy(), np.asarray(qvel).copy()


def run_mps(engine, qpos, qvel, ctrl, targets, substeps: int, warmup: int, repeats: int):
    for _ in range(warmup):
        engine.set_state(qpos, qvel, ctrl)
        for action_targets in targets[:1]:
            _advance_mps(engine, action_targets, substeps)

    elapsed_runs = []
    qpos_history = qvel_history = None
    for _ in range(repeats):
        engine.set_state(qpos, qvel, ctrl)
        qpos_samples = []
        qvel_samples = []
        start = time.perf_counter()
        for action_targets in targets:
            qpos_out, qvel_out = _advance_mps(engine, action_targets, substeps)
            qpos_samples.append(qpos_out)
            qvel_samples.append(qvel_out)
        elapsed_runs.append(time.perf_counter() - start)
        qpos_history = np.stack(qpos_samples)
        qvel_history = np.stack(qvel_samples)
    return float(np.median(elapsed_runs)), qpos_history, qvel_history


def _set_warp_state(data, qpos, qvel, ctrl):
    data.qpos.numpy()[:] = qpos
    data.qvel.numpy()[:] = qvel
    data.ctrl.numpy()[:] = ctrl
    data.qacc_warmstart.numpy()[:] = 0


def run_warp_cpu(model, qpos, qvel, ctrl, targets, substeps: int, warmup: int, repeats: int):
    import mujoco_warp
    import warp as wp

    nworld = qpos.shape[0]
    warp_model = mujoco_warp.put_model(model)
    warm_data = mujoco_warp.make_data(model, nworld=nworld, nconmax=512, njmax=1024)
    _set_warp_state(warm_data, qpos, qvel, ctrl)
    for action_targets in targets[: max(1, warmup)]:
        warm_data.ctrl.numpy()[:] = action_targets
        for _ in range(substeps):
            mujoco_warp.step(warp_model, warm_data)
        wp.synchronize()

    elapsed_runs = []
    qpos_history = qvel_history = None
    for _ in range(repeats):
        data = mujoco_warp.make_data(model, nworld=nworld, nconmax=512, njmax=1024)
        _set_warp_state(data, qpos, qvel, ctrl)
        qpos_samples = []
        qvel_samples = []
        start = time.perf_counter()
        for action_targets in targets:
            data.ctrl.numpy()[:] = action_targets
            for _ in range(substeps):
                mujoco_warp.step(warp_model, data)
            wp.synchronize()
            qpos_samples.append(np.asarray(data.qpos.numpy()).copy())
            qvel_samples.append(np.asarray(data.qvel.numpy()).copy())
        elapsed_runs.append(time.perf_counter() - start)
        qpos_history = np.stack(qpos_samples)
        qvel_history = np.stack(qvel_samples)
    return float(np.median(elapsed_runs)), qpos_history, qvel_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50, help="20 ms control transitions")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--contact-row-cap", type=int, default=512)
    parser.add_argument("--iterations", type=int, help="override the Menagerie solver iteration count on both engines")
    args = parser.parse_args()
    if args.worlds < 1 or args.steps < 1 or args.warmup < 0 or args.repeats < 1 or args.contact_row_cap < 1:
        parser.error("worlds, steps, repeats, and contact-row-cap must be positive; warmup cannot be negative")
    if args.iterations is not None and args.iterations < 1:
        parser.error("iterations must be positive")

    import mujoco
    from mujoco_warp.mps.robot import BatchedEngine

    model, original_integrator = prepare_mps_compatible_model(args.xml)
    if args.iterations is not None:
        model.opt.iterations = args.iterations
    substeps = int(round(CONTROL_DT / model.opt.timestep))
    if not np.isclose(substeps * model.opt.timestep, CONTROL_DT, rtol=0.0, atol=1e-8):
        raise ValueError("Menagerie timestep must divide the 20 ms control interval")

    qpos, qvel, ctrl = make_initial_state(model, args.worlds)
    targets = make_actions(args.worlds, args.steps, model, substeps)
    engine = BatchedEngine(model, nworld=args.worlds, contact_row_cap=args.contact_row_cap)
    mps_elapsed, mps_qpos, mps_qvel = run_mps(
        engine, qpos, qvel, ctrl, targets, substeps, args.warmup, args.repeats
    )
    cpu_elapsed, cpu_qpos, cpu_qvel = run_warp_cpu(
        model, qpos, qvel, ctrl, targets, substeps, args.warmup, args.repeats
    )

    qpos_diff = np.abs(mps_qpos - cpu_qpos)
    qvel_diff = np.abs(mps_qvel - cpu_qvel)
    qpos_peak = tuple(int(i) for i in np.unravel_index(np.argmax(qpos_diff), qpos_diff.shape))
    qvel_peak = tuple(int(i) for i in np.unravel_index(np.argmax(qvel_diff), qvel_diff.shape))
    qpos_final_error = float(np.max(qpos_diff[-1]))
    qvel_final_error = float(np.max(qvel_diff[-1]))
    finite = np.isfinite(qpos_diff).all() and np.isfinite(qvel_diff).all()
    mps_rate = args.worlds * args.steps / mps_elapsed
    cpu_rate = args.worlds * args.steps / cpu_elapsed
    removed = ",".join(sorted(UNSUPPORTED_PAIR_NAMES))

    print(
        f"model=Menagerie-G1-MPS-proxy worlds={args.worlds} control_steps={args.steps} "
        f"substeps={substeps} repeats={args.repeats} action_profile=sinusoidal(0.25)"
    )
    print(
        f"integrator={mujoco.mjtIntegrator(original_integrator)}->Euler "
        f"timestep={model.opt.timestep:g}s iterations={model.opt.iterations} contacts={model.npair}/49 "
        f"MPS-row-cap={engine.rowcap} CPU-njmax={model.njmax} CPU-nconmax={model.nconmax}"
    )
    print(f"removed_unsupported_self_pairs={removed}")
    print(f"Warp-MPS median envsteps/s={mps_rate:.1f} wall={mps_elapsed:.3f}s")
    print(
        f"Warp-CPU median envsteps/s={cpu_rate:.1f} wall={cpu_elapsed:.3f}s "
        f"speedup={cpu_elapsed / mps_elapsed:.2f}x"
    )
    print(
        f"finite={finite} MPS/CPU peak qpos={float(np.max(qpos_diff)):.3e}@{qpos_peak} "
        f"qvel={float(np.max(qvel_diff)):.3e}@{qvel_peak}; "
        f"final qpos={qpos_final_error:.3e} qvel={qvel_final_error:.3e}"
    )
    if not finite:
        raise SystemExit("Warp-MPS or Warp CPU produced non-finite states")


if __name__ == "__main__":
    main()
