"""Compare origin-G1 MPS physics against Warp CPU on the same adapted model."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from supermimic.mps_g1 import MPSG1, prepare_origin_reference_model


DEFAULT_XML = Path(__file__).resolve().parents[1] / "data/g1_description/g1_29dof.xml"


def _set_cpu_state(data, qpos, qvel, ctrl):
    data.qpos.numpy()[:] = qpos
    data.qvel.numpy()[:] = qvel
    data.ctrl.numpy()[:] = ctrl
    data.qacc_warmstart.numpy()[:] = 0


def make_action_sequence(num_envs, steps, num_actions, control_dt):
    t = np.arange(steps, dtype=np.float32)[:, None, None] * control_dt
    env_phase = np.arange(num_envs, dtype=np.float32)[None, :, None] * 0.013
    joint_phase = np.arange(num_actions, dtype=np.float32)[None, None, :] * 0.17
    return (0.25 * np.sin(2.0 * np.pi * t + env_phase + joint_phase)).astype(np.float32)


def run_cpu_reference(model, qpos, qvel, target_sequence, substeps, warmup_steps, repeats):
    import mujoco_warp
    import warp as wp

    nworld = qpos.shape[0]
    warp_model = mujoco_warp.put_model(model)
    warm_data = mujoco_warp.make_data(model, nworld=nworld)
    _set_cpu_state(warm_data, qpos, qvel, target_sequence[0])
    for step in range(warmup_steps):
        warm_data.ctrl.numpy()[:] = target_sequence[step % len(target_sequence)]
        for _ in range(substeps):
            mujoco_warp.step(warp_model, warm_data)
        wp.synchronize()

    elapsed_runs = []
    qpos_out = qvel_out = qpos_history = qvel_history = None
    for _ in range(repeats):
        data = mujoco_warp.make_data(model, nworld=nworld)
        _set_cpu_state(data, qpos, qvel, target_sequence[0])
        qpos_samples = []
        qvel_samples = []
        start = time.perf_counter()
        for targets in target_sequence:
            data.ctrl.numpy()[:] = targets
            for _ in range(substeps):
                mujoco_warp.step(warp_model, data)
            wp.synchronize()
            qpos_out = np.array(data.qpos.numpy())
            qvel_out = np.array(data.qvel.numpy())
            qpos_samples.append(qpos_out)
            qvel_samples.append(qvel_out)
        elapsed_runs.append(time.perf_counter() - start)
        qpos_history = np.stack(qpos_samples)
        qvel_history = np.stack(qvel_samples)
    return float(np.median(elapsed_runs)), qpos_out, qvel_out, qpos_history, qvel_history


def compare_origin_integrator(xml_path, substeps, action_sequence):
    import mujoco

    model, _ = prepare_origin_reference_model(xml_path)
    reference = mujoco.MjData(model)
    probe = MPSG1(xml_path, num_envs=1, substeps=substeps)
    reference.qpos[:] = probe.default_qpos[0]
    reference.qvel[:] = probe.default_qvel[0]
    peak_qpos_error = 0.0
    peak_qvel_error = 0.0

    for actions in action_sequence:
        reference.ctrl[:] = probe.default_joint_pos + probe.action_scales * actions[0]
        for _ in range(substeps):
            mujoco.mj_step(model, reference)
        mps_qpos, mps_qvel = probe.step_action(actions)
        peak_qpos_error = max(peak_qpos_error, float(np.max(np.abs(mps_qpos[0] - reference.qpos))))
        peak_qvel_error = max(peak_qvel_error, float(np.max(np.abs(mps_qvel[0] - reference.qvel))))

    final_qpos_error = float(np.max(np.abs(mps_qpos[0] - reference.qpos)))
    final_qvel_error = float(np.max(np.abs(mps_qvel[0] - reference.qvel)))
    return final_qpos_error, final_qvel_error, peak_qpos_error, peak_qvel_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--steps", type=int, default=50, help="20 ms control transitions")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.worlds < 1 or args.steps < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("worlds, steps, and repeats must be positive; warmup cannot be negative")

    sim = MPSG1(args.xml, args.worlds)
    control_dt = sim.substeps * sim.info.timestep
    actions = make_action_sequence(args.worlds, args.steps, sim.model.nu, control_dt)
    targets = sim.default_joint_pos[None, None, :] + sim.action_scales[None, None, :] * actions
    qpos, qvel = sim.default_qpos, sim.default_qvel

    for step in range(args.warmup):
        sim.step_action(actions[step % args.steps])
    mps_runs = []
    mps_qpos_history = mps_qvel_history = None
    for _ in range(args.repeats):
        sim.reset()
        qpos_samples = []
        qvel_samples = []
        start = time.perf_counter()
        for action in actions:
            mps_qpos, mps_qvel = sim.step_action(action)
            qpos_samples.append(mps_qpos)
            qvel_samples.append(mps_qvel)
        mps_runs.append(time.perf_counter() - start)
        mps_qpos_history = np.stack(qpos_samples)
        mps_qvel_history = np.stack(qvel_samples)
    mps_elapsed = float(np.median(mps_runs))

    cpu_elapsed, cpu_qpos, cpu_qvel, cpu_qpos_history, cpu_qvel_history = run_cpu_reference(
        sim.model, qpos, qvel, targets, sim.substeps, args.warmup, args.repeats
    )
    position_diff = np.abs(mps_qpos_history - cpu_qpos_history)
    velocity_diff = np.abs(mps_qvel_history - cpu_qvel_history)
    position_error = float(np.max(position_diff))
    velocity_error = float(np.max(velocity_diff))
    position_peak = tuple(int(i) for i in np.unravel_index(np.argmax(position_diff), position_diff.shape))
    velocity_peak = tuple(int(i) for i in np.unravel_index(np.argmax(velocity_diff), velocity_diff.shape))
    final_position_error = float(np.max(np.abs(mps_qpos - cpu_qpos)))
    final_velocity_error = float(np.max(np.abs(mps_qvel - cpu_qvel)))
    origin_errors = compare_origin_integrator(args.xml, sim.substeps, actions[:, :1, :])
    finite = all(
        np.isfinite(x).all()
        for x in (mps_qpos_history, mps_qvel_history, cpu_qpos_history, cpu_qvel_history)
    )
    finite = finite and np.isfinite(origin_errors).all()

    mps_rate = args.worlds * args.steps / mps_elapsed
    cpu_rate = args.worlds * args.steps / cpu_elapsed
    print(
        f"model=origin-g1 worlds={args.worlds} control_steps={args.steps} "
        f"substeps={sim.substeps} repeats={args.repeats} action_profile=sinusoidal(0.25)"
    )
    print(
        f"integrator={sim.info.original_integrator}->Euler timestep={sim.info.timestep:g}s "
        f"armature_dofs={sim.info.restored_armature_dofs}"
    )
    print(
        f"contacts=ground+{len(sim.info.foot_sphere_ids)} foot spheres; "
        f"disabled cylinder geoms={list(sim.info.disabled_cylinder_ids)}"
    )
    print(f"MPS median envsteps/s={mps_rate:.1f} wall={mps_elapsed:.3f}s")
    print(
        f"Warp-CPU median envsteps/s={cpu_rate:.1f} wall={cpu_elapsed:.3f}s "
        f"speedup={cpu_elapsed / mps_elapsed:.2f}x"
    )
    print(
        f"finite={finite} MPS/Warp peak error qpos={position_error:.3e}@{position_peak} "
        f"qvel={velocity_error:.3e}@{velocity_peak}; "
        f"final qpos={final_position_error:.3e} qvel={final_velocity_error:.3e}"
    )
    print("MPS/Warp final-state tolerances: qpos<=1e-3, qvel<=5e-2; peak errors are diagnostic")
    print(
        "Euler vs implicitfast diagnostic only (same filtered model, synthetic actions, 1 world): "
        f"final qpos={origin_errors[0]:.3e} qvel={origin_errors[1]:.3e}; "
        f"peak qpos={origin_errors[2]:.3e} qvel={origin_errors[3]:.3e}"
    )
    if not finite or final_position_error > 1e-3 or final_velocity_error > 5e-2:
        raise SystemExit("MPS/Warp final-state validation failed")


if __name__ == "__main__":
    main()
