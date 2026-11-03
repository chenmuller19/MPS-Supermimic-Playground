"""MPS batched G1 environment with MLX-resident state and reference resets."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from .g1_contract import G1_JOINT_NAMES, action_scale, default_joint_pos_list
from .mlx_geometry import mat_to_quat, quat_to_mat
from .mlx_mdp import MAX_STEP_REWARD, flatten_state

CONTROL_DT = 0.02


def _uniform(shape, low: float, high: float) -> mx.array:
    return mx.random.uniform(shape=shape, low=low, high=high, dtype=mx.float32)


def _put_rows(array: mx.array, rows: np.ndarray, values: mx.array) -> mx.array:
    indices = mx.array(rows, dtype=mx.int32)
    if array.ndim > 1:
        indices = mx.broadcast_to(indices[:, None], values.shape)
    return mx.put_along_axis(array, indices, values, axis=0)


class MPSVectorEnv:
    """Host-orchestrated MPS worlds; simulation state never passes through NumPy per step."""

    def __init__(
        self,
        xml_path: str | Path,
        num_envs: int,
        motion,
        mdp,
        gamma: float = 0.99,
        max_episode_steps: int = 500,
        substeps: int | None = None,
        seed: int | None = None,
    ):
        import mujoco
        from mujoco_warp.mps.robot import BatchedEngine
        from .mps_g1 import prepare_mps_vector_env_model

        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.N, self.motion, self.mdp = num_envs, motion, mdp
        self.rng = np.random.default_rng(seed)
        self.gamma = float(gamma)
        self.max_episode_steps = int(max_episode_steps)
        self.model, self.info = prepare_mps_vector_env_model(xml_path)
        if self.info.original_integrator != "implicitfast":
            raise ValueError(f"expected implicitfast integrator, got {self.info.original_integrator}")
        if substeps is None:
            exact_substeps = CONTROL_DT / self.info.timestep
            substeps = int(round(exact_substeps))
            if not np.isclose(exact_substeps, substeps, rtol=0.0, atol=1e-8):
                raise ValueError("physics timestep must divide the 20 ms control period")
        if substeps < 1 or not np.isclose(self.info.timestep * substeps, CONTROL_DT, rtol=0.0, atol=1e-8):
            raise ValueError("MPS substeps do not match the 20 ms control period")
        self.substeps = int(substeps)

        self.floor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, self.info.floor_geom_name
        )
        if self.floor_id < 0:
            raise ValueError(f"G1 MJCF is missing floor geom {self.info.floor_geom_name!r}")
        self.cylinder_ids = np.asarray(self.info.disabled_cylinder_ids, np.int32)
        if len(self.cylinder_ids):
            old_half_length = self.model.geom_size[self.cylinder_ids, 1].copy()
            radius = self.model.geom_size[self.cylinder_ids, 0]
            self.model.geom_size[self.cylinder_ids, 1] = np.maximum(old_half_length - radius, 0.0)
            self.model.geom_type[self.cylinder_ids] = mujoco.mjtGeom.mjGEOM_CAPSULE
            self.model.geom_conaffinity[self.cylinder_ids] = 1
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))

        contact_row_cap = 1024 if self.info.collision_mode == "explicit_pairs" else 96
        self.engine = BatchedEngine(
            self.model, nworld=num_envs, contact_row_cap=contact_row_cap
        )
        self.contact_row_cap = int(self.engine.rig.contact_row_cap)
        self.default_q = mx.array(default_joint_pos_list(), dtype=mx.float32)
        self.action_scales = mx.array([action_scale(name) for name in G1_JOINT_NAMES], dtype=mx.float32)
        self.torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.torso_id < 0:
            raise ValueError("G1 MJCF is missing torso_link")

        self.friction_np = self.rng.uniform(0.3, 1.2, size=num_envs).astype(np.float32)
        self.com_np = np.stack(
            (
                self.rng.uniform(-0.025, 0.025, num_envs),
                self.rng.uniform(-0.05, 0.05, num_envs),
                self.rng.uniform(-0.05, 0.05, num_envs),
            ),
            axis=-1,
        ).astype(np.float32)
        self.xi = mx.concatenate(
            (mx.array(self.friction_np)[:, None], mx.zeros((num_envs, 29), dtype=mx.float32), mx.array(self.com_np)),
            axis=-1,
        )
        self.a_prev = mx.zeros((num_envs, 29), dtype=mx.float32)
        self.ep_ret = mx.zeros((num_envs,), dtype=mx.float32)
        self.ep_disc_ret = mx.zeros((num_envs,), dtype=mx.float32)
        self.f = np.zeros(num_envs, np.int32)
        self.ep_step = np.zeros(num_envs, np.int32)
        self.episode_id = np.full(num_envs, -1, np.int64)
        self.start_bin = np.zeros(num_envs, np.int32)
        self.bin_G = np.zeros(motion.num_bins, np.float32)
        self.bin_w_avg = np.full(motion.num_bins, 1.0 / motion.num_bins, np.float32)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self._base_pair_friction = self.engine.rig.pair_friction.copy()
        self._pair_floor = np.flatnonzero(
            (self.engine.rig.pair_g1 == self.floor_id) | (self.engine.rig.pair_g2 == self.floor_id)
        )
        self._base_body_ipos = self.engine.rig.body_ipos.copy()
        self.reset(np.arange(num_envs, dtype=np.int32))

    def _sample_frames(self, count: int):
        scores = -25.0 * self.bin_G / self.G_max
        soft = np.exp(scores - np.max(scores))
        soft /= soft.sum()
        self.bin_w_avg = 0.999 * self.bin_w_avg + 0.001 * soft
        probs = 0.5 / self.motion.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = self.rng.choice(self.motion.num_bins, count, p=probs).astype(np.int32)
        starts = self.motion.bin_start_np[bins]
        ends = np.minimum(starts + 50, self.motion.clip_end_np[self.motion.bin_clip_np[bins]])
        frames = starts + (self.rng.random(count) * (ends - starts)).astype(np.int32)
        return frames, bins

    def _apply_world_randomization(self, ids: np.ndarray) -> None:
        if not len(ids):
            return
        if len(self._pair_floor):
            friction = np.broadcast_to(self._base_pair_friction, (len(ids),) + self._base_pair_friction.shape).copy()
            friction[:, self._pair_floor, :2] = self.friction_np[ids, None, None]
            self.engine.set_world_tables(ids, pair_friction=friction)
        body_ipos = np.broadcast_to(self._base_body_ipos, (len(ids),) + self._base_body_ipos.shape).copy()
        body_ipos[:, self.torso_id] += self.com_np[ids]
        self.engine.set_world_tables(ids, body_ipos=body_ipos)

    def reset(self, env_ids: np.ndarray, frames: np.ndarray | None = None) -> None:
        ids = np.asarray(env_ids, np.int32).reshape(-1)
        if not len(ids):
            return
        if frames is None:
            frames, bins = self._sample_frames(len(ids))
        else:
            frames = np.asarray(frames, np.int32)
            bins = self.motion.frame_bin_np[frames]
        self._apply_world_randomization(ids)
        fm = mx.array(frames, dtype=mx.int32)
        p, r, v, w, q, qd = self.motion.ref_state(fm)
        n = len(ids)
        p = p + mx.stack((_uniform((n,), -0.05, 0.05), _uniform((n,), -0.05, 0.05), _uniform((n,), -0.01, 0.01)), -1)
        rv = mx.stack((_uniform((n,), -0.1, 0.1), _uniform((n,), -0.1, 0.1), _uniform((n,), -0.2, 0.2)), -1)
        from .mlx_geometry import so3_exp

        r = so3_exp(rv) @ r
        v = v + mx.stack((_uniform((n,), -0.5, 0.5), _uniform((n,), -0.5, 0.5), _uniform((n,), -0.2, 0.2)), -1)
        w = w + mx.stack((_uniform((n,), -0.52, 0.52), _uniform((n,), -0.52, 0.52), _uniform((n,), -0.78, 0.78)), -1)
        q = mx.clip(q + _uniform((n, 29), -0.1, 0.1), self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(r)
        qpos = mx.zeros((self.N, self.model.nq), dtype=mx.float32)
        qvel = mx.zeros((self.N, self.model.nv), dtype=mx.float32)
        ctrl = mx.zeros((self.N, self.model.nu), dtype=mx.float32)
        root_vel_local = (mx.swapaxes(r, -1, -2) @ w[..., None]).squeeze(-1)
        local_qpos = mx.concatenate((p, quat, q), axis=-1)
        local_qvel = mx.concatenate((v, root_vel_local, qd), axis=-1)
        qpos = _put_rows(qpos, ids, local_qpos)
        qvel = _put_rows(qvel, ids, local_qvel)
        ctrl = _put_rows(ctrl, ids, q)
        mask = np.zeros(self.N, np.bool_)
        mask[ids] = True
        self.engine.reset_worlds(mx.array(mask), qpos=qpos, qvel=qvel, ctrl=ctrl)

        offsets = _uniform((n, 29), -0.01, 0.01)
        xi_rows = self.xi[mx.array(ids)]
        xi_rows = mx.concatenate((xi_rows[:, :1], offsets, xi_rows[:, 30:]), axis=-1)
        self.xi = _put_rows(self.xi, ids, xi_rows)
        self.a_prev = _put_rows(self.a_prev, ids, mx.zeros((n, 29), dtype=mx.float32))
        self.ep_ret = _put_rows(self.ep_ret, ids, mx.zeros((n,), dtype=mx.float32))
        self.ep_disc_ret = _put_rows(self.ep_disc_ret, ids, mx.zeros((n,), dtype=mx.float32))
        self.f[ids] = frames
        self.ep_step[ids] = 0
        self.episode_id[ids] += 1
        self.start_bin[ids] = bins

    def state(self):
        qpos, qvel = self.engine.qpos, self.engine.qvel
        p = qpos[:, :3]
        r = quat_to_mat(qpos[:, 3:7])
        v = qvel[:, :3]
        w = (r @ qvel[:, 3:6, None]).squeeze(-1)
        return p, r, v, w, qpos[:, 7:36], qvel[:, 6:35]

    def step(self, action: mx.array) -> dict:
        if action.shape != (self.N, 29):
            raise ValueError(f"expected action shape {(self.N, 29)}, got {action.shape}")
        s = self.state()
        frames = self.f.copy()
        ep_steps = self.ep_step.copy()
        episodes = self.episode_id.copy()
        xi, a_prev = self.xi, self.a_prev
        target = self.default_q + xi[:, 1:30] + self.action_scales * mx.clip(action, -20.0, 20.0)
        for substep in range(self.substeps):
            self.engine.step(ctrl=target if substep == 0 else None)
        s1 = self.state()
        next_frames = frames + 1
        fm = mx.array(next_frames, dtype=mx.int32)
        reward = self.mdp.reward(s1, fm, action, a_prev)
        term = self.mdp.termination(s1, fm)
        clip_end = self.motion.clip_end_np[self.motion.clip_id_np[frames]]
        timeout = (next_frames >= clip_end) | (ep_steps + 1 >= self.max_episode_steps)
        done = term | mx.array(timeout)
        self.ep_disc_ret = self.ep_disc_ret + (self.gamma ** mx.array(ep_steps, dtype=mx.float32)) * reward
        self.ep_ret = self.ep_ret + reward
        mx.eval(self.ep_disc_ret, self.ep_ret)
        self.f = next_frames
        self.ep_step = ep_steps + 1
        self.a_prev = action
        mx.eval(reward, term, done, *s1)
        done_np = np.asarray(done, dtype=np.bool_)
        term_np = np.asarray(term, dtype=np.bool_)

        transition = {
            "s": flatten_state(s),
            "s1": flatten_state(s1),
            "a": action,
            "a_prev": a_prev,
            "xi": xi,
            "r": reward,
            "f": frames,
            "clip_end": clip_end,
            "ep_step": ep_steps,
            "done": done_np,
            "episode": episodes,
        }
        ids = np.flatnonzero(done_np)
        if len(ids):
            mx.eval(self.ep_disc_ret, self.ep_ret)
            disc_ret = np.asarray(self.ep_disc_ret)[ids]
            total_ret = np.asarray(self.ep_ret)[ids]
            length = self.ep_step[ids].astype(np.float32)
            boot = np.where(term_np[ids], 0.0, (self.gamma**length) * (total_ret / length) / (1 - self.gamma))
            returns = disc_ret + boot
            bins = self.start_bin[ids]
            for b, value in zip(bins, returns):
                self.bin_G[b] = 0.8 * self.bin_G[b] + 0.2 * value
            transition["episode_len"] = length
            transition["episode_ret_norm"] = total_ret / (length * MAX_STEP_REWARD)
            transition["term"] = term_np
            self.reset(ids)
        return transition
