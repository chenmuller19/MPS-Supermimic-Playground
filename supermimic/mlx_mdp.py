"""MLX state, observation, reward, and termination heads for SuperMimic."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .mlx_geometry import mat_to_6d, mat_to_quat, quat_to_mat, rot_angle_sq, so3_exp, so3_log, yaw_mat

STATE_DIM = 71
OBS_DIM = 99
XI_DIM = 33
NUM_J = 29
INV_DIM = 1 + 6 + 3 + 3 + 29 + 29
CRITIC_EXTRA = 29 + 14 * 3 + 3 + 6 + XI_DIM
CONTROL_DT = 0.02
MAX_STEP_REWARD = 5 * CONTROL_DT

OBS_NOISE = mx.array(
    [0.01] * 29 + [0.5] * 29 + [0.2] * 3 + [0.05] * 3 + [0.0] * 29 + [0.05] * 6,
    dtype=mx.float32,
)
OBS_SCALE = mx.array([1.0] * 29 + [0.1] * 29 + [0.25] * 3 + [1.0] * 3 + [1.0] * 29 + [1.0] * 6)


@dataclass
class RewardCfg:
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    term_height: float = 0.25
    term_tilt: float = 0.8


def flatten_state(state) -> mx.array:
    p, r, v, w, q, qd = state
    return mx.concatenate((p, mat_to_quat(r), v, w, q, qd), axis=-1)


def unflatten_state(x: mx.array):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:42], x[..., 42:71]


def invariant_features(state) -> mx.array:
    p, r, v, w, q, qd = state
    h = yaw_mat(r)
    rt = mx.swapaxes(r, -1, -2)
    return mx.concatenate(
        (p[..., 2:3], mat_to_6d(mx.swapaxes(h, -1, -2) @ r), (rt @ v[..., None]).squeeze(-1),
         (rt @ w[..., None]).squeeze(-1), q, qd),
        axis=-1,
    )


def normalize_xi(xi: mx.array) -> mx.array:
    return mx.concatenate(((xi[..., :1] - 0.75) / 0.45, xi[..., 1:30] / 0.01, xi[..., 30:33] / 0.05), axis=-1)


class MDP:
    def __init__(self, kin, motion, default_q: mx.array, cfg: RewardCfg | None = None):
        self.kin, self.motion = kin, motion
        self.cfg = cfg or RewardCfg()
        self.default_q = mx.array(default_q, dtype=mx.float32)
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo = mid - half * self.cfg.soft_limit_factor
        self.soft_hi = mid + half * self.cfg.soft_limit_factor
        self.obs_noise = OBS_NOISE
        self.obs_scale = OBS_SCALE

    def obs(self, state, a_prev: mx.array, xi: mx.array, frame: mx.array) -> mx.array:
        p, r, v, w, q, qd = state
        rt = mx.swapaxes(r, -1, -2)
        dq0 = xi[..., 1:30]
        gravity_b = -r[..., 2, :]
        rel = mat_to_6d(rt @ self.motion.root_R[frame])
        return mx.concatenate(
            (q - self.default_q - dq0, qd, (rt @ w[..., None]).squeeze(-1), gravity_b, a_prev, rel),
            axis=-1,
        )

    def add_noise(self, obs: mx.array) -> mx.array:
        noise = mx.random.uniform(shape=obs.shape, low=-1.0, high=1.0, dtype=obs.dtype)
        return obs + noise * self.obs_noise

    def reward(
        self,
        state,
        frame: mx.array,
        action: mx.array,
        a_prev: mx.array,
        terms: bool = False,
        return_body_positions: bool = False,
    ):
        cfg, motion = self.cfg, self.motion
        p, r, v, w, q, _ = state
        bp, br, bv, bw = self.kin.forward(*state)
        rp, rr = motion.root_p[frame], motion.root_R[frame]
        d_r = yaw_mat(r @ mx.swapaxes(rr, -1, -2))
        anchor = mx.concatenate((p[..., :2], rp[..., 2:3]), axis=-1)
        ref_bp = anchor[..., None, :] + (d_r[..., None, :, :] @ (motion.body_p[frame] - rp[..., None, :])[..., None]).squeeze(-1)
        ref_br = d_r[..., None, :, :] @ motion.body_R[frame]
        e_pos = mx.mean(mx.sum((ref_bp - bp) ** 2, axis=-1), axis=-1)
        e_ori = mx.mean(rot_angle_sq(ref_br, br), axis=-1)
        e_lv = mx.mean(mx.sum((motion.body_v[frame] - bv) ** 2, axis=-1), axis=-1)
        e_av = mx.mean(mx.sum((motion.body_w[frame] - bw) ** 2, axis=-1), axis=-1)
        e_rp = mx.sum((rp - p) ** 2, axis=-1)
        e_ro = rot_angle_sq(rr, r)
        tracking = (
            mx.exp(-e_pos / cfg.body_pos_std**2)
            + mx.exp(-e_ori / cfg.body_ori_std**2)
            + mx.exp(-e_lv / cfg.body_linvel_std**2)
            + mx.exp(-e_av / cfg.body_angvel_std**2)
            + 0.5 * mx.exp(-e_rp / cfg.root_pos_std**2)
            + 0.5 * mx.exp(-e_ro / cfg.root_ori_std**2)
        )
        action_rate = mx.sum((action - a_prev) ** 2, axis=-1)
        joint_limit = mx.sum(mx.maximum(q - self.soft_hi, 0) + mx.maximum(self.soft_lo - q, 0), axis=-1)
        reward = (tracking + cfg.w_action_rate * action_rate + cfg.w_joint_limit * joint_limit) * CONTROL_DT
        if terms:
            return reward, {
                "track": tracking * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": action_rate,
                "joint_limit": joint_limit,
            }
        if return_body_positions:
            return reward, bp
        return reward

    def termination(self, state, frame: mx.array) -> mx.array:
        p, r = state[:2]
        rp, rr = self.motion.root_p[frame], self.motion.root_R[frame]
        e_z = mx.abs(rp[..., 2] - p[..., 2])
        e_tilt = mx.abs(rr[..., 2, 2] - r[..., 2, 2])
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(self, state, frame: mx.array, tz: float = 0.02, tt: float = 0.04) -> mx.array:
        p, r = state[:2]
        rp, rr = self.motion.root_p[frame], self.motion.root_R[frame]
        e_z = mx.abs(rp[..., 2] - p[..., 2])
        e_tilt = mx.abs(rr[..., 2, 2] - r[..., 2, 2])
        alive = (1 - mx.sigmoid((e_z - self.cfg.term_height) / tz)) * (1 - mx.sigmoid((e_tilt - self.cfg.term_tilt) / tt))
        return 1 - alive

    def critic_features(
        self,
        state,
        frame: mx.array,
        xi: mx.array,
        body_positions: mx.array | None = None,
    ) -> mx.array:
        motion = self.motion
        p, r, v, w, q, qd = state
        h = yaw_mat(r)
        ht = mx.swapaxes(h, -1, -2)
        rp, rr = motion.root_p[frame], motion.root_R[frame]
        bp = self.kin.forward(*state)[0] if body_positions is None else body_positions
        d_r = yaw_mat(r @ mx.swapaxes(rr, -1, -2))
        anchor = mx.concatenate((p[..., :2], rp[..., 2:3]), axis=-1)
        ref_bp = anchor[..., None, :] + (d_r[..., None, :, :] @ (motion.body_p[frame] - rp[..., None, :])[..., None]).squeeze(-1)
        e_body = (ht[..., None, :, :] @ (ref_bp - bp)[..., None]).squeeze(-1).reshape(p.shape[:-1] + (-1,))
        e_root = (ht @ (rp - p)[..., None]).squeeze(-1)
        rel = mat_to_6d(mx.swapaxes(r, -1, -2) @ rr)
        return mx.concatenate((invariant_features(state), motion.q[frame] - q, e_body, e_root, rel, xi), axis=-1)
