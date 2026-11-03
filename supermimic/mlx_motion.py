"""50 Hz reference motions and GPU-resident goals for the MLX trainer."""

from __future__ import annotations

import glob
import os

import mlx.core as mx
import numpy as np

from .mlx_geometry import mat_to_6d, quat_to_mat, so3_log, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50


def _gradient(x: np.ndarray, dt: float) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    return np.gradient(x, dt, axis=0, edge_order=1).astype(np.float32, copy=False)


def _slerp_np(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    t = t[:, None]
    w0 = np.where(sin_theta < 1e-6, 1 - t, np.sin((1 - t) * theta) / np.maximum(sin_theta, 1e-12))
    w1 = np.where(sin_theta < 1e-6, t, np.sin(t * theta) / np.maximum(sin_theta, 1e-12))
    q = w0 * q0 + w1 * q1
    return (q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)).astype(np.float32)


class MotionLib:
    """Motion table using the original CSV, resampling, finite differences, and bin contract."""

    def __init__(self, files: list[str], kin, src_fps: float = 30.0, fk_chunk: int = 8192):
        if not files:
            raise FileNotFoundError("no motion CSV files were found")
        self.names = [os.path.splitext(os.path.basename(path))[0] for path in files]
        roots, quats, joints, lengths = [], [], [], []
        for path in files:
            raw = np.loadtxt(path, delimiter=",", dtype=np.float32)
            raw = np.atleast_2d(raw)
            if raw.shape[1] < 36 or raw.shape[0] < 2:
                raise ValueError(f"{path} must have at least two rows and 36 CSV columns")
            p_src, q_xyzw, q_src = raw[:, :3], raw[:, 3:7], raw[:, 7:36]
            t_src = np.arange(len(raw), dtype=np.float32)
            duration = (len(raw) - 1) / src_fps
            count = int(np.floor(duration / CONTROL_DT)) + 1
            times = np.arange(count, dtype=np.float32) * CONTROL_DT * src_fps
            i0 = np.minimum(np.floor(times).astype(np.int32), len(raw) - 2)
            alpha = np.clip(times - i0, 0, 1)
            p = p_src[i0] * (1 - alpha[:, None]) + p_src[i0 + 1] * alpha[:, None]
            q = q_src[i0] * (1 - alpha[:, None]) + q_src[i0 + 1] * alpha[:, None]
            quat_src = q_xyzw[:, [3, 0, 1, 2]]
            quat = _slerp_np(quat_src[i0], quat_src[i0 + 1], alpha)
            if not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(quat).all():
                raise ValueError(f"non-finite values in motion clip {path}")
            roots.append(p.astype(np.float32))
            quats.append(quat)
            joints.append(q.astype(np.float32))
            lengths.append(count)

        self.clip_len_np = np.asarray(lengths, np.int32)
        self.clip_start_np = np.cumsum(np.r_[0, self.clip_len_np[:-1]], dtype=np.int32)
        self.clip_end_np = self.clip_start_np + self.clip_len_np - 1
        self.num_frames = int(self.clip_len_np.sum())
        self._goal_tables: dict[int, mx.array] = {}
        self.clip_id_np = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(lengths)])
        self.clip_len = mx.array(self.clip_len_np)
        self.clip_start = mx.array(self.clip_start_np)
        self.clip_end = mx.array(self.clip_end_np)
        self.clip_id = mx.array(self.clip_id_np)

        p_np = np.concatenate(roots)
        quat_np = np.concatenate(quats)
        q_np = np.concatenate(joints)
        self.root_p = mx.array(p_np)
        self.root_R = quat_to_mat(mx.array(quat_np))
        self.q = mx.array(q_np)
        v_np = np.concatenate([_gradient(p, CONTROL_DT) for p in roots])
        qd_np = np.concatenate([_gradient(q, CONTROL_DT) for q in joints])
        self.root_v = mx.array(v_np)
        self.qd = mx.array(qd_np)

        ws = []
        offset = 0
        for length in lengths:
            r = self.root_R[offset : offset + length]
            first = so3_log(r[1] @ mx.swapaxes(r[0], -1, -2)) / CONTROL_DT
            last = so3_log(r[-1] @ mx.swapaxes(r[-2], -1, -2)) / CONTROL_DT
            if length > 2:
                mid = so3_log(r[2:] @ mx.swapaxes(r[:-2], -1, -2)) / (2 * CONTROL_DT)
                w = mx.concatenate((first[None], mid, last[None]), axis=0)
            else:
                w = mx.stack((first, last), axis=0)
            mx.eval(w)
            ws.append(w)
            offset += length
        self.root_w = mx.concatenate(ws, axis=0)

        body_p, body_R, body_v, body_w = [], [], [], []
        for start in range(0, self.num_frames, fk_chunk):
            end = min(start + fk_chunk, self.num_frames)
            out = kin.forward(
                self.root_p[start:end], self.root_R[start:end], self.root_v[start:end], self.root_w[start:end],
                self.q[start:end], self.qd[start:end],
            )
            mx.eval(*out)
            body_p.append(out[0])
            body_R.append(out[1])
            body_v.append(out[2])
            body_w.append(out[3])
        self.body_p, self.body_R = mx.concatenate(body_p), mx.concatenate(body_R)
        self.body_v, self.body_w = mx.concatenate(body_v), mx.concatenate(body_w)
        rt = mx.swapaxes(self.root_R, -1, -2)
        self.root_v_local = (rt @ self.root_v[..., None]).squeeze(-1)
        self.root_w_local = (rt @ self.root_w[..., None]).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]

        bin_clip, bin_start = [], []
        for clip, length in enumerate(lengths):
            for start in range(0, length, BIN_FRAMES):
                bin_clip.append(clip)
                bin_start.append(int(self.clip_start_np[clip]) + start)
        self.bin_clip_np = np.asarray(bin_clip, np.int32)
        self.bin_start_np = np.asarray(bin_start, np.int32)
        self.num_bins = len(bin_start)
        self.bin_clip = mx.array(self.bin_clip_np)
        self.bin_start = mx.array(self.bin_start_np)
        self.frame_bin_np = np.zeros(self.num_frames, np.int32)
        for b, (clip, start) in enumerate(zip(bin_clip, bin_start)):
            end = min(start + BIN_FRAMES, int(self.clip_end_np[clip]) + 1)
            self.frame_bin_np[start:end] = b
        self.frame_bin = mx.array(self.frame_bin_np)

    @classmethod
    def from_dir(cls, path: str, kin, src_fps: float = 30.0):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        return cls(files, kin, src_fps=src_fps)

    def clip_end_for_frame(self, frame: mx.array) -> mx.array:
        return self.clip_end[self.clip_id[frame]]

    def ref_state(self, frame: mx.array):
        return self.root_p[frame], self.root_R[frame], self.root_v[frame], self.root_w[frame], self.q[frame], self.qd[frame]

    def precompute_goals(
        self, horizon: int, max_bytes: int = 256 * 1024 * 1024
    ) -> mx.array | None:
        if horizon in self._goal_tables:
            return self._goal_tables[horizon]
        table_bytes = self.num_frames * (73 * horizon) * np.dtype(np.float32).itemsize
        if table_bytes > max_bytes:
            return None
        frames = mx.arange(self.num_frames, dtype=mx.int32)
        table = self._compute_goals(frames, horizon)
        mx.eval(table)
        self._goal_tables[horizon] = table
        return table

    def goal(self, frame: mx.array, horizon: int = 10) -> mx.array:
        frame = mx.array(frame, dtype=mx.int32)
        table = self._goal_tables.get(horizon)
        if table is not None:
            return table[frame]
        return self._compute_goals(frame, horizon)

    def _compute_goals(self, frame: mx.array, horizon: int) -> mx.array:
        end = self.clip_end_for_frame(frame)
        future = frame[..., None] + mx.arange(1, horizon + 1, dtype=mx.int32)
        recorded = future <= end[..., None]
        future = mx.minimum(future, end[..., None])
        p0, r0 = self.root_p[frame], self.root_R[frame]
        p1, r1 = self.root_p[future], self.root_R[future]
        v = mx.where(recorded[..., None], self.root_v_local[future], 0.0)
        w = mx.where(recorded[..., None], self.root_w_local[future], 0.0)
        qd = mx.where(recorded[..., None], self.qd[future], 0.0)
        h = yaw_mat(r0)
        ht = mx.swapaxes(h, -1, -2)[..., None, :, :]
        dp = (ht @ (p1 - p0[..., None, :])[..., None]).squeeze(-1)
        rot = mat_to_6d(ht @ r1)
        feats = mx.concatenate((dp, rot, v, w, self.q[future], qd * 0.1), axis=-1)
        return feats.reshape(frame.shape + (-1,))
