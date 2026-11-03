"""Differentiable MLX geometry and G1 forward kinematics."""

from __future__ import annotations

import mlx.core as mx

from .g1_contract import G1_JOINT_NAMES, TRACKED_BODIES


def quat_to_mat(q: mx.array) -> mx.array:
    q = q / mx.maximum(mx.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = (q[..., i] for i in range(4))
    return mx.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: mx.array) -> mx.array:
    """Convert rotation matrices to canonical wxyz quaternions."""
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    qw = mx.sqrt(mx.maximum(1 + m00 + m11 + m22, 1e-10)) / 2
    qx = mx.sqrt(mx.maximum(1 + m00 - m11 - m22, 1e-10)) / 2
    qy = mx.sqrt(mx.maximum(1 - m00 + m11 - m22, 1e-10)) / 2
    qz = mx.sqrt(mx.maximum(1 - m00 - m11 + m22, 1e-10)) / 2
    cands = mx.stack(
        (
            mx.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
            mx.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
            mx.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
            mx.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1),
        ),
        axis=-2,
    )
    idx = mx.stop_gradient(mx.argmax(mx.stack((qw, qx, qy, qz), -1), axis=-1))
    q = mx.squeeze(mx.take_along_axis(cands, idx[..., None, None], axis=-2), axis=-2)
    q = q / mx.maximum(mx.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    return q * mx.where(q[..., :1] < 0, -1.0, 1.0)


def cross(a: mx.array, b: mx.array) -> mx.array:
    return mx.stack(
        (a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
         a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
         a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]),
        axis=-1,
    )


def skew(v: mx.array) -> mx.array:
    x, y, z = (v[..., i] for i in range(3))
    o = mx.zeros_like(x)
    return mx.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(v.shape[:-1] + (3, 3))


def axis_angle_to_mat(axis: mx.array, angle: mx.array) -> mx.array:
    c, s = mx.cos(angle), mx.sin(angle)
    k = skew(axis)
    eye = mx.broadcast_to(mx.eye(3), angle.shape + (3, 3))
    return eye + s[..., None, None] * k + (1 - c)[..., None, None] * (k @ k)


def so3_exp(v: mx.array) -> mx.array:
    theta2 = mx.sum(v * v, axis=-1, keepdims=True)
    theta = mx.sqrt(theta2 + 1e-12)
    a = mx.where(theta2 < 1e-8, 1 - theta2 / 6, mx.sin(theta) / theta)
    b = mx.where(theta2 < 1e-8, 0.5 - theta2 / 24, (1 - mx.cos(theta)) / (theta2 + 1e-12))
    k = skew(v)
    eye = mx.broadcast_to(mx.eye(3), k.shape)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def quat_to_rotvec(q: mx.array) -> mx.array:
    q = q * mx.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half_sq = mx.sum(v * v, axis=-1, keepdims=True)
    sin_half = mx.sqrt(mx.maximum(sin_half_sq, 1e-12))
    half = mx.arctan2(sin_half, q[..., :1])
    regular_scale = 2 * half / sin_half
    small_scale = 2.0 / mx.maximum(q[..., :1], 1e-6)
    scale = mx.where(sin_half_sq < 1e-12, small_scale, regular_scale)
    return v * scale


def so3_log(r: mx.array) -> mx.array:
    return quat_to_rotvec(mat_to_quat(r))


def rot_angle_sq(ra: mx.array, rb: mx.array) -> mx.array:
    tr = mx.sum(ra * rb, axis=(-1, -2))
    c = mx.clip((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return mx.arccos(c) ** 2


def yaw_mat(r: mx.array) -> mx.array:
    fx, fy = r[..., 0, 0], r[..., 1, 0]
    rx, ry = r[..., 0, 1], r[..., 1, 1]
    forward_sq = fx * fx + fy * fy
    right_sq = rx * rx + ry * ry
    use_right_axis = mx.stop_gradient(forward_sq < 1e-4)
    forward_norm = mx.sqrt(mx.maximum(forward_sq, 1e-4))
    right_norm = mx.sqrt(mx.maximum(right_sq, 1e-4))
    c = mx.where(use_right_axis, ry / right_norm, fx / forward_norm)
    s = mx.where(use_right_axis, -rx / right_norm, fy / forward_norm)
    o, one = mx.zeros_like(c), mx.ones_like(c)
    return mx.stack((c, -s, o, s, c, o, o, o, one), -1).reshape(r.shape[:-2] + (3, 3))


def mat_to_6d(r: mx.array) -> mx.array:
    return r[..., :, :2].reshape(r.shape[:-2] + (6,))


def quat_slerp(q0: mx.array, q1: mx.array, t: mx.array) -> mx.array:
    dot = mx.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = mx.where(dot < 0, -q1, q1)
    dot = mx.clip(mx.abs(dot), 0, 1)
    theta = mx.arccos(dot)
    sin_theta = mx.sin(theta)
    t = t[..., None]
    w0 = mx.where(sin_theta < 1e-6, 1 - t, mx.sin((1 - t) * theta) / (sin_theta + 1e-12))
    w1 = mx.where(sin_theta < 1e-6, t, mx.sin(t * theta) / (sin_theta + 1e-12))
    q = w0 * q0 + w1 * q1
    return q / mx.maximum(mx.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


class G1Kinematics:
    """MLX forward kinematics from the MJCF shared with the MPS simulator."""

    def __init__(self, xml_path: str, tracked: tuple[str, ...] = TRACKED_BODIES):
        import mujoco
        import numpy as np

        model = mujoco.MjModel.from_xml_path(xml_path)
        joint_index = {name: i for i, name in enumerate(G1_JOINT_NAMES)}
        bodies, parents, joints, axes, positions, quaternions = [], [], [], [], [], []
        body_to_local = {}
        for body in range(1, model.nbody):
            if model.body_jntnum[body] == 0:
                continue
            name = model.body(body).name
            jid = int(model.body_jntadr[body])
            body_to_local[body] = len(bodies)
            bodies.append(name)
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                parents.append(-1)
                joints.append(-1)
            else:
                parent = int(model.body_parentid[body])
                if parent not in body_to_local:
                    raise RuntimeError(f"fixed parent for moving body {name}")
                parents.append(body_to_local[parent])
                joints.append(joint_index[model.joint(jid).name])
            axes.append(model.jnt_axis[jid].copy())
            positions.append(model.body_pos[body].copy())
            quaternions.append(model.body_quat[body].copy())

        missing = sorted(set(tracked) - set(bodies))
        if missing:
            raise ValueError(f"tracked bodies missing from G1 MJCF: {missing}")
        self.body_names = bodies
        self.parent = parents
        self.joint_idx = joints
        self.axes = mx.array(np.asarray(axes, np.float32))
        self.off_p = mx.array(np.asarray(positions, np.float32))
        self.off_R = quat_to_mat(mx.array(np.asarray(quaternions, np.float32)))
        self.tracked_idx = [bodies.index(name) for name in tracked]
        self.num_bodies = len(bodies)
        lo, hi = np.zeros(29, np.float32), np.zeros(29, np.float32)
        for jid in range(model.njnt):
            name = model.joint(jid).name
            if name in joint_index:
                lo[joint_index[name]], hi[joint_index[name]] = model.jnt_range[jid]
        self.q_lo, self.q_hi = mx.array(lo), mx.array(hi)

    def forward(self, p, r, v, w, q, qd, all_bodies: bool = False):
        positions = [None] * self.num_bodies
        rotations = [None] * self.num_bodies
        linear = [None] * self.num_bodies
        angular = [None] * self.num_bodies
        for i in range(self.num_bodies):
            parent = self.parent[i]
            if parent < 0:
                positions[i], rotations[i], linear[i], angular[i] = p, r, v, w
                continue
            j = self.joint_idx[i]
            rp = rotations[parent]
            pi = positions[parent] + (rp @ self.off_p[i][..., None]).squeeze(-1)
            roff = rp @ self.off_R[i]
            ri = roff @ axis_angle_to_mat(self.axes[i], q[..., j])
            axis_w = (roff @ self.axes[i][..., None]).squeeze(-1)
            wi = angular[parent] + axis_w * qd[..., j : j + 1]
            vi = linear[parent] + cross(angular[parent], pi - positions[parent])
            positions[i], rotations[i], linear[i], angular[i] = pi, ri, vi, wi
        indices = range(self.num_bodies) if all_bodies else self.tracked_idx
        return tuple(
            mx.stack([values[i] for i in indices], axis=-3 if k == 1 else -2)
            for k, values in enumerate((positions, rotations, linear, angular))
        )
