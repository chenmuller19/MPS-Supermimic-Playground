"""SuperMimic transition adapter for UniLab motion-tracking environments.

The adapter consumes the Manager-Based motion command's public tensor carriers.  It keeps
MuJoCo/UniLab as the simulator owner while translating one step into the physical state,
reference frame, context, and future-goal contract used by SuperMimic.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .mdp import NUM_J, STATE_DIM, XI_DIM, flatten_state
from .robot import G1_JOINT_NAMES, action_scale as supermimic_action_scale
from .rotations import mat_to_6d, quat_to_mat, yaw_mat


@dataclass
class UniLabTransition:
    state: tuple[torch.Tensor, ...]
    next_state: tuple[torch.Tensor, ...]
    action: torch.Tensor
    action_prev: torch.Tensor
    xi: torch.Tensor
    frame: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor


class UniLabMotionAdapter:
    """Translate a G1 MotionCommand into the SuperMimic tensor contract."""

    def __init__(self, env, command_name: str = "motion") -> None:
        self.env = env
        self.command = env.command_manager.get_term(command_name)
        self.robot = self.command.robot
        names = tuple(self.command.cfg.body_names)
        if "pelvis" not in names:
            raise ValueError("SuperMimic requires pelvis in the UniLab motion body list")
        self.root_body_index = names.index("pelvis")
        self.body_names = names
        self.device = env.device
        self.num_envs = env.num_envs
        self.default_q = self.robot.data.default_joint_pos(self.device).float()
        joint_names = tuple(self.robot.joint_names)
        if joint_names != tuple(G1_JOINT_NAMES):
            raise ValueError("UniLab G1 joint order does not match SuperMimic G1_JOINT_NAMES")
        self.action_term = env.action_manager.get_term("joint_pos")
        scale = self.action_term.scale
        self.action_scale = torch.as_tensor(scale, device=self.device, dtype=torch.float32)
        if self.action_scale.ndim == 0:
            self.action_scale = self.action_scale.expand(self.num_envs, NUM_J)
        if self.action_scale.ndim == 1:
            self.action_scale = self.action_scale.expand(self.num_envs, -1)
        if self.action_scale.shape != (self.num_envs, NUM_J):
            raise ValueError("UniLab joint_pos action scale does not match G1 action order")
        self.supermimic_alpha = torch.tensor(
            [supermimic_action_scale(name) for name in G1_JOINT_NAMES], device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        if self.default_q.ndim == 1:
            self.default_q = self.default_q.unsqueeze(0).expand(self.num_envs, -1)
        if self.default_q.shape[-1] != NUM_J:
            raise ValueError(f"expected {NUM_J} joints, got {self.default_q.shape[-1]}")
        self._zero_com = torch.zeros((self.num_envs, 3), device=self.device)
        motion = self.command.motion
        self.ref_joint_pos = torch.as_tensor(motion.joint_pos, device=self.device, dtype=torch.float32)
        self.ref_joint_vel = torch.as_tensor(motion.joint_vel, device=self.device, dtype=torch.float32)
        self.ref_body_pos_w = torch.as_tensor(motion.body_pos_w, device=self.device, dtype=torch.float32)
        self.ref_body_quat_w = torch.as_tensor(motion.body_quat_w, device=self.device, dtype=torch.float32)
        self.ref_body_lin_vel_w = torch.as_tensor(motion.body_lin_vel_w, device=self.device, dtype=torch.float32)
        self.ref_body_ang_vel_w = torch.as_tensor(motion.body_ang_vel_w, device=self.device, dtype=torch.float32)
        self._reference_cache = {self.device.type + str(self.device.index): (
            self.ref_joint_pos, self.ref_joint_vel, self.ref_body_pos_w,
            self.ref_body_quat_w, self.ref_body_lin_vel_w, self.ref_body_ang_vel_w,
        )}

    @property
    def frame(self) -> torch.Tensor:
        return self.command.time_steps.to(device=self.device, dtype=torch.long).clone()

    @property
    def clip_end(self) -> torch.Tensor:
        return self.command.tensor_current_clip_end_frames.to(device=self.device, dtype=torch.long).clone()

    def _physical(self):
        p = self.command.robot_body_pos_w[:, self.root_body_index].clone()
        R = quat_to_mat(self.command.robot_body_quat_w[:, self.root_body_index].clone())
        v = self.command.robot_body_lin_vel_w[:, self.root_body_index].clone()
        w = self.command.robot_body_ang_vel_w[:, self.root_body_index].clone()
        q = self.command.robot_joint_pos.clone()
        qd = self.command.robot_joint_vel.clone()
        return p, R, v, w, q, qd

    def _reference(self, frames: torch.Tensor):
        target = frames.device
        key = target.type + str(target.index)
        tables = self._reference_cache.get(key)
        if tables is None:
            tables = tuple(value.to(target) for value in (
                self.ref_joint_pos, self.ref_joint_vel, self.ref_body_pos_w,
                self.ref_body_quat_w, self.ref_body_lin_vel_w, self.ref_body_ang_vel_w,
            ))
            self._reference_cache[key] = tables
        joint_pos, joint_vel, body_pos, body_quat, body_lin_vel, body_ang_vel = tables
        rows = frames.to(dtype=torch.long, device=target)
        body_idx = self.root_body_index
        p = body_pos[rows, body_idx]
        R = quat_to_mat(body_quat[rows, body_idx])
        v = body_lin_vel[rows, body_idx]
        w = body_ang_vel[rows, body_idx]
        q = joint_pos[rows]
        qd = joint_vel[rows]
        return p, R, v, w, q, qd

    def state(self):
        return self._physical()

    def xi(self) -> torch.Tensor:
        # Context follows the actual affine path used by MotionJointPositionAction:
        # friction is exposed by the task owner when available, while joint bias is the
        # sum of motion reset bias and encoder bias. COM remains zero until UniLab exposes
        # a tensor carrier for the backend reset transaction.
        joint_bias = self.command.joint_default_bias.clone()
        encoder = self.robot.data.encoder_bias_tensor.to(self.device, dtype=torch.float32)
        friction = getattr(self.command, "foot_friction", None)
        if not isinstance(friction, torch.Tensor):
            friction = torch.ones((self.num_envs, 1), device=self.device)
        else:
            friction = friction.reshape(self.num_envs, -1).mean(-1, keepdim=True)
        return torch.cat((friction, joint_bias - encoder, self._zero_com), dim=-1)

    def to_unilab_action(self, action: torch.Tensor) -> torch.Tensor:
        """Convert SuperMimic normalized action to UniLab raw action coordinates."""
        scale = self.action_scale.to(action.device, dtype=action.dtype)
        if scale.ndim == 1:
            scale = scale.unsqueeze(0)
        # UniLab action target is default + scale * raw + motion_bias - encoder_bias.
        action_offset = torch.as_tensor(self.action_term.offset, device=action.device, dtype=action.dtype)
        # UniLab adds motion_default_bias - encoder_bias after the affine action;
        # retain that same xi offset as SuperMimic's target instead of cancelling it.
        raw = action * self.supermimic_alpha.to(action.device, action.dtype) - action_offset
        return (raw / scale.clamp_min(1e-6)).contiguous()

    def future_goal(self, frames: torch.Tensor, horizon: int, clip_end: torch.Tensor | None = None) -> torch.Tensor:
        target = frames.device
        end_source = self.clip_end if clip_end is None else clip_end
        end = end_source.to(target, dtype=torch.long)
        offsets = torch.arange(1, horizon + 1, device=target, dtype=torch.long)
        frames = frames.to(target, dtype=torch.long)
        requested = frames[:, None] + offsets[None, :]
        future = torch.minimum(requested, end[:, None])
        recorded = requested <= end[:, None]
        root = self._reference(frames)
        fp, fR, fv, fw, fq, fqd = self._reference(future.reshape(-1))
        batch = frames.shape[0]
        fp = fp.reshape(batch, horizon, 3)
        fR = fR.reshape(batch, horizon, 3, 3)
        fv = fv.reshape(batch, horizon, 3)
        fw = fw.reshape(batch, horizon, 3)
        fq = fq.reshape(batch, horizon, NUM_J)
        fqd = fqd.reshape(batch, horizon, NUM_J)
        p0, R0 = root[0], root[1]
        Ht = yaw_mat(R0).transpose(-1, -2)[:, None]
        dp = (Ht @ (fp - p0[:, None]).unsqueeze(-1)).squeeze(-1)
        rot6 = mat_to_6d(Ht @ fR)
        Rt = fR.transpose(-1, -2)
        vlocal = (Rt @ fv.unsqueeze(-1)).squeeze(-1)
        wlocal = (Rt @ fw.unsqueeze(-1)).squeeze(-1)
        vlocal = torch.where(recorded[..., None], vlocal, torch.zeros_like(vlocal))
        wlocal = torch.where(recorded[..., None], wlocal, torch.zeros_like(wlocal))
        fqd = torch.where(recorded[..., None], fqd, torch.zeros_like(fqd))
        return torch.cat((dp, rot6, vlocal, wlocal, fq, fqd * 0.1), dim=-1).reshape(frames.shape[0], -1)

    def transition(self, action: torch.Tensor, action_prev: torch.Tensor, before, reward, done) -> UniLabTransition:
        after = self.state()
        return UniLabTransition(before, after, action, action_prev, self.xi(), self.frame, reward, done)

    def flat_state(self) -> torch.Tensor:
        """Pack the authoritative CPU state once before an MPS transfer."""
        return self.flat(self.state())

    @staticmethod
    def flat(state) -> torch.Tensor:
        result = flatten_state(state)
        if result.shape[-1] != STATE_DIM:
            raise RuntimeError(f"invalid SuperMimic state width {result.shape[-1]}")
        return result
