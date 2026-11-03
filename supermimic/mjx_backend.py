"""Batched MuJoCo XLA (MJX) G1 simulator for the SuperMimic contract.

The simulator owns MuJoCo state on a JAX device.  It exposes normalized
SuperMimic actions and returns the physical state fields required by the
existing Torch MDP and learner through one explicit host transfer per step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
import torch

from .mdp import NUM_J
from .motion import CONTROL_DT
from .robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list
from .rotations import quat_to_mat


@dataclass
class MJXStep:
    """One simulator result in the SuperMimic physical-state contract."""

    state: tuple[torch.Tensor, ...]
    qpos: torch.Tensor
    qvel: torch.Tensor


class MJXG1Simulator:
    """Run batched G1 MuJoCo dynamics through a JAX MJX device."""

    def __init__(
        self,
        xml_path: str | Path,
        num_envs: int,
        backend: str = "mps",
        device: jax.Device | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if backend not in {"cpu", "mps"}:
            raise ValueError("backend must be 'cpu' or 'mps'")
        if backend == "mps" and not any(device.platform == "mps" for device in jax.devices()):
            raise RuntimeError("jax-mps is required for the MJX mps backend")
        self.backend = backend
        self.num_envs = num_envs
        self.device = device or (jax.devices("mps")[0] if backend == "mps" else jax.devices("cpu")[0])
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        actuator_names = tuple(self.model.actuator(i).name for i in range(self.model.nu))
        if self.model.nu != NUM_J or actuator_names != tuple(G1_JOINT_NAMES):
            raise ValueError("MJCF actuator order does not match the SuperMimic G1 contract")
        model = mjx.put_model(self.model, impl="jax", device=self.device)
        single = mjx.make_data(model, impl="jax", device=self.device)
        self._model = model
        substeps = int(round(CONTROL_DT / self.model.opt.timestep))
        if not jnp.isclose(substeps * self.model.opt.timestep, CONTROL_DT):
            raise ValueError("MJCF timestep must divide the SuperMimic control timestep")
        self._substeps = substeps
        self._single = single
        self._data = jax.tree_util.tree_map(
            lambda value: jnp.broadcast_to(value, (num_envs,) + value.shape), single
        )
        def step_control(data):
            for _ in range(self._substeps):
                data = mjx.step(self._model, data)
            return data

        self._step = jax.jit(jax.vmap(step_control))
        self._forward = jax.jit(jax.vmap(lambda data: mjx.forward(self._model, data)))
        self._default_q = jnp.asarray(default_joint_pos_list(), dtype=jnp.float32)
        self._action_scale = jnp.asarray([action_scale(name) for name in G1_JOINT_NAMES], dtype=jnp.float32)
        self._default_qpos = jnp.asarray(self._data.qpos)
        self._default_qvel = jnp.zeros_like(self._data.qvel)
        self.reset()

    @property
    def data(self):
        """Return the current batched MJX data."""
        return self._data

    def reset(self, env_ids: torch.Tensor | None = None) -> MJXStep:
        """Reset selected environments to the configured G1 default pose."""
        if env_ids is None:
            qpos = self._default_qpos
            qpos = qpos.at[:, 2].set(0.75)
            qpos = qpos.at[:, 3].set(1.0)
            qpos = qpos.at[:, 7:36].set(self._default_q)
            self._data = self._data.replace(qpos=qpos, qvel=self._default_qvel, ctrl=qpos[:, 7:36])
        else:
            ids = jnp.asarray(env_ids.detach().cpu().numpy(), dtype=jnp.int32)
            qpos = self._data.qpos.at[ids].set(self._default_qpos[ids])
            qpos = qpos.at[ids, 2].set(0.75)
            qpos = qpos.at[ids, 3].set(1.0)
            qpos = qpos.at[ids, 7:36].set(self._default_q)
            ctrl = self._data.ctrl.at[ids].set(qpos[ids, 7:36])
            self._data = self._data.replace(qpos=qpos, qvel=self._data.qvel.at[ids].set(0.0), ctrl=ctrl)
        return self._torch_result()

    def step(self, action: torch.Tensor, joint_offset: torch.Tensor | None = None) -> MJXStep:
        """Apply a SuperMimic action and advance one 50 Hz control transition."""
        if action.shape != (self.num_envs, NUM_J):
            raise ValueError(f"expected action shape {(self.num_envs, NUM_J)}, got {tuple(action.shape)}")
        action_jax = jnp.asarray(action.detach().cpu().numpy().copy(), dtype=jnp.float32)
        offset = jnp.zeros_like(action_jax) if joint_offset is None else jnp.asarray(
            joint_offset.detach().cpu().numpy().copy(), dtype=jnp.float32
        )
        if offset.shape != action_jax.shape:
            raise ValueError(f"expected joint_offset shape {tuple(action.shape)}, got {offset.shape}")
        ctrl = self._default_q + offset + self._action_scale * action_jax
        self._data = self._data.replace(ctrl=ctrl)
        self._data = self._step(self._data)
        self._data.qpos.block_until_ready()
        return self._torch_result()

    def set_state(self, qpos: torch.Tensor, qvel: torch.Tensor) -> MJXStep:
        """Set the full free-root/joint state for motion-frame resets."""
        if qpos.shape != (self.num_envs, self.model.nq):
            raise ValueError(f"expected qpos shape {(self.num_envs, self.model.nq)}")
        if qvel.shape != (self.num_envs, self.model.nv):
            raise ValueError(f"expected qvel shape {(self.num_envs, self.model.nv)}")
        qpos_jax = jnp.asarray(qpos.detach().cpu().numpy().copy(), dtype=jnp.float32)
        qvel_jax = jnp.asarray(qvel.detach().cpu().numpy().copy(), dtype=jnp.float32)
        ctrl = qpos_jax[:, 7 : 7 + NUM_J]
        self._data = self._data.replace(qpos=qpos_jax, qvel=qvel_jax, ctrl=ctrl)
        self._data = self._forward(self._data)
        return self._torch_result()

    def _torch_result(self) -> MJXStep:
        qpos = torch.from_numpy(np.array(jax.device_get(self._data.qpos), copy=True)).float()
        qvel = torch.from_numpy(np.array(jax.device_get(self._data.qvel), copy=True)).float()
        state = (
            qpos[:, :3],
            quat_to_mat(qpos[:, 3:7]),
            qvel[:, :3],
            qvel[:, 3:6],
            qpos[:, 7:36],
            qvel[:, 6:35],
        )
        return MJXStep(state=state, qpos=qpos, qvel=qvel)
