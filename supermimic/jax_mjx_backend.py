"""Pure-JAX MJX G1 backend for the no-Torch SuperMimic path.

This module deliberately has no Torch dependency.  Actions, qpos/qvel, and all
MJX state remain JAX arrays on the selected PJRT device.  The current Torch
adapter remains available as a parity reference; the JAX learner should consume
this backend directly.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

# MJX 3.3.7 uses clip(a_min=...) while JAX 0.11 renamed those keywords.
if "a_min" not in __import__("inspect").signature(jnp.clip).parameters:
    class _MJXJaxNumpyCompat:
        def __getattr__(self, name):
            return getattr(jnp, name)

        @staticmethod
        def clip(a, a_min=None, a_max=None, **kwargs):
            return jnp.clip(a, min=kwargs.pop("min", a_min), max=kwargs.pop("max", a_max), **kwargs)

    from mujoco.mjx._src import collision_driver, forward, passive
    _compat = _MJXJaxNumpyCompat()
    collision_driver.jp = _compat
    forward.jp = _compat
    passive.jp = _compat

from .robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list
from .robot import DAMPING_RATIO, JOINT_MOTORS, NATURAL_FREQ

CONTROL_DT = 0.02
NUM_J = 29


class JAXMJXG1:
    """Batched G1 MJX state with one 50 Hz control transition per ``step_jax``."""

    def __init__(self, xml_path: str | Path, num_envs: int, backend: str = "mps") -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if backend not in {"cpu", "mps"}:
            raise ValueError("backend must be 'cpu' or 'mps'")
        devices = jax.devices(backend)
        self.device = devices[0]
        self.num_envs = num_envs
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        names = tuple(self.model.actuator(i).name for i in range(self.model.nu))
        if self.model.nu != NUM_J or names != tuple(G1_JOINT_NAMES):
            raise ValueError("MJCF actuator order does not match the SuperMimic G1 contract")
        # Menagerie provides the collision geometry and explicit contact pairs. Override only
        # the actuator gains/limits so the origin controller contract remains unchanged.
        for i,name in enumerate(G1_JOINT_NAMES):
            key=name.removeprefix('left_').removeprefix('right_').removesuffix('_joint')
            effort,_,arm=JOINT_MOTORS[key]; kp=arm*NATURAL_FREQ**2; kd=2*DAMPING_RATIO*arm*NATURAL_FREQ
            self.model.actuator_gainprm[i,0]=kp; self.model.actuator_biasprm[i,1]=-kp; self.model.actuator_biasprm[i,2]=-kd
            self.model.actuator_forcerange[i]=(-effort,effort); self.model.actuator_forcelimited[i]=1; self.model.actuator_ctrllimited[i]=0
        self._model = mjx.put_model(self.model, impl="jax", device=self.device)
        self.torso_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,"torso_link")
        self._torso_geoms=jnp.asarray(self.model.geom_bodyid==self.torso_id)
        self._ground_geoms=jnp.asarray(self.model.geom_bodyid==0)
        self._base_geom_friction=jnp.asarray(self._model.geom_friction)
        self._base_body_ipos=jnp.asarray(self._model.body_ipos)
        self._geom_friction_batch=jnp.broadcast_to(self._base_geom_friction,(num_envs,)+self._base_geom_friction.shape)
        self._body_ipos_batch=jnp.broadcast_to(self._base_body_ipos,(num_envs,)+self._base_body_ipos.shape)
        self._model_batch=self._model.replace(geom_friction=self._geom_friction_batch,body_ipos=self._body_ipos_batch)
        axes=jax.tree_util.tree_map(lambda _:None,self._model).replace(geom_friction=0,body_ipos=0)
        data = mjx.make_data(self._model, impl="jax", device=self.device)
        substeps = int(round(CONTROL_DT / self.model.opt.timestep))
        if substeps * self.model.opt.timestep != CONTROL_DT:
            raise ValueError("MJCF timestep must divide CONTROL_DT exactly")
        self.substeps = substeps
        self._data = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), data)
        self.default_q = jnp.asarray(default_joint_pos_list(), dtype=jnp.float32)
        self.action_scale = jnp.asarray([action_scale(name) for name in G1_JOINT_NAMES], dtype=jnp.float32)
        self._default_qpos = self._data.qpos
        self._default_qvel = jnp.zeros_like(self._data.qvel)

        def control_step(model,d):
            def substep(data,_):
                return mjx.step(model,data),None
            d,_=jax.lax.scan(substep,d,None,length=self.substeps)
            return d

        self._step=jax.jit(jax.vmap(control_step,in_axes=(axes,0)))
        self._forward=jax.jit(jax.vmap(lambda model,d:mjx.forward(model,d),in_axes=(axes,0)))
        self.reset_jax()

    @property
    def data(self):
        return self._data

    @property
    def qpos(self):
        return self._data.qpos

    @property
    def qvel(self):
        return self._data.qvel

    def reset_jax(self, qpos=None, qvel=None):
        """Reset all environments, optionally supplying complete batched state."""
        if qpos is None:
            qpos = self._default_qpos.at[:, 2].set(0.75)
            qpos = qpos.at[:, 3].set(1.0)
            qpos = qpos.at[:, 7:36].set(self.default_q)
            qvel = self._default_qvel
        elif qvel is None:
            raise ValueError("qvel is required when qpos is supplied")
        if qpos.shape != (self.num_envs, self.model.nq) or qvel.shape != (self.num_envs, self.model.nv):
            raise ValueError("qpos/qvel batch shapes do not match the MJX model")
        self._data = self._forward(self._model_batch, self._data.replace(qpos=qpos, qvel=qvel, ctrl=qpos[:, 7:36]))
        return self._data

    def model_batch_from_xi(self, xi):
        """Build a batched MJX model without mutating simulator state."""
        xi=jnp.asarray(xi,dtype=jnp.float32)
        if xi.shape!=(self.num_envs,33): raise ValueError("xi must have shape (N,33)")
        fr=jnp.broadcast_to(xi[:,0,None,None],(self.num_envs,self.model.ngeom,3))
        fr=fr.at[:,:,1:].set(self._base_geom_friction[None,:,1:])
        fr=jnp.where(self._ground_geoms[None,:,None],self._base_geom_friction[None,:,:],fr)
        ip=jnp.broadcast_to(self._base_body_ipos,(self.num_envs,)+self._base_body_ipos.shape).at[:,self.torso_id].add(xi[:,30:33])
        return self._model.replace(geom_friction=fr,body_ipos=ip)

    def set_xi_jax(self, xi):
        """Apply origin friction and torso COM offsets per batched MJX model."""
        self._model_batch=self.model_batch_from_xi(xi)
        return self._model_batch

    def reset_masked_data(self, model_batch, current, mask, qpos, qvel):
        """Return reset data without mutating the simulator, for scan/cond use."""
        mask=jnp.asarray(mask,dtype=jnp.bool_)
        if mask.shape!=(self.num_envs,): raise ValueError("reset mask must have one value per environment")
        qpos=jnp.where(mask[:,None],qpos,current.qpos); qvel=jnp.where(mask[:,None],qvel,current.qvel)
        ctrl=jnp.where(mask[:,None],qpos[:,7:36],current.ctrl)
        return jax.lax.cond(jnp.any(mask),lambda d:self._forward(model_batch,d.replace(qpos=qpos,qvel=qvel,ctrl=ctrl)),lambda d:d,current)

    def reset_masked_jax(self, mask, qpos, qvel):
        """Reset selected batch members while retaining the others' integrated state."""
        self._data=self.reset_masked_data(self._model_batch,self._data,mask,qpos,qvel)
        return self._data

    def step_data(self, model_batch, data, action, joint_offset=None):
        """Purely step supplied MJX data, for use inside jax.lax.scan."""
        action = jnp.asarray(action, dtype=jnp.float32)
        if action.shape != (self.num_envs, NUM_J):
            raise ValueError(f"expected action shape {(self.num_envs, NUM_J)}, got {action.shape}")
        offset = jnp.zeros_like(action) if joint_offset is None else jnp.asarray(joint_offset, dtype=jnp.float32)
        if offset.shape != action.shape:
            raise ValueError("joint_offset shape must match action shape")
        ctrl = self.default_q + offset + self.action_scale * action
        return self._step(model_batch, data.replace(ctrl=ctrl))

    def step_jax(self, action, joint_offset=None):
        """Apply normalized action and return the JAX MJX data after 20 ms."""
        self._data=self.step_data(self._model_batch,self._data,action,joint_offset)
        return self._data
