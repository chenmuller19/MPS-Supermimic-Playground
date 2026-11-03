"""Regression tests for the MPS implicitfast correction."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import mujoco_warp

pytest.importorskip("mlx.core")
from mujoco_warp.mps.robot import BatchedEngine  # noqa: E402


@pytest.mark.parametrize(
  ("forcerange", "force_interior", "qpos_value", "qvel_value", "ctrl_value", "m_layout", "tree"),
  [
    ((-100.0, 100.0), True, 0.2, 0.4, 0.7, "dense", False),
    ((-2.0, 2.0), False, 0.2, 0.4, 0.7, "dense", False),
    ((-10.0, 10.0), False, 0.0, 0.0, 0.125, "dense", False),
    ((-100.0, 100.0), True, 0.2, 0.4, 0.7, "packed", False),
    ((-100.0, 100.0), True, 0.2, 0.4, 0.7, "packed", True),
  ],
)
def test_implicitfast_position_actuator_matches_warp_and_native(
  forcerange, force_interior, qpos_value, qvel_value, ctrl_value, m_layout, tree
):
  """The implicit matrix includes unsaturated servo and passive damping derivatives."""
  xml = f"""
  <mujoco>
    <option timestep="0.002" integrator="implicitfast" gravity="0 0 0" cone="pyramidal"/>
    <worldbody>
      <body>
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.4"/>
        <geom type="capsule" size="0.04 0.12" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <position name="servo" joint="hinge" kp="80" kv="8"
                forcelimited="true" forcerange="{forcerange[0]} {forcerange[1]}"/>
    </actuator>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  qpos = np.array([[qpos_value]], np.float32)
  qvel = np.array([[qvel_value]], np.float32)
  ctrl = np.array([[ctrl_value]], np.float32)

  engine = BatchedEngine(mjm, nworld=1, m_layout=m_layout, tree=tree)
  assert not engine.coop
  engine.set_state(qpos, qvel, ctrl)
  engine._dyn_dispatch(kin=False)
  expected_derivative = -mjm.dof_damping[0]
  if force_interior:
    expected_derivative += mjm.actuator_biasprm[0, 2] * mjm.actuator_gear[0, 0] ** 2
  np.testing.assert_allclose(np.asarray(engine.smooth_vel_deriv)[0, 0], expected_derivative, atol=1e-6)

  warp_model = mujoco_warp.put_model(mjm)
  warp_data = mujoco_warp.make_data(mjm, nworld=1)
  warp_data.qpos.numpy()[:] = qpos
  warp_data.qvel.numpy()[:] = qvel
  warp_data.ctrl.numpy()[:] = ctrl
  mujoco_warp.step(warp_model, warp_data)

  native_data = mujoco.MjData(mjm)
  native_data.qpos[:] = qpos[0]
  native_data.qvel[:] = qvel[0]
  native_data.ctrl[:] = ctrl[0]
  mujoco.mj_step(mjm, native_data)

  engine.step()
  mps_qpos, mps_qvel = engine.get_state()
  np.testing.assert_allclose(mps_qpos, warp_data.qpos.numpy(), atol=2e-6)
  np.testing.assert_allclose(mps_qvel, warp_data.qvel.numpy(), atol=2e-5)
  np.testing.assert_allclose(mps_qpos[0], native_data.qpos, atol=2e-6)
  np.testing.assert_allclose(mps_qvel[0], native_data.qvel, atol=2e-5)
  np.testing.assert_allclose(np.asarray(engine.qacc), warp_data.qacc.numpy(), rtol=5e-7, atol=1e-4)
  np.testing.assert_allclose(np.asarray(engine.warm), warp_data.qacc_warmstart.numpy(), rtol=5e-7, atol=1e-4)


def test_implicitfast_contact_keeps_constrained_acceleration_outputs():
  """Free-body contact rows integrate implicitly without replacing qacc/warmstart."""
  xml = """
  <mujoco>
    <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81" cone="pyramidal"/>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 0.1"/>
      <body pos="0 0 0.04">
        <joint type="free" damping="0.4"/>
        <geom name="ball" type="sphere" size="0.05" mass="1"/>
      </body>
    </worldbody>
    <contact><pair geom1="floor" geom2="ball" condim="1"/></contact>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  qpos = np.array(mjm.qpos0, np.float32)[None]
  qvel = np.zeros((1, mjm.nv), np.float32)
  qvel[0, 2] = -0.1

  engine = BatchedEngine(mjm, nworld=1)
  engine.set_state(qpos, qvel)
  engine._dyn_dispatch(kin=False)
  assert int(engine.ncon[0]) == 1
  np.testing.assert_array_equal(np.asarray(engine.smooth_vel_deriv), np.zeros((1, mjm.nv), np.float32))

  warp_model = mujoco_warp.put_model(mjm)
  warp_data = mujoco_warp.make_data(mjm, nworld=1)
  warp_data.qpos.numpy()[:] = qpos
  warp_data.qvel.numpy()[:] = qvel
  mujoco_warp.step(warp_model, warp_data)

  native_data = mujoco.MjData(mjm)
  native_data.qpos[:] = qpos[0]
  native_data.qvel[:] = qvel[0]
  mujoco.mj_step(mjm, native_data)

  engine.step()
  mps_qpos, mps_qvel = engine.get_state()
  np.testing.assert_allclose(mps_qpos, warp_data.qpos.numpy(), atol=3e-5)
  np.testing.assert_allclose(mps_qvel, warp_data.qvel.numpy(), atol=3e-4)
  np.testing.assert_allclose(mps_qpos[0], native_data.qpos, atol=3e-5)
  np.testing.assert_allclose(mps_qvel[0], native_data.qvel, atol=3e-4)
  np.testing.assert_allclose(np.asarray(engine.qacc), warp_data.qacc.numpy(), rtol=5e-7, atol=1e-4)
  np.testing.assert_allclose(np.asarray(engine.warm), warp_data.qacc_warmstart.numpy(), rtol=5e-7, atol=1e-4)
