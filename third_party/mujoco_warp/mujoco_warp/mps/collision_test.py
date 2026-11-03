"""Regression tests for the MPS capsule-box and box-box narrowphase."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import mujoco_warp

pytest.importorskip("mlx.core")
from mujoco_warp.mps.rig import OP_BOX_BOX  # noqa: E402
from mujoco_warp.mps.rig import OP_CAPSULE_BOX  # noqa: E402
from mujoco_warp.mps.rig import POINTS_PER_OPCODE  # noqa: E402
from mujoco_warp.mps.robot import BatchedEngine  # noqa: E402


def _pair_model(
  kind: str,
  pos: tuple[float, float, float],
  angle: float | tuple[float, ...] = 0.0,
  reverse_pair: bool = False,
):
  """Builds a one-pair model with one static and one free geom."""
  if kind == "capsule_box":
    geom1 = '<geom name="cap" type="capsule" size="0.1 0.35"/>'
    geom2 = '<geom name="box" type="box" size="0.3 0.3 0.3"/>'
    op = OP_CAPSULE_BOX
  else:
    geom1 = '<geom name="a" type="box" size="0.3 0.2 0.25"/>'
    geom2 = '<geom name="b" type="box" size="0.3 0.2 0.25"/>'
    op = OP_BOX_BOX

  first_name, second_name = ("box", "cap") if kind == "capsule_box" and reverse_pair else ("cap", "box")
  if kind == "box_box":
    first_name, second_name = ("b", "a") if reverse_pair else ("a", "b")
  if not np.isscalar(angle) and len(angle) == 4:
    quat = angle
  else:
    angles = (0.0, 0.0, angle) if np.isscalar(angle) else angle
    cr, cp, cy = np.cos(np.asarray(angles) / 2.0)
    sr, sp, sy = np.sin(np.asarray(angles) / 2.0)
    quat = (
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    )
  xml = f"""
  <mujoco>
    <option timestep="0.002" integrator="Euler" gravity="0 0 0" cone="pyramidal"/>
    <worldbody>
      {geom1}
      <body name="moving" pos="{pos[0]} {pos[1]} {pos[2]}"
            quat="{quat[0]} {quat[1]} {quat[2]} {quat[3]}">
        <freejoint/>
        {geom2}
      </body>
    </worldbody>
    <contact><pair geom1="{first_name}" geom2="{second_name}" condim="1" friction="0 0 0"/></contact>
  </mujoco>
  """
  return mujoco.MjModel.from_xml_string(xml), op


def _run_pair(
  kind: str,
  pos: tuple[float, float, float],
  angle: float | tuple[float, ...] = 0.0,
  reverse_pair: bool = False,
):
  """Runs both the MPS and Warp CPU narrowphase for one explicit pair."""
  mjm, op = _pair_model(kind, pos, angle, reverse_pair)
  qpos = np.asarray(mjm.qpos0, np.float32)[None]
  qvel = np.zeros((1, mjm.nv), np.float32)

  warp_model = mujoco_warp.put_model(mjm)
  warp_data = mujoco_warp.make_data(mjm, nworld=1)
  warp_data.qpos.numpy()[:] = qpos
  warp_data.qvel.numpy()[:] = qvel
  mujoco_warp.forward(warp_model, warp_data)

  engine = BatchedEngine(mjm, nworld=1)
  engine.set_state(qpos, qvel)
  engine._dyn_dispatch(kin=True)
  assert engine.rig.npairc == 1
  assert int(engine.rig.pair_op[0]) == op
  assert int(engine.rig.pair_condim[0]) == 1  # The explicit pair overrides each geom's condim=3.
  assert int(engine.rig.pair_ndim[0]) == 1
  assert engine.rig.contact_row_cap == POINTS_PER_OPCODE[op]
  assert int(engine.overflow[0]) == 0
  return mjm, engine, warp_data


def _contact_rows(engine: BatchedEngine, mjm: mujoco.MjModel) -> np.ndarray:
  nrow = int(engine.nefc[0])
  return engine.dense_J()[0, :nrow, : mjm.nv]


@pytest.mark.parametrize(
  ("pos", "reverse_pair", "normal_axis", "expected_ncon"),
  [
    ((0.0, 0.0, 0.5), False, 2, 1),
    ((0.36, 0.36, 0.0), False, -1, 1),
    ((0.36, 0.36, 0.0), True, -1, 1),
  ],
)
def test_capsule_box_face_edge_and_reversed_explicit_pair(pos, reverse_pair, normal_axis, expected_ncon):
  """Capsule-box face and shallow-edge contacts keep the capsule-to-box normal."""
  mjm, engine, warp_data = _run_pair("capsule_box", pos, reverse_pair=reverse_pair)
  assert int(engine.ncon[0]) == expected_ncon
  assert int(warp_data.nacon.numpy()[0]) > 0
  assert int(engine.ncon[0]) <= 2
  assert int(engine.nefc[0]) == int(engine.ncon[0])

  # Geometry-type ordering is stable even when an explicit pair reverses its inputs.
  g1, g2 = map(int, engine.rig.pair_g1[:1].tolist() + engine.rig.pair_g2[:1].tolist())
  assert int(mjm.geom_type[g1]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
  assert int(mjm.geom_type[g2]) == int(mujoco.mjtGeom.mjGEOM_BOX)

  normal = _contact_rows(engine, mjm)[0, :3]
  np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-5)
  if normal_axis >= 0:
    assert normal[normal_axis] > 0.8
    assert np.linalg.norm(np.delete(normal, normal_axis)) < 1e-4
  else:
    assert normal[0] > 0.5 and normal[1] > 0.5
    assert abs(normal[2]) < 1e-4


def test_capsule_box_separated_pair_emits_no_contact_rows():
  """Broadphase candidates that fail capsule-box narrowphase emit no rows."""
  mjm, engine, warp_data = _run_pair("capsule_box", (0.6, 0.6, 0.0))
  assert int(engine.ncon[0]) == 0
  assert int(engine.nefc[0]) == 0
  assert int(warp_data.nacon.numpy()[0]) == 0


@pytest.mark.parametrize(
  ("pos", "rotation", "expected_ncon", "expected_normal"),
  [
    ((0.4, 0.0, 0.0), 0.0, 4, None),
    ((0.45, 0.15, 0.05), 0.35, 4, None),
    (
      (-0.4492906481, -0.3652798831, 0.1884488314),
      (0.2805831730, -0.4358958602, -0.8096224069, -0.2752804756),
      1,
      (-0.76611072, -0.64270866, 0.0),
    ),
  ],
)
def test_box_box_sat_manifold(pos, rotation, expected_ncon, expected_normal):
  """Box-box SAT matches MJX face manifolds and one-point edge reduction."""
  mjm, engine, warp_data = _run_pair("box_box", pos, rotation)
  ncon = int(engine.ncon[0])
  assert ncon == expected_ncon
  assert int(warp_data.nacon.numpy()[0]) > 0
  assert int(engine.nefc[0]) == ncon
  normal = _contact_rows(engine, mjm)[0, :3]
  np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-5)
  assert np.dot(normal, np.asarray(pos, np.float32)) > 0.0
  if expected_normal is not None:
    np.testing.assert_allclose(normal, expected_normal, atol=2e-5)

  # Native Warp core may retain a different number of points than MJX; compare
  # exact rows only where the aligned four-point manifold counts also match.
  if int(warp_data.nacon.numpy()[0]) == ncon == 4:
    rows_warp = warp_data.efc.J.numpy()[0, :ncon, : mjm.nv]
    rows_mps = _contact_rows(engine, mjm)
    keys_warp = tuple(rows_warp[:, i] for i in range(rows_warp.shape[1] - 1, -1, -1))
    keys_mps = tuple(rows_mps[:, i] for i in range(rows_mps.shape[1] - 1, -1, -1))
    np.testing.assert_allclose(rows_mps[np.lexsort(keys_mps)], rows_warp[np.lexsort(keys_warp)], atol=1e-5)


def test_box_box_separated_pair_emits_no_contact_rows():
  """SAT rejects a separated box pair after the conservative broadphase."""
  mjm, engine, warp_data = _run_pair("box_box", (0.8, 0.0, 0.0))
  assert int(engine.ncon[0]) == 0
  assert int(engine.nefc[0]) == 0
  assert int(warp_data.nacon.numpy()[0]) == 0
