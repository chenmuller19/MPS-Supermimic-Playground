from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from supermimic.mlx_geometry import (
    axis_angle_to_mat, mat_to_quat, quat_to_mat, quat_to_rotvec, so3_exp, so3_log, yaw_mat,
)
from supermimic.mlx_networks import MLP, Policy, RunningNorm, StableELU
from supermimic.mlx_train import _first_nonfinite_step, _nonfinite_gradient_paths
from supermimic.mlx_replay import ReplayBuffer


REPO = Path(__file__).resolve().parents[1]


def _synthetic_motion(xml: Path, directory: Path):
    from supermimic.g1_contract import default_joint_pos_list
    from supermimic.mlx_geometry import G1Kinematics
    from supermimic.mlx_motion import MotionLib

    raw = np.zeros((100, 36), dtype=np.float32)
    raw[:, 2] = 0.76
    raw[:, 6] = 1.0
    raw[:, 7:] = default_joint_pos_list()
    motion_path = directory / "synthetic_g1.csv"
    np.savetxt(motion_path, raw, delimiter=",")
    kin = G1Kinematics(str(xml))
    return kin, MotionLib([str(motion_path)], kin)


def test_rotation_roundtrip_and_gradient():
    vectors = mx.array([[0.2, -0.1, 0.05], [0.0, 0.0, 0.0]], dtype=mx.float32)
    recovered = so3_log(so3_exp(vectors))
    mx.eval(recovered)
    np.testing.assert_allclose(np.asarray(recovered), np.asarray(vectors), atol=2e-5)

    rotations = quat_to_mat(mx.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
    quaternions = mat_to_quat(rotations)
    mx.eval(quaternions)
    np.testing.assert_allclose(np.asarray(quat_to_mat(quaternions)), np.asarray(rotations), atol=2e-5)

    grad = mx.grad(lambda x: mx.sum(so3_log(so3_exp(x)) ** 2))(vectors[:1])
    mx.eval(grad)
    assert np.isfinite(np.asarray(grad)).all()
    np.testing.assert_allclose(np.asarray(grad), 2 * np.asarray(vectors[:1]), atol=2e-4)

    identity_quat_grad = mx.grad(
        lambda q: mx.sum(quat_to_rotvec(q))
    )(mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32))
    mx.eval(identity_quat_grad)
    assert np.isfinite(np.asarray(identity_quat_grad)).all()
    np.testing.assert_allclose(np.asarray(identity_quat_grad), [0, 2, 2, 2], atol=2e-5)

    zero_grad = mx.grad(lambda x: mx.sum(so3_log(so3_exp(x))))(mx.zeros((3,), dtype=mx.float32))
    mx.eval(zero_grad)
    assert np.isfinite(np.asarray(zero_grad)).all()
    np.testing.assert_allclose(np.asarray(zero_grad), np.ones(3), atol=2e-5)


def test_stable_elu_keeps_large_positive_gradients_finite():
    values = mx.array([-90.0, -24.0, -1.0, 0.0, 1.0, 90.0, 92.0], dtype=mx.float32)
    actual = StableELU()(values)
    expected = nn.ELU()(values)
    grads = mx.grad(lambda x: mx.sum(StableELU()(x)))(values)
    mx.eval(actual, expected, grads)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=1e-6)
    assert np.isfinite(np.asarray(grads)).all()
    np.testing.assert_allclose(np.asarray(grads), [0, 0, np.exp(-1), 1, 1, 1, 1], atol=1e-6)


def test_yaw_extraction_is_stable_near_vertical_pitch():
    axis = mx.array([0.0, 1.0, 0.0], dtype=mx.float32)
    for degrees in (89.9, 89.99, 90.0, 90.01):
        angle = mx.array(np.deg2rad(degrees), dtype=mx.float32)
        grad = mx.grad(
            lambda value: mx.sum(yaw_mat(axis_angle_to_mat(axis, value)))
        )(angle)
        mx.eval(grad)
        assert np.isfinite(np.asarray(grad)).all()
        assert abs(float(np.asarray(grad))) < 2.0

    vertical = yaw_mat(axis_angle_to_mat(axis, mx.array(np.pi / 2, dtype=mx.float32)))
    mx.eval(vertical)
    np.testing.assert_allclose(np.asarray(vertical), np.eye(3), atol=2e-6)


def test_nonfinite_policy_diagnostics_identify_paths_and_steps():
    gradients = {"encoder": {"weight": mx.array([[1.0, float("nan")]])}}
    assert _nonfinite_gradient_paths(gradients) == ["encoder.weight: 1/2"]
    rollout = mx.array([[0.0, 1.0], [2.0, float("nan")], [float("inf"), 0.0]])
    assert _first_nonfinite_step(rollout) == 1
    assert _first_nonfinite_step((mx.zeros((3, 2)), rollout)) == 1


def test_checkpointed_policy_preserves_parameter_gradients():
    policy = Policy(obs_dim=12, goal_dim=24, hist=2)
    obs = mx.random.normal((4, 2, 12))
    goal = mx.random.normal((4, 24))
    params = policy.trainable_parameters()

    def eager_loss(values):
        policy.update(values)
        return mx.mean(policy(obs, goal) ** 2)

    checkpointed = mx.checkpoint(
        lambda values, observations, goals: (
            policy.update(values),
            policy(observations, goals),
        )[1]
    )
    eager_grads = mx.grad(eager_loss)(params)
    checkpointed_grads = mx.grad(lambda values: mx.mean(checkpointed(values, obs, goal) ** 2))(params)
    eager_leaves = tree_flatten(eager_grads)
    checkpointed_leaves = tree_flatten(checkpointed_grads)
    assert len(eager_leaves) == len(checkpointed_leaves)
    for (_, eager), (_, rematerialized) in zip(eager_leaves, checkpointed_leaves):
        mx.eval(eager, rematerialized)
        np.testing.assert_allclose(np.asarray(eager), np.asarray(rematerialized), rtol=0, atol=0)


def test_fp16_mlp_matmul_keeps_fp32_master_parameters_and_gradients():
    reference = MLP(24, [32, 16], 8, nn.ELU)
    mixed = MLP(24, [32, 16], 8, nn.ELU, matmul_dtype=mx.float16)
    mixed.update(reference.trainable_parameters())
    inputs = mx.random.normal((12, 24))

    reference_output = reference(inputs)
    mixed_output = mixed(inputs)
    mx.eval(reference_output, mixed_output)
    assert mixed_output.dtype == mx.float32
    np.testing.assert_allclose(
        np.asarray(mixed_output), np.asarray(reference_output), rtol=2e-2, atol=2e-2
    )

    params = mixed.trainable_parameters()

    def mixed_loss(values):
        mixed.update(values)
        return mx.mean(mixed(inputs) ** 2)

    grads = tree_flatten(mx.grad(mixed_loss)(params))
    assert grads
    for _, grad in grads:
        mx.eval(grad)
        assert grad.dtype == mx.float32
        assert np.isfinite(np.asarray(grad)).all()
    for _, param in tree_flatten(mixed.trainable_parameters()):
        assert param.dtype == mx.float32


def test_mlp_float32_path_matches_sequential_module():
    mlp = MLP(12, [16, 8], 4, nn.ELU)
    inputs = mx.random.normal((3, 12))
    expected = mlp.layers(inputs)
    actual = mlp(inputs)
    mx.eval(expected, actual)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0, atol=0)


def test_shared_goal_encoding_preserves_outputs_and_parameter_gradients():
    policy = Policy(obs_dim=12, goal_dim=24, hist=2)
    obs = mx.random.normal((5, 2, 12))
    goal_table = mx.random.normal((7, 24))
    indices = mx.array([2, 2, 6, 1, 2], dtype=mx.int32)
    selected_goals = goal_table[indices]

    eager_output = policy(obs, selected_goals)
    cached_output = policy.decode(obs, policy.encode_goal(goal_table)[indices])
    mx.eval(eager_output, cached_output)
    np.testing.assert_allclose(np.asarray(eager_output), np.asarray(cached_output), rtol=1e-5, atol=1e-5)

    params = policy.trainable_parameters()

    def decode_checkpoint_fn(values, observations, encoded_goal):
        policy.update(values)
        return policy.decode(observations, encoded_goal)

    checkpointed_decode = mx.checkpoint(decode_checkpoint_fn)

    def eager_loss(values):
        policy.update(values)
        return mx.mean(policy(obs, selected_goals) ** 2)

    def cached_loss(values):
        policy.update(values)
        encoded = policy.encode_goal(goal_table)
        return mx.mean(checkpointed_decode(values, obs, encoded[indices]) ** 2)

    eager_grads = tree_flatten(mx.grad(eager_loss)(params))
    cached_grads = tree_flatten(mx.grad(cached_loss)(params))
    assert len(eager_grads) == len(cached_grads)
    for (_, eager), (_, cached) in zip(eager_grads, cached_grads):
        mx.eval(eager, cached)
        np.testing.assert_allclose(np.asarray(eager), np.asarray(cached), rtol=2e-5, atol=2e-5)


def test_running_norm_matches_population_statistics():
    norm = RunningNorm(2, eps=0.1)
    norm.update(mx.array([[1.0, 2.0], [3.0, 6.0]]))
    norm.update(mx.array([[5.0, 10.0]]))
    mx.eval(norm.mean, norm.var)
    np.testing.assert_allclose(np.asarray(norm.mean), [3.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(norm.var), [8.0 / 3.0, 32.0 / 3.0], atol=1e-6)
    assert norm.count == 3


def _transition(frame: int, done: bool):
    return {
        "s": mx.full((1, 71), frame, dtype=mx.float32),
        "s1": mx.full((1, 71), frame + 1, dtype=mx.float32),
        "a": mx.zeros((1, 29), dtype=mx.float32),
        "a_prev": mx.zeros((1, 29), dtype=mx.float32),
        "xi": mx.zeros((1, 33), dtype=mx.float32),
        "r": mx.zeros((1,), dtype=mx.float32),
        "f": np.array([frame], np.int32),
        "clip_end": np.array([20], np.int32),
        "ep_step": np.array([frame], np.int32),
        "done": np.array([done], np.bool_),
        "episode": np.array([0], np.int64),
    }


def test_replay_done_metadata_masks_post_terminal_transitions():
    replay = ReplayBuffer(capacity=8, num_envs=1)
    replay.add(_transition(0, True))
    replay.add(_transition(1, False))
    replay.add(_transition(2, False))
    batch = replay.sample_windows(batch=1, window=3)
    mx.eval(batch["done"], batch["mask"])
    np.testing.assert_array_equal(np.asarray(batch["done"]), [[True, False, False]])
    np.testing.assert_array_equal(np.asarray(batch["mask"]), [[1.0, 0.0, 0.0]])


def test_mps_g1_reset_and_step(tmp_path):
    from supermimic.g1_contract import default_joint_pos_list
    from supermimic.mlx_mdp import MDP, normalize_xi, unflatten_state
    from supermimic.mlx_mps_env import MPSVectorEnv
    import mujoco

    xml = REPO / "data" / "menagerie_unitree_g1" / "scene_mjx.xml"
    kin, motion = _synthetic_motion(xml, tmp_path)
    mdp = MDP(kin, motion, mx.array(default_joint_pos_list(), dtype=mx.float32))
    env = MPSVectorEnv(xml, 2, motion, mdp)
    state = env.state()
    assert [part.shape for part in state] == [(2, 3), (2, 3, 3), (2, 3), (2, 3), (2, 29), (2, 29)]
    assert env.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert env.model.geom_type[env.cylinder_ids].tolist() == [mujoco.mjtGeom.mjGEOM_CAPSULE] * len(env.cylinder_ids)

    transition = env.step(mx.zeros((2, 29), dtype=mx.float32))
    mx.eval(transition["s1"], transition["r"])
    assert transition["s"].shape == transition["s1"].shape == (2, 71)
    assert np.isfinite(np.asarray(transition["r"])).all()
    assert transition["done"].shape == (2,)

    frame = mx.array(transition["f"] + 1, dtype=mx.int32)
    state = unflatten_state(transition["s1"])
    reward, body_positions = mdp.reward(
        state, frame, transition["a"], transition["a_prev"], return_body_positions=True
    )
    reference_reward = mdp.reward(state, frame, transition["a"], transition["a_prev"])
    features = mdp.critic_features(state, frame, normalize_xi(transition["xi"]))
    cached_features = mdp.critic_features(
        state, frame, normalize_xi(transition["xi"]), body_positions=body_positions
    )
    repeated_goals = mx.stack([motion.goal(frame + t, 2) for t in range(3)], axis=0)
    frame_seq = frame[None] + mx.arange(3, dtype=mx.int32)[:, None]
    fallback_goals = motion.goal(frame_seq, 2)
    mx.eval(repeated_goals, fallback_goals)
    assert motion.precompute_goals(2, max_bytes=1) is None
    motion.precompute_goals(2)
    cached_goals = motion.goal(frame_seq, 2)
    mx.eval(reward, reference_reward, features, cached_features, cached_goals)
    np.testing.assert_allclose(np.asarray(reward), np.asarray(reference_reward), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(features), np.asarray(cached_features), rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(repeated_goals), np.asarray(fallback_goals), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(repeated_goals), np.asarray(cached_goals), rtol=1e-6, atol=1e-6)


def test_full_body_proxy_scene_keeps_pairs_and_contact_rows(tmp_path):
    import mujoco

    from supermimic.g1_contract import default_joint_pos_list
    from supermimic.mlx_mdp import MDP
    from supermimic.mlx_mps_env import MPSVectorEnv

    xml = REPO / "data" / "menagerie_unitree_g1" / "scene_mjx.xml"
    kin, motion = _synthetic_motion(xml, tmp_path)
    mdp = MDP(kin, motion, mx.array(default_joint_pos_list(), dtype=mx.float32))
    env = MPSVectorEnv(xml, 1, motion, mdp, seed=0)

    assert env.info.collision_mode == "explicit_pairs"
    assert env.info.explicit_contact_pairs == 49
    assert env.substeps == 5
    assert env.engine.rig.npairc == 49
    assert 96 < env.contact_row_cap < 1024
    assert env.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert env.model.geom_contype[env.floor_id] == env.model.geom_conaffinity[env.floor_id] == 0

    required = ("head_floor", "torso_floor", "pelvis_floor", "left_hand_floor", "right_hand_floor")
    rig_pairs = {
        tuple(sorted((int(geom1), int(geom2))))
        for geom1, geom2 in zip(env.engine.rig.pair_g1, env.engine.rig.pair_g2)
    }
    for name in required:
        pair_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_PAIR, name)
        expected = tuple(sorted((int(env.model.pair_geom1[pair_id]), int(env.model.pair_geom2[pair_id]))))
        assert expected in rig_pairs

    default_q = mx.array(default_joint_pos_list(), dtype=mx.float32)[None, :]
    qpos = mx.concatenate(
        (mx.array([[0.0, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0]], dtype=mx.float32), default_q),
        axis=-1,
    )
    qvel = mx.zeros((1, env.model.nv), dtype=mx.float32)
    env.engine.reset_worlds(mx.array([True]), qpos=qpos, qvel=qvel, ctrl=default_q)
    for substep in range(env.substeps):
        env.engine.step(ctrl=default_q if substep == 0 else None)
    mx.eval(env.engine.ncon, env.engine.overflow)
    assert int(np.asarray(env.engine.ncon)[0]) > 0
    assert int(np.asarray(env.engine.overflow)[0]) == 0
