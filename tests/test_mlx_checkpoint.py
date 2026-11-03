import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import supermimic.mlx_checkpoint as checkpoint
from supermimic.mlx_networks import RunningNorm


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def __call__(self, value):
        return self.linear(value)


class _TinyWorldModel:
    def __init__(self):
        self.net = _TinyModule()
        self.in_norm = RunningNorm(2)
        self.out_norm = RunningNorm(2)


class _TinyPolicy(_TinyModule):
    def __init__(self):
        super().__init__()
        self.obs_norm = RunningNorm(2)
        self.goal_norm = RunningNorm(2)


class _TinyCritic(_TinyModule):
    def __init__(self):
        super().__init__()
        self.norm = RunningNorm(2)


def _models():
    return _TinyWorldModel(), _TinyPolicy(), _TinyCritic()


def _args():
    return SimpleNamespace(
        xml="/sim/scene_mjx.xml",
        substeps=5,
        physics={"collision_mode": "explicit_pairs", "explicit_contact_pairs": 49},
    )


def _norms(models):
    wm, policy, critic = models
    return (wm.in_norm, wm.out_norm, policy.obs_norm, policy.goal_norm, critic.norm)


def test_snapshot_saves_are_immutable_and_latest_resolves(tmp_path):
    models = _models()
    for index, norm in enumerate(_norms(models)):
        norm.mean = mx.array([index + 1, index + 2], dtype=mx.float32)
        norm.var = mx.array([index + 3, index + 4], dtype=mx.float32)
        norm.count = float(index + 10)
    expected_norms = [
        (np.asarray(norm.mean).copy(), np.asarray(norm.var).copy(), norm.count)
        for norm in _norms(models)
    ]
    expected_first_bias = np.asarray(models[1].linear.bias).copy()

    first = checkpoint.save_checkpoint_snapshot(tmp_path, 20, 2.5, _args(), *models)
    first_model, _ = checkpoint.resolve_checkpoint_files(first)
    first_bytes = first_model.read_bytes()
    models[1].linear.bias = mx.array([3.0, 4.0], dtype=mx.float32)
    second = checkpoint.save_checkpoint_snapshot(tmp_path, 40, 5.0, _args(), *models)

    assert first != second
    assert first_model.read_bytes() == first_bytes
    latest_model, latest_metadata = checkpoint.resolve_checkpoint_files(tmp_path)
    assert latest_model == second / checkpoint.MODEL_FILENAME
    assert json.loads(latest_metadata.read_text(encoding="utf-8"))["iteration"] == 40
    assert checkpoint.resolve_checkpoint_files(first)[0] == first_model

    restored = _models()
    checkpoint.load_models(first_model, *restored)
    mx.eval(*[norm.mean for norm in _norms(restored)])
    for actual, (mean, var, count) in zip(_norms(restored), expected_norms):
        np.testing.assert_array_equal(np.asarray(actual.mean), mean)
        np.testing.assert_array_equal(np.asarray(actual.var), var)
        assert actual.count == count
    np.testing.assert_array_equal(np.asarray(restored[1].linear.bias), expected_first_bias)


def test_snapshot_failure_does_not_advance_latest(tmp_path, monkeypatch):
    models = _models()
    first = checkpoint.save_checkpoint_snapshot(tmp_path, 10, 1.0, _args(), *models)
    latest_before = (tmp_path / "latest").resolve()

    def fail_save(*_args, **_kwargs):
        raise OSError("simulated checkpoint write failure")

    monkeypatch.setattr(checkpoint, "save_models", fail_save)
    with pytest.raises(OSError, match="simulated checkpoint write failure"):
        checkpoint.save_checkpoint_snapshot(tmp_path, 20, 2.0, _args(), *models)

    assert (tmp_path / "latest").resolve() == latest_before == first
    committed = [path for path in (tmp_path / "checkpoints").iterdir() if not path.name.startswith(".")]
    assert committed == [first]


def test_legacy_flat_checkpoint_resolution(tmp_path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    models = _models()
    model_path = legacy / checkpoint.MODEL_FILENAME
    checkpoint.save_models(model_path, *models)
    metadata_path = legacy / checkpoint.METADATA_FILENAME
    metadata_path.write_text(json.dumps({"kind": "inference-only-model"}), encoding="utf-8")

    assert checkpoint.resolve_checkpoint_files(legacy) == (model_path, metadata_path)
    assert checkpoint.resolve_checkpoint_files(model_path) == (model_path, metadata_path)
