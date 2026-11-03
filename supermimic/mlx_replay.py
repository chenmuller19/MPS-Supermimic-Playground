"""GPU-resident ring replay for MPS samples and MLX learning."""

from __future__ import annotations

import numpy as np
import mlx.core as mx


_ARRAY_FIELDS = ("s", "s1", "a", "a_prev", "xi", "r")
_META_FIELDS = ("f", "clip_end", "ep_step", "done", "episode")


class ReplayBuffer:
    """Time-major ring buffer that retains MLX arrays without copying simulator state to NumPy."""

    def __init__(self, capacity: int, num_envs: int):
        if capacity < 2 or num_envs < 1:
            raise ValueError("replay capacity must be at least 2 and num_envs must be positive")
        self.T, self.N = int(capacity), int(num_envs)
        self.rows: list[dict | None] = [None] * self.T
        self.meta = {
            "f": np.zeros((self.T, self.N), np.int32),
            "clip_end": np.zeros((self.T, self.N), np.int32),
            "ep_step": np.zeros((self.T, self.N), np.int32),
            "done": np.zeros((self.T, self.N), np.bool_),
            "episode": np.zeros((self.T, self.N), np.int64),
        }
        self.ptr, self.size = 0, 0

    def add(self, transition: dict) -> None:
        row = {}
        for key in _ARRAY_FIELDS:
            value = mx.stop_gradient(mx.array(transition[key]))
            if value.shape[0] != self.N:
                raise ValueError(f"replay field {key!r} must have leading dimension {self.N}")
            row[key] = value
        for key in _META_FIELDS:
            value = np.asarray(transition[key])
            if value.shape != (self.N,):
                raise ValueError(f"replay metadata {key!r} must have shape {(self.N,)}")
            self.meta[key][self.ptr] = value
        self.rows[self.ptr] = row
        mx.eval(*row.values())
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def _physical(self, age):
        return (self.ptr - self.size + np.asarray(age)) % self.T

    def _gather(self, ages: np.ndarray, envs: np.ndarray, field: str) -> mx.array:
        ages = np.asarray(ages).reshape(-1)
        envs = np.asarray(envs).reshape(-1)
        parts, positions = [], []
        for age in np.unique(ages):
            selected = np.flatnonzero(ages == age)
            physical = int(self._physical(age))
            record = self.rows[physical]
            if record is None:
                raise RuntimeError("replay row was unexpectedly empty")
            parts.append(record[field][mx.array(envs[selected], dtype=mx.int32)])
            positions.append(selected)
        joined = mx.concatenate(parts, axis=0)
        order = np.argsort(np.concatenate(positions))
        return joined[mx.array(order, dtype=mx.int32)]

    def _gather_meta(self, ages: np.ndarray, envs: np.ndarray, field: str) -> mx.array:
        return mx.array(self.meta[field][self._physical(ages), envs])

    def _sample_grouped(self, valid_age_env: np.ndarray, batch: int, groups: int = 16):
        valid_counts = valid_age_env.sum(axis=1)
        ages = np.flatnonzero(valid_counts)
        if not len(ages):
            raise ValueError("replay contains no valid samples for the requested window")
        n_groups = min(max(1, int(groups)), batch)
        chosen = np.random.choice(ages, n_groups, replace=n_groups > len(ages), p=valid_counts[ages] / valid_counts[ages].sum())
        quotas = np.full(n_groups, batch // n_groups, np.int32)
        quotas[: batch % n_groups] += 1
        samples_age, samples_env = [], []
        for age, count in zip(chosen, quotas):
            envs = np.flatnonzero(valid_age_env[age])
            samples_age.extend([int(age)] * int(count))
            samples_env.extend(np.random.choice(envs, int(count), replace=int(count) > len(envs)).tolist())
        return np.asarray(samples_age, np.int32), np.asarray(samples_env, np.int32)

    def sample_windows(self, batch: int, window: int, groups: int = 16) -> dict:
        if self.size < window:
            raise ValueError(f"replay has {self.size} rows; need {window} for a world-model window")
        max_age = self.size - window
        ages = np.arange(max_age + 1, dtype=np.int32)
        env_mask = np.ones((len(ages), self.N), np.bool_)
        start, env = self._sample_grouped(env_mask, batch, groups)
        seq_ages = start[:, None] + np.arange(window, dtype=np.int32)[None, :]
        seq_env = np.broadcast_to(env[:, None], seq_ages.shape)
        out = {}
        for field in ("s", "s1", "a", "xi"):
            flat = self._gather(seq_ages, seq_env, field)
            out[field] = flat.reshape((batch, window) + flat.shape[1:])
        out["done"] = self._gather_meta(seq_ages, seq_env, "done")
        alive = mx.ones((batch,), dtype=mx.float32)
        masks = [alive]
        for t in range(window - 1):
            alive = alive * (1 - out["done"][:, t].astype(mx.float32))
            masks.append(alive)
        out["mask"] = mx.stack(masks, axis=1)
        return out

    def sample_starts(self, batch: int, hist: int, horizon: int | None = None, groups: int = 16):
        if self.size == 0:
            raise ValueError("cannot sample starts from an empty replay")
        age_rows = self._physical(np.arange(self.size, dtype=np.int32))
        valid = np.ones((self.size, self.N), np.bool_)
        if horizon is not None:
            frame = self.meta["f"][age_rows]
            clip_end = self.meta["clip_end"][age_rows]
            valid &= frame + horizon <= clip_end
        ages, envs = self._sample_grouped(valid, batch, groups)
        physical = self._physical(ages)
        ep_step = self.meta["ep_step"][physical, envs]
        back = np.minimum(np.arange(hist - 1, -1, -1, dtype=np.int32)[None], ep_step[:, None])
        back = np.minimum(back, ages[:, None])
        hist_ages = ages[:, None] - back
        hist_envs = np.broadcast_to(envs[:, None], hist_ages.shape)

        start = {
            "s": self._gather(ages, envs, "s"),
            "a_prev": self._gather(ages, envs, "a_prev"),
            "xi": self._gather(ages, envs, "xi"),
            "f": self._gather_meta(ages, envs, "f"),
            "clip_end": self._gather_meta(ages, envs, "clip_end"),
        }
        history = {}
        for key in ("s", "a_prev", "xi"):
            values = self._gather(hist_ages, hist_envs, key)
            history[key] = values.reshape((batch, hist) + values.shape[1:])
        for key in ("f",):
            history[key] = self._gather_meta(hist_ages, hist_envs, key).reshape((batch, hist))
        return start, history

    def state_dict(self) -> dict:
        return {
            "T": self.T,
            "N": self.N,
            "ptr": self.ptr,
            "size": self.size,
            "rows": self.rows,
            "meta": self.meta,
        }

    def load_state_dict(self, state: dict) -> None:
        if int(state["T"]) != self.T or int(state["N"]) != self.N:
            raise ValueError("replay checkpoint dimensions do not match this run")
        if len(state["rows"]) != self.T:
            raise ValueError("replay checkpoint row count does not match capacity")
        self.rows = state["rows"]
        for key in _META_FIELDS:
            source = np.asarray(state["meta"][key])
            if source.shape != self.meta[key].shape:
                raise ValueError(f"replay metadata {key!r} does not match checkpoint dimensions")
            self.meta[key][...] = source
        self.ptr, self.size = int(state["ptr"]), int(state["size"])
