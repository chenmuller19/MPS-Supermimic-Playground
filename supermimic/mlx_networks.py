"""MLX world model, deterministic policy, critic, and running normalizers."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .mlx_geometry import so3_exp, so3_log, yaw_mat
from .mlx_mdp import INV_DIM, NUM_J, XI_DIM, invariant_features

WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J


class RunningNorm:
    """Cumulative population mean/variance, matching the Torch buffer contract."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        self.mean = mx.zeros((dim,), dtype=mx.float32)
        self.var = mx.ones((dim,), dtype=mx.float32)
        self.count = 0.0
        self.max_count = float(max_count)
        self.eps = float(eps)

    def update(self, x: mx.array) -> None:
        if self.count >= self.max_count:
            return
        x = mx.stop_gradient(x.reshape((-1, x.shape[-1])).astype(mx.float32))
        n = x.shape[0]
        if n == 0:
            return
        mean, var = mx.mean(x, axis=0), mx.var(x, axis=0)
        total = self.count + n
        delta = mean - self.mean
        self.var = (self.var * self.count + var * n + delta * delta * self.count * n / total) / total
        self.mean = self.mean + delta * n / total
        self.count = total
        mx.eval(self.mean, self.var)

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.mean) / mx.sqrt(self.var + self.eps**2)

    def std(self) -> mx.array:
        return mx.sqrt(self.var + self.eps**2)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count, "eps": self.eps, "max_count": self.max_count}

    def load_state_dict(self, state: dict) -> None:
        self.mean = mx.array(state["mean"], dtype=mx.float32)
        self.var = mx.array(state["var"], dtype=mx.float32)
        self.count = float(state["count"])
        self.eps = float(state.get("eps", self.eps))
        self.max_count = float(state.get("max_count", self.max_count))


class StableELU(nn.Module):
    """ELU that never exponentiates positive activations during backprop."""

    def __call__(self, x: mx.array) -> mx.array:
        negative = mx.expm1(mx.minimum(x, 0.0))
        return mx.where(x >= 0.0, x, negative)


class MLP(nn.Module):
    def __init__(
        self,
        inp: int,
        hidden: tuple[int, ...] | list[int],
        out: int,
        activation,
        layer_norm: bool = False,
        matmul_dtype=mx.float32,
    ):
        super().__init__()
        layers = []
        width = inp
        for size in hidden:
            layers.append(nn.Linear(width, size))
            if layer_norm:
                layers.append(nn.LayerNorm(size, eps=1e-5))
            layers.append(activation())
            width = size
        layers.append(nn.Linear(width, out))
        self.layers = nn.Sequential(*layers)
        self.matmul_dtype = matmul_dtype

    def __call__(self, x: mx.array) -> mx.array:
        if self.matmul_dtype == mx.float32:
            return self.layers(x)
        for layer in self.layers.layers:
            if isinstance(layer, nn.Linear):
                dtype = self.matmul_dtype
                y = mx.matmul(x.astype(dtype), mx.swapaxes(layer.weight.astype(dtype), -1, -2))
                if layer.bias is not None:
                    y = y + layer.bias.astype(dtype)
                x = y.astype(mx.float32)
            else:
                x = layer(x)
        return x


class WorldModel:
    def __init__(
        self,
        hidden: tuple[int, ...] = (2048, 2048, 2048),
        matmul_dtype=mx.float32,
    ):
        self.in_dim = INV_DIM + NUM_J + XI_DIM
        self.net = MLP(
            self.in_dim, hidden, WM_OUT, nn.SiLU, layer_norm=True, matmul_dtype=matmul_dtype
        )
        output = self.net.layers.layers[-1]
        output.weight = mx.zeros_like(output.weight)
        output.bias = mx.zeros_like(output.bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)

    def inputs(self, state, action: mx.array, xi_n: mx.array) -> mx.array:
        return mx.concatenate((invariant_features(state), action, xi_n), axis=-1)

    @staticmethod
    def targets(state, next_state) -> mx.array:
        p, r, v, w, q, qd = state
        p1, r1, v1, w1, q1, qd1 = next_state
        h = yaw_mat(r)
        rt = mx.swapaxes(r, -1, -2)
        return mx.concatenate(
            ((mx.swapaxes(h, -1, -2) @ (p1 - p)[..., None]).squeeze(-1), so3_log(rt @ r1),
             (rt @ (v1 - v)[..., None]).squeeze(-1), (rt @ (w1 - w)[..., None]).squeeze(-1), q1 - q, qd1 - qd),
            axis=-1,
        )

    def step(self, state, action: mx.array, xi_n: mx.array, net: nn.Module | None = None):
        p, r, v, w, q, qd = state
        network = self.net if net is None else net
        y = network(self.in_norm(self.inputs(state, action, xi_n)))
        d = y * self.out_norm.std() + self.out_norm.mean
        h = yaw_mat(r)
        dp, dth = d[..., :3], d[..., 3:6]
        dv, dw = d[..., 6:9], d[..., 9:12]
        dq, dqd = d[..., 12:41], d[..., 41:70]
        return (
            p + (h @ dp[..., None]).squeeze(-1),
            r @ so3_exp(dth),
            v + (r @ dv[..., None]).squeeze(-1),
            w + (r @ dw[..., None]).squeeze(-1),
            q + dq,
            qd + dqd,
        )

    @staticmethod
    def state_error(predicted, target) -> mx.array:
        p, r, v, w, q, qd = target
        ph, rh, vh, wh, qh, qdh = predicted
        h = yaw_mat(r)
        rt = mx.swapaxes(r, -1, -2)
        return mx.concatenate(
            ((mx.swapaxes(h, -1, -2) @ (ph - p)[..., None]).squeeze(-1), so3_log(rt @ rh),
             (rt @ (vh - v)[..., None]).squeeze(-1), (rt @ (wh - w)[..., None]).squeeze(-1), qh - q, qdh - qd),
            axis=-1,
        )


class Policy(nn.Module):
    """Deterministic latent-goal actor with ten-frame observation history."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        matmul_dtype=mx.float32,
    ):
        super().__init__()
        self.hist = hist
        self.enc = MLP(
            goal_dim, [2048, 1024, 512, 256], z_dim, StableELU, matmul_dtype=matmul_dtype
        )
        self.dec = MLP(
            z_dim + obs_dim * hist,
            [2048, 1024, 512, 256],
            NUM_J,
            StableELU,
            matmul_dtype=matmul_dtype,
        )
        output = self.dec.layers.layers[-1]
        output.weight = output.weight * 0.01
        output.bias = mx.zeros_like(output.bias)
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def encode_goal(self, goal: mx.array) -> mx.array:
        return mx.tanh(self.enc(self.goal_norm(goal)))

    def decode(self, obs_hist: mx.array, encoded_goal: mx.array) -> mx.array:
        obs = self.obs_norm(obs_hist).reshape((obs_hist.shape[0], -1))
        return self.dec(mx.concatenate((encoded_goal, obs), axis=-1))

    def __call__(self, obs_hist: mx.array, goal: mx.array) -> mx.array:
        return self.decode(obs_hist, self.encode_goal(goal))


class Critic(nn.Module):
    def __init__(self, in_dim: int, matmul_dtype=mx.float32):
        super().__init__()
        self.net = MLP(in_dim, [256, 128], 1, StableELU, matmul_dtype=matmul_dtype)
        self.norm = RunningNorm(in_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(self.norm(x)).squeeze(-1)
