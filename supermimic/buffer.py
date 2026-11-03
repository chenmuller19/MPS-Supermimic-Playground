"""Fixed-size GPU replay buffer of simulator transitions, laid out as (time, env)."""

from __future__ import annotations

import torch

from .mdp import STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "clip_end": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(self, T: int, N: int, device: str):
        self.T, self.N, self.device = T, N, device
        self.data = {}
        for k, (shape, dt) in FIELDS.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=device)
        self.ptr, self.size = 0, 0

    def add(self, tr: dict):
        for k in FIELDS:
            self.data[k][self.ptr] = tr[k]
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def state_dict(self) -> dict:
        return {"T": self.T, "N": self.N, "ptr": self.ptr, "size": self.size,
                "data": {name: value.detach().cpu().clone() for name, value in self.data.items()}}

    def load_state_dict(self, state: dict) -> None:
        if int(state["T"]) != self.T or int(state["N"]) != self.N:
            raise ValueError("replay checkpoint dimensions do not match this run")
        for name in FIELDS:
            source = state["data"][name]
            if source.shape != self.data[name].shape or source.dtype != self.data[name].dtype:
                raise ValueError(f"replay field {name!r} does not match checkpoint contract")
            self.data[name].copy_(source.to(self.device))
        self.ptr, self.size = int(state["ptr"]), int(state["size"])

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map age-ordered time index (0 = oldest) to the storage row."""
        return (self.ptr - self.size + age) % self.T

    def sample_windows(self, B: int, k: int):
        """k-step windows; returns dict of (B, k, ...) tensors and a validity mask (B, k)."""
        t0 = torch.randint(0, self.size - k + 1, (B,), device=self.device)
        env = torch.randint(0, self.N, (B,), device=self.device)
        ages = t0.unsqueeze(-1) + torch.arange(k, device=self.device)
        rows = self._row(ages)
        e = env.unsqueeze(-1).expand_as(rows)
        out = {key: self.data[key][rows, e] for key in ("s", "s1", "a", "xi", "done")}
        # transition j is valid if no episode ended at transitions 0..j-1
        d = out["done"].float()
        mask = torch.cumprod(torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1)
        out["mask"] = mask
        return out

    def sample_starts(self, B: int, hist: int, horizon: int | None = None):
        """Start points for imagination plus their observation history.

        When ``horizon`` is provided, reject starts too close to the recorded clip end so
        imagined reference indexing stays inside the motion table.
        """
        if self.size == 0:
            raise ValueError("cannot sample starts from an empty replay")
        if horizon is not None:
            ages = torch.arange(self.size, device=self.device)
            rows = self._row(ages)
            valid = self.data["f"][rows] + horizon <= self.data["clip_end"][rows]
            candidates = valid.nonzero(as_tuple=False)
            if candidates.numel() == 0:
                raise ValueError("replay contains no valid starts for requested imagination horizon")
            chosen = torch.randint(candidates.shape[0], (B,), device=self.device)
            age, env = candidates[chosen].unbind(-1)
        else:
            age = torch.randint(0, self.size, (B,), device=self.device)
            env = torch.randint(0, self.N, (B,), device=self.device)
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=self.device)  # hist-1 ... 0
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))  # stay inside the episode
        back = torch.minimum(back, age.unsqueeze(-1))  # stay inside the buffer
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        start = {k: self.data[k][row, env] for k in ("s", "a_prev", "xi", "f", "clip_end")}
        hist_d = {k: self.data[k][hrows, henv] for k in ("s", "a_prev", "xi", "f", "clip_end")}
        return start, hist_d
