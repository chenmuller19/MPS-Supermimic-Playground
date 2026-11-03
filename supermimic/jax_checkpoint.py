"""Orbax checkpoint helpers for the pure-JAX SuperMimic learner."""

from __future__ import annotations

from pathlib import Path

import orbax.checkpoint as ocp


def save(path: str | Path, state) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ocp.PyTreeCheckpointer().save(str(path), state, force=True)


def restore(path: str | Path, item=None):
    path = Path(path).expanduser().resolve()
    return ocp.PyTreeCheckpointer().restore(str(path), item=item)
