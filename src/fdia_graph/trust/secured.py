"""A copy of a timeline with a set of meters secured: the attacker locked out of them, so on every
frame their observed reading is the benign one, and the stored temporal features recomputed from
the pinned scan. The estimators and the localizers then read the copy like any timeline, which is
how a trusted-meter selection is measured on state estimation and localization rather than on the
residual test alone (`TrustSelector.score`)."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import h5py
import numpy as np

from .. import schema
from ..formulas.projection import meter_positions
from ..formulas.temporal import SWING_WINDOW, recent_change_scale, swing_zscore, temporal_delta

if TYPE_CHECKING:
    from ..dataset import FdiaGraph


def secured_copy(
    ds: FdiaGraph,
    n_branch: int,
    mask: np.ndarray,
    meters: Sequence[int] | np.ndarray,
    out: str,
    name: Optional[str] = None,
) -> str:
    """Write `out`, a copy of the timeline behind `ds` in which the meters `meters` (indices into
    the masked measurement vector of a fitted estimator, the layout of `TrustSelector.order`) read
    their benign value on every frame, with the tamper masks and the two stored temporal features
    following. `mask` is the estimator's [4N + 2E] slot mask and `n_branch` its E. Registers the
    file under `name` when given, so `fg.load(name)` reads it. Returns the path."""
    if not ds.has_benign:
        raise ValueError("secured_copy needs a timeline with the benign layer")
    nodes, edges = meter_positions(ds.N, n_branch, mask, np.asarray(meters, int))
    shutil.copyfile(ds.path, out)
    with h5py.File(out, "r+") as f:
        _pin(f, nodes, edges)
        _retemporal(f)
    if name is not None:
        from ..registry import register_local

        register_local(name, out, meta={"secured": [int(m) for m in meters], "source": ds.path})
    return out


def _pin(f: h5py.File, nodes: list[tuple[int, int]], edges: list[tuple[int, int]]) -> None:
    """The secured channels read the benign layer; they are no longer tampered."""
    nx, bnx, tn = f[schema.NODE_X], f[schema.NODE_BENIGN], f[schema.NODE_TAMPER]
    for bus, col in nodes:
        nx[:, bus, col] = bnx[:, bus, col]
        tn[:, bus, col] = False
    ex, bex, te = f[schema.EDGE_X], f[schema.EDGE_BENIGN], f[schema.EDGE_TAMPER]
    for branch, col in edges:
        ex[:, branch, col] = bex[:, branch, col]
        te[:, branch, col] = False


def _retemporal(f: h5py.File) -> None:
    """`temporal_delta` and `swing` from the pinned scan, through the writer's own kernels: against
    the previous emitted frame, the swing scale over the true states (the clean layer is the pool
    the timeline walked, frame for frame). The stored layers are float32 where the writer had
    float64 scans, so an unpinned channel's feature agrees with the stored one to about six digits."""
    nx = np.asarray(f[schema.NODE_X], np.float32)
    scale = recent_change_scale(np.asarray(f[schema.NODE_CLEAN], np.float64), SWING_WINDOW, nx.shape[1])
    every = np.ones(nx.shape[1], bool)
    delta = np.zeros(nx.shape[:2] + (2,), np.float32)
    swing = np.zeros_like(delta)
    for t in range(len(nx)):
        prev = nx[t - 1] if t else nx[t]
        delta[t] = temporal_delta(nx[t], prev, every)
        swing[t] = swing_zscore(nx[t], prev, scale[t], every)
    f[schema.TEMPORAL_DELTA][...] = delta
    f[schema.SWING][...] = swing
