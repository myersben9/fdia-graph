"""A copy of a timeline with a set of meters secured: the attacker locked out of them, so on every
frame their observed reading is the benign one, and the stored temporal features recomputed from
the pinned observed frames (`timeline.write_temporal_layers`). The estimators and the localizers then read the copy like any timeline, which is
how a trusted-meter selection is measured on state estimation and localization rather than on the
residual test alone (`TrustSelector.score`). The copy is rewritten in blocks of frames, so a
72,000-frame timeline of any ladder system fits in bounded memory."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Optional

import h5py
import numpy as np

from .. import schema
from ..formulas.projection import meter_positions
from ..timeline import write_temporal_layers

if TYPE_CHECKING:
    from ..dataset import FdiaGraph
    from .base import TrustSelector

BLOCK = 2000  # frames rewritten at a time


def secured_copy(
    selector: TrustSelector, ds: FdiaGraph, out: str, name: Optional[str] = None, k: Optional[int] = None
) -> str:
    """Write `out`, a copy of the timeline behind `ds` in which the first `k` meters the fitted
    `selector` chose (default: all of them) read their benign value on every frame, with the tamper
    masks and the two stored temporal features following. Registers the file under `name` when
    given, so `fg.load(name)` reads it. Returns the path."""
    est = selector.est
    return _secured_copy(ds, est.E, est.mask, selector.select(k), out, name)


def _secured_copy(
    ds: FdiaGraph, n_branch: int, mask: np.ndarray, meters: np.ndarray, out: str, name: Optional[str]
) -> str:
    """The copy for `meters` given as indices into an estimator's masked measurement vector, `mask`
    its [4N + 2E] slot mask and `n_branch` its E."""
    ds.require("benign_layer", by="secured_copy")
    nodes, edges = meter_positions(ds.N, n_branch, mask, np.asarray(meters, int))
    shutil.copyfile(ds.path, out)
    with h5py.File(out, "r+") as f:
        _pin(f, nodes, edges)
        write_temporal_layers(f, BLOCK)
    if name is not None:
        from ..registry import register_local

        register_local(name, out, meta={"secured": [int(m) for m in meters], "source": ds.path})
    return out


def _pin(f: h5py.File, nodes: list[tuple[int, int]], edges: list[tuple[int, int]]) -> None:
    """The secured channels read the benign layer and are no longer tampered, one block of frames
    at a time."""
    if not nodes and not edges:
        return
    layers = [
        (f[schema.NODE_X], f[schema.NODE_BENIGN], f[schema.NODE_TAMPER], nodes),
        (f[schema.EDGE_X], f[schema.EDGE_BENIGN], f[schema.EDGE_TAMPER], edges),
    ]
    T = f[schema.NODE_X].shape[0]
    for a in range(0, T, BLOCK):
        b = min(a + BLOCK, T)
        for observed, benign, tamper, positions in layers:
            if not positions:
                continue
            x, bx, tm = observed[a:b], benign[a:b], tamper[a:b]
            for i, col in positions:
                x[:, i, col] = bx[:, i, col]
                tm[:, i, col] = False
            observed[a:b], tamper[a:b] = x, tm
