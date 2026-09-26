"""The timeline as sequences: sliding windows over a time-ordered view, and the episode table."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import schema
from ..models.choices import (  # noqa: F401  re-exported beside the code that reads them
    Label,
    Layer,
)
from ..models.config import WindowSpec
from ..models.data import EpisodeTable
from .base import DatasetBase


def check_window_args(T: int, W: int, stride: int, label: str) -> None:
    """A window request as its model (`models.config.WindowSpec`), which checks it."""
    WindowSpec(T, W, stride, label)


def window_labels(y: np.ndarray, starts: range, W: int, label: str) -> np.ndarray:
    """Per-window labels: every frame ("frame"), attacked at any frame ("any"), or the last frame."""
    if label == "frame":
        return np.stack([y[s : s + W] for s in starts])
    if label == "last":
        return np.stack([y[s + W - 1] for s in starts])
    return np.stack([y[s : s + W].max(0) for s in starts])


class SequenceMixin(DatasetBase):
    def _check_timeline(self, what: str) -> None:
        """A sequence view needs consecutive frames in time order: a timeline file, `order="time"`,
        and a view that keeps a contiguous span (a split or the whole file, not a family subset)."""
        self.require("timeline", "time_order", "consecutive", by=what)

    def windows(
        self,
        W: int,
        stride: int = 1,
        label: str = "any",
        layer: str = "node_x",
        per_bus: bool = False,
        copy: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a length-W window over this view's frames. Returns (Xw [n, W, N, 4], yw) in self.units.

        label: "frame" -> per-frame per-bus labels yw [n, W, N]; "any" -> window-level per-bus label
        yw [n, N] (bus attacked at ANY frame in the window); "last" -> the label at the final frame.
        layer: the measurement layer windowed, "node_x" (observed), "benign" or "clean".
        per_bus: one sequence per bus instead, Xw [n*N, W, 4] and yw [n*N] (or [n*N, W] per frame),
        what a per-bus recurrent model consumes; wrap in torch.as_tensor for PyTorch.
        copy: False returns Xw as a read-only strided view of the frames, no memory per window (a
        72k-frame IEEE-118 timeline at W=60 is about 8 GB as a copy); per_bus always copies.
        """
        self._check_timeline("windows")
        T = len(self.idx)
        layer = WindowSpec(T, W, stride, label, layer).layer
        a = self.export([layer, "y"])
        nx, y = a[layer], a["y"]
        starts = range(0, T - W + 1, stride)
        # [T-W+1, N, C, W] view -> window axis second -> every stride-th start
        Xw = np.moveaxis(np.lib.stride_tricks.sliding_window_view(nx, W, axis=0), -1, 1)[::stride]
        yw = window_labels(y, starts, W, label)
        if per_bus:  # the reshape can itself be a view of the windows (stride 1), so own the memory
            Xb, yb = per_bus_sequences(Xw, yw, label)
            return np.require(Xb, requirements=["O", "W"]), yb
        return (np.array(Xw) if copy else Xw), yw

    @property
    def episodes(self) -> EpisodeTable:
        """The attack episodes whose onset falls in this view: onset (a file row), length, family,
        and the attacked buses of each. Timeline files only."""
        if self._episodes is None:
            raise AttributeError(f"{self.path} has no episodes (a record shard, not a timeline)")
        e = self._episodes
        keep = np.isin(e.onset, self.idx)
        return EpisodeTable(
            onset=e.onset[keep],
            length=e.length[keep],
            family=e.family[keep],
            buses=[b for b, k in zip(e.buses, keep) if k],
        )


def per_bus_sequences(Xw: np.ndarray, yw: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Whole-grid windows [n, W, N, C] as one sequence per bus [n*N, W, C]; per-frame labels keep
    the window axis ([n*N, W]), window labels flatten to [n*N]."""
    n, W, N, C = Xw.shape
    X = Xw.transpose(0, 2, 1, 3).reshape(n * N, W, C)
    y = yw.transpose(0, 2, 1).reshape(n * N, W) if label == "frame" else yw.reshape(n * N)
    return X, y


def read_episodes(f, has_group: bool) -> Optional[EpisodeTable]:
    """The episodes/ group as a table, or None when the file has none."""
    if not has_group:
        return None
    g = f[schema.Group.EPISODES]
    ptr, idx = g[schema.EPISODE_BUS_PTR][:], g[schema.EPISODE_BUS_IDX][:]
    return EpisodeTable(
        onset=g[schema.EPISODE_ONSET][:].astype(np.int64),
        length=g[schema.EPISODE_LENGTH][:].astype(np.int64),
        family=g[schema.EPISODE_FAMILY][:].astype(np.int64),
        buses=[idx[ptr[k] : ptr[k + 1]].astype(np.int64) for k in range(len(ptr) - 1)],
    )
