"""The timeline as sequences: sliding windows over a time-ordered view, and the episode table."""

from __future__ import annotations

import numbers
from typing import Optional

import numpy as np

from ..models.data import EpisodeTable
from .base import DatasetBase


def check_window_args(T: int, W: int, stride: int, label: str) -> None:
    """Reject a window request the docstring of `windows` does not allow, before any slicing."""
    if label not in ("frame", "any", "last"):
        raise ValueError(f"label must be 'frame', 'any' or 'last', got {label!r}")
    integral = all(isinstance(v, numbers.Integral) and not isinstance(v, bool) for v in (W, stride))
    if not integral or not 1 <= W <= T or stride < 1:
        raise ValueError(
            f"need integers 1 <= W <= {T} frames and stride >= 1, got W={W!r}, stride={stride!r}"
        )


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
        if not self.is_timeline:
            raise ValueError(f"{what} needs a timeline file; {self.path} is a record shard")
        if self._perm is not None:
            raise ValueError(f"{what} needs order='time'; this view is a random permutation")
        if len(self.idx) and not np.all(np.diff(self.idx) == 1):
            raise ValueError(f"{what} needs consecutive frames; a families= or heldout= view is not")

    def windows(
        self, W: int, stride: int = 1, label: str = "any", layer: str = "node_x"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slide a length-W window over this view's frames. Returns (Xw [n, W, N, 4], yw) in self.units.

        label: "frame" -> per-frame per-bus labels yw [n, W, N]; "any" -> window-level per-bus label
        yw [n, N] (bus attacked at ANY frame in the window); "last" -> the label at the final frame.
        layer: the measurement layer windowed, "node_x" (observed), "benign" or "clean".
        """
        self._check_timeline("windows")
        if layer not in ("node_x", "benign", "clean"):
            raise ValueError(f"layer must be 'node_x', 'benign' or 'clean', got {layer!r}")
        T = len(self.idx)
        check_window_args(T, W, stride, label)
        a = self.to_numpy([layer, "y"])
        nx, y = a[layer], a["y"]
        starts = range(0, T - W + 1, stride)
        return np.stack([nx[s : s + W] for s in starts]), window_labels(y, starts, W, label)

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


def read_episodes(f, has_group: bool) -> Optional[EpisodeTable]:
    """The episodes/ group as a table, or None when the file has none."""
    if not has_group:
        return None
    g = f["episodes"]
    ptr, idx = g["bus_ptr"][:], g["bus_idx"][:]
    return EpisodeTable(
        onset=g["onset"][:].astype(np.int64),
        length=g["length"][:].astype(np.int64),
        family=g["family"][:].astype(np.int64),
        buses=[idx[ptr[k] : ptr[k + 1]].astype(np.int64) for k in range(len(ptr) - 1)],
    )
