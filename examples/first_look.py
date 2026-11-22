#!/usr/bin/env python
"""One overload attack as an operator would see it: the figure at the top of the README.

It generates the 600-frame IEEE-14 timeline of docs/guides/tour.md (one overload episode), reads the
flow meters of the two branches the episode targets, and plots what the meters would have read
without the attack against what they reported, with each branch's rating.

Run from the repository root: python examples/first_look.py [out.png]
(needs fdia-graph[generate] and matplotlib; downloads the IEEE-14 pool on first use, and registers the
timeline as `tour` in the local cache, as examples/tour.py does with the same settings)
"""

from __future__ import annotations

import sys

import h5py
import matplotlib

matplotlib.use("Agg")  # a file is written; no window is opened
import matplotlib.pyplot as plt
import numpy as np

import fdia_graph as fg
from fdia_graph import schema

FRAMES, SEED = 600, 123
BEFORE, AFTER = 40, 40  # frames shown on each side of the attack window
INK, MUTED, ATTACK = "#111111", "#6b7280", "#D22030"  # black for the flow without the attack, red for the reported one


def apparent(flows: np.ndarray) -> np.ndarray:
    """|S| in MVA of from-end flows [..., 2] = (P MW, Q MVAr)."""
    return np.hypot(flows[..., 0], flows[..., 1])


def main(out: str) -> None:
    path = fg.generate("ieee14", name="tour", frames=FRAMES, seed=SEED, families=("Am",), attacked_frac=0.2)
    with h5py.File(path, "r") as f:
        episodes = f[schema.Group.EPISODES]  # this seed places one episode, an overload on two branches
        onset, length = int(episodes[schema.EPISODE_ONSET][0]), int(episodes[schema.EPISODE_LENGTH][0])
        lines = episodes[schema.EPISODE_AM_LINE][:].astype(int)
        ratings = episodes[schema.EPISODE_AM_RATING][:]
        frames = slice(onset - BEFORE, onset + length + AFTER)
        reported = apparent(f[schema.EDGE_X][frames][:, lines])
        without = apparent(f[schema.EDGE_BENIGN][frames][:, lines])
        ends = f[schema.EDGE_INDEX][:][:, lines].T
    t = np.arange(frames.start, frames.stop)

    fig, axes = plt.subplots(1, len(lines), figsize=(9.0, 3.6), dpi=160, sharex=True)
    for ax, k in zip(axes, range(len(lines))):
        ax.axvspan(onset, onset + length - 1, color=ATTACK, alpha=0.07, lw=0)
        ax.axhline(ratings[k], color=MUTED, lw=1.0, ls=(0, (1, 3)))
        ax.plot(t, without[:, k], color=INK, lw=1.3, label="without the attack")
        ax.plot(t, reported[:, k], color=ATTACK, lw=1.8, label="reported to the operator")
        ax.text(t[2], ratings[k], "rating", color=MUTED, fontsize=8, va="bottom")
        # IEEE-14 numbers its buses in order from one, so a bus number is its position plus one
        ax.set_title(f"Branch {int(ends[k, 0]) + 1} to {int(ends[k, 1]) + 1}", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("frame", fontsize=9, color=INK)
        ax.tick_params(labelsize=8, colors=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("apparent flow (MVA)", fontsize=9, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    fig.suptitle(
        "An overload attack on IEEE-14: the meters report a flow climbing to the rating",
        fontsize=11, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}: frames {frames.start} to {frames.stop - 1}, attack {onset} to {onset + length - 1}")
    print(f"reported flow at the last attacked frame: {np.round(reported[BEFORE + length - 1], 2).tolist()} MVA")
    print(f"ratings: {np.round(ratings, 2).tolist()} MVA")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docs/figures/first_look.png")
