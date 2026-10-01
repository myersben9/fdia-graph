"""Record a data release's statistics in the results store (experiment `data.release_stats`), the
numbers docs/reference/EXAMPLES.md renders in its "Dataset statistics" section.

    python tools/experiments/release_stats.py --release v0.8.3

Per system, read from the release's timeline: the frames, buses, branches and episodes, and per
split the frames and the frames of each attack family (benign included). The operating-state
distributions come from `docs/figures/fig_dataset_stats.csv`, the data behind the figure of that
section, as quantiles per system. Downloads the timelines it does not have cached.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

import fdia_graph as fg  # noqa: E402
from fdia_graph import schema  # noqa: E402
from fdia_graph.download import ensure_local  # noqa: E402
from fdia_graph.registry import resolve  # noqa: E402
from fdia_graph.results import Run, Store  # noqa: E402

SYSTEMS = ("ieee14", "ieee30", "ieee57", "ieee89", "ieee118", "ieee145", "ieee200", "ieee300")
SPLITS = ("train", "val", "test")
STATS = os.path.join(ROOT, "docs", "figures", "fig_dataset_stats.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--release", default="v0.8.3")
    args = ap.parse_args()
    store = Store(os.path.join(ROOT, "results"))
    with Run(
        "data.release_stats", data_release=args.release, settings={"release": args.release}, store=store
    ) as run:
        for system in SYSTEMS:
            with h5py.File(ensure_local(resolve(system, release=args.release)), "r") as f:
                split = f[schema.SPLIT][:]
                family = f[schema.FAMILY][:]
                seq = f[schema.SEQ_ID][:]
                run.add("frames", len(split), system=system, split="all")
                run.add("buses", f[schema.NODE_X].shape[1], system=system)
                run.add("branches", f[schema.EDGE_X].shape[1], system=system)
                run.add("episodes", len(np.unique(seq[seq >= 0])), system=system, split="all")
                for code, name in enumerate(SPLITS):
                    inside = split == code
                    run.add("frames", int(inside.sum()), system=system, split=name)
                    for fid, fam in fg.FAMILIES.items():
                        run.add(
                            "frames",
                            int((inside & (family == fid)).sum()),
                            system=system,
                            split=name,
                            family=fam,
                        )
        with open(STATS, newline="") as f:
            for row in csv.DictReader(f):
                for q in ("p1", "p25", "p50", "p75", "p99", "min", "max"):
                    run.add(
                        "quantile",
                        float(row[q]),
                        system=row["system"],
                        family="",
                        quantity=row["quantity"],
                        q=q,
                    )


if __name__ == "__main__":
    main()
