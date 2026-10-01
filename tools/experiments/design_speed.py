"""Record what the parallel overload Am design stage buys (experiment `generation.parallel_design`):
an Am-only IEEE-14 timeline over the first `--frames` pool frames, generated with each worker count of
`--workers` on one seed, its wall time, the episodes built and whether every file is byte-identical
to the one-worker file (it must be: each design draws from its own keyed stream).

    python tools/experiments/design_speed.py --frames 3000 --workers 1 16

The numbers render in docs/guides/generation.md ("Designing in parallel").
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fdia_graph import schema  # noqa: E402
from fdia_graph.generation import _load_states  # noqa: E402
from fdia_graph.results import Run, Store  # noqa: E402
from fdia_graph.timeline import generate_timeline  # noqa: E402

SEED = 123


def _datasets(path: str) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with h5py.File(path, "r") as f:
        f.visititems(
            lambda n, o: out.__setitem__(n, o[()].tobytes()) if isinstance(o, h5py.Dataset) else None
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 16])
    args = ap.parse_args()
    X = _load_states(14, None)[: args.frames]
    store = Store(os.path.join(ROOT, "results"))
    settings = {"frames": args.frames, "families": "Am", "seed": SEED, "workers": args.workers}
    run = Run("generation.parallel_design", system="ieee14", settings=settings, seed=SEED, store=store)
    with tempfile.TemporaryDirectory() as tmp, run:
        first = None
        for w in args.workers:
            out = os.path.join(tmp, f"am_w{w}.h5")
            t0 = time.time()
            generate_timeline(14, states=X, families=("Am",), seed=SEED, workers=w, out=out)
            run.add("seconds", time.time() - t0, method=f"workers={w}", frames=args.frames)
            with h5py.File(out, "r") as f:
                run.add("episodes", int(np.sum(f.attrs[schema.Attr.EPISODES_BUILT])), method=f"workers={w}")
            data = _datasets(out)
            first = first or data
            run.add("identical", float(data == first), method=f"workers={w}")
            print(
                f"workers {w}: {time.time() - t0:.0f} s, identical to the first: {data == first}", flush=True
            )


# the workers start by importing this module: the guard keeps them from running it again
if __name__ == "__main__":
    main()
