"""Build the release timeline (`fdia_graph.generation.nygrid.build.production_builder`) from a pool.

    python tools/nyfed/build_timeline.py OUT.h5 --pool out/pool_2019_v3.h5 [--families Am,At]
                                         [--interface SPR/DUN-SOUTH] [--workers 12] [--survey-cache DIR]
    python tools/nyfed/build_timeline.py --fetch      # download the inputs under ./data first

Run from the folder whose `data/` holds the inputs (`fetch_inputs`); the summary is written next to OUT
as JSON.
"""

from __future__ import annotations

import os

# One BLAS thread per process: the workers inherit it, and many threaded workers oversubscribe the machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from pathlib import Path

from fdia_graph.generation.nygrid.build import fetch_inputs, production_builder


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--families", default="Am,At")
    ap.add_argument(
        "--interface",
        default="SPR/DUN-SOUTH",
        help="comma list of NYISO interfaces; empty targets branch ratings",
    )
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--survey", type=int, default=12, help="onset survey grid in frames, 0 off")
    ap.add_argument("--survey-cache", default=None, help="folder that keeps the survey by chunk (resumable)")
    ap.add_argument("--fetch", action="store_true", help="download the inputs under ./data and stop")
    args = ap.parse_args()
    if args.fetch:
        print(fetch_inputs("."), flush=True)
        return
    if not args.out:
        ap.error("OUT is required unless --fetch")
    t0 = time.perf_counter()
    b = production_builder(
        args.pool,
        tuple(args.families.split(",")),
        interfaces=tuple(x for x in args.interface.split(",") if x),
        workers=args.workers,
        survey=args.survey,
        survey_cache=args.survey_cache,
    )
    summary = b.build(Path(args.out))
    summary["run_seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(summary), flush=True)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
