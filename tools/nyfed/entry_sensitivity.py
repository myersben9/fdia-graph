"""Where the inadvertent interchange enters New York, and what it does to the other interfaces (N72): the trusting
operator rerun on every attack day the damage study wrote, once per entry pattern of `response.ENTRY`, one row per
day file and pattern (`summarize`, plus the variant and the overshoot).

    python tools/nyfed/entry_sensitivity.py OUT.csv out/damage_study out/damage_study_b [--workers 8]

Run from the folder whose `data/` holds the inputs.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from fdia_graph.nyfed.operator.response import ENTRY, operator_day, respond, summarize


def _job(
    path: str,
    pool: str,
    data: str,
    calib: tuple[float, float] | None,
    entries: list[str],
    closed_loop: bool = False,
) -> list[dict]:
    day = operator_day(Path(path), pool, data)
    onset, overshoot = Path(path).stem.split("_")[1:3]
    rows = []
    for entry in entries:
        s = summarize(respond(day, mode="trusting", calib=calib, entry=entry, closed_loop=closed_loop))
        rows.append(
            {
                "file": Path(path).name,
                "onset": int(onset),
                "overshoot_mw": float(overshoot),
                "variant": "at_limit" if overshoot == "0" else "damaging",
                "entry": entry,
                **s,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("studies", nargs="+", help="damage_study.py output folders")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default="data/NYgrid/Data")
    ap.add_argument("--calib", default="results/nygrid.tie_check.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--closed-loop", action="store_true", help="the operator reads the true flows (N76)")
    ap.add_argument("--entries", nargs="*", default=list(ENTRY), help="the patterns of `ENTRY` to run")
    args = ap.parse_args()
    cal = json.loads(Path(args.calib).read_text())
    calib = (cal["mean_mw"], cal["sd_mw"])
    files = sorted({str(p) for s in args.studies for p in Path(s, "days").glob("am_*_*.h5")})
    print(f"{len(files)} attack days, {len(ENTRY)} entry patterns", flush=True)
    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [
            ex.submit(_job, f, args.pool, args.data, calib, args.entries, args.closed_loop) for f in files
        ]
        for k, fut in enumerate(as_completed(futs), 1):
            rows += fut.result()
            if k % 10 == 0 or k == len(files):
                pd.DataFrame(rows).to_csv(args.out, index=False)
                print(f"{k}/{len(files)} [{time.perf_counter() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
