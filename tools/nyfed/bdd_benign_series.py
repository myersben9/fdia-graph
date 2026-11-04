"""The benign weighted residual scan by scan (N79): for days drawn from a span of 2019, every solved five-minute scan
drawn with the plan's own meter noise (its constant biases and a fresh jitter) and fitted by AC weighted least
squares, so a persistence rule on the residual test can be calibrated against whole benign days.

    python tools/nyfed/bdd_benign_series.py OUT.csv --days 40 --months 1-3 [--workers 8]

Run from the folder whose `data/` holds the inputs. One row per scan: the day, the scan of the day and the residual.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from fdia_graph.nyfed.detect import reference_bus, with_fleet, wls
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.io import read_pool
from fdia_graph.nyfed.operator.response import operator_day
from fdia_graph.nyfed.scenario.pool import SOLVED
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.noise import NoiseModel

_CTX: dict = {}


def _init(day_path: str, pool_path: str, data: str, draw: int = 0) -> None:
    day = operator_day(Path(day_path), pool_path, data)  # any attack day carries the case and the meter plan
    plan = with_fleet(day.plan, draw)  # draw 0 the plan's own meter fleet, otherwise fresh constant biases
    _CTX["meas"] = Measurement(Network(day.case), plan)
    _CTX["noise"] = NoiseModel(plan)
    _CTX["ref"] = reference_bus(day.case)
    _CTX["pool"] = read_pool(pool_path)


def _job(day: str, frames: list[int], seed: int) -> list[dict]:
    meas, noise, ref, pool = _CTX["meas"], _CTX["noise"], _CTX["ref"], _CTX["pool"]
    g = np.random.default_rng(seed)
    rows = []
    for k, f in enumerate(frames):
        vm, va = pool.vm[f], pool.va[f]
        if not np.isfinite(vm).all():
            rows.append({"day": day, "scan": k, "j": np.nan})
            continue
        z = noise.sample(meas.h(vm, va), g)
        rows.append({"day": day, "scan": k, "j": wls(meas, z, noise.sigmas(z), vm, va, ref).j})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default="data/NYgrid/Data")
    ap.add_argument("--days", type=int, default=40)
    ap.add_argument("--months", default="1-3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20190301)
    ap.add_argument("--draw", type=int, default=0, help="0 the plan's meter biases, otherwise a fresh fleet")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.months.split("-"))
    pool = read_pool(args.pool)
    t = pool.time.astype("datetime64[s]")
    month = t.astype("datetime64[M]").astype(int) % 12 + 1
    days = np.unique(t[(month >= lo) & (month <= hi)].astype("datetime64[D]"))
    pick = (
        days
        if args.days <= 0
        else np.random.default_rng(args.seed).choice(days, size=min(args.days, len(days)), replace=False)
    )
    any_day = sorted(glob.glob("out/damage_study/days/am_*.h5"))[0]
    jobs = []
    for i, d in enumerate(sorted(pick)):
        frames = np.flatnonzero((t.astype("datetime64[D]") == d) & (pool.solve_kind == SOLVED)).tolist()
        jobs.append((str(d), frames, args.seed + i))
    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(any_day, args.pool, args.data, args.draw)
    ) as ex:
        futs = [ex.submit(_job, *j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            rows += fut.result()
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"{k}/{len(jobs)} [{time.perf_counter() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
