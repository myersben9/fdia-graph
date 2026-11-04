"""The residual test with a persistence rule, and the operator's answers to its alarm (N79): alarm when `k` of the last
`n` scans of the closed-loop residual exceed `theta`, the threshold fitted on benign January to March days
(`bdd_benign_series.py`) so that no fit day raises a false alarm. It reads the per-scan residuals that
`closed_loop_bdd.py` wrote, finds the first alarm of every attack window and of its benign twin, and reruns the operator
answering the alarm three ways (`respond(after=...)`).

    python tools/nyfed/bdd_rule.py OUT.csv IN.csv [--theta 428.7 --k 2 --n 3] [--mode trusting] [--workers 8]

Run from the folder whose `data/` holds the inputs.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fdia_graph.nyfed.operator.response import operator_day, respond, summarize

KEEP = (
    "inadvertent_mw",
    "inadvertent_mwh",
    "iface_down_mw",
    "iface_up_mw",
    "other_iface_up_mw",
    "margin_lost_mw",
)
AFTER = ("benign", "freeze", "freeze_cs")


def first_alarm(j: np.ndarray, theta: float, k: int, n: int) -> int:
    """The first scan at which `k` of the last `n` residuals exceed `theta` (-1 none)."""
    over = np.nan_to_num(j, nan=0.0) > theta
    count = np.convolve(over.astype(int), np.ones(n, int))[: len(over)]
    hit = np.flatnonzero(count >= k)
    return int(hit[0]) if len(hit) else -1


def _job(row: dict, study_days: list[str], pool: str, data: str, calib, mode: str, rule) -> dict:
    path = next(str(Path(s) / row["file"]) for s in study_days if (Path(s) / row["file"]).exists())
    day = operator_day(Path(path), pool, data)
    theta, k, n = rule
    first = first_alarm(np.array(json.loads(row["j_attacked"]), float), theta, k, n)
    benign_first = first_alarm(np.array(json.loads(row["j_benign"]), float), theta, k, n)
    out = {
        "file": row["file"],
        "variant": row["variant"],
        "mode": mode,
        "first": first,
        "benign_first": benign_first,
    }
    full = summarize(respond(day, mode=mode, calib=calib, closed_loop=True))
    out.update({f"full_{x}": full[x] for x in KEEP})
    if first >= 0:
        for after in AFTER:
            s = summarize(
                respond(
                    day, mode=mode, calib=calib, closed_loop=True, alarm_at=day.onset + first, after=after
                )
            )
            out.update({f"{after}_{x}": s[x] for x in KEEP})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("inp", help="closed_loop_bdd.py output with the per-scan residuals")
    ap.add_argument("--studies", nargs="+", default=["out/damage_study", "out/damage_study_b"])
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default="data/NYgrid/Data")
    ap.add_argument("--calib", default="results/nygrid.tie_check.json")
    ap.add_argument("--mode", default="trusting")
    ap.add_argument("--theta", type=float, default=428.7)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cal = json.loads(Path(args.calib).read_text())
    calib = (cal["mean_mw"], cal["sd_mw"])
    rows = pd.read_csv(args.inp).drop_duplicates("file").to_dict("records")
    days = [str(Path(s) / "days") for s in args.studies]
    out: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [
            ex.submit(_job, r, days, args.pool, args.data, calib, args.mode, (args.theta, args.k, args.n))
            for r in rows
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            out.append(fut.result())
            pd.DataFrame(out).to_csv(args.out, index=False)
            print(f"{i}/{len(rows)} [{time.perf_counter() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
