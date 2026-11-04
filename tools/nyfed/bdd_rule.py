"""The residual test with a persistence rule, and the operator's answers to its alarm (N79): alarm when `k` of the last
`n` scans of the closed-loop residual exceed `theta`, the threshold fitted on benign January to March days
(`bdd_benign_series.py`) so that no fit day raises a false alarm. It reads the per-scan residuals that
`closed_loop_bdd.py` wrote, finds the first alarm of every attack window and of its benign twin, and reruns the operator
answering the alarm three ways (`respond(after=...)`).

    python tools/nyfed/bdd_rule.py OUT.csv IN.csv [--theta 428.7 --k 2 --n 3] [--mode trusting]
                                   [--dispatch cost] [--workers 8]

Run from the folder whose `data/` holds the inputs.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fdia_graph.nyfed.detect import PersistenceRule
from fdia_graph.nyfed.experiments import add_study_arguments, load_calib, run_parallel
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


def _job(
    row: dict, study_days: list[str], pool: str, data: str, calib, mode: str, rule, dispatch: str = "movement"
) -> dict:
    path = next(str(Path(s) / row["file"]) for s in study_days if (Path(s) / row["file"]).exists())
    day = operator_day(Path(path), pool, data)
    first = rule.first_alarm(np.array(json.loads(row["j_attacked"]), float))
    benign_first = rule.first_alarm(np.array(json.loads(row["j_benign"]), float))
    out = {
        "file": row["file"],
        "variant": row["variant"],
        "mode": mode,
        "dispatch": dispatch,
        "first": first,
        "benign_first": benign_first,
    }
    full = summarize(respond(day, mode=mode, calib=calib, closed_loop=True, dispatch=dispatch))
    out.update({f"full_{x}": full[x] for x in KEEP})
    if first >= 0:
        for after in AFTER:
            s = summarize(
                respond(
                    day,
                    mode=mode,
                    calib=calib,
                    closed_loop=True,
                    alarm_at=day.onset + first,
                    after=after,
                    dispatch=dispatch,
                )
            )
            out.update({f"{after}_{x}": s[x] for x in KEEP})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    add_study_arguments(ap, studies=False)
    ap.add_argument("inp", help="closed_loop_bdd.py output with the per-scan residuals")
    ap.add_argument("--studies", nargs="+", default=["out/damage_study", "out/damage_study_b"])
    ap.add_argument("--mode", default="trusting")
    ap.add_argument("--theta", type=float, default=428.7)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    calib = load_calib(args.calib)
    rule = PersistenceRule(args.theta, args.k, args.n)
    inp = pd.read_csv(args.inp).drop_duplicates("file")
    # the residuals follow the operator's own trajectory, so they must come from the dispatch asked for here; a CSV
    # written before the option existed is the least-movement one
    found = set(inp["dispatch"]) if "dispatch" in inp else {"movement"}
    if found != {args.dispatch}:
        raise SystemExit(f"{args.inp} holds residuals of the {sorted(found)} dispatch, not {args.dispatch!r}")
    rows = inp.to_dict("records")
    if args.limit:
        rows = rows[: args.limit]
    days = [str(Path(s) / "days") for s in args.studies]
    tasks = [(r, days, args.pool, args.data, calib, args.mode, rule, args.dispatch) for r in rows]
    run_parallel(_job, tasks, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
