"""Where the inadvertent interchange enters New York, and what it does to the other interfaces (N72): the trusting
operator rerun on every attack day the damage study wrote, once per entry pattern of `response.ENTRY`, one row per
day file and pattern (`summarize`, plus the variant and the overshoot).

    python tools/nyfed/entry_sensitivity.py OUT.csv out/damage_study out/damage_study_b [--dispatch cost]
                                             [--workers 8]

Run from the folder whose `data/` holds the inputs.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from pathlib import Path

from fdia_graph.nyfed.experiments import add_study_arguments, load_calib, run_parallel, study_days, variant
from fdia_graph.nyfed.operator.response import ENTRY, operator_day, respond, summarize


def _job(
    path: str,
    pool: str,
    data: str,
    calib: tuple[float, float] | None,
    entries: list[str],
    closed_loop: bool = False,
    dispatch: str = "movement",
) -> list[dict]:
    day = operator_day(Path(path), pool, data)
    onset, overshoot = Path(path).stem.split("_")[1:3]
    rows = []
    for entry in entries:
        s = summarize(
            respond(
                day, mode="trusting", calib=calib, entry=entry, closed_loop=closed_loop, dispatch=dispatch
            )
        )
        rows.append(
            {
                "file": Path(path).name,
                "onset": int(onset),
                "overshoot_mw": float(overshoot),
                "variant": variant(path),
                "entry": entry,
                "dispatch": dispatch,
                **s,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    add_study_arguments(ap)
    ap.add_argument("--closed-loop", action="store_true", help="the operator reads the true flows (N76)")
    ap.add_argument("--entries", nargs="*", default=list(ENTRY), help="the patterns of `ENTRY` to run")
    args = ap.parse_args()
    calib = load_calib(args.calib)
    files = study_days(args.studies, args.limit)
    print(f"{len(files)} attack days, {len(args.entries)} entry patterns", flush=True)
    tasks = [(f, args.pool, args.data, calib, args.entries, args.closed_loop, args.dispatch) for f in files]
    run_parallel(_job, tasks, args.out, workers=args.workers, every=10)


if __name__ == "__main__":
    main()
