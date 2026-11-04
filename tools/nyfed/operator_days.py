"""The operator half of the damage study on attack days already built: every day file of each study runs through
each operator, and one row per day and operator gets `summarize`'s consequence with the attack's overshoot and
tampered meters, the columns of `damage_study.py`'s damage.csv. It reruns a study under another operator choice
(`--dispatch cost`) without designing the attacks again, so the attacks, the onsets and the draws stay those of
the study.

    python tools/nyfed/operator_days.py OUT.csv STUDY [STUDY ...] [--modes trusting accounting oracle]
                                        [--dispatch cost] [--workers 8]

Run from the folder whose `data/` holds the inputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py

from fdia_graph.nyfed.experiments import add_study_arguments, load_calib, run_parallel, study_days, variant
from fdia_graph.nyfed.operator.response import MODES, operator_day, respond, summarize


def _job(
    path: str, pool: str, data: str, calib: tuple[float, float], modes: list[str], dispatch: str
) -> list[dict]:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:  # a day without an attack
        return []
    with h5py.File(path, "r") as h:
        meters = int(h["labels/meter_tampered"][:].any(axis=0).sum())
    name = Path(path).stem.split("_")  # am_<onset>_<overshoot>
    tag = {
        "file": Path(path).name,
        "study": str(Path(path).parents[1]),
        "onset": int(name[1]),
        "time": str(day.time[day.onset].astype("datetime64[s]")),
        "variant": variant(path),
        "overshoot_mw": float(name[2]),
        "meters": meters,
        "dispatch": dispatch,
    }
    return [
        {
            **tag,
            "mode": mode,
            **summarize(respond(day, mode=mode, calib=calib, closed_loop=True, dispatch=dispatch)),
        }
        for mode in modes
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_study_arguments(ap)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = ap.parse_args()
    files = study_days(args.studies, args.limit)
    if not files:
        raise SystemExit(f"no attack days (days/am_*_*.h5) under {args.studies}")
    calib = load_calib(args.calib)
    tasks = [(f, args.pool, args.data, calib, args.modes, args.dispatch) for f in files]
    run_parallel(_job, tasks, args.out, workers=args.workers, every=10)


if __name__ == "__main__":
    main()
