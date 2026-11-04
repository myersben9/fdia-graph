"""The interchange check against the checkout operators already run: when does each catch an attack, and how much
inadvertent interchange has the trusting operator taken by then?

Balancing authorities agree the energy of every tie after each clock hour (NERC BAL-006, the hourly checkout of the
joint operating agreements). On the readings the operator has, that is the hourly mean of the tie gap (the
interchange read at the New York ends less the neighbours' readings) checked against the same benign calibration as
the scan-rate check, and known only once its hour has ended. For every attack day this reports the scan the
scan-rate check alarms (`interchange_alarm`), the last scan of the first hour whose mean gap passes the threshold
(the earliest the checkout can know), and the inadvertent interchange the trusting operator has taken before each
(MWh, attacked less benign).

    python tools/nyfed/hourly_checkout.py OUT.csv STUDY [STUDY ...] [--workers 8]

Run from the folder whose `data/` holds the inputs. One row per attack day.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fdia_graph.nyfed.experiments import add_study_arguments, load_calib, run_parallel, study_days, variant
from fdia_graph.nyfed.operator.response import interchange_alarm, operator_day, respond, tie_gap
from fdia_graph.nyfed.operator.sced import FRAME_HOURS

SIGMAS = 4.0  # the threshold of the scan-rate check (N62), in standard deviations of the benign gap


def hourly_alarm(time: np.ndarray, gap: np.ndarray, onset: int, calib: tuple[float, float]) -> int | None:
    """The last scan of the first clock hour at or after the onset's whose mean gap is more than `SIGMAS` benign
    standard deviations off the benign mean, or None: when an hourly checkout of the same readings would know."""
    hour = np.asarray(time).astype("datetime64[s]").astype("datetime64[h]")
    mean, sd = calib
    for h in np.unique(hour[onset:]):
        scans = np.flatnonzero(hour == h)
        if abs(float(np.nanmean(gap[scans])) - mean) > SIGMAS * sd:
            return int(scans[-1])
    return None


def _job(path: str, pool: str, data: str, calib: tuple[float, float]) -> dict | None:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:
        return None
    r = respond(day, mode="trusting", calib=calib, closed_loop=True)
    # [T] MWh, the inadvertent interchange the attack adds in each scan
    inadv = np.clip(np.abs(r.attacked.uncovered) - np.abs(r.benign.uncovered), 0, None) * FRAME_HOURS
    gap = tie_gap(day, day.z)
    scan = interchange_alarm(day, gap, tie_gap(day, day.z_benign), calib)
    hour = hourly_alarm(day.time, gap, day.onset, calib)

    def taken(at: int | None) -> float:
        """The inadvertent energy up to and including scan `at`, all of it when there is no alarm."""
        return float(inadv[: (len(inadv) if at is None else at + 1)].sum())

    return {
        "file": Path(path).name,
        "onset": int(Path(path).stem.split("_")[1]),
        "variant": variant(path),
        "scan_alarm": np.nan if scan is None else scan - day.onset,
        "hourly_alarm": np.nan if hour is None else hour - day.onset,
        "inadvertent_mwh": float(inadv.sum()),
        "taken_by_scan_mwh": taken(scan),
        "taken_by_hourly_mwh": taken(hour),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_study_arguments(ap)
    args = ap.parse_args()
    files = study_days(args.studies, args.limit)
    if not files:
        raise SystemExit(f"no attack days (days/am_*_*.h5) under {args.studies}")
    calib = load_calib(args.calib)
    run_parallel(
        _job, [(f, args.pool, args.data, calib) for f in files], args.out, workers=args.workers, every=10
    )


if __name__ == "__main__":
    main()
