"""The interchange check's false alarms over 2019 (N62, N67): the benign gap between the interchange read
at the New York ends of the ties and the neighbours' readings, every solved five-minute frame of the pool,
both ends through meters of their accuracy class. The alarm's mean and standard deviation are fitted on
January to March and the alarm rate is counted on April to December, out of sample.

    python tools/nyfed/tie_check.py --pool out/pool_2019_v3.h5 [--out results]

Run from the folder whose `data/` holds the inputs. Writes `nygrid.tie_check.json`: the calibration, the
threshold (MW) and the false alarms per scan and per day.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from fdia_graph.nyfed.build import production_builder
from fdia_graph.nyfed.config import Settings, rng
from fdia_graph.nyfed.operator.response import NEIGHBOUR_SEED, neighbour_reads, tie_meters
from fdia_graph.nyfed.scenario.pool import SOLVED
from fdia_graph.nyfed.sensing.meters import JITTER_FRAC

CHUNK = 4032  # two weeks of frames at a time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    b = production_builder(args.pool, ("Am",), workers=1, survey=0)
    case, plan, net, pool = b.att.net.case, b.att.meas.plan, b.att.net, b.pool
    br, end, rows = tie_meters(case, plan)
    sr, sa, bias = plan.sigma_rel[rows], plan.sigma_abs[rows], plan.bias[rows]
    rel = sr > 0
    g_ny = rng(Settings(), "tie_check")
    gaps, times = [], []
    for k in range(0, len(pool.time), CHUNK):
        sl = slice(k, k + CHUNK)
        ok = pool.solve_kind[sl] == SOLVED
        vm, va = pool.vm[sl][ok], pool.va[sl][ok]
        pf, _, pt, _ = net.flows(vm, va)
        near = np.where(end == 0, pf[:, br], pt[:, br])  # New York's end, leaving New York
        far = np.where(end == 0, -pt[:, br], -pf[:, br])  # the neighbour's end, the same direction
        jitter = JITTER_FRAC * sr * np.abs(near) + np.where(rel, sa, JITTER_FRAC * sa)
        read = np.where(rel, near * (1.0 + bias), near + bias) + g_ny.normal(0.0, 1.0, near.shape) * jitter
        other = neighbour_reads(plan, rows, far, np.random.default_rng([NEIGHBOUR_SEED, k]))
        gaps.append(read.sum(axis=1) - other.sum(axis=1))
        times.append(pool.time[sl][ok])
    gap, time = np.concatenate(gaps), np.concatenate(times).astype("datetime64[s]")
    month = time.astype("datetime64[M]").astype(int) % 12 + 1
    fit, test = month <= 3, month > 3
    mean, sd = float(gap[fit].mean()), float(gap[fit].std())
    alarm = np.abs(gap - mean) > 4.0 * sd
    days = time[test].astype("datetime64[D]")
    per_day = np.array([alarm[test][days == d].any() for d in np.unique(days)])
    out = {
        "frames": int(len(gap)),
        "fit_frames": int(fit.sum()),
        "test_frames": int(test.sum()),
        "mean_mw": round(mean, 2),
        "sd_mw": round(sd, 2),
        "threshold_mw": round(4.0 * sd, 1),
        "false_alarms_per_scan": float(alarm[test].mean()),
        "false_alarm_scans": int(alarm[test].sum()),
        "days_with_a_false_alarm_pct": round(100.0 * float(per_day.mean()), 2),
        "test_days": int(len(per_day)),
        "ties": int(len(br)),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "nygrid.tie_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
