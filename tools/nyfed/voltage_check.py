"""The voltage-limit alarm as a detector (N74): the largest voltage-magnitude reading of every scan over 2019, through
the meters of the plan (constant bias and fresh jitter), an alarm threshold set on January to March at the largest
benign reading plus a guard, its false-alarm days on April to December, and the attacks of the damage studies it
catches (any reading of their window above the threshold).

    python tools/nyfed/voltage_check.py OUT.json STUDY [STUDY ...] [--guard 0.002]

Run from the folder whose `data/` holds the inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from fdia_graph.nyfed.build import production_builder
from fdia_graph.nyfed.config import Settings, rng
from fdia_graph.nyfed.scenario.pool import SOLVED
from fdia_graph.nyfed.sensing.meters import JITTER_FRAC, KINDS

CHUNK = 4032


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("studies", nargs="+")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--guard", type=float, default=0.002, help="pu added to the largest benign reading")
    args = ap.parse_args()
    b = production_builder(args.pool, ("Am",), workers=1, survey=0)
    plan, pool = b.att.meas.plan, b.pool
    v = np.flatnonzero(plan.kind == KINDS.index("v"))
    bus, sa, bias = plan.bus[v], plan.sigma_abs[v], plan.bias[v]
    g = rng(Settings(), "voltage_check")
    tops, times = [], []
    for k in range(0, len(pool.time), CHUNK):
        sl = slice(k, k + CHUNK)
        ok = pool.solve_kind[sl] == SOLVED
        vm = pool.vm[sl][ok][:, bus]
        read = vm + bias + g.normal(0.0, 1.0, vm.shape) * JITTER_FRAC * sa
        tops.append(read.max(axis=1))
        times.append(pool.time[sl][ok])
    top, time = np.concatenate(tops), np.concatenate(times).astype("datetime64[s]")
    month = time.astype("datetime64[M]").astype(int) % 12 + 1
    fit, test = month <= 3, month > 3
    threshold = float(top[fit].max()) + args.guard
    days = time[test].astype("datetime64[D]")
    alarm = top[test] > threshold
    alarm_days = int(len(np.unique(days[alarm])))
    out = {
        "threshold_pu": round(threshold, 4),
        "guard_pu": args.guard,
        "benign_max_fit_pu": round(float(top[fit].max()), 4),
        "benign_max_test_pu": round(float(top[test].max()), 4),
        "test_days": int(len(np.unique(days))),
        "false_alarm_scans": int(alarm.sum()),
        "false_alarm_days": alarm_days,
        "studies": {},
    }
    for study in args.studies:
        caught, margins = [], []
        for path in sorted(Path(study, "days").glob("am_*.h5")):
            with h5py.File(path, "r") as h:
                kind = np.asarray(KINDS)[h["meters/kind"][:]] == "v"
                on = h["frames/episode"][:] >= 0
                if not on.any():
                    continue
                peak = float(np.nanmax(h["frames/z"][:][on][:, kind]))
                margins.append(peak - threshold)
                caught.append(peak > threshold)
        m = np.array(margins)
        out["studies"][study] = {
            "attacks": len(m),
            "caught": int(np.sum(caught)),
            "margin_median_pu": round(float(np.median(m)), 4),
            "margin_min_pu": round(float(m.min()), 4),
        }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
