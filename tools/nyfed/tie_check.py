"""The interchange check's false alarms over 2019 (N62, N67): the benign gap between the interchange read
at the New York ends of the ties and the neighbours' readings, every solved five-minute frame of the pool,
both ends through meters of their accuracy class. The alarm's mean and standard deviation are fitted on
January to March and the alarm rate is counted on April to December, out of sample.

    python tools/nyfed/tie_check.py --pool out/pool_2019_v3.h5 [--out results] [--draws 20]

Run from the folder whose `data/` holds the inputs. Writes `nygrid.tie_check.json`: the calibration, the
threshold (MW) and the false alarms per scan and per day. `--draws N` repeats the check over N fresh draws of
both ends' constant meter biases, each calibrated on its own January to March, and writes
`nygrid.tie_check_draws.json` (N73): one realization of the biases is one meter fleet, not the check's rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fdia_graph.nyfed.build import production_builder
from fdia_graph.nyfed.config import Settings, rng
from fdia_graph.nyfed.experiments import DATA, POOL, study_days, variant
from fdia_graph.nyfed.operator.response import NEIGHBOUR_SEED, neighbour_reads, tie_meters
from fdia_graph.nyfed.scenario.pool import SOLVED
from fdia_graph.nyfed.sensing.meters import JITTER_FRAC

CHUNK = 4032  # two weeks of frames at a time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--data", default=DATA, help="NYgrid's data tables")
    ap.add_argument("--out", default="results")
    ap.add_argument("--draws", type=int, default=0)
    ap.add_argument(
        "--attacks", nargs="*", default=[], help="damage_study.py folders whose attacks each draw detects"
    )
    args = ap.parse_args()
    # the builder reads NYgrid from <nygrid_dir>/Data, so --data names that Data folder
    settings = Settings(nygrid_dir=Path(args.data).parent)
    b = production_builder(args.pool, ("Am",), workers=1, survey=0, settings=settings)
    case, plan, net, pool = b.att.net.case, b.att.meas.plan, b.att.net, b.pool
    br, end, rows = tie_meters(case, plan)
    sr, sa, bias = plan.sigma_rel[rows], plan.sigma_abs[rows], plan.bias[rows]
    rel = sr > 0
    g_ny = rng(Settings(), "tie_check")
    gaps, times, nears, fars = [], [], [], []
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
        nears.append(near)
        fars.append(far)
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
    if args.draws:
        _draws(args, plan, rows, np.concatenate(nears), np.concatenate(fars), time, fit, test, _attacks(args))


def _attacks(args) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Each attack day of `--attacks`: its variant, the clock of its attacked scans and the false interchange it
    adds there (the read interchange less the benign one, the meters' noise cancelling against the twin)."""
    from fdia_graph.nyfed.operator.response import operator_day, tie_gap

    out = []
    for study in args.attacks:
        for path in study_days([study]):
            day = operator_day(Path(path), args.pool, args.data)
            if day.onset < 0:
                continue
            w = slice(day.onset, day.onset + day.length)
            false = (tie_gap(day, day.z) - tie_gap(day, day.z_benign))[w]
            out.append((variant(path), np.asarray(day.time[w]).astype("datetime64[s]"), false))
    return out


def _draws(args, plan, rows, near, far, time, fit, test, attacks=()) -> None:
    """The check over fresh draws of both ends' constant biases (N73), the same accuracy classes and jitter."""
    sr, sa = plan.sigma_rel[rows], plan.sigma_abs[rows]
    rel = sr > 0
    scale = np.sqrt(1.0 - JITTER_FRAC**2) * np.where(rel, sr, sa)
    days = time[test].astype("datetime64[D]")
    uniq = np.unique(days)
    out = []
    for d in range(args.draws):
        g = np.random.default_rng([NEIGHBOUR_SEED, 7919, d])

        def reading(x: np.ndarray) -> np.ndarray:
            bias = g.normal(0.0, scale)
            jitter = JITTER_FRAC * sr * np.abs(x) + np.where(rel, sa, JITTER_FRAC * sa)
            return np.where(rel, x * (1.0 + bias), x + bias) + g.normal(0.0, 1.0, x.shape) * jitter

        r_near, r_far = reading(near), reading(far)
        gap = r_near.sum(axis=1) - r_far.sum(axis=1)
        mean, sd = float(gap[fit].mean()), float(gap[fit].std())
        alarm = np.abs(gap - mean) > 4.0 * sd
        per_day = np.array([alarm[test][days == u].any() for u in uniq])
        # reconciled: each tie's two meters fitted against each other on the fit months (a gain and an offset
        # per tie, as the hourly checkout reconciles them), the corrected gap then tested as before
        corr = np.empty_like(r_near)
        for k in range(r_near.shape[1]):
            a_, b_ = np.polyfit(r_far[fit, k], r_near[fit, k], 1)
            corr[:, k] = (r_near[:, k] - b_) / a_
        gap_r = corr.sum(axis=1) - r_far.sum(axis=1)

        # normalized by each scan's expected spread from the meters' accuracy classes at the flows read, since the
        # relative jitter grows with the flow and a spread fitted in winter understates summer's
        def spread(x: np.ndarray) -> np.ndarray:
            return JITTER_FRAC * sr * np.abs(x) + np.where(rel, sa, JITTER_FRAC * sa)

        sd_t = np.sqrt((spread(r_near) ** 2 + spread(r_far) ** 2).sum(axis=1))
        z = gap_r / sd_t
        m_r, s_r = float(z[fit].mean()), float(z[fit].std())
        alarm_r = np.abs(z - m_r) > 4.0 * s_r
        s_r = s_r * float(np.median(sd_t[test]))  # the median threshold in MW, for the record
        days_r = np.array([alarm_r[test][days == u].any() for u in uniq])
        roll_scans, roll_days, roll_thr = int(alarm_r[test].sum()), int(days_r.sum()), [4.0 * s_r]
        detect: dict[str, list] = {"at_limit": [], "damaging": []}
        for form, clock, false in attacks:  # each attack against this draw's benign gap and threshold
            idx = np.searchsorted(time, clock)
            ok = (idx < len(time)) & (time[np.minimum(idx, len(time) - 1)] == clock)
            hit = np.flatnonzero(ok & (np.abs(gap[np.minimum(idx, len(time) - 1)] + false - mean) > 4.0 * sd))
            detect[form].append(int(hit[0]) if len(hit) else -1)
        out.append(
            {
                "draw": d,
                "detected_at_limit_pct": round(
                    100.0 * float(np.mean([h >= 0 for h in detect["at_limit"]])), 1
                )
                if detect["at_limit"]
                else None,
                "detected_damaging_pct": round(
                    100.0 * float(np.mean([h >= 0 for h in detect["damaging"]])), 1
                )
                if detect["damaging"]
                else None,
                "delay_at_limit_median": float(np.median([h for h in detect["at_limit"] if h >= 0]))
                if any(h >= 0 for h in detect["at_limit"])
                else None,
                "delay_damaging_median": float(np.median([h for h in detect["damaging"] if h >= 0]))
                if any(h >= 0 for h in detect["damaging"])
                else None,
                "delay_max": max([h for v in detect.values() for h in v] + [-1]),
                "threshold_mw": round(4.0 * sd, 1),
                "false_alarm_scans": int(alarm[test].sum()),
                "days_with_a_false_alarm": int(per_day.sum()),
                "reconciled_false_alarm_scans": roll_scans,
                "reconciled_days_with_a_false_alarm": int(roll_days),
                "reconciled_threshold_mw": round(float(np.median(roll_thr)), 1),
            }
        )
        print(out[-1], flush=True)
    summary = {
        "draws": out,
        "test_frames": int(test.sum()),
        "test_days": int(len(uniq)),
        "threshold_mw_range": [min(o["threshold_mw"] for o in out), max(o["threshold_mw"] for o in out)],
        "false_alarm_scans_total": int(sum(o["false_alarm_scans"] for o in out)),
        "draws_with_any_false_alarm": int(sum(o["false_alarm_scans"] > 0 for o in out)),
        "days_with_a_false_alarm_total": int(sum(o["days_with_a_false_alarm"] for o in out)),
        "reconciled_false_alarm_scans_total": int(sum(o["reconciled_false_alarm_scans"] for o in out)),
        "reconciled_days_with_a_false_alarm_total": int(
            sum(o["reconciled_days_with_a_false_alarm"] for o in out)
        ),
        "reconciled_draws_with_any_false_alarm": int(sum(o["reconciled_false_alarm_scans"] > 0 for o in out)),
    }
    (Path(args.out) / "nygrid.tie_check_draws.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "draws"}, indent=1))


if __name__ == "__main__":
    main()
