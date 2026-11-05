"""Does the attack stay stealthy once the operator has acted on it (N76 to N78)? The attack vectors are designed on
the benign trajectory, while the operator's dispatch, regulation and the inadvertent interchange move the true
state. For every scan of the attack window, with the meters' own noise realization e (the benign scan less the
benign readings, constant bias and jitter):

- benign: the benign scan, fitted by AC weighted least squares on the full accuracy-class variance;
- attacked: h(x_closed) + a + e, with x_closed the AC power flow at the operator's units and the neighbours' supply,
  fitted the same way.

Each weighted residual is tested against two thresholds: the 99 percent point of the chi-square distribution, and
the 99th percentile of the benign residual over January to March (scans drawn with the plan's own meter noise), the
threshold an operator would calibrate since a constant bias keeps the residual nearly fixed within a window. At the
first scan past the calibrated threshold the operator responds three ways (`respond(after=...)`): it reads the scans
without the attack (benign), it holds its dispatch while its AGC keeps reading the tie meters it reads (freeze), or it
holds its dispatch with the AGC on the common-source tie values (freeze_cs).

    python tools/nyfed/closed_loop_bdd.py OUT.csv STUDY [STUDY ...] [--mode trusting] [--dispatch cost] [--workers 8]
                                          [--draws 5]

With `--draws N` it also tests every attack against N further meter fleets (fresh constant biases of the same
accuracy classes, as bdd_benign_series.py --draw), one `j_attacked_dK` column each, and skips the three answers.

Run from the folder whose `data/` holds the inputs. One row per attack window.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from fdia_graph.nyfed.attacks.attacker import wrap_angles
from fdia_graph.nyfed.detect import fleet_bias, reference_bus, wls
from fdia_graph.nyfed.experiments import add_study_arguments, load_calib, run_parallel, study_days, variant
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.io import read_pool
from fdia_graph.nyfed.operator.response import TrueGrid, entry_share, operator_day, respond, summarize
from fdia_graph.nyfed.operator.run import AC_DISPATCH
from fdia_graph.nyfed.scenario.pool import SOLVED
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.noise import NoiseModel

KEEP = (
    "inadvertent_mw",
    "inadvertent_mwh",
    "iface_down_mw",
    "iface_up_mw",
    "other_iface_up_mw",
    "margin_lost_mw",
)
AFTER = ("benign", "freeze", "freeze_cs")


def _calibrate(day_path: str, pool_path: str, data: str, frames: int, seed: int) -> tuple[float, list[float]]:
    """The 99th percentile of the benign weighted residual over `frames` solved scans of January to March, each
    drawn with the plan's meter noise (its constant biases and a fresh jitter)."""
    day = operator_day(Path(day_path), pool_path, data)
    meas = Measurement(Network(day.case), day.plan)
    noise = NoiseModel(day.plan)
    ref = reference_bus(day.case)
    pool = read_pool(pool_path)
    month = pool.time.astype("datetime64[M]").astype(int) % 12 + 1
    ok = np.flatnonzero((month <= 3) & (pool.solve_kind == SOLVED))
    g = np.random.default_rng(seed)
    js = []
    for k in g.choice(ok, size=min(frames, len(ok)), replace=False):
        vm, va = pool.vm[k], pool.va[k]
        if not np.isfinite(vm).all():
            continue
        exact = meas.h(vm, va)
        z = noise.sample(exact, g)
        js.append(wls(meas, z, noise.sigmas(z), vm, va, ref).j)
    return float(np.percentile(js, 99)), js


def _job(
    path: str,
    pool: str,
    data: str,
    calib: tuple[float, float],
    mode: str,
    thr_cal: float,
    draws: int = 0,
    dispatch: str = "movement",
) -> dict | None:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:
        return None
    c = day.case
    resp = respond(day, mode=mode, calib=calib, closed_loop=True, dispatch=dispatch)
    a = resp.attacked
    grid = TrueGrid(day, ac=dispatch in AC_DISPATCH)  # only its AC power flows are used here
    share = entry_share(c, "capacity")
    meas = Measurement(Network(c), day.plan)
    noise_model = NoiseModel(day.plan)
    ref = reference_bus(c)
    j_att, j_ben, dof = [], [], np.nan
    rel = day.plan.sigma_rel > 0
    fleets = [fleet_bias(day.plan, d) for d in range(1, draws + 1)]
    j_draw: list[list[float]] = [[] for _ in fleets]
    for t in range(day.onset, day.onset + day.length):
        x = grid.run_state(a, t, share)  # the true state once the operator has acted
        if x is None:
            j_att.append(np.nan)
            j_ben.append(np.nan)
            for jd in j_draw:
                jd.append(np.nan)
            continue
        noise = wrap_angles(day.z_benign[t] - meas.h(day.vm[t], day.va[t]), day.plan.kind)
        z_b = day.z_benign[t]
        z_a = meas.h(*x) + (day.z[t] - z_b) + noise
        sigma = noise_model.sigmas(z_b)
        benign = wls(meas, z_b, sigma, day.vm[t], day.va[t], ref)
        j_att.append(wls(meas, z_a, sigma, *x, ref).j)
        j_ben.append(benign.j)
        exact = meas.h(day.vm[t], day.va[t])
        own = np.where(rel, exact * day.plan.bias, day.plan.bias)
        for bias, jd in zip(fleets, j_draw):  # the same jitter, the other fleet's constant bias
            z_d = z_a - own + np.where(rel, exact * bias, bias)
            jd.append(wls(meas, z_d, sigma, *x, ref).j)
        dof = len(z_b) - benign.rank
    ja_, jb_ = np.array(j_att), np.array(j_ben)
    thr_chi = chi2.ppf(0.99, dof) if np.isfinite(dof) else np.inf
    row = {
        "file": Path(path).name,
        "onset": day.onset,
        "mode": mode,
        "dispatch": dispatch,
        "variant": variant(path),
        "scans": int(np.isfinite(ja_).sum()),
        "dof": dof,
        "thr_chi": thr_chi,
        "thr_cal": thr_cal,
        "j_attacked": json.dumps([round(float(x), 2) for x in ja_]),
        "j_benign": json.dumps([round(float(x), 2) for x in jb_]),
    }
    for d, jd in enumerate(j_draw, 1):
        row[f"j_attacked_d{d}"] = json.dumps([round(float(x), 2) for x in jd])
    for name, thr in (("chi", thr_chi), ("cal", thr_cal)):
        fa, fb = np.nan_to_num(ja_) > thr, np.nan_to_num(jb_) > thr
        row.update(
            {
                f"{name}_fired_scans": int(fa.sum()),
                f"{name}_first": int(np.argmax(fa)) if fa.any() else -1,
                f"{name}_benign_scans": int(fb.sum()),
            }
        )
    full = summarize(resp)
    row.update({f"full_{k}": full[k] for k in KEEP})
    first = row["cal_first"]
    if first >= 0 and not draws:  # the operator responding to its own calibrated residual alarm, three ways
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
            row.update({f"{after}_{k}": s[k] for k in KEEP})
    return row


def _threshold(cal_json: Path, first_day: str, pool: str, data: str, frames: int) -> float:
    """The calibrated per-scan threshold, read from `cal_json`, or calibrated now on benign January to March scans
    and written there."""
    if cal_json.exists():
        return float(json.loads(cal_json.read_text())["threshold"])
    thr, js = _calibrate(first_day, pool, data, frames, 20190101)
    summary = {"threshold": thr, "frames": len(js), "median": float(np.median(js)), "max": float(np.max(js))}
    cal_json.write_text(json.dumps({**summary, "chi2_99_400": float(chi2.ppf(0.99, 400))}, indent=1))
    return thr


def main() -> None:
    ap = argparse.ArgumentParser()
    add_study_arguments(ap)
    ap.add_argument("--mode", default="trusting")
    ap.add_argument(
        "--cal-frames", type=int, default=600, help="benign January to March scans for the threshold"
    )
    ap.add_argument("--cal-json", default="results/nygrid.bdd_calibration.json")
    ap.add_argument("--draws", type=int, default=0, help="also test against N further meter fleets")
    args = ap.parse_args()
    calib = load_calib(args.calib)
    files = study_days(args.studies, args.limit)
    if not files:
        raise SystemExit(f"no attack days (days/am_*_*.h5) under {args.studies}")
    thr_cal = _threshold(Path(args.cal_json), files[0], args.pool, args.data, args.cal_frames)
    print(f"{len(files)} attack days, calibrated threshold {thr_cal:.1f}", flush=True)
    tasks = [(f, args.pool, args.data, calib, args.mode, thr_cal, args.draws, args.dispatch) for f in files]
    run_parallel(_job, tasks, args.out, workers=args.workers)


if __name__ == "__main__":
    main()
