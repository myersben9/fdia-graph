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

    python tools/nyfed/closed_loop_bdd.py OUT.csv STUDY [STUDY ...] [--mode trusting] [--workers 8] [--draws 5]

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
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from fdia_graph.nyfed.attacks.attacker import wrap_angles
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.io import read_pool
from fdia_graph.nyfed.operator.response import _entry_share, _TrueGrid, operator_day, respond, summarize
from fdia_graph.nyfed.scenario.pool import SOLVED
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.meters import JITTER_FRAC
from fdia_graph.nyfed.sensing.noise import NoiseModel

REF = 3  # bus type of the reference bus
KEEP = (
    "inadvertent_mw",
    "inadvertent_mwh",
    "iface_down_mw",
    "iface_up_mw",
    "other_iface_up_mw",
    "margin_lost_mw",
)
AFTER = ("benign", "freeze", "freeze_cs")


def _wls(meas: Measurement, z: np.ndarray, sigma: np.ndarray, vm: np.ndarray, va: np.ndarray, ref: int):
    """Gauss-Newton weighted least squares from (vm, va), the reference angle fixed; a small damping keeps the
    buses no meter sees from making the step singular. Returns the weighted residual sum and the rank used."""
    kind = meas.plan.kind
    keep = np.ones(2 * len(vm), bool)
    keep[len(vm) + ref] = False
    w = 1.0 / sigma
    for _ in range(12):
        r = wrap_angles(z - meas.h(vm, va), kind) * w
        jac = meas.jacobian(vm, va)[:, keep] * w[:, None]
        step = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(keep.sum()), jac.T @ r)
        dx = np.zeros(2 * len(vm))
        dx[keep] = step
        vm, va = vm + dx[: len(vm)], va + dx[len(vm) :]
        if np.abs(step).max() < 1e-7:
            break
    r = wrap_angles(z - meas.h(vm, va), kind) * w
    return float(r @ r), int(np.linalg.matrix_rank(jac, tol=1e-6))


def _calibrate(day_path: str, pool_path: str, data: str, frames: int, seed: int) -> tuple[float, list[float]]:
    """The 99th percentile of the benign weighted residual over `frames` solved scans of January to March, each
    drawn with the plan's meter noise (its constant biases and a fresh jitter)."""
    day = operator_day(Path(day_path), pool_path, data)
    meas = Measurement(Network(day.case), day.plan)
    noise = NoiseModel(day.plan)
    ref = int(np.flatnonzero(day.case.bus_type == REF)[0])
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
        j, _ = _wls(meas, z, noise.sigmas(z), vm.copy(), va.copy(), ref)
        js.append(j)
    return float(np.percentile(js, 99)), js


def _fleet_bias(plan, draw: int) -> np.ndarray:
    """The constant biases of another meter fleet, drawn exactly as bdd_benign_series.py --draw draws them."""
    rel = plan.sigma_rel > 0
    scale = np.sqrt(1.0 - JITTER_FRAC**2) * np.where(rel, plan.sigma_rel, plan.sigma_abs)
    return np.random.default_rng([20190101, 4219, draw]).normal(0.0, scale)


def _job(
    path: str, pool: str, data: str, calib: tuple[float, float], mode: str, thr_cal: float, draws: int = 0
) -> dict | None:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:
        return None
    c = day.case
    resp = respond(day, mode=mode, calib=calib, closed_loop=True)
    a = resp.attacked
    grid = _TrueGrid(day)
    share = _entry_share(c, "capacity")
    meas = Measurement(Network(c), day.plan)
    ref = int(np.flatnonzero(c.bus_type == REF)[0])
    sr, sa = day.plan.sigma_rel, day.plan.sigma_abs
    j_att, j_ben, dof = [], [], np.nan
    rel = day.plan.sigma_rel > 0
    fleets = [_fleet_bias(day.plan, d) for d in range(1, draws + 1)]
    j_draw: list[list[float]] = [[] for _ in fleets]
    for t in range(day.onset, day.onset + day.length):
        units = day.unit_p[t] + a.dev[t] + a.reg[t] + a.uncovered[t] * share
        r = grid.solver.power_flow(
            grid.pd[t],
            grid.qd[t],
            units,
            np.ones(len(units), bool),
            day.vm[t, c.unit_bus],
            np.zeros(len(c.dc_f)),
            start=(day.vm[t], day.va[t]),
        )
        if r is None:
            j_att.append(np.nan)
            j_ben.append(np.nan)
            for jd in j_draw:
                jd.append(np.nan)
            continue
        noise = wrap_angles(day.z_benign[t] - meas.h(day.vm[t], day.va[t]), day.plan.kind)
        z_b = day.z_benign[t]
        z_a = meas.h(r["vm"], r["va"]) + (day.z[t] - z_b) + noise
        sigma = sr * np.abs(z_b) + sa
        jb, rank = _wls(meas, z_b, sigma, day.vm[t].copy(), day.va[t].copy(), ref)
        ja, _ = _wls(meas, z_a, sigma, r["vm"].copy(), r["va"].copy(), ref)
        j_att.append(ja)
        j_ben.append(jb)
        exact = meas.h(day.vm[t], day.va[t])
        own = np.where(rel, exact * day.plan.bias, day.plan.bias)
        for bias, jd in zip(fleets, j_draw):  # the same jitter, the other fleet's constant bias
            z_d = z_a - own + np.where(rel, exact * bias, bias)
            jd.append(_wls(meas, z_d, sigma, r["vm"].copy(), r["va"].copy(), ref)[0])
        dof = len(z_b) - rank
    ja_, jb_ = np.array(j_att), np.array(j_ben)
    thr_chi = chi2.ppf(0.99, dof) if np.isfinite(dof) else np.inf
    row = {
        "file": Path(path).name,
        "onset": day.onset,
        "mode": mode,
        "variant": "at_limit" if Path(path).stem.endswith("_0") else "damaging",
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
                    day, mode=mode, calib=calib, closed_loop=True, alarm_at=day.onset + first, after=after
                )
            )
            row.update({f"{after}_{k}": s[k] for k in KEEP})
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("studies", nargs="+")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default="data/NYgrid/Data")
    ap.add_argument("--calib", default="results/nygrid.tie_check.json")
    ap.add_argument("--mode", default="trusting")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N attack days (a quick check)")
    ap.add_argument(
        "--cal-frames", type=int, default=600, help="benign January to March scans for the threshold"
    )
    ap.add_argument("--cal-json", default="results/nygrid.bdd_calibration.json")
    ap.add_argument("--draws", type=int, default=0, help="also test against N further meter fleets")
    args = ap.parse_args()
    cal = json.loads(Path(args.calib).read_text())
    calib = (cal["mean_mw"], cal["sd_mw"])
    files = sorted({str(p) for s in args.studies for p in Path(s, "days").glob("am_*_*.h5")})
    if args.limit:
        files = files[: args.limit]
    t0 = time.perf_counter()
    cal_path = Path(args.cal_json)
    if cal_path.exists():
        thr_cal = float(json.loads(cal_path.read_text())["threshold"])
    else:
        thr_cal, js = _calibrate(files[0], args.pool, args.data, args.cal_frames, 20190101)
        cal_path.write_text(
            json.dumps(
                {
                    "threshold": thr_cal,
                    "frames": len(js),
                    "median": float(np.median(js)),
                    "max": float(np.max(js)),
                    "chi2_99_400": float(chi2.ppf(0.99, 400)),
                },
                indent=1,
            )
        )
    print(
        f"{len(files)} attack days, calibrated threshold {thr_cal:.1f} [{time.perf_counter() - t0:.0f}s]",
        flush=True,
    )
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [
            ex.submit(_job, f, args.pool, args.data, calib, args.mode, thr_cal, args.draws) for f in files
        ]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is not None:
                rows.append(r)
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"{k}/{len(files)} [{time.perf_counter() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
