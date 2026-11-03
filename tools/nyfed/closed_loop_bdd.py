"""Does the attack stay stealthy once the operator has acted on it (N76)? The attack vectors are designed on the
benign trajectory, while the trusting operator's dispatch, regulation and the inadvertent interchange move the true
state. For every attacked scan: the closed-loop true state (an AC power flow at the operator's units and the
neighbours' supply), the scan the estimator then receives, h(x_closed) + a, and its weighted least squares fit. The
residual that fit leaves is the attack's noncentrality in the chi-square test; with the meters' noise added the test
fires with the probability of a noncentral chi-square past the 99th percentile of the central one.

    python tools/nyfed/closed_loop_bdd.py OUT.csv STUDY [STUDY ...] [--workers 8]

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
from scipy.stats import chi2, ncx2

warnings.filterwarnings("ignore")

from fdia_graph.nyfed.attacks.attacker import wrap_angles
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.response import _entry_share, _TrueGrid, operator_day, respond
from fdia_graph.nyfed.sensing.measure import Measurement

REF = 3  # bus type of the reference bus


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
    rank = int(np.linalg.matrix_rank(jac, tol=1e-6))
    return float(r @ r), rank


def _job(path: str, pool: str, data: str, calib: tuple[float, float]) -> list[dict]:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:
        return []
    c = day.case
    resp = respond(day, mode="trusting", calib=calib, closed_loop=True)
    a = resp.attacked
    grid = _TrueGrid(day)
    share = _entry_share(c, "capacity")
    meas = Measurement(Network(c), day.plan)
    ref = int(np.flatnonzero(c.bus_type == REF)[0])
    sr, sa = day.plan.sigma_rel, day.plan.sigma_abs
    rows = []
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
            continue
        att = day.z[t] - day.z_benign[t]
        z_cl = meas.h(r["vm"], r["va"]) + att
        z_ol = meas.h(day.vm[t], day.va[t]) + att
        sigma = sr * np.abs(z_cl) + sa
        j_cl, rank = _wls(meas, z_cl, sigma, r["vm"].copy(), r["va"].copy(), ref)
        j_ol, _ = _wls(meas, z_ol, sigma, day.vm[t].copy(), day.va[t].copy(), ref)
        dof = len(z_cl) - rank
        thr = chi2.ppf(0.99, dof)
        rows.append(
            {
                "file": Path(path).name,
                "scan": t - day.onset,
                "dof": dof,
                "lambda_closed": j_cl,
                "lambda_open": j_ol,
                "p_detect_closed": float(ncx2.sf(thr, dof, j_cl)) if j_cl > 0 else 0.01,
                "p_detect_open": float(ncx2.sf(thr, dof, j_ol)) if j_ol > 0 else 0.01,
                "moved_mw": float(np.abs(a.dev[t] + a.reg[t]).sum()),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("studies", nargs="+")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default="data/NYgrid/Data")
    ap.add_argument("--calib", default="results/nygrid.tie_check.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N attack days (a quick check)")
    args = ap.parse_args()
    cal = json.loads(Path(args.calib).read_text())
    calib = (cal["mean_mw"], cal["sd_mw"])
    files = sorted({str(p) for s in args.studies for p in Path(s, "days").glob("am_*_*.h5")})
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files)} attack days", flush=True)
    rows: list[dict] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_job, f, args.pool, args.data, calib) for f in files]
        for k, fut in enumerate(as_completed(futs), 1):
            rows += fut.result()
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"{k}/{len(files)} [{time.perf_counter() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
