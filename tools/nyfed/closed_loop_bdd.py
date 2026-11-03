"""Does the attack stay stealthy once the operator has acted on it (N76, N77)? The attack vectors are designed on the
benign trajectory, while the operator's dispatch, regulation and the inadvertent interchange move the true state.
For every scan of the attack window, with the meters' own noise realization e (the benign scan less the benign
readings, constant bias and jitter):

- benign: the benign scan, fitted by AC weighted least squares on the full accuracy-class variance, and its
  chi-square test at 99 percent;
- attacked: h(x_closed) + a + e, with x_closed the AC power flow at the operator's units and the neighbours' supply,
  fitted and tested the same way.

It then reruns the oracle operator with its fallback at the first scan the residual test fires, which gives the
damage an operator that acts on its own residual alarm still takes.

    python tools/nyfed/closed_loop_bdd.py OUT.csv STUDY [STUDY ...] [--mode trusting] [--workers 8]

Run from the folder whose `data/` holds the inputs. One row per attack window: the scans the test fires on, the
first of them, the benign firings over the same window and the damage of the residual-alarm oracle.
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
from scipy.stats import chi2

warnings.filterwarnings("ignore")

from fdia_graph.nyfed.attacks.attacker import wrap_angles
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.response import _entry_share, _TrueGrid, operator_day, respond, summarize
from fdia_graph.nyfed.sensing.measure import Measurement

REF = 3  # bus type of the reference bus
KEEP = (
    "inadvertent_mw",
    "inadvertent_mwh",
    "iface_down_mw",
    "iface_up_mw",
    "other_iface_up_mw",
    "margin_lost_mw",
)


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


def _job(path: str, pool: str, data: str, calib: tuple[float, float], mode: str) -> dict | None:
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
    fired, benign_fired, j_att, j_ben = [], [], [], []
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
            fired.append(False)
            benign_fired.append(False)
            continue
        noise = wrap_angles(day.z_benign[t] - meas.h(day.vm[t], day.va[t]), day.plan.kind)
        att = day.z[t] - day.z_benign[t]
        z_b = day.z_benign[t]
        z_a = meas.h(r["vm"], r["va"]) + att + noise
        sigma = sr * np.abs(z_b) + sa
        jb, rank = _wls(meas, z_b, sigma, day.vm[t].copy(), day.va[t].copy(), ref)
        ja, _ = _wls(meas, z_a, sigma, r["vm"].copy(), r["va"].copy(), ref)
        thr = chi2.ppf(0.99, len(z_b) - rank)
        fired.append(ja > thr)
        benign_fired.append(jb > thr)
        j_att.append(ja)
        j_ben.append(jb)
    fired_a, benign_a = np.array(fired), np.array(benign_fired)
    first = int(np.argmax(fired_a)) if fired_a.any() else -1
    row = {
        "file": Path(path).name,
        "mode": mode,
        "variant": "at_limit" if Path(path).stem.endswith("_0") else "damaging",
        "scans": len(fired_a),
        "fired_scans": int(fired_a.sum()),
        "first_fired": first,
        "benign_fired_scans": int(benign_a.sum()),
        "benign_any": bool(benign_a.any()),
        "j_attacked_median": float(np.median(j_att)) if j_att else np.nan,
        "j_benign_median": float(np.median(j_ben)) if j_ben else np.nan,
        "dof": len(day.z_benign[0]) - rank if j_ben else np.nan,
    }
    full = summarize(resp)
    row.update({f"full_{k}": full[k] for k in KEEP})
    if first >= 0:  # the oracle acting on its own residual alarm
        s = summarize(respond(day, mode="oracle", calib=calib, closed_loop=True, alarm_at=day.onset + first))
        row.update({f"bdd_{k}": s[k] for k in KEEP})
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
        futs = [ex.submit(_job, f, args.pool, args.data, calib, args.mode) for f in files]
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is not None:
                rows.append(r)
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"{k}/{len(files)} [{time.perf_counter() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
