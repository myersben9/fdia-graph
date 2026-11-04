"""An attacker that follows the closed loop (N82), and the stealthy attack before the operator acts. For every
attack window and every scan:

- open: the designed scan itself (the benign scan plus the attack vector), the residual before the operator has
  moved anything;
- adaptive: the attacker reads the meters it holds, so it knows the closed-loop true state x' around its support,
  and it moves that state by the designed false-state increment, a' = h(x' + (x~ - x)) - h(x'). The increment
  touches only the support's buses, so a' moves only the meters around them, and z = h(x') + a' + e is consistent
  with the state x' + (x~ - x) by construction.

The false state x~ is recovered from the stored scans (h(x~) = h(x) + a, fitted without noise). The closed-loop state
x' is the operator's response to the designed attack (`respond(closed_loop=True)`), so the adaptive attacker is
taken one pass: its readings move the operator only by the change of its increment's flows, which the closed-loop
operator already reads through its mismatch term. Every residual is fitted by AC weighted least squares on the full
accuracy-class variance with each window's own noise e, as in closed_loop_bdd.py, and judged by the persistence rule
of bdd_rule.py. Also reported: the meters a' moves beyond their noise that the designed attack does not (meters the
attacker would need but does not hold), the largest change of a pretended injection against the designed one, and
how far the adaptive attack's pretended change at each New York bus goes beyond the designed attack's, as a share of a
scale for that bus's bound (10 percent of its load at a load bus, 3 percent of its units' capacity at a generator
bus). The designed attack meets the attacker's exact bounds, so a small excess says the adaptive one nearly does; the
scale is only a yardstick, not the bounds themselves.

    python tools/nyfed/adaptive_bdd.py OUT.csv STUDY [STUDY ...] [--mode trusting] [--workers 8]

Run from the folder whose `data/` holds the inputs. One row per attack window.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from bdd_rule import first_alarm  # noqa: E402
from closed_loop_bdd import REF, _wls  # noqa: E402

from fdia_graph.nyfed.attacks.attacker import wrap_angles  # noqa: E402
from fdia_graph.nyfed.grid.network import Network  # noqa: E402
from fdia_graph.nyfed.operator.response import _entry_share, _TrueGrid, operator_day, respond  # noqa: E402
from fdia_graph.nyfed.sensing.measure import Measurement  # noqa: E402


def _false_state(meas: Measurement, z: np.ndarray, vm: np.ndarray, va: np.ndarray, ref: int):
    """The state whose readings are `z` (noise free, so the fit is exact), by Gauss-Newton from (vm, va) with the
    reference angle fixed. Returns (vm, va, largest remaining mismatch)."""
    kind = meas.plan.kind
    keep = np.ones(2 * len(vm), bool)
    keep[len(vm) + ref] = False
    for _ in range(20):
        r = wrap_angles(z - meas.h(vm, va), kind)
        jac = meas.jacobian(vm, va)[:, keep]
        step = np.linalg.lstsq(jac, r, rcond=None)[0]
        dx = np.zeros(2 * len(vm))
        dx[keep] = step
        vm, va = vm + dx[: len(vm)], va + dx[len(vm) :]
        if np.abs(step).max() < 1e-9:
            break
    return vm, va, float(np.abs(wrap_angles(z - meas.h(vm, va), kind)).max())


def _job(path: str, pool: str, data: str, calib, mode: str, rule) -> dict | None:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:
        return None
    c, plan = day.case, day.plan
    resp = respond(day, mode=mode, calib=calib, closed_loop=True)
    a_resp = resp.attacked
    grid = _TrueGrid(day)
    share = _entry_share(c, "capacity")
    meas = Measurement(Network(c), plan)
    net = Network(c)
    ref = int(np.flatnonzero(c.bus_type == REF)[0])
    sr, sa = plan.sigma_rel, plan.sigma_abs
    j_open, j_adapt, extra, dinj, fit_err, excess = [], [], [], [], [], []
    gen_bus = np.zeros(len(c.vm), bool)
    gen_bus[c.unit_bus] = True
    cap_g = np.zeros(len(c.vm))
    np.add.at(cap_g, c.unit_bus, c.pmax)
    ny = c.is_ny.astype(bool)
    for t in range(day.onset, day.onset + day.length):
        z_b = day.z_benign[t]
        a = wrap_angles(day.z[t] - z_b, plan.kind)
        sigma = sr * np.abs(z_b) + sa
        j_open.append(_wls(meas, day.z[t].astype(float), sigma, day.vm[t].copy(), day.va[t].copy(), ref)[0])
        units = day.unit_p[t] + a_resp.dev[t] + a_resp.reg[t] + a_resp.uncovered[t] * share
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
            j_adapt.append(np.nan)
            continue
        h_true = meas.h(day.vm[t], day.va[t])
        vm_f, va_f, err = _false_state(meas, h_true + a, day.vm[t].copy(), day.va[t].copy(), ref)
        fit_err.append(err)
        dvm, dva = vm_f - day.vm[t], va_f - day.va[t]
        h_cl = meas.h(r["vm"], r["va"])
        a_ad = wrap_angles(meas.h(r["vm"] + dvm, r["va"] + dva) - h_cl, plan.kind)
        noise = wrap_angles(z_b - h_true, plan.kind)
        z = h_cl + a_ad + noise
        j_adapt.append(_wls(meas, z, sigma, r["vm"].copy(), r["va"].copy(), ref)[0])
        beyond = np.abs(a_ad) > sr * np.abs(h_cl) + sa
        designed = np.abs(a) > sigma
        extra.append(int((beyond & ~designed).sum()))
        p_d = net.injections(vm_f, va_f)[0] - net.injections(day.vm[t], day.va[t])[0]
        p_a = net.injections(r["vm"] + dvm, r["va"] + dva)[0] - net.injections(r["vm"], r["va"])[0]
        dinj.append(float(np.abs(p_a - p_d).max()))
        load = np.maximum(-net.injections(r["vm"], r["va"])[0], 0.0)
        bound = np.where(gen_bus, 0.03 * cap_g, 0.1 * load)
        live = ny & (bound > 1.0) & ((np.abs(p_d) > 0.5) | (np.abs(p_a) > 0.5))
        if live.any():
            excess.append(float(((np.abs(p_a[live]) - np.abs(p_d[live])) / bound[live]).max()))
    jo, ja = np.array(j_open), np.array(j_adapt)
    theta, k, n = rule
    return {
        "file": Path(path).name,
        "onset": day.onset,
        "mode": mode,
        "variant": "at_limit" if Path(path).stem.endswith("_0") else "damaging",
        "first_open": first_alarm(jo, theta, k, n),
        "first_adaptive": first_alarm(ja, theta, k, n),
        "extra_meters_max": int(max(extra)) if extra else -1,
        "dinj_max_mw": float(max(dinj)) if dinj else np.nan,
        "fit_err_max": float(max(fit_err)) if fit_err else np.nan,
        "bound_excess": float(max(excess)) if excess else np.nan,
        "j_open": json.dumps([round(float(x), 2) for x in jo]),
        "j_adaptive": json.dumps([round(float(x), 2) for x in ja]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("studies", nargs="+")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default="data/NYgrid/Data")
    ap.add_argument("--calib", default="results/nygrid.tie_check.json")
    ap.add_argument("--mode", default="trusting")
    ap.add_argument("--theta", type=float, default=428.7)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N attack days (a quick check)")
    args = ap.parse_args()
    cal = json.loads(Path(args.calib).read_text())
    calib = (cal["mean_mw"], cal["sd_mw"])
    files = sorted({str(p) for s in args.studies for p in Path(s, "days").glob("am_*_*.h5")})
    if args.limit:
        files = files[: args.limit]
    rule = (args.theta, args.k, args.n)
    t0 = time.perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_job, f, args.pool, args.data, calib, args.mode, rule) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is not None:
                rows.append(r)
                print(
                    f"{i}/{len(files)} {r['file']} open {r['first_open']} adaptive {r['first_adaptive']} "
                    f"extra {r['extra_meters_max']} dinj {r['dinj_max_mw']:.1f} excess {r['bound_excess']:.3f} "
                    f"[{time.perf_counter() - t0:.0f}s]",
                    flush=True,
                )
            pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
