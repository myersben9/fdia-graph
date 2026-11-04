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

    python tools/nyfed/adaptive_bdd.py OUT.csv STUDY [STUDY ...] [--mode trusting] [--dispatch cost] [--workers 8]

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

from fdia_graph.nyfed.attacks.attacker import wrap_angles
from fdia_graph.nyfed.detect import PersistenceRule, exact_state, reference_bus, wls
from fdia_graph.nyfed.experiments import add_study_arguments, load_calib, run_parallel, study_days, variant
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.response import TrueGrid, entry_share, operator_day, respond
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.noise import NoiseModel


def _job(path: str, pool: str, data: str, calib, mode: str, rule, dispatch: str = "movement") -> dict | None:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:
        return None
    c, plan = day.case, day.plan
    resp = respond(day, mode=mode, calib=calib, closed_loop=True, dispatch=dispatch)
    a_resp = resp.attacked
    grid = TrueGrid(day)
    share = entry_share(c, "capacity")
    meas = Measurement(Network(c), plan)
    net = Network(c)
    ref = reference_bus(c)
    noise_model = NoiseModel(plan)
    j_open, j_adapt, extra, dinj, fit_err, excess = [], [], [], [], [], []
    gen_bus = np.zeros(len(c.vm), bool)
    gen_bus[c.unit_bus] = True
    cap_g = np.zeros(len(c.vm))
    np.add.at(cap_g, c.unit_bus, c.pmax)
    ny = c.is_ny.astype(bool)
    for t in range(day.onset, day.onset + day.length):
        z_b = day.z_benign[t]
        a = wrap_angles(day.z[t] - z_b, plan.kind)
        sigma = noise_model.sigmas(z_b)
        j_open.append(wls(meas, day.z[t].astype(float), sigma, day.vm[t], day.va[t], ref).j)
        x = grid.run_state(a_resp, t, share)  # the true state once the operator has acted
        if x is None:
            j_adapt.append(np.nan)
            continue
        h_true = meas.h(day.vm[t], day.va[t])
        vm_f, va_f, err = exact_state(meas, h_true + a, day.vm[t].copy(), day.va[t].copy(), ref)
        fit_err.append(err)
        dvm, dva = vm_f - day.vm[t], va_f - day.va[t]
        vm_cl, va_cl = x
        h_cl = meas.h(vm_cl, va_cl)
        a_ad = wrap_angles(meas.h(vm_cl + dvm, va_cl + dva) - h_cl, plan.kind)
        noise = wrap_angles(z_b - h_true, plan.kind)
        z = h_cl + a_ad + noise
        j_adapt.append(wls(meas, z, sigma, vm_cl, va_cl, ref).j)
        beyond = np.abs(a_ad) > noise_model.sigmas(h_cl)
        designed = np.abs(a) > sigma
        extra.append(int((beyond & ~designed).sum()))
        p_d = net.injections(vm_f, va_f)[0] - net.injections(day.vm[t], day.va[t])[0]
        p_a = net.injections(vm_cl + dvm, va_cl + dva)[0] - net.injections(vm_cl, va_cl)[0]
        dinj.append(float(np.abs(p_a - p_d).max()))
        load = np.maximum(-net.injections(vm_cl, va_cl)[0], 0.0)
        bound = np.where(gen_bus, 0.03 * cap_g, 0.1 * load)
        live = ny & (bound > 1.0) & ((np.abs(p_d) > 0.5) | (np.abs(p_a) > 0.5))
        if live.any():
            excess.append(float(((np.abs(p_a[live]) - np.abs(p_d[live])) / bound[live]).max()))
    jo, ja = np.array(j_open), np.array(j_adapt)
    return {
        "file": Path(path).name,
        "onset": day.onset,
        "mode": mode,
        "dispatch": dispatch,
        "variant": variant(path),
        "first_open": rule.first_alarm(jo),
        "first_adaptive": rule.first_alarm(ja),
        "extra_meters_max": int(max(extra)) if extra else -1,
        "dinj_max_mw": float(max(dinj)) if dinj else np.nan,
        "fit_err_max": float(max(fit_err)) if fit_err else np.nan,
        "bound_excess": float(max(excess)) if excess else np.nan,
        "j_open": json.dumps([round(float(x), 2) for x in jo]),
        "j_adaptive": json.dumps([round(float(x), 2) for x in ja]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    add_study_arguments(ap)
    ap.add_argument("--mode", default="trusting")
    ap.add_argument("--theta", type=float, default=428.7)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    calib = load_calib(args.calib)
    rule = PersistenceRule(args.theta, args.k, args.n)
    files = study_days(args.studies, args.limit)
    tasks = [(f, args.pool, args.data, calib, args.mode, rule, args.dispatch) for f in files]
    run_parallel(
        _job,
        tasks,
        args.out,
        workers=args.workers,
        describe=lambda r: (
            f"{r['file']} open {r['first_open']} adaptive {r['first_adaptive']} extra {r['extra_meters_max']} "
            f"dinj {r['dinj_max_mw']:.1f} excess {r['bound_excess']:.3f}"
        ),
    )


if __name__ == "__main__":
    main()
