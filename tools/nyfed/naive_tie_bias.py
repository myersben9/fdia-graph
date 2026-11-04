"""What stealth buys (N80): the naive baseline that falsifies only the New York-end tie meters. For every attack
window it adds to the benign scan the change the stealthy attack makes on those meters alone, so the operator's
interchange reading and the interchange check see exactly what they see under the stealthy attack, and fits the scan
by AC weighted least squares on the full accuracy-class variance, as closed_loop_bdd.py does. The residuals are taken
before the operator acts, so they show how soon the residual test catches an attack that is not consistent with a
state. The first alarm follows the persistence rule of bdd_rule.py.

    python tools/nyfed/naive_tie_bias.py OUT.csv STUDY [STUDY ...] [--theta 428.7 --k 2 --n 3] [--workers 8]

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

from fdia_graph.nyfed.detect import PersistenceRule, reference_bus, wls
from fdia_graph.nyfed.experiments import add_study_arguments, run_parallel, study_days, variant
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.response import operator_day, tie_meters
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.noise import NoiseModel


def _job(path: str, pool: str, data: str, rule: PersistenceRule) -> dict | None:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:
        return None
    c, plan = day.case, day.plan
    meas = Measurement(Network(c), plan)
    ref = reference_bus(c)
    _, _, rows = tie_meters(c, plan)
    noise_model = NoiseModel(plan)
    js, bias_mw = [], []
    for t in range(day.onset, day.onset + day.length):
        z_b = day.z_benign[t]
        a = np.zeros_like(z_b)
        a[rows] = (day.z[t] - z_b)[rows]  # the stealthy attack's tie readings, nothing else
        z = z_b + a
        js.append(wls(meas, z, noise_model.sigmas(z_b), day.vm[t], day.va[t], ref).j)
        bias_mw.append(float(np.abs(a[rows]).sum()))
    j_ = np.array(js)
    return {
        "file": Path(path).name,
        "onset": day.onset,
        "variant": variant(path),
        "ties": int(len(rows)),
        "tie_change_max": float(np.max(bias_mw)),
        "j_naive": json.dumps([round(float(x), 2) for x in j_]),
        "first": rule.first_alarm(j_),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    add_study_arguments(ap, calib=False)
    ap.add_argument("--theta", type=float, default=428.7)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    rule = PersistenceRule(args.theta, args.k, args.n)
    tasks = [(f, args.pool, args.data, rule) for f in study_days(args.studies, args.limit)]
    run_parallel(
        _job, tasks, args.out, workers=args.workers, describe=lambda r: f"{r['file']} first {r['first']}"
    )


if __name__ == "__main__":
    main()
