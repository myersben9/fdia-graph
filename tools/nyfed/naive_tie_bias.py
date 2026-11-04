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

from fdia_graph.nyfed.grid.network import Network  # noqa: E402
from fdia_graph.nyfed.operator.response import operator_day, tie_meters  # noqa: E402
from fdia_graph.nyfed.sensing.measure import Measurement  # noqa: E402


def _job(path: str, pool: str, data: str, rule: tuple[float, int, int]) -> dict | None:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:
        return None
    c, plan = day.case, day.plan
    meas = Measurement(Network(c), plan)
    ref = int(np.flatnonzero(c.bus_type == REF)[0])
    _, _, rows = tie_meters(c, plan)
    sr, sa = plan.sigma_rel, plan.sigma_abs
    js, bias_mw = [], []
    for t in range(day.onset, day.onset + day.length):
        z_b = day.z_benign[t]
        a = np.zeros_like(z_b)
        a[rows] = (day.z[t] - z_b)[rows]  # the stealthy attack's tie readings, nothing else
        z = z_b + a
        j, _ = _wls(meas, z, sr * np.abs(z_b) + sa, day.vm[t].copy(), day.va[t].copy(), ref)
        js.append(j)
        bias_mw.append(float(np.abs(a[rows]).sum()))
    j_ = np.array(js)
    theta, k, n = rule
    return {
        "file": Path(path).name,
        "onset": day.onset,
        "variant": "at_limit" if Path(path).stem.endswith("_0") else "damaging",
        "ties": int(len(rows)),
        "tie_change_max": float(np.max(bias_mw)),
        "j_naive": json.dumps([round(float(x), 2) for x in j_]),
        "first": first_alarm(j_, theta, k, n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("studies", nargs="+")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default="data/NYgrid/Data")
    ap.add_argument("--theta", type=float, default=428.7)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N attack days (a quick check)")
    args = ap.parse_args()
    files = sorted({str(p) for s in args.studies for p in Path(s, "days").glob("am_*_*.h5")})
    if args.limit:
        files = files[: args.limit]
    rule = (args.theta, args.k, args.n)
    t0 = time.perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(_job, f, args.pool, args.data, rule) for f in files]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r is not None:
                rows.append(r)
                print(
                    f"{i}/{len(files)} {r['file']} first {r['first']} [{time.perf_counter() - t0:.0f}s]",
                    flush=True,
                )
            pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
