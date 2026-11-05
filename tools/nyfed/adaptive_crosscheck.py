"""Does the attacker that follows the closed loop (adaptive_bdd.py, N82) still show the interchange check its false
interchange and the voltage alarm its high reading? For every attack window, against the trusting operator: the
largest gap between the summed New York-end tie P change of the adaptive vector and the designed one, and the largest
voltage reading of each (noise free), to compare with the voltage alarm (N74).

    python tools/nyfed/adaptive_crosscheck.py OUT.csv STUDY [--pool out/pool_2019_v3.h5] [--data data/NYgrid/Data]
                                              [--dispatch cost]

Run from the folder whose `data/` holds the inputs. One row per attack window.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fdia_graph.nyfed.attacks.attacker import wrap_angles
from fdia_graph.nyfed.detect import exact_state, reference_bus
from fdia_graph.nyfed.experiments import DATA, add_dispatch_argument, study_days
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.response import TrueGrid, entry_share, operator_day, respond, tie_meters
from fdia_graph.nyfed.operator.run import AC_DISPATCH
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.meters import KINDS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("study")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default=DATA)
    add_dispatch_argument(ap)
    args = ap.parse_args()
    with open(args.out, "w") as out:
        out.write("file,tie_diff_max,tie_designed_max,vmax_designed,vmax_adaptive,dispatch\n")
        for p in map(Path, study_days([args.study])):
            day = operator_day(p, args.pool, args.data)
            c, plan = day.case, day.plan
            resp = respond(
                day, mode="trusting", calib=None, closed_loop=True, dispatch=args.dispatch
            ).attacked
            grid, share = TrueGrid(day, ac=args.dispatch in AC_DISPATCH), entry_share(c, "capacity")
            meas = Measurement(Network(c), plan)
            ref = reference_bus(c)
            _, _, rows = tie_meters(c, plan)
            vrows = np.flatnonzero(plan.kind == KINDS.index("v"))
            tie_d, tie_a, v_d, v_a = [], [], [], []
            for t in range(day.onset, day.onset + day.length):
                x = grid.run_state(resp, t, share)  # the true state once the operator has acted
                if x is None:
                    continue
                h_true = meas.h(day.vm[t], day.va[t])
                a = wrap_angles(day.z[t] - day.z_benign[t], plan.kind)
                vm_f, va_f, _ = exact_state(meas, h_true + a, day.vm[t].copy(), day.va[t].copy(), ref)
                h_cl = meas.h(*x)
                h_ad = meas.h(x[0] + vm_f - day.vm[t], x[1] + va_f - day.va[t])
                tie_d.append(a[rows].sum())
                tie_a.append((h_ad - h_cl)[rows].sum())
                v_d.append((h_true + a)[vrows].max())
                v_a.append(h_ad[vrows].max())
            td, ta = np.array(tie_d), np.array(tie_a)
            out.write(
                f"{p.name},{np.abs(ta - td).max():.2f},{np.abs(td).max():.1f},{max(v_d):.5f},{max(v_a):.5f},"
                f"{args.dispatch}\n"
            )
            out.flush()
            print(f"{p.name}: tie gap {np.abs(ta - td).max():.1f} MW, vmax {max(v_a):.4f}", flush=True)


if __name__ == "__main__":
    main()
