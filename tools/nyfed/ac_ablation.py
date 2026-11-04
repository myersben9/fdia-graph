"""Drop-one test of the AC changes with the Python projection (Ipopt), the solver the pool uses. The
MATLAB version is tools/matlab/nygrid/ablation.m. One variant per process:

    python tools/ac_ablation.py full|noJ|noL|noE|projOnly OUT.csv [--hours N]

full      zone J split + 765 kV line + external areas at base load, then the projection
noJ       without the zone J split
noL       without the 765 kV line
noE       external areas scaled with NY load as NYgrid does
projOnly  the projection on NYgrid's case with none of the three changes

Writes one row per hour (ok, NY redispatch, NY generation, NY voltage range) and prints a summary.
Run from the nyfed repo root.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import fdia_graph.nyfed.scenario.dispatch as dispatch
from fdia_graph.nyfed.grid.fixes import restore_765kv, split_zone_j
from fdia_graph.nyfed.grid.nygrid import modify_mpc, nygrid_units, read_npcc
from fdia_graph.nyfed.scenario.inputs import read_nygrid
from fdia_graph.nyfed.scenario.solver import AcSolver

DATA = Path.cwd() / "data" / "NYgrid" / "Data"


def case_for(variant: str):
    base = nygrid_units(modify_mpc(read_npcc(DATA / "npcc.mat")), DATA)
    split = variant not in ("noJ", "projOnly")
    line = variant not in ("noL", "projOnly")
    case = split_zone_j(base) if split else base
    return restore_765kv(case) if line else case


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=["full", "noJ", "noL", "noE", "projOnly"])
    ap.add_argument("out")
    ap.add_argument("--hours", type=int, default=0)
    a = ap.parse_args()
    case = case_for(a.variant)
    hours = read_nygrid(DATA, 2019)
    if a.hours:
        hours = hours[: a.hours]
    # noE and projOnly keep NYgrid's own external scaling: no fix holding the external areas at base load
    d = dispatch.hourly_dispatch(case, hours, hold_externals=a.variant not in ("noE", "projOnly"))
    solver = AcSolver(case)
    ny_unit = case.is_ny[case.unit_bus]
    rows, t0 = [], time.perf_counter()
    for k, h in enumerate(hours):
        r = solver.project(d.pd[k], d.qd[k], d.unit_p[k], d.unit_on[k], d.hvdc_p[k])
        row = {"ts": str(h.time)[:13], "ok": r is not None, "ny_gen": float(d.unit_p[k][ny_unit].sum())}
        if r is not None:
            dp = r["unit_p"] - d.unit_p[k]
            row.update(
                redisp_ny=float(np.abs(dp[ny_unit]).sum() / 2),
                redisp_ext=float(np.abs(dp[~ny_unit]).sum() / 2),
                ny_vmin=float(r["vm"][case.is_ny].min()),
                ny_vmax=float(r["vm"][case.is_ny].max()),
            )
        rows.append(row)
        if k < 3 or k % 500 == 0:
            print(
                f"{a.variant} {row['ts']} ok={row['ok']} redisp_ny={row.get('redisp_ny', float('nan')):.0f} "
                f"[{time.perf_counter() - t0:.0f}s]",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    ok = df[df.ok]
    # No feasible hour leaves no redispatch columns, so the median is NaN rather than an error.
    share = ok.redisp_ny / ok.ny_gen if len(ok) else pd.Series(dtype=float)
    print(
        f"SUMMARY {a.variant}: {int(df.ok.sum())}/{len(df)} hours feasible, median NY redispatch "
        f"{share.median() * 100:.2f}% [{time.perf_counter() - t0:.0f}s]",
        flush=True,
    )


if __name__ == "__main__":
    main()
