"""Summaries of the MATLAB AC feasibility runs in tools/matlab/nygrid, written next to their per-hour tables.

    python tools/nygrid_ac_summary.py year       # results/nygrid.ac_year.csv  -> nygrid.ac_year.json
    python tools/nygrid_ac_summary.py ablation   # results/ablation_parts/part_*.csv -> nygrid.ac_ablation.csv + .json
    python tools/nygrid_ac_summary.py ablation_py  # out/ablation/{variant}.csv -> nygrid.ac_ablation_py.csv + .json

The year run is ac_year.m (the recipe against NYgrid's raw 2019 operating points). The ablation run is
ablation.m, which drops one change of the recipe at a time to show which ones AC feasibility needs.
ablation_py is the same test with the Python projection (tools/ac_ablation.py, Ipopt), the solver the pool
uses. Ipopt solves hours MATLAB's MIPS fails on, so the paper quotes this one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _pct(x: pd.Series) -> float:
    return round(100 * float(x.mean()), 2)


def _num(x: float, digits: int) -> float | None:
    """Rounded value, or None (JSON null) for a variant with no feasible hour."""
    return None if pd.isna(x) else round(float(x), digits)


def year() -> dict:
    """Headline numbers of the recipe over 2019. Repeated hours (test runs kept in the worker files) are
    dropped so every hour counts once."""
    path = RESULTS / "nygrid.ac_year.csv"
    df = pd.read_csv(path).drop_duplicates("ts").sort_values("ts")
    df.to_csv(path, index=False)
    ok = df[df.proj_ok == 1]
    share = ok.redisp_ny / ok.ny_gen
    return {
        "hours": len(df),
        "raw_ac_converged": int(df.raw_ac_ok.sum()),
        "fixes_without_projection_converged": int(df.fix_plain_ok.sum()),
        "recipe_ac_feasible": int(df.proj_ok.sum()),
        "recipe_feasible_pct": _pct(df.proj_ok),
        "redispatch_ny_mw_median": round(ok.redisp_ny.median(), 1),
        "redispatch_ny_mw_p95": round(ok.redisp_ny.quantile(0.95), 1),
        "redispatch_ny_pct_of_ny_gen_median": round(100 * share.median(), 2),
        "redispatch_ny_pct_of_ny_gen_p95": round(100 * share.quantile(0.95), 2),
        "ny_vmin": round(ok.ny_vmin.min(), 4),
        "ny_vmax": round(ok.ny_vmax.max(), 4),
        "ac_max_angle_diff_deg_p95": round(ok.ac_dthmax.quantile(0.95), 1),
        "ac_losses_mw_median": round(ok.loss_ac.median(), 1),
        "interface_mae_dc_median": round(ok.dc_ifmae.median(), 4),
        "interface_mae_ac_median": round(ok.ac_ifmae.median(), 4),
        "infeasible_hours": df.loc[df.proj_ok != 1, "ts"].tolist(),
    }


def ablation() -> dict:
    """Per variant: hours AC-feasible, the re-dispatch it costs, and how many hours the full recipe solves
    that the variant loses (and the reverse)."""
    parts = sorted((RESULTS / "ablation_parts").glob("part_*.csv"))
    df = pd.concat([pd.read_csv(p, dtype={"nyload": str}) for p in parts], ignore_index=True)
    err = df[df.nyload.str.startswith("ERR")]
    df = df.drop(err.index).astype({"nyload": float}).drop_duplicates(["ts", "variant"])
    df = df.sort_values(["ts", "variant"])
    df.to_csv(RESULTS / "nygrid.ac_ablation.csv", index=False)

    wide = df.pivot(index="ts", columns="variant", values="ok")
    out = {"hours": int(wide.shape[0]), "hours_with_errors": int(err.ts.nunique()), "variants": {}}
    for v, g in df.groupby("variant"):
        ok = g[g.ok == 1]
        share = ok.redisp_ny / ok.ny_gen
        out["variants"][v] = {
            "ac_feasible": int(g.ok.sum()),
            "feasible_pct": _pct(g.ok),
            "lost_vs_full": int(((wide.full == 1) & (wide[v] == 0)).sum()),
            "gained_vs_full": int(((wide.full == 0) & (wide[v] == 1)).sum()),
            "redispatch_ny_mw_median": _num(ok.redisp_ny.median(), 1),
            "redispatch_ny_pct_of_ny_gen_median": _num(100 * share.median(), 2),
            "redispatch_ext_mw_median": _num(ok.redisp_ext.median(), 1),
            "ny_vmin": _num(ok.ny_vmin.min(), 4),
        }
    return out


def ablation_py() -> dict:
    """The Python drop-one test, in the same layout as `ablation`."""
    variants = ["full", "noJ", "noL", "noE", "projOnly"]
    df = pd.concat(
        [pd.read_csv(RESULTS.parent / "out" / "ablation" / f"{v}.csv").assign(variant=v) for v in variants],
        ignore_index=True,
    ).sort_values(["ts", "variant"])
    df.to_csv(RESULTS / "nygrid.ac_ablation_py.csv", index=False)

    wide = df.pivot(index="ts", columns="variant", values="ok")
    out = {"hours": int(wide.shape[0]), "solver": "ipopt", "variants": {}}
    for v in variants:
        g = df[df.variant == v]
        ok = g[g.ok]
        share = ok.redisp_ny / ok.ny_gen
        out["variants"][v] = {
            "ac_feasible": int(g.ok.sum()),
            "feasible_pct": _pct(g.ok),
            "lost_vs_full": int((wide.full & ~wide[v]).sum()),
            "gained_vs_full": int((~wide.full & wide[v]).sum()),
            "redispatch_ny_mw_median": _num(ok.redisp_ny.median(), 1),
            "redispatch_ny_pct_of_ny_gen_median": _num(100 * share.median(), 2),
            "redispatch_ny_pct_of_ny_gen_p95": _num(100 * share.quantile(0.95), 2),
            "redispatch_ext_mw_median": _num(ok.redisp_ext.median(), 1),
            "ny_vmin": _num(ok.ny_vmin.min(), 4),
            "ny_vmax": _num(ok.ny_vmax.max(), 4),
        }
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ablation"
    summary = {"year": year, "ablation": ablation, "ablation_py": ablation_py}[which]()
    (RESULTS / f"nygrid.ac_{which}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "infeasible_hours"}, indent=1))
