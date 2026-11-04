"""What an idealized operator would predict for each attack, against what the closed-loop operator causes.

The consequence studies of load redistribution attacks take one snapshot: the operator solves a least-cost DC
dispatch on the false reading, and the true flows follow from that dispatch with the reference bus balancing the
true load. There is no area control error, no regulation and no neighbor, so the shortfall a false load drop leaves
is met at the reference bus. This tool solves that dispatch at the last rise scan of each attack day (`DcSced` with
the least-cost objective and no ramp history), attacked less benign, and reports the true change of every interface
it predicts, and again with the shortfall drawn from the neighbors' units by capacity as the closed loop draws it
(`shared_*`), which separates the slack from the loop. Beside it, the same day through the closed-loop trusting operator (`respond`) gives the largest true
rise and drop of the attacked interface, the largest rise of any other and the inadvertent interchange, which the
snapshot cannot see.

    python tools/nyfed/idealized_operator.py OUT.csv STUDY [STUDY ...] [--workers 8]

Run from the folder whose `data/` holds the inputs. One row per attack day.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fdia_graph.nyfed.experiments import add_study_arguments, load_calib, run_parallel, study_days, variant
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.policy import entry_share
from fdia_graph.nyfed.operator.response import operator_day, respond, seen_interfaces, summarize
from fdia_graph.nyfed.operator.run import QUICK_FUELS, QUICK_MAX_MW, QUICK_ZONES
from fdia_graph.nyfed.operator.sced import DcSced
from fdia_graph.nyfed.operator.truegrid import TrueGrid
from fdia_graph.nyfed.sensing.meters import KINDS

RISE = 16  # scans of the rise; the snapshot is taken at the last of them


def snapshot(day, slack: str = "reference") -> np.ndarray:
    """`[I]` each interface's true flow change (MW) the single-snapshot operator predicts at the last rise scan:
    a least-cost DC dispatch on the false reading, attacked less benign, its injections balanced at the reference
    bus, whose shift factors are zero."""
    c = day.case
    t = day.onset + RISE - 1
    pf = Network(c).flows(day.vm[t : t + 1], day.va[t : t + 1])[0]
    true_iface = np.array([it.model_flow(pf)[0] for it in day.interfaces])
    limit = true_iface + np.array([it.limit[t] - it.flow[t] for it in day.interfaces])
    p = np.flatnonzero(day.plan.kind == KINDS.index("p"))
    load_change = -float((day.z[t, p] - day.z_benign[t, p]).sum())
    fuel, zone = c.fuel.astype(str), c.zone.astype(str)[c.unit_bus]
    quick = np.isin(fuel, QUICK_FUELS) & np.isin(zone, QUICK_ZONES) & (c.pmax <= QUICK_MAX_MW)
    load = np.clip(TrueGrid(day).pd, 0.0, None) * c.is_ny
    sced = DcSced(c, day.unit_p, day.cost, day.interfaces, load, quick_start=quick, objective="cost")
    seen_a = seen_interfaces(day.plan, day.interfaces, day.z[t : t + 1])[0]
    seen_b = seen_interfaces(day.plan, day.interfaces, day.z_benign[t : t + 1])[0]
    a = sced.step(t, seen_a, limit, load_change, None)
    b = sced.step(t, seen_b, limit, 0.0, None)
    inj = np.zeros(len(c.bus_id))
    np.add.at(inj, c.unit_bus, a.dev - b.dev)
    inj += a.shed - b.shed  # shed load raises the bus's net injection
    if (
        slack == "neighbors"
    ):  # the shortfall from the neighbors' units by capacity, as in the closed loop (N69)
        share = entry_share(c, "capacity")
        np.add.at(inj, c.unit_bus, -inj.sum() * share)
    return sced.sf @ inj


def _job(path: str, pool: str, data: str, calib: tuple[float, float]) -> dict | None:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:
        return None
    predicted = snapshot(day)
    shared = snapshot(day, "neighbors")
    s = summarize(respond(day, mode="trusting", calib=calib, closed_loop=True))
    return {
        "file": Path(path).name,
        "study": str(Path(path).parents[1]),
        "onset": int(Path(path).stem.split("_")[1]),
        "variant": variant(path),
        "snapshot_attacked_mw": float(predicted[0]),
        "snapshot_other_up_mw": float(max(predicted[1:].max(), 0.0)) if len(predicted) > 1 else 0.0,
        "shared_attacked_mw": float(shared[0]),
        "shared_other_up_mw": float(max(shared[1:].max(), 0.0)) if len(shared) > 1 else 0.0,
        "closed_up_mw": s["iface_up_mw"],
        "closed_down_mw": -s["iface_down_mw"],
        "closed_other_up_mw": s["other_iface_up_mw"],
        "closed_inadvertent_mw": s["inadvertent_mw"],
        "closed_inadvertent_mwh": s["inadvertent_mwh"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_study_arguments(ap)
    args = ap.parse_args()
    files = study_days(args.studies, args.limit)
    if not files:
        raise SystemExit(f"no attack days (days/am_*_*.h5) under {args.studies}")
    calib = load_calib(args.calib)
    run_parallel(
        _job, [(f, args.pool, args.data, calib) for f in files], args.out, workers=args.workers, every=10
    )


if __name__ == "__main__":
    main()
