"""Build the P5 pool for a time window as NYISO's real-time dispatch (N32, N39), in chunks run in parallel.

    python tools/run_pool.py START END OUT.h5 [--loads LOADS.npz] [--workers N] [--chunk-days D]
                                             [--frames N] [--compare-hours] [--seam-frames N]

START/END are local-clock stamps (END exclusive), e.g. 2019-01-07 2019-01-14. Each chunk (a week by
default) starts from NYgrid's hourly AC point at its first hour and dispatches its frames in order; chunks
run in `--workers` spawn processes. The seams are then repaired so the year is one continuous dispatch:
chunk c's first frames are solved again from chunk c-1's last state until the re-solved thermal dispatch
meets the stored one within 0.5 MW per unit (at most `--seam-frames`, 288 = 24 h), and the re-solved frames
replace the stored ones. A seam that does not merge in that time has its whole chunk solved again from the
chunk before, and the next seam is repaired again from the new end. `--loads` is the cached LoadSeries of the year (built with LoadModel when
missing). `--frames` stops after N frames (a smoke run). `--compare-hours` also projects every hour of the
window and prints the dispatch cost against NYgrid's hourly AC point at the same prices. The grid keeps
NYgrid's unit reactive limits (N38). Run from the nyfed repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import time

# One thread per worker and per Ipopt child: the OPF is small, so threads do not speed one solve (621 against
# 685 ms per frame measured), and many workers each running threaded solves oversubscribe the machine.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from fdia_graph.engine.attacks.attacker import HYDRO_RAMP_PER_MIN, IMPORT_RAMP_MW_PER_MIN
from fdia_graph.federated.utilities import assign_clients
from fdia_graph.generation.nygrid.config import Settings, rng
from fdia_graph.generation.nygrid.dispatch import hourly_dispatch
from fdia_graph.generation.nygrid.fixes import restore_765kv, split_zone_j
from fdia_graph.generation.nygrid.inputs import fetch_nyiso, read_nygrid
from fdia_graph.generation.nygrid.io import read_pool, write_pool
from fdia_graph.generation.nygrid.loads import LoadModel, LoadSeries
from fdia_graph.generation.nygrid.npcc import modify_mpc, nygrid_units, read_npcc
from fdia_graph.generation.nygrid.pool import SOLVED, Pool, PoolBuilder
from fdia_graph.generation.nygrid.solver import AcSolver

ROOT = Path.cwd()
DATA = ROOT / "data" / "NYgrid" / "Data"
TERR = ROOT / "data" / "territories"
FRAME_MIN = 5.0
SEAM_TOL_MW = 0.5  # a seam is merged once every thermal unit is within this of the stored dispatch
HELD = ("Nuclear", "Hydro", "Import")  # follow NYgrid's hourly output (N39)
EXTRAS = ("slack_mw", "cost", "ramp_bound", "q_fallback", "anchor_dev")


def log(*a):
    print(*a, flush=True)


def _case():
    # The case the MATLAB recipe and the P4 tests use: zone J split, 765 kV line, no HVDC links.
    return restore_765kv(split_zone_j(nygrid_units(modify_mpc(read_npcc(DATA / "npcc.mat")), DATA)))


def unit_ramps(case) -> np.ndarray:
    """[G] MW per frame: the per-unit rules of attacks.generator_ramps (N34), unit by unit as the OPF
    constrains each unit: maxRampAgc, hydro without data 10% of nameplate per minute, imports 20 MW/min,
    nuclear 0, other units without data unbounded. Under N39 only the thermal units' ramps bind."""
    r = np.asarray(case.ramp, float).copy()
    r = np.where((case.fuel == "Hydro") & np.isnan(r), HYDRO_RAMP_PER_MIN * case.pmax, r)
    r = np.where((case.fuel == "Import") & np.isnan(r), IMPORT_RAMP_MW_PER_MIN, r)
    r = np.where(case.fuel == "Nuclear", 0.0, r) * FRAME_MIN
    return np.where(np.isnan(r), np.inf, r)


def _builder(time_, p, q):
    """A PoolBuilder over these frames, with the hours around them."""
    case = _case()
    hours = read_nygrid(DATA, 2019)
    stamps = pd.DatetimeIndex([h.time for h in hours])
    t = pd.DatetimeIndex(time_)
    h_idx = np.flatnonzero(
        (stamps >= t[0].floor("h") - pd.Timedelta("1h")) & (stamps <= t[-1].floor("h") + pd.Timedelta("1h"))
    )
    picked = [hours[i] for i in h_idx]
    disp = hourly_dispatch(case, picked)
    renew = np.array([h.renew_mw for h in picked])
    b = PoolBuilder(AcSolver(case), LoadSeries(time=time_, p=p, q=q), disp, renew, unit_ramps(case))
    return b, stamps[h_idx]


def _chunk(time_, p, q, compare):
    """One chunk: dispatch and solve its frames in order from its first hour's AC point."""
    b, stamps = _builder(time_, p, q)
    pool = b.build()
    extra = {k: getattr(b, k) for k in EXTRAS}
    extra["final"] = b.final
    extra["proj_ok"] = {str(stamps[h])[:13]: r is not None for h, r in b.hour_points.items()}
    if compare:  # NYgrid's hourly AC point of every hour, priced like the frames
        d, hourly = b.dispatch, {}
        for k in range(len(d.hour)):
            r = b.solver.project(d.pd[k], d.qd[k], d.unit_p[k], d.unit_on[k], d.hvdc_p[k])
            hourly[str(stamps[k])] = None if r is None else float(d.unit_cost[k] @ r["unit_p"])
        extra["hourly_cost"] = hourly
    return pool, b.report(), extra


def _repair(time_, p, q, start, stored, thermal):
    """Chunk c's first frames solved again from chunk c-1's last state. Returns the re-solved pool and its
    extras, and per frame the largest thermal-unit gap to the stored dispatch (NaN where either is unsolved)."""
    b, _ = _builder(time_, p, q)
    pool = b.build(start=start)
    gap = np.nanmax(np.abs(pool.unit_p[:, thermal] - stored[:, thermal]), axis=1)
    gap[~(np.isfinite(pool.unit_p).all(1) & np.isfinite(stored).all(1))] = np.nan
    return pool, {k: getattr(b, k) for k in EXTRAS}, gap, b.final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start"), ap.add_argument("end"), ap.add_argument("out")
    ap.add_argument("--loads", default=str(ROOT / "out" / "loads_2019_v2.npz"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunk-days", type=float, default=7)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--compare-hours", action="store_true")
    ap.add_argument("--seam-frames", type=int, default=288)
    args = ap.parse_args()
    t0 = time.perf_counter()

    case = _case()
    ramp = unit_ramps(case)
    held = np.isin(case.fuel, HELD)
    thermal = ~held
    log(
        f"case {len(case.bus_id)} buses {len(case.f)} branches {len(case.unit_bus)} units; re-dispatched (thermal) "
        f"{thermal.sum()}, held at NYgrid's hourly output {held.sum()}; NYgrid reactive limits (N38)"
    )

    # Loads over the full year (bus sizes come from the yearly zone peaks), cached since the aggregation
    # is the slow part, then cut to the window.
    cache = Path(args.loads)
    if cache.exists():
        z = np.load(cache)
        series = LoadSeries(time=z["time"], p=z["p"], q=z["q"])
    else:
        own = assign_clients(
            case,
            json.load(open(TERR / "nys_electric_territories.geojson", encoding="utf-8")),
            TERR / "ny_bus_locations.csv",
        )
        model = LoadModel(case, own, DATA, TERR / "ny_bus_locations.csv", ROOT / "data" / "loads")
        series = model.profiles(fetch_nyiso(2019, ROOT / "data" / "nyiso"), rng(Settings(), "loads"))
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, time=series.time, p=series.p, q=series.q)
    log(
        f"loads {cache.name} {series.p.shape}, NaN frames {np.isnan(series.p).any(axis=1).sum()}  [{time.perf_counter() - t0:.0f}s]"
    )

    t = pd.DatetimeIndex(series.time)
    window = np.flatnonzero((t >= args.start) & (t < args.end))
    if args.frames:
        window = window[: args.frames]
    size = int(round(args.chunk_days * 288))
    chunks = [window[c : c + size] for c in range(0, len(window), size)]
    log(f"{len(window)} frames in {len(chunks)} chunks of {size}, {args.workers} workers")

    results = {}
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {
            ex.submit(_chunk, series.time[idx], series.p[idx], series.q[idx], args.compare_hours): k
            for k, idx in enumerate(chunks)
        }
        for fut in as_completed(futs):
            k = futs[fut]
            results[k] = list(fut.result())
            idx, rep, extra = chunks[k], results[k][1], results[k][2]
            log(
                f"{t[idx[0]]} .. {t[idx[-1]]}: "
                + json.dumps({k2: v for k2, v in rep.items() if k2 != "no_load_times"})
                + f"  [{time.perf_counter() - t0:.0f}s]"
            )
            if k == 0:
                pool = results[k][0]
                for f in np.flatnonzero(pool.solve_kind == SOLVED)[:3]:
                    log(
                        f"  frame {pool.time[f]}: V {pool.vm[f].min():.4f}-{pool.vm[f].max():.4f}, cost {extra['cost'][f]:.0f} $/h, "
                        f"ramp-held {extra['ramp_bound'][f]:.3f}, reference bus residual {extra['slack_mw'][f]:.2e} MW"
                    )

        # Seams: chunk c's first frames again from chunk c-1's last state (unchanged by seam c-1's repair,
        # which ends long before chunk c-1 does), so the repairs run in parallel too.
        seams = {}
        futs = {}
        stored = {c: results[c][0].unit_p.copy() for c in results}  # each chunk's dispatch as first solved
        for c in range(1, len(chunks)):
            start = results[c - 1][2]["final"]
            n = min(args.seam_frames, len(chunks[c]))
            idx = chunks[c][:n]
            if start is None:
                seams[c] = {"merged_at": None, "reason": "previous chunk has no solved frame"}
                continue
            futs[
                ex.submit(
                    _repair,
                    series.time[idx],
                    series.p[idx],
                    series.q[idx],
                    start,
                    results[c][0].unit_p[:n],
                    thermal,
                )
            ] = c

        def splice(c, fixed, extra, gap, final):
            """Put the re-solved frames of chunk c in place up to where they meet the stored dispatch."""
            hit = np.flatnonzero(gap <= SEAM_TOL_MW)
            k = int(hit[0]) if len(hit) else len(gap)  # frames [0, k) are replaced
            pool, old_extra = results[c][0], results[c][2]
            results[c][0] = Pool(
                **{
                    name: (
                        np.concatenate([getattr(fixed, name)[:k], getattr(pool, name)[k:]])
                        if name != "time"
                        else pool.time
                    )
                    for name in Pool.__dataclass_fields__
                }
            )
            for key in EXTRAS:
                old_extra[key] = np.concatenate([extra[key][:k], old_extra[key][k:]])
            if k == len(chunks[c]):  # the whole chunk was solved again, so its end is new
                old_extra["final"] = final
            return {
                "merged_at": k if len(hit) else None,
                "frames_replaced": k,
                "gap_at_merge_mw": float(gap[k]) if len(hit) else None,
                "gap_first_frame_mw": float(gap[0]) if np.isfinite(gap[0]) else None,
            }

        for fut in as_completed(futs):
            c = futs[fut]
            seams[c] = splice(c, *fut.result())
            log(f"seam {t[chunks[c][0]]}: " + json.dumps(seams[c]) + f"  [{time.perf_counter() - t0:.0f}s]")

        # A seam that did not merge in --seam-frames: solve its whole chunk again from the chunk before, then
        # repair the next seam from the new end, in time order.
        todo = sorted(
            c
            for c, v in seams.items()
            if v.get("merged_at") is None and results[c - 1][2]["final"] is not None
        )
        while todo:
            c = todo.pop(0)
            idx = chunks[c]
            out = ex.submit(
                _repair,
                series.time[idx],
                series.p[idx],
                series.q[idx],
                results[c - 1][2]["final"],
                stored[c],
                thermal,
            ).result()
            seams[c] = {**splice(c, *out), "whole_chunk": True}
            log(
                f"seam {t[chunks[c][0]]} (whole chunk): "
                + json.dumps(seams[c])
                + f"  [{time.perf_counter() - t0:.0f}s]"
            )
            if c + 1 < len(chunks) and seams[c]["frames_replaced"] == len(idx):
                n = min(args.seam_frames, len(chunks[c + 1]))
                nidx = chunks[c + 1][:n]
                out = ex.submit(
                    _repair,
                    series.time[nidx],
                    series.p[nidx],
                    series.q[nidx],
                    results[c][2]["final"],
                    stored[c + 1][:n],
                    thermal,
                ).result()
                seams[c + 1] = splice(c + 1, *out)
                log(
                    f"seam {t[chunks[c + 1][0]]} (again): "
                    + json.dumps(seams[c + 1])
                    + f"  [{time.perf_counter() - t0:.0f}s]"
                )
                if seams[c + 1]["merged_at"] is None and c + 1 not in todo:
                    todo = sorted(todo + [c + 1])

    order = sorted(results)
    pools = [results[k][0] for k in order]
    reports = [results[k][1] for k in order]
    extras = [results[k][2] for k in order]
    full = Pool(*[np.concatenate([getattr(p, name) for p in pools]) for name in Pool.__dataclass_fields__])
    write_pool(args.out, full)
    back = read_pool(args.out)
    assert np.array_equal(back.solve_kind, full.solve_kind) and np.allclose(back.vm, full.vm, equal_nan=True)
    assert np.allclose(back.unit_p, full.unit_p, equal_nan=True)
    assert (back.time == full.time.astype("datetime64[s]")).all()

    ok = full.solve_kind == SOLVED
    slack = np.abs(np.concatenate([e["slack_mw"] for e in extras]))[ok]
    cost = np.concatenate([e["cost"] for e in extras])
    held_share = np.concatenate([e["ramp_bound"] for e in extras])[ok]
    qfb = np.concatenate([e["q_fallback"] for e in extras])
    # Ramps over the whole year: each solved frame against the solved frame before it, the pair the OPF
    # constrains (a failed frame keeps the dispatch where it was).
    u = full.unit_p[ok]
    excess = np.abs(np.diff(u, axis=0)) - ramp
    th = thermal & np.isfinite(ramp)
    worst = np.unravel_index(np.argmax(excess[:, th]), excess[:, th].shape)
    proj = {}
    for e in extras:
        proj.update(e["proj_ok"])
    summary = {
        "window": [args.start, args.end],
        "frames": int(len(full.time)),
        "solved": int(ok.sum()),
        "solved_pct": round(100 * float(ok.mean()), 3),
        "kinds": {int(k): int((full.solve_kind == k).sum()) for k in np.unique(full.solve_kind)},
        "no_load_times": sum((r["no_load_times"] for r in reports), []),
        "vm_min": float(np.nanmin(full.vm[ok])),
        "vm_max": float(np.nanmax(full.vm[ok])),
        "ny_vm_min": float(np.nanmin(full.vm[ok][:, case.is_ny])),
        "ny_vm_max": float(np.nanmax(full.vm[ok][:, case.is_ny])),
        "slack_mw_median": float(np.median(slack)),
        "slack_mw_p95": float(np.quantile(slack, 0.95)),
        "slack_mw_max": float(slack.max()),
        "cost_median": float(np.nanmedian(cost[ok])),
        "anchor_dev_mw_median": float(np.nanmedian(np.concatenate([e["anchor_dev"] for e in extras])[ok])),
        "anchor_dev_mw_p99": float(
            np.nanpercentile(np.concatenate([e["anchor_dev"] for e in extras])[ok], 99)
        ),
        "anchor_dev_mw_max": float(np.nanmax(np.concatenate([e["anchor_dev"] for e in extras])[ok])),
        "ramp_bound_share": float(np.mean(held_share)),
        "q_fallback_frames": int(qfb.sum()),
        "thermal_ramp_excess_mw_max": float(excess[:, th].max()) if len(u) > 1 else None,
        "thermal_ramp_excess_at": [str(full.time[ok][worst[0] + 1]), int(np.flatnonzero(th)[worst[1]])]
        if len(u) > 1
        else None,
        "seams": {str(t[chunks[c][0]]): v for c, v in sorted(seams.items())},
        # a seam is continuous once merged, or once its whole chunk was solved again from the chunk before
        "seams_unmerged": [
            str(t[chunks[c][0]])
            for c, v in seams.items()
            if v["merged_at"] is None and v["frames_replaced"] < len(chunks[c])
        ],
        "ms_per_frame_worker": round(1000 * sum(r["seconds"] for r in reports) / len(full.time), 2),
        "hours_failed": sum((r["hours_failed"] for r in reports), []),
        "start_hours": len(proj),
        "start_hours_ok": int(sum(proj.values())),
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    if args.compare_hours:
        hourly = {}
        for e in extras:
            hourly.update(e["hourly_cost"])
        hc = pd.Series({pd.Timestamp(k): v for k, v in hourly.items()}, dtype=float)
        fc = pd.Series(cost, index=pd.DatetimeIndex(full.time))[ok].resample("1h").mean()
        both = pd.concat([fc.rename("rtd"), hc.rename("hourly")], axis=1).dropna()
        rel = both.rtd / both.hourly - 1
        summary["cost_vs_hourly_point"] = {
            "hours": int(len(both)),
            "median_pct": round(100 * float(rel.median()), 2),
            "p10_pct": round(100 * float(rel.quantile(0.1)), 2),
            "p90_pct": round(100 * float(rel.quantile(0.9)), 2),
            "total_pct": round(100 * float(both.rtd.sum() / both.hourly.sum() - 1), 2),
        }
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    log("SUMMARY " + json.dumps(summary))


if __name__ == "__main__":
    main()
