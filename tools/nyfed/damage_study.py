"""The damage study (N58, N59): what the operator does, and what it costs, under the overload attack as
built (the interface reading taken to its limit) and under its damaging variant (taken as far past the
limit as every bound allows), over onsets drawn at random among the near-limit hours.

    python tools/nyfed/damage_study.py OUT_DIR --pool out/pool_2019_v3.h5 [--onsets 60] [--workers 8]
                                       [--voll 0] [--interface SPR/DUN-SOUTH]

Run from the folder whose `data/` holds the inputs. Each onset's two attacks are written as single-episode
day files under OUT_DIR/days (resumable: a day already written is reused), the operator runs on each, and
OUT_DIR/damage.csv gets one row per onset and variant (`summarize`, plus the overshoot and the meters).
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

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fdia_graph.nyfed.build import production_builder
from fdia_graph.nyfed.config import Settings, rng
from fdia_graph.nyfed.operator.response import operator_day, respond, summarize
from fdia_graph.nyfed.scenario.pool import SOLVED

DAY = 288  # five-minute scans
_B = None  # each worker's builder


def _init(pool: str, interface: str, consistent: bool = False) -> None:
    global _B
    _B = production_builder(
        pool, ("Am",), interfaces=(interface,), workers=1, survey=0, hold_external=consistent
    )


def _day_file(out: Path, onset: int, overshoot: float) -> tuple[Path, dict] | None:
    """The single-episode day of `onset` with the attack at `overshoot` (MW past the limit), written once."""
    b = _B
    assert b is not None
    fam = b.families["Am"]
    t = b.pool.time.astype("datetime64[s]")
    # the calendar day of the onset, slid later where the window runs past midnight, so every attack fits
    lo = int(np.flatnonzero(t == t[onset].astype("datetime64[D]"))[0])
    lo = min(max(lo, onset + b._lengths()["Am"] - DAY), len(t) - DAY)
    path = out / f"am_{onset}_{int(round(overshoot))}.h5"
    if path.exists():  # a reused day: its meter count read back from the labels
        with h5py.File(path, "r") as h:
            return path, {"meters_tampered": int(h["labels/meter_tampered"][:].any(axis=0).sum())}
    fam.overshoot = overshoot
    try:
        return path, b.build_episode(path, onset, "Am", lo, lo + DAY)
    except ValueError:
        return None
    finally:
        fam.overshoot = 0.0


def _job(onset: int, out: str, pool: str, data: str, interface: str, voll: float | None) -> list[dict]:
    b = _B
    assert b is not None
    days = Path(out) / "days"
    days.mkdir(parents=True, exist_ok=True)
    L = b._lengths()["Am"]
    w = slice(onset, onset + L)
    cached = days / f"overshoot_{onset}.json"  # the search is the slow part, so a rerun reuses it
    if cached.exists():
        overshoot, design_s = json.loads(cached.read_text())
    else:
        t0 = time.perf_counter()
        _, overshoot = b.families["Am"].max_overshoot(
            (onset, b.pool.vm[w], b.pool.va[w]), rng(Settings(), f"Am{onset}")
        )
        design_s = time.perf_counter() - t0
        cached.write_text(json.dumps([overshoot, design_s]))
    rows = []
    for variant, mw in (("at_limit", 0.0), ("damaging", overshoot)):
        made = _day_file(days, onset, mw)
        if made is None:
            continue
        path, info = made
        day = operator_day(path, pool, data)
        for defended in (False, True):  # the operator as it is, and with the interchange check (N65)
            s = summarize(respond(day, defended=defended), voll)
            rows.append(
                {
                    "onset": onset,
                    "time": str(b.pool.time.astype("datetime64[s]")[onset]),
                    "variant": variant,
                    "overshoot_mw": mw,
                    "meters": info.get("meters_tampered"),
                    "design_s": round(design_s, 1),
                    **s,
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default="data/NYgrid/Data")
    ap.add_argument("--interface", default="SPR/DUN-SOUTH")
    ap.add_argument("--onsets", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--voll", type=float, default=None, help="$/MWh for the shed energy (a cited value)")
    ap.add_argument("--consistent", action="store_true", help="the interchange-consistent attacker (N62)")
    ap.add_argument(
        "--seed-key", default="damage", help="the stream of the draw (another key draws other onsets)"
    )
    ap.add_argument(
        "--rerun", action="store_true", help="run the operator again on every onset (designs reused)"
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _init(args.pool, args.interface, args.consistent)
    b = _B
    assert b is not None
    fam = b.families["Am"]
    L = b._lengths()["Am"]
    mask = fam.onset_mask(b.pool.vm, b.pool.va, L)
    ok = np.flatnonzero(mask)
    ok = ok[[bool((b.pool.solve_kind[o : o + L] == SOLVED).all()) for o in ok]]
    pick = np.sort(rng(Settings(), args.seed_key).choice(ok, size=min(args.onsets, len(ok)), replace=False))
    print(f"{len(ok)} near-limit onsets with solved windows, {len(pick)} drawn", flush=True)
    done = out / "damage.csv"
    rows = [] if args.rerun or not done.exists() else pd.read_csv(done).to_dict("records")
    todo = [int(o) for o in pick if int(o) not in {r["onset"] for r in rows}]
    t0 = time.perf_counter()
    init = (args.pool, args.interface, args.consistent)
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as ex:
        futs = [ex.submit(_job, o, str(out), args.pool, args.data, args.interface, args.voll) for o in todo]
        for k, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            pd.DataFrame(rows).to_csv(done, index=False)
            print(f"{k}/{len(todo)} [{time.perf_counter() - t0:.0f}s]", flush=True)
    (out / "damage.json").write_text(
        json.dumps({"onsets": [int(o) for o in pick], "near_limit": len(ok)}, indent=1)
    )


if __name__ == "__main__":
    main()
