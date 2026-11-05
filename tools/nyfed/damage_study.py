"""The damage study (N58, N59): what the operator does, and what it costs, under the overload attack as
built (the interface reading taken to its limit) and under its damaging variant (taken as far past the
limit as every bound allows), over onsets drawn at random among the near-limit hours.

    python tools/nyfed/damage_study.py OUT_DIR --pool out/pool_2019_v3.h5 [--onsets 60] [--workers 8]
                                       [--voll 0] [--interface SPR/DUN-SOUTH] [--dispatch cost]
                                       [--load-cap 0.1] [--band 0.03] [--lag 0] [--onset-list ONSETS.json]

Run from the folder whose `data/` holds the inputs. Each onset's two attacks are written as single-episode
day files under OUT_DIR/days (resumable: a day already written is reused), the operator runs on each, and
OUT_DIR/damage.csv gets one row per onset and variant (`summarize`, plus the overshoot and the meters). A folder
holds one operator dispatch, so `--dispatch cost` results go in a folder of their own.

The attacker's sensitivities: `--load-cap` its cap on a pretended load change (0.1, N28), `--band` the basepoint band
as a share of the upper limit (0.03, N33), and `--lag` (frames) the age of what it knows: the attack is designed on
the window `lag` frames before the onset (states, hours and headroom, a persistence forecast) and added to the true
scans (`TimelineBuilder.build_episode(lag=...)`). `--onset-list` (a JSON list of pool frames) designs those onsets
in place of the random draw, so a sensitivity runs on the same onsets as the study it is compared with. A folder
holds one attacker form, so each setting goes in a folder of its own.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from fdia_graph.nyfed.attacks.attacker import BASEPOINT_BAND
from fdia_graph.nyfed.build import production_builder
from fdia_graph.nyfed.config import Settings, rng
from fdia_graph.nyfed.experiments import DATA, add_dispatch_argument, load_calib
from fdia_graph.nyfed.operator.response import MODES, operator_day, respond, summarize
from fdia_graph.nyfed.scenario.pool import SOLVED

DAY = 288  # five-minute scans
_B = None  # each worker's builder


def _init(
    pool: str,
    interface: str,
    consistent: bool = False,
    v_margin: float = 0.0,
    gen_v_tol: float | None = None,
    load_cap: float | None = None,
    band: float = BASEPOINT_BAND,
) -> None:
    global _B
    _B = production_builder(
        pool,
        ("Am",),
        interfaces=(interface,),
        workers=1,
        survey=0,
        hold_external=consistent,
        v_margin=v_margin,  # N71: the attacker's voltage limits widened
        gen_v_tol=gen_v_tol,  # N75: generator voltages held near their schedule
        load_cap=load_cap,
        band=band,
    )


def _day_file(out: Path, onset: int, overshoot: float, lag: int = 0) -> tuple[Path, dict] | None:
    """The single-episode day of `onset` with the attack at `overshoot` (MW past the limit), written once."""
    b = _B
    assert b is not None
    fam = b.families["Am"]
    t = b.pool.time.astype("datetime64[s]")
    # the calendar day of the onset, slid later where the window runs past midnight, so every attack fits
    lo = int(np.flatnonzero(t == t[onset].astype("datetime64[D]"))[0])
    lo = min(max(lo, onset + b.episode_lengths()["Am"] - DAY), len(t) - DAY)
    path = out / f"am_{onset}_{int(round(overshoot))}.h5"
    if path.exists():  # a reused day: its meter count read back from the labels
        with h5py.File(path, "r") as h:
            return path, {"meters_tampered": int(h["labels/meter_tampered"][:].any(axis=0).sum())}
    try:
        return path, b.build_episode(
            path, onset, "Am", lo, lo + DAY, using=fam.with_overshoot(overshoot), lag=lag
        )
    except ValueError:  # no attack at this overshoot, or the window touches an unsolved frame
        return None


def _job(
    onset: int,
    out: str,
    pool: str,
    data: str,
    interface: str,
    voll: float | None,
    calib: tuple | None,
    closed_loop: bool = False,
    dispatch: str = "movement",
    lag: int = 0,
) -> list[dict]:
    b = _B
    assert b is not None
    days = Path(out) / "days"
    days.mkdir(parents=True, exist_ok=True)
    L = b.episode_lengths()["Am"]
    w = slice(onset - lag, onset - lag + L)  # the window the attacker knows: the true one without a lag
    if onset < lag or not (b.pool.solve_kind[w] == SOLVED).all():
        return []  # the attacker knows no solved window to design on
    cached = days / f"overshoot_{onset}.json"  # the search is the slow part, so a rerun reuses it
    if cached.exists():
        overshoot, design_s = json.loads(cached.read_text())
    else:
        t0 = time.perf_counter()
        _, overshoot = b.families["Am"].max_overshoot(
            (onset - lag, b.pool.vm[w], b.pool.va[w]), rng(Settings(), f"Am{onset}")
        )
        design_s = time.perf_counter() - t0
        cached.write_text(json.dumps([overshoot, design_s]))
    rows = []
    for variant, mw in (("at_limit", 0.0), ("damaging", overshoot)):
        made = _day_file(days, onset, mw, lag)
        if made is None:
            continue
        path, info = made
        day = operator_day(path, pool, data)
        for mode in MODES:  # the trusting, the accounting and the oracle operator (N66)
            s = summarize(
                respond(day, mode=mode, calib=calib, closed_loop=closed_loop, dispatch=dispatch), voll
            )
            rows.append(
                {
                    "onset": onset,
                    "time": str(b.pool.time.astype("datetime64[s]")[onset]),
                    "variant": variant,
                    "dispatch": dispatch,
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
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--interface", default="SPR/DUN-SOUTH")
    ap.add_argument("--onsets", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--voll", type=float, default=None, help="$/MWh for the shed energy (a cited value)")
    ap.add_argument("--consistent", action="store_true", help="the interchange-consistent attacker (N62)")
    ap.add_argument("--calib", default=None, help="tie_check.py's json: the alarm calibrated on other months")
    ap.add_argument(
        "--seed-key", default="damage", help="the stream of the draw (another key draws other onsets)"
    )
    ap.add_argument(
        "--rerun", action="store_true", help="run the operator again on every onset (designs reused)"
    )
    ap.add_argument(
        "--v-margin",
        type=float,
        default=0.0,
        help="p.u. the attacker's bus voltage limits are widened by (N71)",
    )
    ap.add_argument(
        "--closed-loop",
        action="store_true",
        help="the operator reads the true flows after its dispatch (N76)",
    )
    ap.add_argument(
        "--gen-v-tol",
        type=float,
        default=None,
        help="p.u. a generator bus's false voltage may leave its schedule (N75)",
    )
    ap.add_argument(
        "--load-cap", type=float, default=None, help="the cap on a pretended load change (0.1, N28)"
    )
    ap.add_argument(
        "--band",
        type=float,
        default=BASEPOINT_BAND,
        help="the basepoint band's share of the upper limit (N33)",
    )
    ap.add_argument(
        "--lag", type=int, default=0, help="frames: the attacker designs on the window this much earlier"
    )
    ap.add_argument(
        "--onset-list", default=None, help="a JSON list of the onsets to design, in place of the draw"
    )
    add_dispatch_argument(ap)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # the designs cached here belong to one attacker form and the results to one dispatch: never mix them
    marker = out / "attacker.json"
    form = {
        "consistent": args.consistent,
        **({"v_margin": args.v_margin} if args.v_margin else {}),
        **({"gen_v_tol": args.gen_v_tol} if args.gen_v_tol is not None else {}),
        **({"dispatch": args.dispatch} if args.dispatch != "movement" else {}),
        **({"load_cap": args.load_cap} if args.load_cap is not None else {}),
        **({"band": args.band} if args.band != BASEPOINT_BAND else {}),
        **({"lag": args.lag} if args.lag else {}),
    }
    if marker.exists() and json.loads(marker.read_text()) != form:
        raise SystemExit(
            f"{out} holds another attacker form's designs or another dispatch; use another folder"
        )
    marker.write_text(json.dumps(form))
    init = (
        args.pool,
        args.interface,
        args.consistent,
        args.v_margin,
        args.gen_v_tol,
        args.load_cap,
        args.band,
    )
    _init(*init)
    b = _B
    assert b is not None
    fam = b.families["Am"]
    L = b.episode_lengths()["Am"]
    mask = fam.onset_mask(b.pool.vm, b.pool.va, L)
    ok = np.flatnonzero(mask)
    ok = ok[[bool((b.pool.solve_kind[o : o + L] == SOLVED).all()) for o in ok]]
    if args.onset_list:
        pick = np.sort(np.asarray(json.loads(Path(args.onset_list).read_text()), int))
        print(f"{len(pick)} onsets from {args.onset_list}", flush=True)
    else:
        pick = np.sort(
            rng(Settings(), args.seed_key).choice(ok, size=min(args.onsets, len(ok)), replace=False)
        )
        print(f"{len(ok)} near-limit onsets with solved windows, {len(pick)} drawn", flush=True)
    done = out / "damage.csv"
    rows = [] if args.rerun or not done.exists() else pd.read_csv(done).to_dict("records")
    todo = [int(o) for o in pick if int(o) not in {r["onset"] for r in rows}]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=init) as ex:
        calib = load_calib(args.calib) if args.calib else None
        futs = [
            ex.submit(
                _job,
                o,
                str(out),
                args.pool,
                args.data,
                args.interface,
                args.voll,
                calib,
                args.closed_loop,
                args.dispatch,
                args.lag,
            )
            for o in todo
        ]
        for k, f in enumerate(as_completed(futs), 1):
            rows += f.result()
            pd.DataFrame(rows).to_csv(done, index=False)
            print(f"{k}/{len(todo)} [{time.perf_counter() - t0:.0f}s]", flush=True)
    (out / "damage.json").write_text(
        json.dumps({"onsets": [int(o) for o in pick], "near_limit": len(ok)}, indent=1)
    )


if __name__ == "__main__":
    main()
