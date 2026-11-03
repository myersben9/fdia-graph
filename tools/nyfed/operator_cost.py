"""The operator's response to each single-episode day and what it costs (`fdia_graph.nyfed.operator`).

    python tools/nyfed/operator_cost.py out/single_episode/*.h5 --pool out/pool_2019_v3.h5
                                        [--delay 0,1,3] [--over 0,50,100,200] [--out RESULTS_DIR]

Run from the folder whose `data/` holds the inputs. Writes `nygrid.operator_cost.csv` (one row per day,
delay and overshoot: the attack's consequence, attacked less benign, from `summarize`) and
`nygrid.operator_trace.csv` (per scan of each day at delay 0 and no overshoot: the seen and true
interface flow, the dispatch moved, the regulation, the true interface change and the costs, both
twins), with a JSON summary, into `--out` (default `results`).
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.response import operator_day, respond, seen_interfaces, summarize


def _trace(name: str, day, r) -> pd.DataFrame:
    """Per scan of one day: what the operator saw and did, and the consequence, in both twins."""
    rows = {"day": name, "time": day.time.astype("datetime64[s]").astype(str)}
    rows["attacked"] = np.zeros(len(day.time), bool)
    rows["attacked"][r.onset : r.onset + r.length] = True
    rows["seen_attacked_mw"] = seen_interfaces(day.plan, day.interfaces, day.z)[:, 0]
    rows["seen_benign_mw"] = seen_interfaces(day.plan, day.interfaces, day.z_benign)[:, 0]
    it = day.interfaces[0]
    rows["real_headroom_mw"] = it.limit - it.flow
    rows["tie_gap_mw"] = r.tie_attacked - r.tie_benign
    rows["alarm_scans"] = r.alarm  # scans from the onset to the interchange alarm
    rows["true_mw"] = it.model_flow(Network(day.case).flows(day.vm, day.va)[0])
    rows["limit_mw"] = rows["true_mw"] + rows["real_headroom_mw"]  # the operator's limit (N52)
    for tag, tw in (("a", r.attacked), ("b", r.benign)):
        rows[f"{tag}_dispatch_mw"] = tw.dev.sum(axis=1)
        rows[f"{tag}_moved_mw"] = np.abs(tw.dev).sum(axis=1)
        rows[f"{tag}_regulation_mw"] = tw.reg.sum(axis=1)
        rows[f"{tag}_inadvertent_mw"] = tw.uncovered
        rows[f"{tag}_shed_mw"] = tw.shed.sum(axis=1)
        rows[f"{tag}_shortage_usd"] = tw.shortage
        rows[f"{tag}_iface_change_mw"] = tw.iface_change[:, 0]
        rows[f"{tag}_n1_change_mw"] = tw.n1_change
        rows[f"{tag}_fuel_usd"] = tw.fuel
        rows[f"{tag}_settled_usd"] = tw.settled
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="+")
    ap.add_argument("--pool", default="out/pool_2019_v3.h5")
    ap.add_argument("--data", default="data/NYgrid/Data")
    ap.add_argument("--delay", default="0,1,3")
    ap.add_argument("--over", default="0,50,100,200")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, traces = [], []
    t0 = time.perf_counter()
    for path in args.days:
        name = Path(path).stem
        day = operator_day(path, args.pool, args.data)  # every interface enforced, the attacked one first
        for delay in (int(x) for x in args.delay.split(",")):
            for over in (float(x) for x in args.over.split(",")):
                if delay and over:
                    continue  # the delays at the attack as built, the overshoots on time
                r = respond(day, delay, over)
                rows.append({"day": name, **summarize(r)})
                print(json.dumps(rows[-1]), f"[{time.perf_counter() - t0:.0f}s]", flush=True)
                if delay == 0 and over == 0:
                    traces.append(_trace(name, day, r))
    pd.DataFrame(rows).to_csv(out / "nygrid.operator_cost.csv", index=False)
    if traces:  # the trace is the baseline run (delay 0, no overshoot), when the sweep includes it
        pd.concat(traces).to_csv(out / "nygrid.operator_trace.csv", index=False)
    (out / "nygrid.operator_cost.json").write_text(
        json.dumps(
            {
                "days": [Path(p).stem for p in args.days],
                "rows": len(rows),
                "seconds": round(time.perf_counter() - t0),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
