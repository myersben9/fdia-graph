"""[WU26]'s defense on the faithful attack (`trust.wu26`), at the paper's scale: windows drawn by p. 658's
data recipe, the undefended attack of each and the attack under Solution 1's trusted PMUs, one step per
configuration slot. Every window's numbers go to the results store as `wu26.faithful` (docs/wu26).

    python tools/wu26_faithful.py --system 118 --windows 100 --workers 26
    python tools/wu26_faithful.py --system 14 --windows 5 --dry      # a few windows, printed, nothing stored

Per window: the devices tampered at any snapshot without and with the defense (Fig. 12's cost), the rise
and the new devices, eq. (33)'s net-vector channels, the largest change (Fig. 4's scale), the overlap
with Table V's tampered set (IEEE-118) or Fig. 4's (IEEE-14), feasibility and the solve times.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fdia_graph.models.config import Wu26Attack, WuDefenseConfig  # noqa: E402
from fdia_graph.trust.wu26 import wu26_network, wu26_snapshots  # noqa: E402
from fdia_graph.trust.wu26_defense import Wu26DefenseEnv, wu26_solution1  # noqa: E402

# the paper's scenarios (Sec. V): PMUs, attack area, target lines, configuration slots (0-based snapshots),
# the reference tampered set, and the settings it leaves open (ours, Wu26Attack)
SCENARIOS: dict[str, dict] = {
    "118": dict(
        case="case118",
        pmus=[76, 78, 80, 83, 89, 92, 94, 100, 105, 106, 110],  # Fig. 9 / Fig. 11
        area=list(range(74, 113)) + [118],  # Fig. 9's dashed network
        targets=[(84, 85), (99, 100)],
        snapshots=10,
        slots=list(range(10)),  # one trusted PMU per snapshot (p. 662)
        described=[80, 83, 100, 89, 92, 94, 105, 106, 110, 76],  # p. 662's account of the trust order
        reference={
            "SCADA": {84, 85, 89, 94, 99, 100, 101, 103, 105},
            "PMU": {83, 89, 100, 105, 106, 110},
        },  # Table V
        attack=dict(rho=1.5, tau=0.05, dv=0.04, da=0.12),
    ),
    "14-1": dict(
        case="case14",
        pmus=[1, 4, 6, 13],
        area=[2, 3, 4, 5, 6, 11, 12, 13],
        targets=[(3, 4), (6, 11)],
        snapshots=20,
        slots=[1, 3, 5, 7],  # snapshots 2, 4, 6, 8 (p. 659)
        described=[1, 4, 6, 13],  # Fig. 6
        reference={"SCADA": {3, 4, 5, 6, 11}, "PMU": {4, 6}},  # Fig. 4, scenario 1
        attack=dict(rho=1.5, tau=0.5, dv=0.03, da=0.1),
    ),
    "14-2": dict(
        case="case14",
        pmus=[1, 4, 6, 13],
        area=[1, 2, 3, 4, 5, 6, 9],
        targets=[(1, 2), (4, 5)],
        snapshots=20,
        slots=[1, 3, 5, 7],
        described=[4, 6, 1, 13],  # Fig. 6
        reference={"SCADA": {1, 2, 3, 4, 5, 9}, "PMU": {1, 4, 6}},  # Fig. 4, scenario 2
        attack=dict(rho=1.08, tau=0.5, dv=0.03, da=0.1),
    ),
}


def run_window(job: tuple[str, int, list[int]]) -> dict:
    """One window: the undefended attack, then the attack under Solution 1's order and under the order
    the paper reports (Fig. 6 on IEEE-14, p. 662's account on IEEE-118)."""
    name, seed, order = job
    sc = SCENARIOS[name]
    net = wu26_network(sc["case"], sc["pmus"])
    states = wu26_snapshots(net.case, sc["snapshots"], seed)
    pmus = [b - 1 for b in sc["pmus"]]
    config = WuDefenseConfig(pmus=pmus, slots=sc["slots"], unit="devices")
    t0 = time.time()
    attack = Wu26Attack(**sc["attack"])
    env = Wu26DefenseEnv(net, states, [b - 1 for b in sc["area"]], sc["targets"], attack, config)
    base = env.base.devices_any
    row = dict(
        name=name,
        seed=seed,
        devices=len(base),
        channels=env.base.channels_net,
        max_change=env.base.max_magnitude,
        feasible=int(env.base.feasible),
        seconds=time.time() - t0,
        undefended=sorted(base),
        **_overlap(base, sc["reference"]),
    )
    row["arms"] = {}
    for method, sequence in (("solution1", order), ("described", sc["described"])):
        t1 = time.time()
        defended = env.result_of(tuple(pmus.index(b - 1) for b in sequence[: len(sc["slots"])]))
        after = defended.devices_any
        row["arms"][method] = dict(
            devices=len(after),
            channels=defended.channels_net,
            rise=100.0 * (len(after) - len(base)) / max(len(base), 1),
            extras=sorted(after - base),
            feasible=int(defended.feasible),
            seconds=time.time() - t1,
        )
    return row


def _overlap(devices: set[tuple[str, int]], reference: dict[str, set[int]]) -> dict[str, float]:
    """Hits and Jaccard overlap of the attack's SCADA and PMU devices with the paper's tampered set."""
    out: dict[str, float] = {}
    for kind in ("SCADA", "PMU"):
        mine = {b for k, b in devices if k == kind}
        out[f"hits_{kind.lower()}"] = len(mine & reference[kind])
        out[f"jaccard_{kind.lower()}"] = len(mine & reference[kind]) / max(len(mine | reference[kind]), 1)
    return out


def _print(r: dict) -> None:
    arms = "; ".join(
        f"{m} {a['devices']} ({a['rise']:+.1f}%, new {len(a['extras'])}, {a['seconds']:.0f} s)"
        for m, a in r["arms"].items()
    )
    print(
        f"[{r['name']} seed {r['seed']}] undefended {r['devices']} devices, {r['channels']} net channels, "
        f"max {r['max_change']:.3f} pu, reference hits SCADA {r['hits_scada']} PMU {r['hits_pmu']}, "
        f"{r['seconds']:.0f} s | {arms}",
        flush=True,
    )


def store(rows: list[dict], name: str, order: list[int], store_dir: Optional[str]) -> int:
    """Every window as records of `wu26.faithful`: the undefended attack (defended=0) and each trust
    order's (defended=1, method solution1 or described), plus Solution 1's order (Fig. 10 / Fig. 6)."""
    from fdia_graph.results import Run, Store

    sc = SCENARIOS[name]
    system = "ieee118" if name == "118" else "ieee14"
    settings = {k: sc[k] for k in ("attack", "slots", "pmus", "area")}
    with Run(
        "wu26.faithful", system=system, settings=settings, store=Store(store_dir), note=f"scenario {name}"
    ) as run:
        for step, bus in enumerate(order[: len(sc["slots"])]):
            run.add("trusted_pmu", bus, method="solution1", scenario=name, step=step + 1)
        for r in rows:
            base = dict(scenario=name, window=r["seed"], defended=0)
            for metric, key in (
                ("devices", "devices"),
                ("channels", "channels"),
                ("max_change_pu", "max_change"),
                ("jaccard_scada", "jaccard_scada"),
                ("jaccard_pmu", "jaccard_pmu"),
                ("feasible", "feasible"),
                ("seconds", "seconds"),
            ):
                run.add(metric, r[key], **base)
            run.add("reference_hits", r["hits_scada"] + r["hits_pmu"], **base)
            for method, a in r["arms"].items():
                keys = dict(method=method, scenario=name, window=r["seed"], defended=1)
                run.add("devices", a["devices"], **keys)
                run.add("channels", a["channels"], **keys)
                run.add("cost_increase_pct", a["rise"], **keys)
                run.add("extra_devices", len(a["extras"]), **keys)
                run.add("feasible", a["feasible"], **keys)
                run.add("seconds", a["seconds"], **keys)
        return len(run.records)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--system", choices=["118", "14-1", "14-2"], default="118")
    ap.add_argument("--windows", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1000, help="the first window's seed; window i uses seed + i")
    ap.add_argument(
        "--seeds", default="", help="comma-separated window seeds, in place of --seed and --windows"
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--store", default=None, help="results store folder (default: the repository's results/)")
    ap.add_argument("--dry", action="store_true", help="print only; write nothing to the store")
    args = ap.parse_args(argv)
    sc = SCENARIOS[args.system]
    order = wu26_solution1(wu26_network(sc["case"], sc["pmus"]), sc["targets"], n=len(sc["slots"]))
    print(f"Solution 1 order: {order}", flush=True)
    seeds = (
        [int(v) for v in args.seeds.split(",")]
        if args.seeds
        else [args.seed + i for i in range(args.windows)]
    )
    jobs = [(args.system, seed, order) for seed in seeds]
    rows = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(run_window, jobs):
            _print(r)
            rows.append(r)
    for method in ("solution1", "described"):
        rises = sorted(r["arms"][method]["rise"] for r in rows)
        extras = [len(r["arms"][method]["extras"]) for r in rows]
        print(
            f"{method}: rise mean {sum(rises) / len(rises):+.1f}%, median {rises[len(rises) // 2]:+.1f}%, "
            f"new devices mean {sum(extras) / len(extras):.1f}",
            flush=True,
        )
    if not args.dry:
        print(f"stored {store(rows, args.system, order, args.store)} records", flush=True)


if __name__ == "__main__":
    main()
