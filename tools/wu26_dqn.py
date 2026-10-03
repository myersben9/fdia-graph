"""[WU26]'s Solution 2 on the faithful attack (`trust.TrustedPMUsDQN` over `trust.Wu26DefenseEnv`):
training sessions of Algorithm 1 run in parallel, each tested on fresh windows, the paper's "100 tests"
as sessions x test windows (the plan's E8 option 2). Records go to the results store as `wu26.dqn`
(docs/wu26): every test window's cost without and with the trained policy's schedule (Fig. 12), each
step's choice (Fig. 11's selection probabilities) and the decision and training times (p. 662).

    python tools/wu26_dqn.py --sessions 10 --test 10 --workers 10            # the run
    python tools/wu26_dqn.py --sessions 1 --episodes 3 --train 2 --test 1 --dry  # a timing check

The scale is reduced from the paper's where the attack solves make it unaffordable: `--episodes` (250 in
the paper) is recorded with the results.
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
sys.path.insert(0, os.path.join(ROOT, "tools"))

from wu26_faithful import SCENARIOS  # noqa: E402

from fdia_graph.models.config import Wu26Attack, WuDefenseConfig, WuDqnConfig  # noqa: E402
from fdia_graph.trust.wu26 import wu26_network, wu26_snapshots  # noqa: E402
from fdia_graph.trust.wu26_defense import Wu26DefenseEnv, wu26_solution1  # noqa: E402


def _env(name: str, seed: int, tau: Optional[float]) -> Wu26DefenseEnv:
    """One window's MDP, drawn by p. 658's recipe at `seed`."""
    sc = SCENARIOS[name]
    net = wu26_network(sc["case"], sc["pmus"])
    states = wu26_snapshots(net.case, sc["snapshots"], seed)
    attack = Wu26Attack(**{**sc["attack"], **({} if tau is None else {"tau": tau})})
    config = WuDefenseConfig(pmus=[b - 1 for b in sc["pmus"]], slots=sc["slots"], unit="devices")
    return Wu26DefenseEnv(net, states, [b - 1 for b in sc["area"]], sc["targets"], attack, config)


def session(job: tuple[str, int, int, int, int, Optional[float]]) -> dict:
    """Train one DQN (Algorithm 1) on `train` windows, then test it on `test` fresh windows."""
    from fdia_graph.trust.wu_dqn import TrustedPMUsDQN

    name, s, episodes, train, test, tau = job
    sc = SCENARIOS[name]
    t0 = time.time()
    envs = [_env(name, 20000 + 100 * s + i, tau) for i in range(train)]
    dqn = TrustedPMUsDQN(envs, WuDqnConfig(episodes=episodes, seed=123 + s))
    t1 = time.time()
    dqn.fit()
    train_s = time.time() - t1
    print(
        f"[session {s}] trained {episodes} episodes on {train} windows in {train_s / 3600:.2f} h "
        f"({sum(e.solves for e in envs)} solves, {dqn.breaks} line-8 breaks; setup {t1 - t0:.0f} s)",
        flush=True,
    )
    tests = []
    for j in range(test):
        env = _env(name, 50000 + 100 * s + j, tau)
        order = dqn.order(env)  # fills the cache along the greedy path
        t2 = time.time()
        dqn.order(env)  # again, cached: the policy's own decision time
        decide_s = time.time() - t2
        base, after = env.base.devices_any, env.result_of(tuple(order)).devices_any
        buses = [int(sc["pmus"][a]) for a in order]
        tests.append(
            dict(
                window=50000 + 100 * s + j,
                devices=len(base),
                defended=len(after),
                rise=100.0 * (len(after) - len(base)) / max(len(base), 1),
                extras=len(after - base),
                order=buses,
                decide_s=decide_s,
            )
        )
        print(
            f"[session {s} test {j}] order {buses} | {len(base)} -> {len(after)} devices "
            f"({tests[-1]['rise']:+.1f}%, new {tests[-1]['extras']}), decision {decide_s * 1e3:.1f} ms",
            flush=True,
        )
    return dict(session=s, episodes=episodes, train=train, train_s=train_s, breaks=dqn.breaks, tests=tests)


def store(results: list[dict], name: str, tau: Optional[float], store_dir: Optional[str]) -> int:
    """Every test as records of `wu26.dqn`, plus each step's selection counts (Fig. 11)."""
    from fdia_graph.results import Run, Store

    sc = SCENARIOS[name]
    settings = dict(
        attack={**sc["attack"], **({} if tau is None else {"tau": tau})},
        episodes=results[0]["episodes"],
        train_windows=results[0]["train"],
        pmus=sc["pmus"],
    )
    system = "ieee118" if name == "118" else "ieee14"
    with Run(
        "wu26.dqn",
        system=system,
        settings=settings,
        store=Store(store_dir),
        note=f"scenario {name}, {results[0]['episodes']} episodes on {results[0]['train']} windows per session, tau={settings['attack']['tau']}",
    ) as run:
        counts: dict[tuple[int, int], int] = {}
        for r in results:
            run.add("seconds", r["train_s"], scenario=name, session=r["session"], phase="train")
            for t in r["tests"]:
                keys = dict(scenario=name, session=r["session"], window=t["window"], method="dqn")
                run.add("devices", t["devices"], defended=0, **keys)
                run.add("devices", t["defended"], defended=1, **keys)
                run.add("cost_increase_pct", t["rise"], defended=1, **keys)
                run.add("extra_devices", t["extras"], defended=1, **keys)
                run.add("seconds", t["decide_s"], phase="decide", **keys)
                for step, bus in enumerate(t["order"]):
                    counts[(step + 1, bus)] = counts.get((step + 1, bus), 0) + 1
        n = sum(len(r["tests"]) for r in results)
        for (step, bus), c in sorted(counts.items()):
            run.add("selection_share", c / n, scenario=name, step=step, pmu=bus, method="dqn")
        return len(run.records)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--system", choices=["118", "14-1", "14-2"], default="118")
    ap.add_argument("--sessions", type=int, default=10)
    ap.add_argument(
        "--episodes", type=int, default=250, help="training episodes per session (250 in the paper)"
    )
    ap.add_argument("--train", type=int, default=4, help="training windows per session")
    ap.add_argument("--test", type=int, default=10, help="test windows per session")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--store", default=None)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)
    sc = SCENARIOS[args.system]
    t = time.time()
    order = wu26_solution1(wu26_network(sc["case"], sc["pmus"]), sc["targets"], n=len(sc["slots"]))
    print(f"Solution 1 order {order} derived in {time.time() - t:.1f} s", flush=True)
    jobs = [(args.system, s, args.episodes, args.train, args.test, args.tau) for s in range(args.sessions)]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(session, jobs))
    rises = sorted(t["rise"] for r in results for t in r["tests"])
    extras = [t["extras"] for r in results for t in r["tests"]]
    print(
        f"DQN: {len(rises)} tests, rise mean {sum(rises) / len(rises):+.1f}%, median {rises[len(rises) // 2]:+.1f}%, "
        f"in 10-20%: {sum(10 <= v <= 20 for v in rises)}, new devices mean {sum(extras) / len(extras):.1f}",
        flush=True,
    )
    if not args.dry:
        print(f"stored {store(results, args.system, args.tau, args.store)} records", flush=True)


if __name__ == "__main__":
    main()
