"""Reproduce [WU26]'s defense results on our generator (docs/plans/WU_DEFENSE_PLAN.md, PR E).

    python tools/wu26_harness.py --system 14 --windows 20 --workers 26 --out docs/wu26
    python tools/wu26_harness.py --system 118 --windows 100 --sessions 10 --workers 26 --out docs/wu26

For each of the paper's scenarios (IEEE-14: lines 3-4 and 6-11, lines 1-2 and 4-5, PMUs 1, 4, 6, 13
trusted at snapshots 2, 4, 6, 8; IEEE-118: lines 84-85 and 99-100, its 11 PMUs, one per snapshot) and
each rating (`--ratings`: "+0.10", each target line's true flow at the window's last snapshot plus
0.10 pu, matched to the paper's Fig. 4 attack magnitudes, the default [E14]; or a margin k such as "1.1",
k times each target line's peak true flow over the window), on the paper's attack area (`wu26_area`:
IEEE-14's every bus but the slack, IEEE-118's Fig. 9 network [D18]), windows of 1-minute attack snapshots are built from the
5-minute operating pool (ours: each bus's load and generation scale recovered from two consecutive pool
states, interpolated linearly in time and re-solved by AC power flow, so every snapshot is a power-flow
solution). On each window the trusted-PMU MDP (`trust.WuDefenseEnv`, the search as the cost oracle) is
played by Solution 1 (`trust.TrustedPMUs`) and by Solution 2 (`trust.TrustedPMUsDQN`, trained on the
other windows of its session, [E8] option 2).

Written to --out, each table and figure beside the CSV its numbers come from:
    table2_ieee{N}.csv   devices and channels undefended and defended, the extra devices, the rise (Table II, Fig. 12)
    orders_ieee{N}.csv   each method's trusted sequence per window (Figs. 6, 10)
    times_ieee{N}.csv    each method's decision time per window (Fig. 8, Table VI)
    fig11_ieee{N}.png    Solution 2's selection probability, PMU by step, over its test windows (Fig. 11)
    fig12_ieee{N}.png    the rise per test window and the extra devices (Fig. 12)
Figures follow the project's rules: no legend and no text inside the plot area; the caption in
docs/wu26/README.md carries the keys. Every row also goes to the results store as a run of
`wu26.reproduction` (`--store`, the repository's `results/` by default), so the docs cite it
through results blocks rather than typed numbers.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

# the paper's trusted orders per IEEE-14 scenario (Fig. 6), a reference for the output only: every
# environment offers the meter plan's PMUs in their canonical order, so no method is steered to them
PAPER_ORDERS = {14: ((1, 4, 6, 13), (4, 6, 1, 13))}
SNAPSHOTS = {14: 20, 118: 10}  # 10 minutes of 30 s snapshots on IEEE-14 is 20; Figs. 10-11 show 10 on 118
POOL_MINUTES = 5  # the pool's cadence
ATTACK_MINUTES = 1  # the attack snapshots' [E4]


# ---- windows of 1-minute snapshots -----------------------------------------------------------------------
def minute_window(system: int, start: int, length: int) -> list[np.ndarray]:
    """`length` AC states one minute apart from pool frame `start` on: each bus's scale (its injection over
    the base case's, the pool's own construction, `profiles.generate_states`) interpolated linearly
    between the two pool frames around each minute and solved again (`profiles._solve_states_chunk`); a
    bus whose base injection is zero keeps scale 1 (ours)."""
    from fdia_graph.generation import _load_states
    from fdia_graph.profiles import _case_buses, _solve_states_chunk

    pool = _load_states(system, None)
    buses = _case_buses(system)
    base = pool_base(system)
    frames = int(np.ceil(length * ATTACK_MINUTES / POOL_MINUTES)) + 1
    net = np.stack([pool[start + i][buses, 1] for i in range(frames)])  # [frames, nbus] active injection
    scale = np.where(np.abs(base) > 1e-9, net / np.where(np.abs(base) > 1e-9, base, 1.0), 1.0)
    minutes = np.arange(length) * ATTACK_MINUTES / POOL_MINUTES
    sf = np.stack([np.interp(minutes, np.arange(frames), scale[:, j]) for j in range(len(buses))], axis=1)
    return list(_solve_states_chunk(system, sf))[:length]


def pool_base(system: int) -> np.ndarray:
    """The base case's net active injection at `profiles._case_buses`, the unit the pool's scales divide."""
    import pandapower as pp
    import pandapower.networks as pn

    from fdia_graph.profiles import _CASE, _case_buses

    net = getattr(pn, _CASE[system])()
    pp.runpp(net)
    order = sorted(net.bus.index)
    p = net.res_bus.reindex(order)["p_mw"].to_numpy()
    return p[_case_buses(system)]


def paper_order(system: int, scenario: int) -> str:
    """The paper's trusted order for the scenario (Fig. 6), written beside ours for comparison."""
    orders = PAPER_ORDERS.get(system)
    return " ".join(map(str, orders[scenario])) if orders else ""


# ---- one window ------------------------------------------------------------------------------------
def rating_settings(rating: str):
    """The overload settings of a `--ratings` token: "+d" the delta ratings (true flow at the window's end
    plus d pu), anything else a margin k on each line's peak flow over the window."""
    from fdia_graph.models.config import OverloadSettings

    if rating.startswith("+"):
        return OverloadSettings(rating_source="delta", rating_delta=float(rating[1:]))
    return OverloadSettings(rating_margin=float(rating))


def build_env(system: int, scenario: int, margin: str, states: list[np.ndarray], budget: int):
    """The trusted-PMU MDP of one window on the paper's metering and attack area (the construction of the
    tests'), its ratings from the `--ratings` token `margin`."""
    from fdia_graph.engine.attacks.overload import WU26_PMUS, WU26_SCENARIOS
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.generation import _load_states
    from fdia_graph.models import WuDefenseConfig
    from fdia_graph.models.frames import FrameKnobs
    from fdia_graph.trust import WuDefenseEnv

    g = FdiaGenerator(system, seed=1, meter_model="hybrid")
    lines = [g.wu26_branch(*pair) for pair in WU26_SCENARIOS[system][scenario]]
    flow = np.asarray(g.meters.flow, bool).copy()
    flow[lines] = True
    if system == 14:
        flow[g.wu26_branch(6, 11)] = True  # the paper meters line 6-11
    g.meters = g.meters._replace(pmu=set(g.wu26_buses(WU26_PMUS[system]).tolist()), flow=flow)
    g.use_line_ratings(rating_settings(margin), np.stack(states))
    limits = g.operating_limits(_load_states(system, None))
    k = FrameKnobs(
        hops=2, limits=limits, min_tamper=True, min_budget=budget, load_cap=0.5, n_lines=2, area=g.wu26_area()
    )
    order = WU26_PMUS[system]
    slots = (
        (1, 3, 5, 7) if system == 14 else tuple(range(SNAPSHOTS[system]))
    )  # Figs. 10-11: one step per snapshot
    pmus = [int(b) for b in g.wu26_buses(tuple(order))]
    return WuDefenseEnv(g, states, g.overload_goal(states, *lines), k, WuDefenseConfig(pmus, slots))


def devices_named(env, result) -> set[str]:
    """The devices a search answer tampers over the window, named as the paper names them."""
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.models.grid import CURRENT, NODE

    g, number = env.g, env.g.base.bus["name"].astype(int).to_numpy()
    if result.devices < 0:
        return set()
    w = _Window(g, env.states, env.goal, env.k, trust=env._schedule(env.trusted))
    prev, names = w.prev, set()
    for t in range(len(env.states)):
        free = w.support_at(t, result.support)
        snap = w._snapshot(t, free, prev) if len(free) else None
        if snap is None:  # nothing free at this snapshot: no channel is tampered there
            continue
        node, edge, cur, prev = snap
        for b, c in zip(*np.nonzero(node)):
            names.add(("PMU " if c in (NODE.v, NODE.theta) and w.pmu[b] else "SCADA ") + str(number[b]))
        for e, _c in zip(*np.nonzero(edge)):
            names.add(f"SCADA {number[g.ei[0, e]]}")
        for e, c in zip(*np.nonzero(cur)) if cur is not None else ():
            names.add(f"PMU {number[g.ei[0, e] if c in (CURRENT.re_from, CURRENT.im_from) else g.ei[1, e]]}")
    return names


def run_solution1(job: tuple) -> dict:
    """Solution 1 on one window: its order, the undefended and defended counts, the extra devices, time."""
    from fdia_graph.trust import TrustedPMUs

    system, scenario, margin, start, budget = job
    states = minute_window(system, start, SNAPSHOTS[system])
    env = build_env(system, scenario, margin, states, budget)
    before = devices_named(env, env.result_of(()))
    sol = TrustedPMUs(env).fit()
    result = env.result_of(env.trusted)
    after = devices_named(env, result)
    number = env.g.base.bus["name"].astype(int).to_numpy()
    base = env.result_of(())
    return dict(
        system=system,
        scenario=scenario,
        margin=margin,
        window=start,
        method="solution1",
        order=" ".join(str(number[env.config.pmus[i]]) for i in sol.order),
        paper_order=paper_order(system, scenario),
        devices_before=base.devices,
        channels_before=base.channels,
        devices_after=result.devices,
        channels_after=result.channels,
        extra=" ".join(sorted(after - before)),
        seconds=round(sol.seconds, 4),
    )


def run_dqn_session(job: tuple) -> list[dict]:
    """One Solution 2 session: train on the session's training windows, test on its test windows."""
    from fdia_graph.models import WuDqnConfig
    from fdia_graph.trust import TrustedPMUsDQN

    system, scenario, margin, train, test, budget, seed = job
    envs = {
        s: build_env(system, scenario, margin, minute_window(system, s, SNAPSHOTS[system]), budget)
        for s in (*train, *test)
    }
    dqn = TrustedPMUsDQN([envs[s] for s in train], WuDqnConfig(seed=seed)).fit()
    rows = []
    for s in test:
        env = envs[s]
        before, base = devices_named(env, env.result_of(())), env.result_of(())
        start = time.perf_counter()
        order = dqn.order(env)
        seconds = time.perf_counter() - start
        result = env.result_of(env.trusted)
        number = env.g.base.bus["name"].astype(int).to_numpy()
        rows.append(
            dict(
                system=system,
                scenario=scenario,
                margin=margin,
                window=s,
                method="dqn",
                order=" ".join(str(number[env.config.pmus[i]]) for i in order),
                paper_order=paper_order(system, scenario),
                devices_before=base.devices,
                channels_before=base.channels,
                devices_after=result.devices,
                channels_after=result.channels,
                extra=" ".join(sorted(devices_named(env, result) - before)),
                seconds=round(seconds, 4),
            )
        )
    return rows


# ---- figures ---------------------------------------------------------------------------------------
def figures(rows: list[dict], system: int, out: str) -> None:
    """Figs. 11 and 12 in the project's style: no legend, no text inside the plot area."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9})
    dqn = [r for r in rows if r["method"] == "dqn" and r["order"]]
    if dqn:
        pmus = sorted({int(p) for r in dqn for p in r["order"].split()})
        steps = max(len(r["order"].split()) for r in dqn)
        prob = np.zeros((steps, len(pmus)))
        for r in dqn:
            for j, p in enumerate(r["order"].split()):
                prob[j, pmus.index(int(p))] += 1 / len(dqn)
        np.savetxt(
            os.path.join(out, f"fig11_ieee{system}.csv"), prob, delimiter=",", header=",".join(map(str, pmus))
        )
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        im = ax.imshow(prob, aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(range(len(pmus)), [str(p) for p in pmus])
        ax.set_yticks(range(steps), [str(j + 1) for j in range(steps)])
        ax.set_xlabel("PMU bus")
        ax.set_ylabel("Configuration step")
        fig.colorbar(im, ax=ax, label="Selection probability")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig11_ieee{system}.png"), dpi=300)
        plt.close(fig)
    # the test windows where an attack survives the defense, so the rise and the extra devices are defined
    tested = [r for r in rows if r["method"] == "dqn" and r["devices_after"] > 0 and r["channels_before"] > 0]
    rise = [(r["channels_after"] - r["channels_before"]) / r["channels_before"] * 100 for r in tested]
    extra = [len(r["extra"].split()) for r in tested]
    if rise:
        rank = np.argsort(rise)
        np.savetxt(  # the figure's data beside it: each ranked test window's rise and extra devices
            os.path.join(out, f"fig12_ieee{system}.csv"),
            np.column_stack([np.asarray(rise)[rank], np.asarray(extra)[rank]]),
            delimiter=",",
            header="rise_percent,extra_devices",
        )
        fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.4))
        a.plot(sorted(rise), "o", color="#1f3b73", markersize=3)
        a.axhspan(10, 20, color="#9aa7c7", alpha=0.35, linewidth=0)  # the paper's 10%-20% band
        a.set_xlabel("Test window (ranked)")
        a.set_ylabel("Attack cost rise (%)")
        b.hist(extra, bins=np.arange(-0.5, max(extra) + 1.5), color="#1f3b73")
        b.set_xlabel("Extra tampered devices")
        b.set_ylabel("Test windows")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"fig12_ieee{system}.png"), dpi=300)
        plt.close(fig)


def write(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--system", type=int, choices=(14, 118), required=True)
    ap.add_argument(
        "--ratings",
        nargs="+",
        default=["+0.10"],
        help='"+d": each line true flow at the window end plus d pu [E14]; "k": k times its peak flow',
    )
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument(
        "--sessions", type=int, default=2, help="DQN sessions; each trains on the others' windows"
    )
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--stride", type=int, default=12, help="pool frames between window starts")
    ap.add_argument("--out", default="docs/wu26")
    ap.add_argument("--store", default=None, help="results store folder (default: the repository's results/)")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    from fdia_graph.engine.attacks.overload import WU26_SCENARIOS

    if args.sessions < 2 or args.windows < args.sessions:  # each session trains on the other folds
        ap.error("--sessions must be at least 2 and at most --windows")
    scenarios = range(len(WU26_SCENARIOS[args.system]))
    starts = [i * args.stride for i in range(args.windows)]
    sol1 = [(args.system, s, m, w, args.budget) for s in scenarios for m in args.ratings for w in starts]
    folds = np.array_split(np.array(starts), args.sessions)
    dqn = [
        (
            args.system,
            s,
            m,
            [int(w) for w in starts if w not in fold],
            [int(w) for w in fold],
            args.budget,
            123 + i,
        )
        for s in scenarios
        for m in args.ratings
        for i, fold in enumerate(folds)
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run_solution1, sol1))
        for part in pool.map(run_dqn_session, dqn):
            rows.extend(part)
    write(rows, os.path.join(args.out, f"table2_ieee{args.system}.csv"))
    write(
        [{k: r[k] for k in ("scenario", "margin", "window", "method", "order", "paper_order")} for r in rows],
        os.path.join(args.out, f"orders_ieee{args.system}.csv"),
    )
    write(
        [{k: r[k] for k in ("scenario", "margin", "window", "method", "seconds")} for r in rows],
        os.path.join(args.out, f"times_ieee{args.system}.csv"),
    )
    figures(rows, args.system, args.out)
    record(rows, args)


def record(rows: list[dict], args: argparse.Namespace) -> None:
    """Every row as records of the run's `wu26.reproduction` experiment in the results store: the
    counts undefended and defended, the cost rise (eq. 33, over channels), the extra devices, the
    decision time and the trusted order, keyed by method, scenario, k and window."""
    from fdia_graph.results import Run, Store

    settings = {k: getattr(args, k) for k in ("ratings", "windows", "sessions", "budget", "stride")}
    run = Run("wu26.reproduction", system=f"ieee{args.system}", settings=settings, store=Store(args.store))
    for r in rows:
        keys = dict(method=r["method"], scenario=r["scenario"], k=r["margin"], window=r["window"])
        for stage in ("before", "after"):
            run.add(
                "devices",
                r[f"devices_{stage}"],
                stage="defended" if stage == "after" else "undefended",
                **keys,
            )
            run.add(
                "channels",
                r[f"channels_{stage}"],
                stage="defended" if stage == "after" else "undefended",
                **keys,
            )
        if (
            r["channels_before"] > 0 and r["channels_after"] > 0
        ):  # -1 is an attack the search found infeasible
            run.add("cost_increase_pct", 100.0 * (r["channels_after"] / r["channels_before"] - 1.0), **keys)
            run.add("extra_devices", len(r["extra"].split()), **keys)
        run.add("decision_seconds", r["seconds"], **keys)
        for step, bus in enumerate(r["order"].split()):
            run.add("trusted_pmu", int(bus), step=step, **keys)
    run.write()


if __name__ == "__main__":
    main()
