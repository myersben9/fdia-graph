#!/usr/bin/env python
"""One overload episode on IEEE-14, step by step: the script behind docs/guides/tour.md.

It generates a 600-frame timeline with one overload (Am) episode, reads the episode back from the
file, and rebuilds it by hand with the calls the generator makes, printing what each step works
with. Step 7 says whether the frame built by hand equals the frame in the file.

Run: python examples/tour.py   (needs fdia-graph[generate]; downloads the IEEE-14 pool on first use)
"""

from __future__ import annotations

import h5py
import numpy as np

import fdia_graph as fg
from fdia_graph.constants.channels import NodeChannel
from fdia_graph.constants.families import AM_CODE
from fdia_graph.engine.attacks.window import AttackWindow
from fdia_graph.generation import _load_states
from fdia_graph.generation.ieee import IeeeSource
from fdia_graph.models.config import TimelineSettings
from fdia_graph.models.frames import AmOverloadDesign

FRAMES, SEED = 600, 123
KNOBS = dict(families=("Am",), attacked_frac=0.2)  # one family, one episode in the train split


def apparent(flows: np.ndarray) -> np.ndarray:
    """|S| in MVA of from-end flows [..., 2] = (P MW, Q MVAr)."""
    return np.hypot(flows[..., 0], flows[..., 1])


def ints(values: np.ndarray) -> list[int]:
    """Positions as plain integers, for printing."""
    return [int(v) for v in values]


def mva(values: np.ndarray) -> list[float]:
    """Flows rounded to 0.01 MVA, for printing."""
    return [round(float(v), 2) for v in values]


def main() -> None:
    # 1. the timeline, by the public call
    path = fg.generate("ieee14", name="tour", frames=FRAMES, seed=SEED, **KNOBS)
    print(f"1. wrote a timeline of {FRAMES} frames, registered as 'tour'")

    # 2. the same source, built by hand: the case, the pool and the meter plan
    X = _load_states("ieee14", None)[:FRAMES]
    source = IeeeSource("ieee14", X, TimelineSettings.of(None, **KNOBS), seed=SEED)
    g, knobs = source.g, source.knobs
    free = g.false_state.free_injection_buses()
    print(f"2. case: {g.n_bus} buses, {g.n_branch} branches, slack at position {g.slack_bus}")
    print(f"   pool: {X.shape} = frames, buses, (|V|, P_inj, Q_inj, theta)")
    print(f"   meters: {len(g.plan.kind)}, PMUs at {ints(g.pmu_buses)}, ", end="")
    print(f"a flow meter on {int(g.flow_metered.sum())} of {g.n_branch} branches")
    print(f"   free injections: {ints(free)}, of which generators {ints(g.false_state.generator_buses())}")
    print(f"   zero-injection buses: {g.zero_inj}")

    # 3. the episode the file holds
    with h5py.File(path, "r") as f:
        ep = f["episodes"]
        k = int(np.flatnonzero(ep["family"][:] == AM_CODE)[0])
        onset, length = int(ep["onset"][k]), int(ep["length"][k])
        lines = ints(ep["am_line"][np.flatnonzero(ep["am_episode"][:] == k)])
        row = int(np.flatnonzero(ep["min_episode"][:] == k)[0])
        ptr = ep["min_support_ptr"][:]
        file_support = ints(ep["min_support_idx"][ptr[row] : ptr[row + 1]])
        file_search = [int(ep[name][row]) for name in ("min_devices", "min_channels", "min_evaluated")]
        last = onset + length - 1
        file_last = f["data/node_x"][last], f["data/edge_x"][last]
        file_y = ints(np.flatnonzero(f["data/y"][last]))
        split = int(f["data/split"][onset])
    print(f"3. episode {k}: onset {onset}, length {length}, split code {split}")
    print(f"   target branches {lines}, between buses {[ints(g.edge_index[:, b]) for b in lines]}")

    # 4. the goal: the flow each target branch must show at each snapshot
    window = [X[u] for u in range(onset, onset + length)]
    eligible = g.overload.eligible_lines(window, knobs.hops)
    goal = g.overload.overload_goal(window, *lines)
    true = apparent(g.emitter.clean_flows_from_states(np.stack(window)))
    print(f"4. eligible branches: {len(eligible)} of {g.n_branch}")
    for j, b in enumerate(goal.lines):
        first, final = goal.targets_at(0)[j], goal.targets_at(length - 1)[j]
        print(f"   branch {b}: true flow {true[0, b]:.2f} to {true[-1, b]:.2f} MVA, ", end="")
        print(f"goal {first:.2f} to {final:.2f} MVA, rating {g.overload.ratings[b]:.2f} MVA")

    # 5. the area, then the search for the support
    area = g.minimizer.window_area(window, goal, knobs)
    found = g.minimizer.min_tamper(window, goal, knobs)
    support = found.support
    costs = AttackWindow(g, window, goal, knobs)
    print(f"5. area: {ints(area)}, boundary {ints(g.area.boundary(area)[1])}")
    print(f"   the whole area costs {costs.cost(area, None)}")
    print(f"   support found: {ints(support)}, {costs.cost(support, None)}")
    print(f"   candidates solved: {found.evaluated}")
    print(f"   boundary of the support: {ints(g.area.boundary(support)[1])}, ", end="")
    print(f"reach {ints(g.false_state.reach(support))}")
    print(f"   the file recorded support {file_support}, devices, channels, solved {file_search}")

    # 6. the false state of the first and the last snapshot
    for i in (0, length - 1):
        x_true = window[i]
        solve = g.false_state.solve_flow_local(
            x_true, support, goal.lines, goal.targets_at(i), knobs.limits, knobs.load_cap
        )
        x_false = solve.x_false
        moved = np.flatnonzero(x_false[:, NodeChannel.V] != x_true[:, NodeChannel.V])
        reached = apparent(g.emitter.clean_flows_from_states(x_false[None])[0, list(goal.lines)])
        change = {int(b): round(float(v), 2) for b, v in enumerate(solve.load_change) if v != 0.0}
        print(f"6. snapshot {i}: converged {solve.converged}, voltages moved at {ints(moved)}")
        print(f"   flow reached {mva(reached)} MVA, pretended load change in MW {change}")

    # 7. the last frame, built as the walk builds it
    design = AmOverloadDesign(goal, goal.targets_at(length - 1), support, found)
    g.scan_key = last  # the walk keys each frame's meter noise by its timestep
    frame, _ = g.overload.overload_step(design, window[-1], length - 1, knobs)
    node, edge = frame.tamper
    print(f"7. last frame: labelled buses {ints(np.flatnonzero(frame.y))}, in the file {file_y}")
    print(f"   node channels written: {int(node.sum())}, at buses {ints(np.flatnonzero(node.any(axis=1)))}")
    print(
        f"   flow channels written: {int(edge.sum())}, on branches {ints(np.flatnonzero(edge.any(axis=1)))}"
    )
    print(f"   PMU current channels written: {int(frame.i_tamper.sum())}")
    same = np.array_equal(frame.node_x, file_last[0]) and np.array_equal(frame.edge_x, file_last[1])
    print(f"   equal to frame {last} of the file: {same}")

    # 8. the file, read back
    ds = fg.load("tour", order="time")
    print(f"8. loaded: {ds.summary()}")
    print(f"   episode buses: {ints(ds.episodes.buses[k])}")


if __name__ == "__main__":
    main()
