"""[WU26]'s case studies on our overload attack: the paper's PMU placement, its two
lines overloaded at once, the ratings 1.2 times each line's peak true flow over the window (the level
at which our device counts matched the paper's in the S_max sensitivity study), and the load and generator bounds.
The paper states no line limits, so the ratings are ours and the comparison is of kind, not of value:
the attack is feasible, reaches both ratings, tampers the paper's devices on IEEE-14, and its largest
per-device change is of the paper's order (0.22 and 0.17 pu there)."""

import os

import numpy as np
import pytest

pytest.importorskip("pandapower")

from fdia_graph.constants.channels import (
    CurrentChannel,  # noqa: E402
    NodeChannel,  # noqa: E402
)
from fdia_graph.engine.attacks.window import AttackWindow  # noqa: E402
from fdia_graph.engine.core import FdiaGenerator  # noqa: E402
from fdia_graph.generation import _load_states  # noqa: E402
from fdia_graph.models.config import OverloadSettings  # noqa: E402
from fdia_graph.models.frames import FrameKnobs  # noqa: E402
from fdia_graph.systems.network import Network

from . import case_studies  # noqa: E402

WINDOW = 10  # snapshots, from the pool's first frame
MARGIN = 1.2  # S_max = MARGIN x each target line's peak true flow over the window
# the paper's tampered devices (MATPOWER bus numbers) and largest per-device change (pu), IEEE-14
PAPER = {
    ((3, 4), (6, 11)): ({3, 4, 5, 6, 11}, {4, 6}, 0.22),
    ((1, 2), (4, 5)): ({1, 2, 3, 4, 5, 9}, {1, 4, 6}, 0.17),
}
# Where our model differs from the paper's, and why. Line 6-11's flow is metered at its from end, bus 6
# (the paper does not say at which end it meters it), and the least-norm false state leaves bus 11's
# injection within noise, so SCADA 11 is not tampered.
KNOWN_DIFFERENCES = {((3, 4), (6, 11)): {"SCADA 11"}}
BAND = (0.05, 1.0)  # the largest per-device change must be of the paper's order, in pu


def _scenario(system: int, scenario: tuple) -> tuple:
    """The generator on the paper's metering, the window, the knobs, the goal and the search result."""
    g = FdiaGenerator(Network.from_ieee(system), seed=1)
    X = _load_states(system, None)
    lines = [case_studies.branch(g, *pair) for pair in scenario]
    flow = g.flow_metered.copy()
    flow[lines] = True  # the operator sees the target lines' flows
    if system == 14:
        flow[case_studies.branch(g, 6, 11)] = True  # the paper meters line 6-11
    g = case_studies.paper_metering(g, case_studies.PMUS[system], flow)
    window = list(X[:WINDOW])
    g.overload.use_line_ratings(OverloadSettings(rating_margin=MARGIN), np.stack(window))
    k = FrameKnobs(2, g.false_state.operating_limits(X), True, 4096, 1.0, 0.5, 2)
    goal = g.overload.overload_goal(window, *lines)
    return g, window, k, goal, g.minimizer.min_tamper(window, goal, k)


def _devices(g, window, k, goal, support) -> tuple[set[str], float]:
    """The devices the held support tampers over the window, named as the paper does ("SCADA 3",
    "PMU 4", MATPOWER numbers), and the largest change on one of their channels (pu, angles in rad)."""
    number = g.case.bus_id
    w = AttackWindow(g, window, goal, k)
    prev, names, worst = w.prev, set(), 0.0
    for t in range(len(window)):
        node, edge, cur, prev = w._snapshot(t, support, prev)
        mag = np.abs(np.asarray(prev.node, float)).copy()
        mag[:, [NodeChannel.P_INJ, NodeChannel.Q_INJ]] /= g.case.base_mva
        mag[:, NodeChannel.THETA] = np.deg2rad(mag[:, NodeChannel.THETA])
        for b, c in zip(*np.nonzero(node)):
            names.add(
                ("PMU " if c in (NodeChannel.V, NodeChannel.THETA) and w.pmu[b] else "SCADA ")
                + str(number[b])
            )
            worst = max(worst, float(mag[b, c]))
        for e, c in zip(*np.nonzero(edge)):
            names.add(f"SCADA {number[g.edge_index[0, e]]}")
            worst = max(worst, abs(float(prev.edge[e, c])) / g.case.base_mva)
        for e, c in zip(*np.nonzero(cur)) if cur is not None else ():
            end = (
                g.edge_index[0, e]
                if c in (CurrentChannel.RE_FROM, CurrentChannel.IM_FROM)
                else g.edge_index[1, e]
            )
            names.add(f"PMU {number[end]}")
            worst = max(worst, abs(float(prev.current[e, c])))
    return names, worst


def _reaches_both_ratings(g, window, k, goal, support) -> None:
    x_false = g.minimizer.goal_state(goal, len(window) - 1, window[-1], support, k)
    assert x_false is not None
    flows = g.emitter.clean_flows_from_states(x_false[None])[0, list(goal.lines)]
    ratings = g.overload.line_ratings()[list(goal.lines)]
    assert np.allclose(np.hypot(flows[:, 0], flows[:, 1]), ratings, rtol=1e-6)


def test_the_scenario_table_maps_to_our_branches_and_buses():
    g14, g118 = FdiaGenerator(Network.from_ieee(14), seed=1), FdiaGenerator(Network.from_ieee(118), seed=1)
    for g, system in ((g14, 14), (g118, 118)):
        for scenario in case_studies.SCENARIOS[system]:
            for a, b in scenario:
                e = case_studies.branch(g, a, b)
                number = g.case.bus_id
                assert {int(number[g.edge_index[0, e]]), int(number[g.edge_index[1, e]])} == {a, b}
    assert len(case_studies.PMUS[14]) == 4 and len(case_studies.PMUS[118]) == 11
    assert set(case_studies.PMUS[118]) <= set(
        case_studies.ATTACK_AREA[118]
    )  # every PMU of Fig. 9 lies in the area
    assert {84, 85, 99, 100} <= set(case_studies.ATTACK_AREA[118])  # and so do the target lines


@pytest.mark.parametrize("scenario", case_studies.SCENARIOS[14])
def test_the_ieee14_scenarios_match_the_paper_in_kind(scenario):
    g, window, k, goal, result = _scenario(14, scenario)
    assert result is not None and result.devices >= 1, "no stealthy two-line overload"
    _reaches_both_ratings(g, window, k, goal, result.support)
    names, worst = _devices(g, window, k, goal, result.support)
    scada, pmu, _ = PAPER[scenario]
    paper = {f"SCADA {b}" for b in scada} | {f"PMU {b}" for b in pmu}
    assert paper - KNOWN_DIFFERENCES.get(scenario, set()) <= names
    assert BAND[0] <= worst <= BAND[1]


@pytest.mark.skipif(not os.environ.get("FDIA_SLOW"), reason="set FDIA_SLOW=1: a minutes-long IEEE-118 search")
def test_the_ieee118_scenario_is_feasible_and_reaches_both_ratings():
    g, window, k, goal, result = _scenario(118, case_studies.SCENARIOS[118][0])
    assert result is not None and result.devices >= 1, "no stealthy two-line overload"
    _reaches_both_ratings(g, window, k, goal, result.support)
