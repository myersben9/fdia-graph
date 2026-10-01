"""[WU26]'s trusted-PMU schedule as a constraint on the fewest-tamper search (docs/plans/WU_DEFENSE_PLAN.md,
PR A): a PMU trusted at slot s keeps its bus's |V| and angle true at every snapshot t >= s (eqs. 26-32),
its branch currents stay untrusted (eq. 27), and one support is held for the window.

The pinned counts are the IEEE-14 case studies on the paper's metering, a window of 20 pool frames with
the PMUs trusted at the paper's snapshots 2, 4, 6 and 8, ratings k times each target line's peak flow
(ours; the paper states none) and the D16 bounds. Table II of the paper reports the attack cost rising
25.6% and 23.9% (scenario 1, Solutions 1 and 2) and 35.2% and 27.0% (scenario 2); at k = 1.1 the counts
below rise 20.0% in devices and 37.5% in channels (5 to 6, 8 to 11) and 28.6% and 21.7% (7 to 9, 23 to
28). At k = 1.2 no candidate's solve reaches both ratings with the four PMUs trusted: the defense stops
the attack there."""

import os

import numpy as np
import pytest

pytest.importorskip("pandapower")

from fdia_graph.engine.attacks.minimize import _Window  # noqa: E402
from fdia_graph.engine.attacks.overload import WU26_PMUS, WU26_SCENARIOS  # noqa: E402
from fdia_graph.engine.core import FdiaGenerator  # noqa: E402
from fdia_graph.generation import _load_states  # noqa: E402
from fdia_graph.models import TrustSchedule  # noqa: E402
from fdia_graph.models.config import OverloadSettings  # noqa: E402
from fdia_graph.models.frames import FrameKnobs  # noqa: E402
from fdia_graph.models.grid import CURRENT, NODE  # noqa: E402
from fdia_graph.models.validation import ConfigError  # noqa: E402

WINDOW = 20  # snapshots from the pool's first frame
SLOTS = [1, 3, 5, 7]  # the paper's snapshots 2, 4, 6, 8, zero-based
ORDER = {0: [1, 4, 6, 13], 1: [4, 6, 1, 13]}  # the paper's Fig. 6, MATPOWER bus numbers
SLOW = pytest.mark.skipif(
    not os.environ.get("FDIA_SLOW"), reason="set FDIA_SLOW=1: IEEE-14 searches of ~20 s"
)


def _setup(scenario: int, margin: float) -> tuple:
    """The generator on the paper's metering, the window, the knobs and the two-line goal (no search)."""
    g = FdiaGenerator(14, seed=1, meter_model="hybrid")
    X = _load_states(14, None)
    lines = [g.wu26_branch(*pair) for pair in WU26_SCENARIOS[14][scenario]]
    flow = np.asarray(g.meters.flow, bool).copy()
    flow[lines] = True
    flow[g.wu26_branch(6, 11)] = True  # the paper meters line 6-11
    g.meters = g.meters._replace(pmu=set(g.wu26_buses(WU26_PMUS[14]).tolist()), flow=flow)
    window = list(X[:WINDOW])
    g.use_line_ratings(OverloadSettings(rating_margin=margin), np.stack(window))
    k = FrameKnobs(2, g.operating_limits(X), True, 4096, 1.0, 0.5, 2)
    return g, window, k, g.overload_goal(window, *lines)


def _schedule(g, scenario: int) -> TrustSchedule:
    return TrustSchedule([int(b) for b in g.wu26_buses(ORDER[scenario])], SLOTS)


# ---- the model -------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad",
    [
        dict(buses=[1, 2], slots=[1]),
        dict(buses=[1, 1], slots=[1, 2]),
        dict(buses=[1.5], slots=[1]),
        dict(buses=[-1], slots=[1]),
        dict(buses=[1], slots=[-1]),
        dict(buses="13", slots=[1, 2]),
    ],
)
def test_the_schedule_refuses_what_it_cannot_mean(bad):
    with pytest.raises(ConfigError):
        TrustSchedule(**bad)


def test_trust_accumulates_from_its_slot():
    # [WU26 eqs. 30-31] [E1]
    s = TrustSchedule([0, 3, 5, 12], [1, 3, 5, 7])
    assert s.pinned(0) == frozenset()
    assert s.pinned(1) == {0} and s.pinned(4) == {0, 3} and s.pinned(19) == {0, 3, 5, 12}


# ---- the constraint --------------------------------------------------------------------------------
def test_a_trusted_pmu_keeps_its_bus_true_from_its_slot_on():
    # [WU26 eqs. 17-18, 27, 29, 32] [E2]
    """Eq. (29): from its slot, a trusted PMU's |V| and angle are exactly true, even when the support
    holds its bus; before its slot the bus may still move."""
    g, window, k, goal = _setup(0, 1.1)
    trust = _schedule(g, 0)
    w = _Window(g, window, goal, k, trust=trust)
    bus4 = int(g.wu26_buses([4])[0])
    S = np.array(sorted(int(b) for b in g.wu26_buses([3, 4, 5, 7, 11])), dtype=np.int64)
    assert bus4 in w.support_at(2, S) and bus4 not in w.support_at(3, S)
    for t in (2, 3, 10):
        Xa = g.goal_state(goal, t, window[t], w.support_at(t, S), k)
        assert Xa is not None
        moved = not np.array_equal(Xa[bus4, [NODE.v, NODE.theta]], window[t][bus4, [NODE.v, NODE.theta]])
        assert moved == (t < 3)


def test_an_empty_schedule_costs_what_no_schedule_costs():
    # [WU26 eqs. 12, 28]
    """No trusted PMU is the search without a schedule: the same cost for every support tried."""
    g, window, k, goal = _setup(1, 1.1)
    bare, empty = _Window(g, window, goal, k), _Window(g, window, goal, k, trust=TrustSchedule([], []))
    for buses in ([2, 4, 5, 6, 7, 8, 9], [2, 5, 6, 11], [2, 3, 4, 5]):
        S = np.array(sorted(int(b) for b in g.wu26_buses(buses)), dtype=np.int64)
        assert bare.cost(S, None) == empty.cost(S, None)


# ---- the case studies ------------------------------------------------------------------------------
def _named(g, window, k, goal, trust, result) -> set[str]:
    number = g.base.bus["name"].astype(int).to_numpy()
    w = _Window(g, window, goal, k, trust=trust)
    prev, names = w.prev, set()
    for t in range(len(window)):
        snap = w._snapshot(t, w.support_at(t, result.support), prev)
        assert snap is not None
        node, edge, cur, prev = snap
        for b, c in zip(*np.nonzero(node)):
            names.add(("PMU " if c in (NODE.v, NODE.theta) and w.pmu[b] else "SCADA ") + str(number[b]))
        for e, _c in zip(*np.nonzero(edge)):
            names.add(f"SCADA {number[g.ei[0, e]]}")
        for e, c in zip(*np.nonzero(cur)):  # a PMU's branch current at its end of the branch
            end = g.ei[0, e] if c in (CURRENT.re_from, CURRENT.im_from) else g.ei[1, e]
            names.add(f"PMU {number[end]}")
    return names


def test_scenario_2_survives_the_schedule_at_a_higher_cost():
    """Lines 1-2 and 4-5, PMUs trusted 4, 6, 1, 13 at k = 1.1: 9 devices and 28 channels against 7 and 23
    undefended (+28.6% and +21.7%; the paper's Table II: 35.2% and 27.0%). The extra devices are SCADA 6
    and PMU 6, whose |V| and angle are trusted from snapshot 4 but whose branch currents are not (eq. 27)."""
    g, window, k, goal = _setup(1, 1.1)
    trust = _schedule(g, 1)
    r = g.min_tamper(window, goal, k, trust=trust)
    assert (r.devices, r.channels) == (9, 28)
    assert {"SCADA 6", "PMU 6"} <= _named(g, window, k, goal, trust, r)


@SLOW
@pytest.mark.parametrize(
    "scenario, before, after",
    [(0, (5, 8), (6, 11)), (1, (7, 23), (9, 28))],
)
def test_the_ieee14_scenarios_at_k_1_1(scenario, before, after):
    """Undefended and defended counts, and the paper's order against the swapped one: the order does not
    change the result (the cheapest support avoids every PMU bus)."""
    g, window, k, goal = _setup(scenario, 1.1)
    bare = g.min_tamper(window, goal, k)
    assert (bare.devices, bare.channels) == before
    r = g.min_tamper(window, goal, k, trust=_schedule(g, scenario))
    assert (r.devices, r.channels) == after
    swapped = TrustSchedule([int(b) for b in g.wu26_buses(ORDER[1 - scenario])], SLOTS)
    r = g.min_tamper(window, goal, k, trust=swapped)
    assert (r.devices, r.channels) == after


@SLOW
@pytest.mark.parametrize("scenario", [0, 1])
def test_the_schedule_stops_the_ieee14_scenarios_at_k_1_2(scenario):
    """At k = 1.2 no candidate reaches both ratings with the four PMUs trusted, while the undefended
    attack does (8 and 10 devices)."""
    g, window, k, goal = _setup(scenario, 1.2)
    assert g.min_tamper(window, goal, k).devices == (8, 10)[scenario]
    assert g.min_tamper(window, goal, k, trust=_schedule(g, scenario)).devices == -1


@pytest.mark.parametrize("bus", [2, 99])  # MATPOWER bus 2 has no PMU in the paper's plan; 99 is no bus
def test_a_trusted_bus_must_carry_a_pmu(bus):
    """The schedule is checked against the grid and the meter plan before the search runs."""
    g, window, k, goal = _setup(0, 1.1)
    trusted = int(g.wu26_buses([bus])[0]) if bus <= 14 else bus
    with pytest.raises(ConfigError):
        g.min_tamper(window, goal, k, trust=TrustSchedule([trusted], [1]))
