"""[WU26]'s trusted-PMU schedule as a constraint on the fewest-tamper search (docs/plans/WU_DEFENSE_PLAN.md,
PR A): a PMU trusted at slot s keeps its bus's |V| and angle true at every snapshot t >= s (eqs. 26-32),
its branch currents stay untrusted (eq. 27), and the support may change at the slots (E13, eq. 28).

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
from fdia_graph.generation import NOISE_FLOOR, _load_states  # noqa: E402
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
    k = FrameKnobs(0.2, NOISE_FLOOR, 6, None, False, True, 2, g.operating_limits(X), True, 4096, 1.0, 0.5, 2)
    return g, window, k, g.overload_goal(window, *lines)


def _schedule(g, scenario: int, per_slot: bool = True) -> TrustSchedule:
    return TrustSchedule([int(b) for b in g.wu26_buses(ORDER[scenario])], SLOTS, per_slot=per_slot)


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


def test_trust_accumulates_from_its_slot_and_cuts_the_window():
    s = TrustSchedule([0, 3, 5, 12], [1, 3, 5, 7])
    assert s.pinned(0) == frozenset()
    assert s.pinned(1) == {0} and s.pinned(4) == {0, 3} and s.pinned(19) == {0, 3, 5, 12}
    assert s.segments(20) == [(0, 1), (1, 3), (3, 5), (5, 7), (7, 20)]
    assert s.segments(4) == [(0, 1), (1, 3), (3, 4)]  # a slot past the window never binds


# ---- the constraint --------------------------------------------------------------------------------
def test_a_trusted_pmu_keeps_its_bus_true_from_its_slot_on():
    """Eq. (29): from its slot, a trusted PMU's |V| and angle are exactly true, even when the support
    holds its bus; before its slot the bus may still move."""
    g, window, k, goal = _setup(0, 1.1)
    trust = _schedule(g, 0)
    w = _Window(g, window, goal, k, trust=trust)
    bus4 = int(g.wu26_buses([4])[0])
    S = np.array(sorted(int(b) for b in g.wu26_buses([3, 4, 5, 7, 11])), dtype=np.int64)
    plan = tuple(S for _ in w.segments)
    assert bus4 in w.support_at(2, plan) and bus4 not in w.support_at(3, plan)
    for t in (2, 3, 10):
        Xa, _ = g.goal_state(goal, t, window[t], w.support_at(t, plan), k)
        assert Xa is not None
        moved = not np.array_equal(Xa[bus4, [NODE.v, NODE.theta]], window[t][bus4, [NODE.v, NODE.theta]])
        assert moved == (t < 3)


def test_an_empty_schedule_costs_what_no_schedule_costs():
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
    plan = result.plan or tuple(result.support for _ in w.segments)
    prev, names = w.prev, set()
    for t in range(len(window)):
        node, edge, cur, prev = w._free_snapshot(t, w.support_at(t, plan), prev)
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
    trust = _schedule(g, 1, per_slot=False)
    r = g.min_tamper(window, goal, k, trust=trust)
    assert (r.devices, r.channels) == (9, 28)
    assert {"SCADA 6", "PMU 6"} <= _named(g, window, k, goal, trust, r)


@SLOW
@pytest.mark.parametrize(
    "scenario, before, after",
    [(0, (5, 8), (6, 11)), (1, (7, 23), (9, 28))],
)
def test_the_ieee14_scenarios_at_k_1_1(scenario, before, after):
    """Undefended and defended counts, held and per slot, and the paper's order against the swapped one:
    at k = 1.1 the per-slot search finds no plan cheaper than the held support, and the order does not
    change the result (the cheapest support avoids every PMU bus)."""
    g, window, k, goal = _setup(scenario, 1.1)
    bare = g.min_tamper(window, goal, k)
    assert (bare.devices, bare.channels) == before
    for per_slot in (False, True):
        r = g.min_tamper(window, goal, k, trust=_schedule(g, scenario, per_slot))
        assert (r.devices, r.channels) == after
    swapped = TrustSchedule([int(b) for b in g.wu26_buses(ORDER[1 - scenario])], SLOTS)
    r = g.min_tamper(window, goal, k, trust=swapped)
    assert (r.devices, r.channels) == after


@SLOW
@pytest.mark.parametrize("scenario", [0, 1])
def test_the_schedule_stops_the_ieee14_scenarios_at_k_1_2(scenario):
    """At k = 1.2 no candidate reaches both ratings with the four PMUs trusted, held or per slot (where
    a plan is also seeded segment by segment when no held support works), while the undefended attack
    does (8 and 10 devices)."""
    g, window, k, goal = _setup(scenario, 1.2)
    assert g.min_tamper(window, goal, k).devices == (8, 10)[scenario]
    for per_slot in (False, True):
        assert g.min_tamper(window, goal, k, trust=_schedule(g, scenario, per_slot)).devices == -1


@pytest.mark.parametrize("bus", [2, 99])  # MATPOWER bus 2 has no PMU in the paper's plan; 99 is no bus
def test_a_trusted_bus_must_carry_a_pmu(bus):
    """The schedule is checked against the grid and the meter plan before the search runs."""
    g, window, k, goal = _setup(0, 1.1)
    trusted = int(g.wu26_buses([bus])[0]) if bus <= 14 else bus
    with pytest.raises(ConfigError):
        g.min_tamper(window, goal, k, trust=TrustSchedule([trusted], [1]))


class _TableWindow:
    """A window whose cost is a table over plans: the per-slot search sees only `segments`, `prev`,
    `cost_plan` and `segment_cost`, so a constructed table isolates its logic from the physics. `alone`
    maps (segment, support, the attack vector the segment starts from) to (its cost on the segment, the
    attack vector it leaves); attack vectors are names, "start" the frame before the window."""

    def __init__(self, costs: dict, alone: dict = {}) -> None:  # noqa: B006  read only
        self.segments, self.prev = [(0, 1), (1, 2)], "start"
        self.channels_first = False
        self.costs, self.alone = costs, alone
        self.converged, self.unsolved, self.last_tried = True, 0, 0

    def cost_plan(self, plan, beat):
        return _beats(self.costs.get(tuple(tuple(int(b) for b in S) for S in plan)), beat)

    def prune_bound(self, lower):
        return lower  # device-first, as the search's default

    def counts(self, cost):
        return cost[0], cost[1]  # the table's costs are device-first

    def segment_cost(self, j, S, beat, prev):
        found = self.alone.get((j, tuple(int(b) for b in S), prev))
        return found if found is not None and _beats(found[0], beat) is not None else None


def _beats(cost, beat):
    return cost if cost is not None and (beat is None or cost < beat) else None


def test_the_support_changes_at_a_slot_when_that_is_cheaper():
    """A constructed window of two segments where A = {1, 2} is the cheapest held support (5 devices) but
    A before the slot and B = {3} after it costs 3: the per-slot search finds the changed plan, and the
    result's support is the union of its segments. The IEEE-14 case studies have no such window (the
    held support is already cheapest there, `test_the_ieee14_scenarios_at_k_1_1`), so the table stands
    in for the physics."""
    from fdia_graph.engine.attacks.minimize import MinimizeMixin

    A, B = np.array([1, 2]), np.array([3])
    window = _TableWindow({((1, 2), (1, 2)): (5, 9, 2), ((1, 2), (3,)): (3, 6, 3), ((3,), (3,)): (7, 12, 1)})
    k = FrameKnobs(0.2, NOISE_FLOOR, 6, None, False, True)
    r = MinimizeMixin._per_slot_result(window, ((5, 9, 2), A), lambda: iter([A, B]), (1, 0, np.arange(4)), k)
    assert [S.tolist() for S in r.plan] == [[1, 2], [3]]
    assert r.support.tolist() == [1, 2, 3]
    assert (r.devices, r.channels) == (3, 6)


def test_a_plan_can_be_feasible_when_no_held_support_is():
    """A constructed window where A = {1, 2} works only before the slot and B = {3} only after it: no
    held support is feasible, so the search seeds a plan with each segment's own cheapest support and
    returns (A, B) instead of reporting the window infeasible. When some segment has no feasible
    support, the window is infeasible (-1, the area as the support)."""
    from fdia_graph.engine.attacks.minimize import MinimizeMixin

    A, B = np.array([1, 2]), np.array([3])
    k = FrameKnobs(0.2, NOISE_FLOOR, 6, None, False, True)
    area = np.arange(4)
    alone = {(0, (1, 2), "start"): ((2, 4, 2), "A"), (1, (3,), "A"): ((3, 5, 1), "B")}
    window = _TableWindow({((1, 2), (3,)): (4, 7, 3)}, alone)
    r = MinimizeMixin._per_slot_result(window, None, lambda: iter([A, B]), (1, 0, area), k)
    assert [S.tolist() for S in r.plan] == [[1, 2], [3]]
    assert r.support.tolist() == [1, 2, 3] and (r.devices, r.channels) == (4, 7)
    stuck = _TableWindow({}, {(0, (1, 2), "start"): ((2, 4, 2), "A")})  # nothing works after the slot
    r = MinimizeMixin._per_slot_result(stuck, None, lambda: iter([A, B]), (1, 0, area), k)
    assert r.devices == -1 and r.support.tolist() == area.tolist() and r.plan == ()


def test_a_segment_starts_from_where_the_one_before_ended():
    """A load goal's stealth bound (At) caps each step between snapshots, so segment 1 can be reachable
    only from the attack vector segment 0 leaves: here B after the slot works from A's end ("A") but not
    from the frame before the window ("start"). The seed carries A's end into segment 1 and finds (A, B);
    measured from "start", segment 1 would have nothing and the window would read infeasible."""
    from fdia_graph.engine.attacks.minimize import MinimizeMixin

    A, B = np.array([1, 2]), np.array([3])
    k = FrameKnobs(0.2, NOISE_FLOOR, 6, None, False, True)
    area = np.arange(4)
    ramp = {(0, (1, 2), "start"): ((2, 4, 2), "A"), (1, (3,), "A"): ((3, 5, 1), "B")}
    window = _TableWindow({((1, 2), (3,)): (4, 7, 3)}, ramp)
    r = MinimizeMixin._per_slot_result(window, None, lambda: iter([A, B]), (1, 0, area), k)
    assert [S.tolist() for S in r.plan] == [[1, 2], [3]] and r.devices == 4
    unprepared = {(0, (1, 2), "start"): ((2, 4, 2), "A"), (1, (3,), "start"): ((3, 5, 1), "B")}
    r = MinimizeMixin._per_slot_result(
        _TableWindow({}, unprepared), None, lambda: iter([A, B]), (1, 0, area), k
    )
    assert r.devices == -1
