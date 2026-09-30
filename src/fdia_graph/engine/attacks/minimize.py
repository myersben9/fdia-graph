"""The fewest-tamper support of an attack window [WU26, eq. 12].

[WU26] builds its attack as the solution of a minimization: the fewest tampered measurements, a
change under a meter's noise not counted, subject to the AC measurement model (satisfied here by
construction, the tampered readings are the readings of a local false state) and the operating
limits. Every tampered reading is a reading of the false state, so which meters move is set by the
support S, the buses whose voltages the false state frees: only meters that depend on a bus of S
can move. The search is over supports.

A candidate S lies in the attacker's area (the region `local_region` returns, the slack never in
it), holds the buses through which it acts on the goal (the targeted load buses of At; a target
line's goal needs only one of its end buses, since a flow changes when either end's voltage moves),
and every other bus of S joins S to a goal bus through S (a bus that does not solves for its own
true injection and moves nothing). Like the region, a support takes in every zero-injection bus of
its boundary (a boundary bus absorbs the changed power, and a bus known to inject nothing cannot).
S is feasible when the local false state with only S free exists at every snapshot of the window,
meets the goal there on its noiseless readings h(x^a), stays inside the operating limits, and, for
a load goal (At), moves no metered channel by more than its accuracy-class sigma from one snapshot
to the next (the stealth bound, the first snapshot measured from the frame before). A flow goal (Am)
has no stealth bound: [WU26]'s model has none, its noise only thresholds the count (the plan's D11). Its cost is the number
of devices (`formulas.attacks.tampered_devices`) with a channel moved beyond its accuracy-class
sigma (`formulas.noise.accuracy_sigma`) at some
snapshot of the window, the union [WU26] counts over its window. A support that moves no device beyond its sigma
at any snapshot is not an attack (an accurate estimator resolves it to the noise floor) and is
treated as infeasible.

The region itself is solved first: the episode was accepted on it, so when the region is feasible
a feasible support exists and bounds the rest. The other candidates follow in order
of size; the search stops when every candidate is solved (the optimum over supports held for the
window), when the best cost reaches the devices the goal forces above noise whatever the support
(the only bound pruned with, the SCADA terminals of the targeted buses at a snapshot whose step
exceeds their noise), or at `k.min_budget` solved candidates (the best found, no claim), and records
which (`MinimizerResult.proven`). A local solve that fails to converge proves nothing about its
support: such candidates are counted (`unsolved`), and the optimum is claimed only among supports
whose solves converged and only when none failed (or the best already sits at the bound). A candidate is dropped early only when its own exact union so far
already exceeds the best. When no held support meets every constraint, the result says so
(`devices = -1`) and the episode runs on the region as without the search.
"""

from __future__ import annotations

import contextlib
import functools
import heapq
import itertools
import threading
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, cast

import numpy as np

from ...formulas.attacks import generator_output, tampered_channels, tampered_devices, within_limits
from ...formulas.network import _row_block, complex_voltages
from ...formulas.noise import (
    PMU_CURRENT_CLASS,
    accuracy_sigma,
    current_sigma,
    paper_current_sigma,
    paper_sigma,
)
from ...models.choices import CostUnit
from ...models.config import TrustSchedule
from ...models.frames import AttackVector, FlowGoal, FrameKnobs, LoadGoal, MinimizerResult
from ...models.grid import NODE
from ...models.inputs import TrustablePmus
from ..base import POWER_NOISE_FLOOR_MW
from .false_state import FalseStateMixin

if TYPE_CHECKING:
    from threadpoolctl import ThreadpoolController

# (devices, channels, support size): the objective, then its tie-breaks; with `FrameKnobs.objective`
# "channels" the first two swap (`_Window.rank`)
_Cost = tuple[int, int, int]
_Best = tuple[_Cost, np.ndarray]  # a solved support and its cost
# the tampered node, flow and PMU current channels (masks; no currents without them in the plan)
_Channels = tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
# one snapshot on a support: the node, flow and PMU current channels moved beyond noise, and its attack vector
_Snapshot = tuple[np.ndarray, np.ndarray, Optional[np.ndarray], AttackVector]
Goal = Union[LoadGoal, FlowGoal]  # what a window must realize: the loads of At, the flow of Am
_Plan = tuple[np.ndarray, ...]  # one support per segment of the window (`TrustSchedule.segments`)
PER_SLOT_PASSES = 2  # rounds of the per-slot search over the segments (it stops once a round gains nothing)


_BLAS: list[Optional[ThreadpoolController]] = []  # the process's BLAS pools, found on the first search
# BLAS limits are process-wide, so concurrent searches share one limit: the first to enter sets it,
# the last to leave restores the caller's setting (a stack of per-call limits would restore out of
# order when calls overlap and leave the process on the wrong count)
_BLAS_LOCK = threading.Lock()
_BLAS_HELD: list[contextlib.AbstractContextManager[object]] = []  # the active limit, while any search runs
_BLAS_USERS = [0]  # searches currently inside the limit


@contextlib.contextmanager
def _one_blas_thread() -> Iterator[None]:
    """Hold the BLAS pools to one thread for a search, restoring them after the last concurrent search
    leaves. The search's products are small (a few hundred buses at most), so on a many-core machine
    starting BLAS threads costs more than the arithmetic (the CHANGELOG's timings; every measured
    episode gave the same answer on one thread as on the default pool). Without threadpoolctl (the
    `generate` extra installs it) the search runs on whatever BLAS is set to. The limit is process-wide,
    so other BLAS work running while a search holds it is on one thread too."""
    with _BLAS_LOCK:
        if not _BLAS:
            try:
                from threadpoolctl import ThreadpoolController

                _BLAS.append(ThreadpoolController())
            except ImportError:
                _BLAS.append(None)
        controller = _BLAS[0]
        if controller is not None and _BLAS_USERS[0] == 0:
            held = controller.limit(limits=1, user_api="blas")
            held.__enter__()
            _BLAS_HELD.append(held)
        _BLAS_USERS[0] += 1
    try:
        yield
    finally:
        with _BLAS_LOCK:
            _BLAS_USERS[0] -= 1
            if _BLAS_USERS[0] == 0 and _BLAS_HELD:
                _BLAS_HELD.pop().__exit__(None, None, None)


class MinimizeMixin(FalseStateMixin):
    """Find the support of an attack window that tampers the fewest devices [WU26, eq. 12]."""

    def min_tamper(
        self,
        states: list[np.ndarray],
        goal: Goal,
        k: FrameKnobs,
        prev: Optional[AttackVector] = None,
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """The fewest-tamper support for the window of `states` (one [N, 4] true state per snapshot)
        and `goal` (one attack design per snapshot), or None when the goal's buses have no area. `prev`
        is the attack vector of the frame before the window (node [N, 4], flow [E, 2]; None or zeros
        when that frame is benign): At's stealth bound measures its first increment from it, since
        episodes may be adjacent (a flow goal has no stealth bound, the plan's D11). `trust` is the
        defender's trusted-PMU schedule [WU26, eqs. 26-32]: from its slot on, a trusted PMU's bus keeps
        its true voltage whatever the support (eq. 29), and with `trust.per_slot` the support may change
        at each slot (`_per_slot`). The search runs with BLAS on one thread (`_one_blas_thread`)."""
        if trust is not None:  # refused before the search runs: every trusted bus carries a PMU
            TrustablePmus(trust.buses, frozenset(self.meters.pmu), self.C)
        with _one_blas_thread():
            return self._min_tamper(states, goal, k, prev, trust)

    def _min_tamper(
        self,
        states: list[np.ndarray],
        goal: Goal,
        k: FrameKnobs,
        prev: Optional[AttackVector],
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """`min_tamper`'s search, on whatever BLAS threads the caller set."""
        seeds, starts, must_hold = self._goal_seeds(goal)
        area = self.local_region(seeds, k.hops)
        if area is None:
            return None
        starts = _starts_in_area(starts, area, must_hold)
        window = _Window(self, states, goal, k, prev=prev, trust=trust)
        region = window.cost(np.asarray(area), None)
        best = None if region is None else (region, np.asarray(area))
        lower = window.lower_bound()
        window.unsolved = 0 if window.converged else 1  # the region's own solve
        best, evaluated, exhausted = self._search(
            window, self._supports(starts, area, must_hold), (best, window.prune_bound(lower)), area, k
        )
        if trust is not None and trust.per_slot:
            candidates = functools.partial(self._supports, starts, area, must_hold)
            return self._per_slot_result(window, best, candidates, (evaluated, lower, np.asarray(area)), k)
        if best is None:  # no support held for the window meets every constraint: say so, keep the region
            return MinimizerResult(np.asarray(area), -1, -1, False, evaluated, lower, window.unsolved)
        (devices, channels), support = window.counts(best[0]), best[1]
        # optimal among supports whose solves converged, and only when none that failed could have beaten it
        proven = window.prune_bound(lower) >= best[0][0] or (exhausted and window.unsolved == 0)
        return MinimizerResult(support, devices, channels, proven, evaluated, lower, window.unsolved)

    @staticmethod
    def _per_slot_result(
        window: _Window,
        best: Optional[_Best],
        candidates: Callable[[], Iterator[np.ndarray]],
        counts: tuple[int, int, np.ndarray],
        k: FrameKnobs,
    ) -> MinimizerResult:
        """The per-slot search's result (`_per_slot`) from the held search's `best` and its `counts`
        (candidates solved, the goal-forced bound, the area): the plan, the support the union of its
        supports, proven only at the bound (a search one segment at a time is not exhaustive over
        plans). When no held support is feasible, a plan can still be (a PMU trusted mid-window can
        rule out every support that works before its slot and after it): each segment's own cheapest
        support seeds the plan (`_segment_seed`), and only when some segment has none is the window
        infeasible."""
        evaluated, lower, area = counts
        start = None if best is None else (best[0], tuple(best[1] for _ in window.segments))
        if start is None:
            start, seeded = _segment_seed(window, candidates, k)
            evaluated += seeded
        if start is None:
            return MinimizerResult(area, -1, -1, False, evaluated, lower, window.unsolved)
        cost, plan, more = _per_slot(window, start, candidates, k)
        devices, channels = window.counts(cost)
        support = np.array(sorted({int(b) for S in plan for b in S}), dtype=np.int64)
        proven = window.prune_bound(lower) >= cost[0]
        return MinimizerResult(
            support, devices, channels, proven, evaluated + more, lower, window.unsolved, plan
        )

    @staticmethod
    def _search(
        window: _Window,
        candidates: Iterator[np.ndarray],
        start: tuple[Optional[_Best], int],
        area: np.ndarray,
        k: FrameKnobs,
    ) -> tuple[Optional[_Best], int, bool]:
        """Solve the candidates in order from the incumbent `start` = (best so far, the goal-forced lower
        bound): (the best, candidates solved including the region, whether the search finished)."""
        best, lower = start
        evaluated = 1  # the region, solved before the search
        for S in candidates:
            if best is not None and best[0][0] <= lower:
                return best, evaluated, True  # at the bound every support shares: nothing tampers fewer
            if evaluated >= k.min_budget:
                return best, evaluated, False
            if np.array_equal(S, area):
                continue
            evaluated += 1
            cost = window.cost(S, best[0] if best is not None else None)  # None unless it beats the best
            window.unsolved += 0 if window.converged else 1
            best = best if cost is None else (cost, S)
        return best, evaluated, True

    def _goal_seeds(self, goal: Goal) -> tuple[np.ndarray, list[frozenset[int]], Optional[frozenset[int]]]:
        """(the buses the area grows from, the candidate supports' starting sets, the buses a candidate
        must hold one of, or None). A load goal acts through its targeted load buses, all of them in
        every support. A flow goal acts on each of its branches through either end (a flow changes
        when one end's voltage moves), so every choice of one end per branch starts its own
        candidates, and a support must hold a free injection (an attackable load or a generator, the
        injections the flow goal frees)."""
        if goal.kind == "load":
            buses = np.unique(self.load_bus[cast(LoadGoal, goal).designs[0].targets])
            return buses, [frozenset(int(b) for b in buses)], None
        lines = list(cast(FlowGoal, goal).lines)
        ends = np.unique(self.ei[:, lines])
        choices = [
            frozenset(int(b) for b in combo) for combo in itertools.product(*(self.ei[:, ln] for ln in lines))
        ]
        starts = list(dict.fromkeys(choices))  # each set once, in a fixed order
        return ends, starts, frozenset(int(b) for b in self.free_injection_buses())

    def goal_state(
        self, goal: Goal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> tuple[Optional[np.ndarray], bool]:
        """(The false state of snapshot t on support S that meets the goal there, inside the operating
        limits, or None; whether the solve converged). Unknowns: |V| and theta of S, every other voltage
        held true. A solve that did not converge is not proof that S is infeasible, so the search counts
        it apart. Each goal kind has its own solve: a load goal is the square local power flow (the
        targeted loads take their new values, every other bus of S keeps its true injection); a flow
        goal frees the attackable loads and generators of S and holds the rest (`solve_flow_local`)."""
        if goal.kind == "flow":
            return self._flow_goal_state(cast(FlowGoal, goal), t, Xt, S, k)
        return self._load_goal_state(cast(LoadGoal, goal), t, Xt, S, k)

    def _flow_goal_state(
        self, goal: FlowGoal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> tuple[Optional[np.ndarray], bool]:
        Xa, converged, dload = self.solve_flow_local(
            Xt, S, goal.lines, goal.targets_at(t), k.limits, k.load_cap
        )
        if Xa is None:
            return None, converged
        if k.limits is not None:  # every generator the attack moves, on S's edge too (D16)
            gen = generator_output(Xt, self.load_base, self.gen_base)
            if not within_limits(Xa, Xt, gen, dload, k.limits, self.touched_buses(S)):
                return None, True
        return Xa, True

    def _load_goal_state(
        self, goal: LoadGoal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> tuple[Optional[np.ndarray], bool]:
        design = goal.designs[t]
        Lp_true, Lq = self.true_load(Xt), self.true_reactive_load(Xt)
        Lp = Lp_true.copy()
        Lp[design.targets] *= design.mult
        Xa = self.solve_local(Xt, S, Lp, Lq)
        if Xa is None:
            return None, False
        if k.limits is not None and not self._within_limits(Xa, Xt, Lp - Lp_true, k.limits, S):
            return None, True
        return Xa, True

    def _supports(
        self, starts: list[frozenset[int]], area: np.ndarray, must_hold: Optional[frozenset[int]] = None
    ) -> Iterator[np.ndarray]:
        """Every candidate support in the area, smallest first: the goal's buses, grown one adjacent area
        bus at a time (so every added bus joins a goal bus through the support), each closed over the
        zero-injection buses of its boundary exactly as `local_region` grows the region (a bus known to
        inject nothing cannot absorb the change, so it moves with the support), each set once. The
        slack is never in a support: the area excludes it."""
        allowed = {int(b) for b in area}
        adj = self._area_adjacency(allowed)
        heap: list[tuple[int, tuple[int, ...]]] = []
        seen: set[frozenset[int]] = set()
        for s in starts:
            _push(heap, seen, self._zero_closed(s))
        while heap:
            _, key = heapq.heappop(heap)
            S = frozenset(key)
            if must_hold is None or S & must_hold:  # a flow goal's support needs a free load
                yield np.array(key, dtype=np.int64)
            self._grow(S, adj, allowed, (heap, seen))

    def _grow(
        self,
        S: frozenset[int],
        adj: dict[int, set[int]],
        allowed: set[int],
        queue: tuple[list[tuple[int, tuple[int, ...]]], set[frozenset[int]]],
    ) -> None:
        """Queue every support one adjacent area bus larger than S, closed over its zero-injection
        boundary, that stays in the area and was not queued before."""
        heap, seen = queue
        for v in sorted({n for b in S for n in adj[b]} - S):
            child = self._zero_closed(S | {v})
            if child <= allowed:
                _push(heap, seen, child)

    def _area_adjacency(self, allowed: set[int]) -> dict[int, set[int]]:
        """The live branches between buses of the area, as neighbour sets."""
        adj: dict[int, set[int]] = {b: set() for b in allowed}
        for a, b in self._live_edges().T:
            if int(a) in allowed and int(b) in allowed:
                adj[int(a)].add(int(b))
                adj[int(b)].add(int(a))
        return adj

    def _zero_closed(self, S: frozenset[int]) -> frozenset[int]:
        """S with every zero-injection bus of its boundary taken in, as `local_region` grows the region."""
        grown, _ = self._grow_over_zero_injection(np.array(sorted(S), dtype=np.int64))
        return frozenset(int(b) for b in grown)


def _segment_seed(
    window: _Window, candidates: Callable[[], Iterator[np.ndarray]], k: FrameKnobs
) -> tuple[Optional[tuple[_Cost, _Plan]], int]:
    """A plan from each segment's cheapest support over the segment's own snapshots
    (`_Window.segment_cost`), joined and costed over the window by the union count [WU26, eq. 28], or
    None when some segment has no feasible support or the joined plan is not an attack; and the
    candidates solved. The segments are seeded in order, each from the attack vector its predecessor's
    chosen support leaves at the segment's first snapshot, so a load goal's stealth bound (At) measures
    a segment's first step from where the previous segment ended, as the whole window does. The seed is
    greedy: an earlier segment's cheapest support can leave a later segment nothing a costlier one would
    have left it, so a None here is "none found", not a proof (the result is never `proven` then). Every
    plan it returns is checked over the whole window by `cost_plan`."""
    plan, evaluated, prev = [], 0, window.prev
    for j in range(len(window.segments)):
        best: Optional[tuple[_Cost, np.ndarray, AttackVector]] = None
        for S in itertools.islice(candidates(), k.min_budget):
            found = window.segment_cost(j, S, None if best is None else best[0], prev)
            evaluated += 1
            window.unsolved += 0 if window.converged else 1
            best = best if found is None else (found[0], S, found[1])
        if best is None:
            return None, evaluated
        plan.append(best[1])
        prev = best[2]
    cost = window.cost_plan(tuple(plan), None)
    return (None if cost is None else (cost, tuple(plan))), evaluated


def _per_slot(
    window: _Window, start: tuple[_Cost, _Plan], candidates: Callable[[], Iterator[np.ndarray]], k: FrameKnobs
) -> tuple[_Cost, _Plan, int]:
    """A support per segment of the trusted schedule (the plan's E13): [WU26, eq. 28] counts the window's
    tampered measurements with each snapshot's deviation taken on its own, so the attacker may move a
    different set of buses once a PMU becomes trusted. From the held support, each segment in turn takes
    the candidate that lowers the window's cost most with the other segments fixed, until a round over
    the segments gains nothing (at most `PER_SLOT_PASSES` rounds, `k.min_budget` candidates per
    segment). `start` is the held support as a plan, or `_segment_seed`'s plan when no held support is
    feasible. Returns (the cost, the plan, candidates solved)."""
    cost, plan = start
    evaluated = 0
    for _ in range(PER_SLOT_PASSES):
        before = cost
        for j in range(len(plan)):
            cost, plan = _best_for_segment(
                window, (cost, plan), j, itertools.islice(candidates(), k.min_budget)
            )
            evaluated += min(k.min_budget, window.last_tried)
        if cost == before:
            break
    return cost, plan, evaluated


def _best_for_segment(
    window: _Window, incumbent: tuple[_Cost, _Plan], j: int, candidates: Iterator[np.ndarray]
) -> tuple[_Cost, _Plan]:
    """The plan with segment j's support the candidate that lowers the window's cost most, the other
    segments as in `incumbent` (kept when no candidate beats it); `window.last_tried` counts the
    candidates solved."""
    cost, plan = incumbent
    window.last_tried = 0
    for S in candidates:
        trial = plan[:j] + (S,) + plan[j + 1 :]
        found = window.cost_plan(trial, cost)
        window.last_tried += 1
        window.unsolved += 0 if window.converged else 1
        if found is not None:
            cost, plan = found, trial
    return cost, plan


def _push(heap: list[tuple[int, tuple[int, ...]]], seen: set[frozenset[int]], S: frozenset[int]) -> None:
    """Queue support S by size, once."""
    if S not in seen:
        seen.add(S)
        heapq.heappush(heap, (len(S), tuple(sorted(S))))


def _starts_in_area(
    starts: list[frozenset[int]], area: np.ndarray, must_hold: Optional[frozenset[int]]
) -> list[frozenset[int]]:
    """A flow goal starts only from its branch ends inside the area (an end at the slack is not one);
    a load goal's starting set is kept as it is."""
    if must_hold is None:
        return starts
    inside = {int(b) for b in area}
    return [s for s in starts if s <= inside]


class _Near(NamedTuple):
    """The branches with an end in a support (`lines`) and what their readings need: their from-end
    and to-end Yf and Yt rows cut to the columns they touch (`_row_block`: block, columns), their
    from buses in ppc order, and whether each is metered."""

    lines: np.ndarray
    yf: tuple[np.ndarray, np.ndarray]
    yt: tuple[np.ndarray, np.ndarray]
    from_ppc: np.ndarray
    metered: np.ndarray


def _near_branches(g: MinimizeMixin, S: np.ndarray) -> _Near:
    """`_Near` of support S: the in-service branches with an end in S."""
    touches = np.isin(g.ei, S).any(axis=0)
    if g.branch.status is not None:
        touches &= np.asarray(g.branch.status) > 0
    lines = np.flatnonzero(touches)
    metered = np.asarray(g.meters.flow, bool)[lines]
    from_ppc = np.asarray(g._from_bus_ppc)[lines]
    return _Near(lines, _row_block(g._Yf, lines), _row_block(g._Yt, lines), from_ppc, metered)


class _Window:
    """One attack window evaluated for candidate supports: the true states, their accuracy sigmas and
    meter masks computed once, the goal's design per snapshot."""

    def __init__(
        self,
        g: MinimizeMixin,
        states: list[np.ndarray],
        goal: Goal,
        k: FrameKnobs,
        stealth_bound: bool = True,
        prev: Optional[AttackVector] = None,
        trust: Optional[TrustSchedule] = None,
    ) -> None:
        # The between-snapshot stealth bound is At's alone (a sub-noise ramp is what At is). [WU26]'s
        # model, eqs. (12)-(25), has no increment constraint: its noise only decides which changes the
        # l0 count ignores, so a flow goal (Am) is never bounded (the plan's D11). `stealth_bound` off
        # is for analysis only.
        self.g, self.states, self.goal, self.k = g, states, goal, k
        self.stealth_bound = stealth_bound and goal.kind == "load"
        self.channels_first = k.objective == CostUnit.CHANNELS.value  # what the cost minimizes first
        self.converged = True  # whether every local solve of the last `cost` call converged
        self._near: Optional[tuple[bytes, _Near]] = None  # the last support's branches (`near`)
        self.unsolved = 0  # candidates of the search whose solve failed to converge
        # the trusted-PMU schedule [WU26, eqs. 29-31]: the buses pinned at each snapshot, the window's
        # segments (a support per segment is a plan), and a cache of snapshot results for the per-slot
        # search, which re-solves one segment while the others repeat (a flow goal's snapshot does not
        # depend on the one before, so its result is a function of its free buses alone)
        self.pinned, self.segments, self._segment_of = _schedule_of(trust, len(states))
        self._cache = _snapshot_cache(trust, goal)
        self.last_tried = 0  # candidates the last per-slot segment search solved (`_best_for_segment`)
        self.node_m, self.edge_m = g.meter_masks()
        self.pmu = np.zeros(g.C, bool)
        self.pmu[sorted(g.meters.pmu)] = True
        # a bus angle is a PMU channel: a SCADA voltmeter reads |V| only, so the attack's tamper count
        # and stealth bound see an angle only where a PMU is. The hybrid meter plan already masks it
        # there (the plan's D10); a v0.8.3-plan generator still writes one at every voltmeter bus.
        self.node_m = self.node_m.copy()
        self.node_m[~self.pmu, NODE.theta] = 0
        flows = g.clean_flows_from_states(np.stack(states))  # [T, E, 2], unmetered zeroed
        self.flows = flows
        # the PMU branch-current channels [WU26, eqs. 19-20]: their mask and noise scale (None without)
        self.i_m = g.current_mask()
        self.currents, self.i_sigma = self._currents(g, states, goal)
        # what a change must exceed to count as tampering, and for At also the most a channel may move
        # between snapshots (the emitter's per-scan jitter is smaller, and is not a detection threshold)
        if goal.kind == "flow":  # the overload attack: [WU26]'s own noise, the l0 threshold only (D8, D11)
            paper = paper_sigma(self.node_m.shape, self.edge_m.shape, self.pmu, g._base_mva)
            self.sigma = [paper for _ in states]
        else:  # At: the meters' rated accuracy (D7)
            self.sigma = [accuracy_sigma(X, F, g.SD, POWER_NOISE_FLOOR_MW) for X, F in zip(states, flows)]
        zero = {int(b) for b in g.zero_inj} - {g.slack_bus}
        self.zero = np.array(sorted(zero), dtype=np.int64)
        # the attack vector of the frame before the window: zero when it is benign
        no_i = None if self.i_m is None else np.zeros(self.i_m.shape)
        if prev is None:
            self.prev = AttackVector(np.zeros(self.node_m.shape), np.zeros(self.edge_m.shape), no_i)
        else:  # a bare (node, edge) pair is an attack vector without currents
            p = AttackVector(*prev)
            cur = no_i if p.current is None else np.asarray(p.current, float)
            self.prev = AttackVector(np.asarray(p.node, float), np.asarray(p.edge, float), cur)

    @staticmethod
    def _currents(
        g: MinimizeMixin, states: list[np.ndarray], goal: Goal
    ) -> tuple[Optional[np.ndarray], Optional[list[np.ndarray]]]:
        """(The true PMU branch currents per snapshot [T, E, 4], computed once since every candidate's
        attack vector subtracts them, as it subtracts `flows`; their scale per snapshot: [WU26]'s
        0.01 pu for the overload attack (D8), the PMU accuracy class at the true currents for At
        (D7)), both None without currents in the meter plan."""
        if g.current_mask() is None:
            return None, None
        true = g.currents_from_states(np.stack(states))
        if goal.kind == "flow":
            return true, [paper_current_sigma((g.E, 4)) for _ in true]
        return true, [current_sigma(i, PMU_CURRENT_CLASS) for i in true]

    def lower_bound(self) -> int:
        """Devices every support tampers, the only bound the search prunes with (`_load_bound`,
        `_flow_bound`)."""
        return self._flow_bound() if self.goal.kind == "flow" else self._load_bound()

    def _flow_bound(self) -> int:
        """A flow goal forces the device metering each goal branch (the SCADA terminal of its from-end
        bus) when the goal's change there must cross that meter's sigma at some snapshot: the change
        delta of the apparent flow splits between P and Q, so one of them moves by at least
        delta / sqrt(2), forced over noise only when that exceeds the larger of the two sigmas."""
        goal = cast(FlowGoal, self.goal)
        forced: set[int] = set()
        for i, line in enumerate(goal.lines):
            if not self.edge_m[line].any():
                continue
            for t in range(len(goal.targets)):
                true = float(np.hypot(*self.flows[t, line]))
                sig = float(np.max(self.sigma[t][1][line]))
                if abs(goal.targets_at(t)[i] - true) / np.sqrt(2.0) > sig:
                    forced.add(int(self.g.ei[0, line]))
                    break
        return len(forced)

    def _load_bound(self) -> int:
        """The load goal's bound: the SCADA terminal of a
        target bus whose injected active power the goal fixes (the load the attacker pretends) and whose
        designed step exceeds that injection meter's accuracy-class sigma at some snapshot. The goal
        leaves a bus's reactive load unchanged, so only the active channel is forced."""
        g, C = self.g, self.g.C
        power = np.zeros(C, bool)
        for t, design in enumerate(cast(LoadGoal, self.goal).designs):
            Lp = g.true_load(self.states[t])
            dload = np.zeros(C)
            np.add.at(dload, g.load_bus[design.targets], Lp[design.targets] * (np.asarray(design.mult) - 1.0))
            sig_node, _ = self.sigma[t]
            power |= (np.abs(dload) > sig_node[:, NODE.p_inj]) & (self.node_m[:, NODE.p_inj] > 0)
        node = np.zeros((C, 4), bool)
        node[:, NODE.p_inj] = power
        return len(tampered_devices(node, np.zeros(self.edge_m.shape, bool), self.pmu, g.ei[0]))

    def cost(self, S: np.ndarray, beat: Optional[_Cost]) -> Optional[_Cost]:
        """The cost of support S held for the window, or None when S is infeasible at a snapshot, moves
        no device beyond its accuracy sigma over the window (not an attack), or cannot beat `beat`
        (stopped as soon as its devices so far exceed it)."""
        return self.cost_plan(tuple(S for _ in self.segments), beat)

    def cost_plan(self, plan: _Plan, beat: Optional[_Cost]) -> Optional[_Cost]:
        """`cost` of a plan, one support per segment of the window: the union of the tampered devices
        over every snapshot [WU26, eq. 28], the support size the number of the plan's buses."""
        self.converged = True
        if any(self._zero_on_boundary(S) for S in plan):
            return None  # never for a candidate of `_supports`, which takes such a bus in; a guard for others
        tampered = self._tampered(plan, beat)
        if tampered is None:
            return None
        union, devices, _ = tampered
        if not devices:
            return None  # within noise at every snapshot: no effect, so not an attack [the sub-noise rule]
        cost = self.rank(devices, _channel_count(union), _plan_size(plan))
        return cost if beat is None or cost < beat else None

    def segment_cost(
        self, j: int, S: np.ndarray, beat: Optional[_Cost], prev: AttackVector
    ) -> Optional[tuple[_Cost, AttackVector]]:
        """The cost of support S over segment j's snapshots alone and the attack vector it leaves at the
        segment's last snapshot, or None when S fails there or cannot beat `beat`. `prev` is the attack
        vector the segment starts from (the preceding segment's last, or the frame before the window's
        for the first): a load goal's stealth bound measures the segment's first step from it, and a flow
        goal ignores it. A segment may move no device beyond noise: only the joined plan must be an
        attack."""
        a, b = self.segments[j]
        self.converged = True
        tampered = self._tampered(tuple(S for _ in self.segments), beat, (range(a, b), prev))
        if tampered is None:
            return None
        union, devices, last = tampered
        cost = self.rank(devices, _channel_count(union), len(S))
        return (cost, last) if beat is None or cost < beat else None

    def support_at(self, t: int, plan: _Plan) -> np.ndarray:
        """The buses free at snapshot t under `plan`: its segment's support less the buses of the PMUs
        trusted by then (eq. 29 holds their deviation at zero, so they keep their true voltage)."""
        S = plan[self._segment_of[t]]
        pinned = self.pinned[t]
        return S if not pinned else S[~np.isin(S, sorted(pinned))]

    def _tampered(
        self, plan: _Plan, beat: Optional[_Cost], span: Optional[tuple[range, AttackVector]] = None
    ) -> Optional[tuple[_Channels, int, AttackVector]]:
        """The union of tampered channels (node, flow, PMU current masks) under `plan` over the window's
        snapshots, its device count and the attack vector of the last snapshot, or None when it fails at
        a snapshot or its devices so far exceed `beat`'s. `span` = (snapshots, the attack vector before
        the first) evaluates part of the window; by default the whole window from the frame before it."""
        union: _Channels = (np.zeros(self.node_m.shape, bool), np.zeros(self.edge_m.shape, bool), None)
        devices = 0
        # the frame before the window (its attack vector, zero when benign), or the given span's start
        snapshots, prev = (range(len(self.states)), self.prev) if span is None else span
        for t in snapshots:
            moved = self._free_snapshot(t, self.support_at(t, plan), prev)
            if moved is None:
                return None
            node, edge, current, prev = moved
            union = _union(union, (node, edge, current))
            devices = self._devices(union)
            first = _channel_count(union) if self.channels_first else devices
            if beat is not None and first > beat[0]:
                return None
        return union, devices, prev

    def rank(self, devices: int, channels: int, size: int) -> _Cost:
        """A cost in the order the search compares it: the objective first (`FrameKnobs.objective`),
        the other count next, the support size last."""
        return (channels, devices, size) if self.channels_first else (devices, channels, size)

    def prune_bound(self, lower: int) -> int:
        """What the cost's first count is known to reach at least: the goal-forced devices `lower` when
        the search minimizes devices, nothing when it minimizes channels (the forced devices bound no
        channel count)."""
        return 0 if self.channels_first else lower

    def counts(self, cost: _Cost) -> tuple[int, int]:
        """(devices, channels) of a cost `rank` built."""
        return (cost[1], cost[0]) if self.channels_first else (cost[0], cost[1])

    def _devices(self, union: _Channels) -> int:
        """How many devices hold a channel of the union (a PMU branch current joins the PMU of the
        bus at its end). A device is a function of its channel alone, so the devices of the union
        are the union of every snapshot's devices."""
        ei = self.g.ei
        return len(tampered_devices(union[0], union[1], self.pmu, ei[0], union[2], ei[1]))

    def _zero_on_boundary(self, S: np.ndarray) -> bool:
        """Whether a zero-injection bus sits on S's boundary (it could not absorb the changed power)."""
        return bool(np.isin(self.g._boundary(S)[1], self.zero).any())

    def _free_snapshot(self, t: int, S: np.ndarray, prev: AttackVector) -> Optional[_Snapshot]:
        """`_snapshot` on the buses S free at snapshot t: none free is no attack there, a zero-injection
        bus that a trusted PMU leaves on S's boundary cannot absorb the change, and under a trusted
        schedule a flow goal's result is cached by (t, S)."""
        if not len(S) or (self.pinned[t] and self._zero_on_boundary(S)):
            return None
        if self._cache is None:
            return self._snapshot(t, S, prev)
        key = (t, S.tobytes())
        if key not in self._cache:
            self.converged = True
            out = self._snapshot(t, S, prev)
            self._cache[key] = (self.converged, out)
        converged, out = self._cache[key]
        if out is None:
            self.converged = converged
        return out

    def _snapshot(self, t: int, S: np.ndarray, prev: AttackVector) -> Optional[_Snapshot]:
        """Snapshot t on support S: (the channels moved beyond their noise, node, flow and PMU branch
        current, and this snapshot's attack vector), or None when S has no false state here or (At)
        breaks the stealth bound against the previous snapshot's attack vector `prev`."""
        g = self.g
        Xa, converged = g.goal_state(self.goal, t, self.states[t], S, self.k)
        if Xa is None:
            self.converged = converged
            return None
        a_node = (np.asarray(Xa, float) - np.asarray(self.states[t], float)).astype(np.float32)
        a_edge, a_cur = self._branch_attack(t, Xa, self.near(S))
        sig_node, sig_edge = self.sigma[t]
        # At's stealth bound: no metered channel moves more than its rated accuracy between snapshots
        scale = self.k.stealth_scale  # the bound's step in multiples of the rated accuracy (D7: 1)
        step = tampered_channels(
            a_node - prev.node,
            a_edge - prev.edge,
            scale * sig_node,
            scale * sig_edge,
            self.node_m,
            self.edge_m,
        )
        if self.stealth_bound and (
            step[0].any() or step[1].any() or self._current_step(t, a_cur, prev, scale)
        ):
            return None
        node, edge = tampered_channels(a_node, a_edge, sig_node, sig_edge, self.node_m, self.edge_m)
        current = self._over(t, a_cur, 1.0)
        return node, edge, current, AttackVector(a_node, a_edge, a_cur)

    def near(self, S: np.ndarray) -> _Near:
        """`_Near` of support S, kept for the last support (a search solves one support at every
        snapshot of the window in turn)."""
        key = np.asarray(S, np.int64).tobytes()
        if self._near is None or self._near[0] != key:
            self._near = (key, _near_branches(self.g, S))
        return self._near[1]

    def _branch_attack(self, t: int, Xa: np.ndarray, near: _Near) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """The attack vector of false state Xa at snapshot t on the flow channels [E, 2] and the PMU
        branch currents [E, 4] (None without currents), as `_attack_vector` and `_current_attack` give
        it: only a branch with an end in the support sees a false voltage, so only the support's
        branches (`near`) are evaluated and every other branch's entry is exactly zero, as it is
        there (its readings come from the same true voltages on both sides)."""
        g = self.g
        lut = g._ppc_row[np.arange(g.C)]
        V = np.zeros(g._n_ppc_buses, complex)
        V[lut] = complex_voltages(Xa[:, NODE.v], Xa[:, NODE.theta])
        (Yf, f_cols), (Yt, t_cols) = near.yf, near.yt
        If = Yf @ V[f_cols]
        Sf = V[near.from_ppc] * np.conj(If) * g._base_mva
        flows = np.stack([np.real(Sf), np.imag(Sf)], axis=1).astype(np.float32)
        flows[~near.metered] = (
            0.0  # an unmetered flow is zero on both sides, as `clean_flows_from_states` has it
        )
        a_edge = np.zeros(self.edge_m.shape, np.float32)
        a_edge[near.lines] = flows - self.flows[t][near.lines]
        if self.currents is None or self.i_m is None:
            return a_edge, None
        It = Yt @ V[t_cols]
        cur = np.stack([np.real(If), np.imag(If), np.real(It), np.imag(It)], axis=1) * self.i_m[near.lines]
        a_cur = np.zeros(self.i_m.shape)
        a_cur[near.lines] = cur - self.currents[t][near.lines]
        return a_edge, a_cur

    def _over(self, t: int, a_cur: Optional[np.ndarray], scale: float) -> Optional[np.ndarray]:
        """The PMU branch-current channels an attack-vector part moves beyond `scale` times their noise
        at snapshot t, metered ones only; None without currents in the plan."""
        if a_cur is None or self.i_m is None or self.i_sigma is None:
            return None
        return (np.abs(a_cur) > scale * self.i_sigma[t]) & (self.i_m > 0)

    def _current_step(self, t: int, a_cur: Optional[np.ndarray], prev: AttackVector, scale: float) -> bool:
        """Whether the attack's step on a PMU branch-current channel breaks the stealth bound."""
        if a_cur is None or prev.current is None:
            return False
        over = self._over(t, a_cur - prev.current, scale)
        return bool(over is not None and over.any())


def _schedule_of(
    trust: Optional[TrustSchedule], T: int
) -> tuple[list[frozenset[int]], list[tuple[int, int]], list[int]]:
    """(The buses pinned at each of the T snapshots, the window's segments, each snapshot's segment)
    under a trusted-PMU schedule; without one nothing is pinned and the window is one segment."""
    pinned = [frozenset[int]() if trust is None else trust.pinned(t) for t in range(T)]
    segments = [(0, T)] if trust is None else trust.segments(T)
    return pinned, segments, [j for j, (a, b) in enumerate(segments) for _ in range(a, b)]


def _snapshot_cache(
    trust: Optional[TrustSchedule], goal: Goal
) -> Optional[dict[tuple[int, bytes], tuple[bool, Optional[_Snapshot]]]]:
    """An empty cache of snapshot results under a trusted schedule and a flow goal, else None: a flow
    goal's snapshot does not depend on the one before (no stealth bound), so its result is a function of
    its free buses alone, and the per-slot search repeats every segment but the one it changes."""
    return {} if trust is not None and goal.kind == "flow" else None


def _channel_count(union: _Channels) -> int:
    """How many channels (node, flow, PMU current) the union holds."""
    return sum(int(m.sum()) for m in union if m is not None)


def _plan_size(plan: _Plan) -> int:
    """How many buses the plan's supports hold together (a held support: its size)."""
    return len({int(b) for S in plan for b in S})


def _union(a: _Channels, b: _Channels) -> _Channels:
    """The channels tampered in either of a and b (a PMU current part is None without currents)."""
    current = b[2] if a[2] is None else (a[2] if b[2] is None else a[2] | b[2])
    return a[0] | b[0], a[1] | b[1], current
