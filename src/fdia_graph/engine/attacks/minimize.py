"""The cheapest support of an attack window [WU26 eq. 12].

[WU26] minimizes the tampered measurements, a change under a meter's noise not counted, subject to the
AC model and the operating limits. Here every tampered reading comes from a local false state, so the
support (the buses whose voltages the false state frees) sets which meters move. The candidates are
the area (`window_area`), then the goal's buses grown one adjacent area bus at a time, smallest first,
at most `FrameKnobs.min_budget`. Each is solved at every snapshot and its tampered devices and channels
counted over the window [WU26 eq. 28]. The cheapest found wins; nothing is claimed about optimality.
A trusted PMU (`TrustSchedule`, [WU26 eqs. 27, 29]) holds its bus true from its slot on. A support
that moves no device beyond noise is not an attack and is skipped."""

from __future__ import annotations

import contextlib
import heapq
import itertools
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, Optional, cast

import numpy as np

from ...models.choices import SupportMethod
from ...models.config import TrustSchedule
from ...models.frames import AttackVector, FlowGoal, FrameKnobs, Goal, LoadGoal, MinimizerResult, SupportCost
from ...models.inputs import TrustablePmus
from ..base import Part
from .window import AttackWindow

if TYPE_CHECKING:
    from threadpoolctl import ThreadpoolController


_blas_pools: list[Optional[ThreadpoolController]] = []  # the process's BLAS pools, found on the first search
# BLAS limits are process-wide, so concurrent searches share one limit: the first to enter sets it,
# the last to leave restores the caller's setting
_blas_lock = threading.Lock()
_blas_held: list[contextlib.AbstractContextManager[object]] = []  # the active limit, while any search runs
_blas_users = [0]  # searches currently inside the limit


@contextlib.contextmanager
def _one_blas_thread() -> Iterator[None]:
    """Hold the BLAS pools to one thread for a search, restored after the last concurrent search leaves:
    the search's products are small, so starting BLAS threads costs more than the arithmetic. Without
    threadpoolctl (the `generate` extra installs it) nothing is changed."""
    with _blas_lock:
        if not _blas_pools:
            try:
                from threadpoolctl import ThreadpoolController

                _blas_pools.append(ThreadpoolController())
            except ImportError:
                _blas_pools.append(None)
        controller = _blas_pools[0]
        if controller is not None and _blas_users[0] == 0:
            held = controller.limit(limits=1, user_api="blas")
            held.__enter__()
            _blas_held.append(held)
        _blas_users[0] += 1
    try:
        yield
    finally:
        with _blas_lock:
            _blas_users[0] -= 1
            if _blas_users[0] == 0 and _blas_held:
                _blas_held.pop().__exit__(None, None, None)


class Minimizer(Part):
    """Find the support of an attack window that tampers the fewest devices [WU26 eq. 12]."""

    def min_tamper(
        self,
        states: list[np.ndarray],
        goal: Goal,
        knobs: FrameKnobs,
        prev: Optional[AttackVector] = None,
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """The cheapest support found for the window of `states` (one [N, 4] true state per snapshot) and
        `goal`, or None when the goal's buses have no area. `prev` is the attack vector of the frame before
        the window (At's stealth bound starts from it); `trust` the trusted-PMU schedule [WU26 eqs. 26-32].
        With `knobs.support_method` "rref" a flow goal takes its support from `rref.RrefSupport` instead."""
        if trust is not None:  # refused before the search runs: every trusted bus carries a PMU
            TrustablePmus(trust.buses, frozenset(self.g.pmu_buses.tolist()), self.g.n_bus)
        with _one_blas_thread():
            if knobs.support_method == SupportMethod.RREF.value and goal.kind == "flow":
                from .rref import RrefSupport

                return RrefSupport(self.g).rref_support(states, cast(FlowGoal, goal), knobs, prev, trust)
            return self._min_tamper(states, goal, knobs, prev, trust)

    def window_area(self, states: list[np.ndarray], goal: Goal, knobs: FrameKnobs) -> Optional[np.ndarray]:
        """The attacker's area for the window [WU26] Sec. III-A: `knobs.area` when the caller gives one,
        else `attack_area` around the goal's buses (`knobs.hops`, or the rules with `knobs.area_rule`)."""
        if knobs.area is not None:
            return self.g.area.explicit_area(np.asarray(knobs.area, np.int64))
        seeds, _, _ = self._goal_seeds(goal)
        lines = list(cast(FlowGoal, goal).lines) if goal.kind == "flow" else []
        return self.g.area.attack_area(seeds, knobs.hops, knobs.area_rule, (lines, np.stack(states)))

    def _min_tamper(
        self,
        states: list[np.ndarray],
        goal: Goal,
        knobs: FrameKnobs,
        prev: Optional[AttackVector],
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """`min_tamper`'s candidate loop (module docstring), on whatever BLAS threads the caller set."""
        area = self.window_area(states, goal, knobs)
        if area is None:
            return None
        _, starts, must_hold = self._goal_seeds(goal)
        starts = _starts_in_area(starts, area, must_hold)
        window = AttackWindow(self.g, states, goal, knobs, prev=prev, trust=trust)
        later = (
            support
            for support in self._supports(starts, area, must_hold)
            if not np.array_equal(support, area)
        )
        best, evaluated = _cheapest(window, itertools.chain([area], later), area, knobs.min_budget)
        if best is None:  # no candidate meets every constraint: say so, keep the area
            return MinimizerResult(area, -1, -1, evaluated)
        cost, support = best
        return MinimizerResult(support, cost.devices, cost.channels, evaluated)

    def _goal_seeds(self, goal: Goal) -> tuple[np.ndarray, list[frozenset[int]], Optional[frozenset[int]]]:
        """(the buses the area grows from, the candidates' starting sets, the buses a candidate must hold
        one of, or None). A load goal starts from its load buses. A flow goal acts through either end of
        each branch, so each choice of ends is a start, and a support must hold a free injection."""
        if goal.kind == "load":
            buses = np.unique(self.g.load_bus[cast(LoadGoal, goal).designs[0].targets])
            return buses, [frozenset(int(b) for b in buses)], None
        lines = list(cast(FlowGoal, goal).lines)
        ends = np.unique(self.g.edge_index[:, lines])
        ends_of = [self.g.edge_index[:, ln] for ln in lines]  # one end of each target line per choice
        choices = [frozenset(int(b) for b in combo) for combo in itertools.product(*ends_of)]
        starts = list(dict.fromkeys(choices))  # each set once, in a fixed order
        return ends, starts, frozenset(int(b) for b in self.g.false_state.free_injection_buses())

    def _supports(
        self, starts: list[frozenset[int]], area: np.ndarray, must_hold: Optional[frozenset[int]] = None
    ) -> Iterator[np.ndarray]:
        """Every candidate support in the area, smallest first, each once: the goal's buses grown one
        adjacent area bus at a time, each closed over the zero-injection buses of its boundary (such a
        bus cannot absorb the change, so it moves with the support)."""
        allowed = {int(b) for b in area}
        adj = self._area_adjacency(allowed)
        heap: list[tuple[int, tuple[int, ...]]] = []
        seen: set[frozenset[int]] = set()
        for s in starts:
            _push(heap, seen, self.zero_closed(s))
        while heap:
            _, key = heapq.heappop(heap)
            support = frozenset(key)
            if must_hold is None or support & must_hold:  # a flow goal's support needs a free injection
                yield np.array(key, dtype=np.int64)
            for v in sorted({n for b in support for n in adj[b]} - support):
                child = self.zero_closed(support | {v})
                if child <= allowed:
                    _push(heap, seen, child)

    def _area_adjacency(self, allowed: set[int]) -> dict[int, set[int]]:
        """The branches between buses of the area, as neighbour sets."""
        adj: dict[int, set[int]] = {b: set() for b in allowed}
        for a, b in self.g.edge_index.T:
            if int(a) in allowed and int(b) in allowed:
                adj[int(a)].add(int(b))
                adj[int(b)].add(int(a))
        return adj

    def zero_closed(self, support: frozenset[int]) -> frozenset[int]:
        """The support with every zero-injection bus of its boundary taken in, as the area is grown."""
        grown, _ = self.g.area.grow_over_zero_injection(np.array(sorted(support), dtype=np.int64))
        return frozenset(int(b) for b in grown)


def _cheapest(
    window: AttackWindow, candidates: Iterator[np.ndarray], area: np.ndarray, budget: int
) -> tuple[Optional[tuple[SupportCost, np.ndarray]], int]:
    """(The cheapest of at most `budget` candidates with its cost, or None; the candidates solved). It
    stops early at one device on one channel, the least an attack tampers: the candidates after the area
    come smallest first, so none later beats it."""
    best: Optional[tuple[SupportCost, np.ndarray]] = None
    evaluated = 0
    for support in itertools.islice(candidates, budget):
        evaluated += 1
        # None unless the support beats the best
        cost = window.cost(support, None if best is None else best[0])
        best = best if cost is None else (cost, support)
        if best is not None and (best[0].devices, best[0].channels) == (1, 1) and best[1] is not area:
            break
    return best, evaluated


def _push(
    heap: list[tuple[int, tuple[int, ...]]], seen: set[frozenset[int]], support: frozenset[int]
) -> None:
    """Queue a support by size, once."""
    if support not in seen:
        seen.add(support)
        heapq.heappush(heap, (len(support), tuple(sorted(support))))


def _starts_in_area(
    starts: list[frozenset[int]], area: np.ndarray, must_hold: Optional[frozenset[int]]
) -> list[frozenset[int]]:
    """A flow goal starts only from its branch ends inside the area (an end at the slack is not one);
    a load goal's starting set is kept as it is."""
    if must_hold is None:
        return starts
    inside = {int(b) for b in area}
    return [s for s in starts if s <= inside]
