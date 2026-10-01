"""The cheapest support of an attack window [WU26 eq. 12].

[WU26] builds its attack as a minimization: the fewest tampered measurements, a change under a meter's
noise not counted [D8], subject to the AC measurement model and the operating limits. Every tampered
reading here is a reading of a local false state, so the model holds by construction and which meters
move is set by the support S, the buses whose voltages the false state frees. The search is a plain
candidate loop over supports:

1. the area: [WU26]'s own when the caller gives it (`FrameKnobs.area`), the region its Section III-A
   principles pick (`FrameKnobs.area_rule` "rules"), or the buses within `hops` of the goal;
2. the candidates: the area itself, then the goal's buses grown one adjacent area bus at a time,
   smallest first (`_supports`), at most `FrameKnobs.min_budget` of them;
3. each candidate: the false state of every snapshot solved on the measurements
   (`FalseStateMixin.solve_flow_local` for a flow goal, `solve_local` for a load goal), its tampered
   devices and channels counted over the window [WU26 eq. 28];
4. the answer: the cheapest candidate found. Nothing is claimed about optimality; [WU26] proves none.

A trusted PMU (`TrustSchedule`, [WU26 eqs. 27, 29]) holds its own bus's |V| and angle true from its slot
on, so the bus drops out of the support from then. A support that moves no device beyond noise at any
snapshot is not an attack (an accurate estimator resolves it to the noise floor) and is skipped.
"""

from __future__ import annotations

import contextlib
import heapq
import itertools
import threading
from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple, Optional, Protocol, Union, cast

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
from ...models.choices import CostUnit, SupportMethod
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
# the tampered node, flow and PMU current channels (masks; no currents without them in the plan)
_Channels = tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
Goal = Union[LoadGoal, FlowGoal]  # what a window must realize: the loads of At, the flow of Am


_BLAS: list[Optional[ThreadpoolController]] = []  # the process's BLAS pools, found on the first search
# BLAS limits are process-wide, so concurrent searches share one limit: the first to enter sets it,
# the last to leave restores the caller's setting
_BLAS_LOCK = threading.Lock()
_BLAS_HELD: list[contextlib.AbstractContextManager[object]] = []  # the active limit, while any search runs
_BLAS_USERS = [0]  # searches currently inside the limit


@contextlib.contextmanager
def _one_blas_thread() -> Iterator[None]:
    """Hold the BLAS pools to one thread for a search, restoring them after the last concurrent search
    leaves: the search's products are small, so on a many-core machine starting BLAS threads costs
    more than the arithmetic. Without threadpoolctl (the `generate` extra installs it) the search runs
    on whatever BLAS is set to."""
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


class SupportStrategy(Protocol):
    """How an attack window's support is chosen (decision B4: a strategy held by the generator): the
    candidate search (`SearchSupport`) or [WU26]'s row reduction (`rref.RrefSupport`). `find` returns
    the support and its counts, None when the goal has no area."""

    def find(
        self,
        states: list[np.ndarray],
        goal: Goal,
        k: FrameKnobs,
        prev: Optional[AttackVector],
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]: ...


class SearchSupport:
    """The candidate search [WU26 eq. 12] as a support strategy (`MinimizeMixin._min_tamper`)."""

    def __init__(self, g: MinimizeMixin) -> None:
        self.g = g

    def find(
        self,
        states: list[np.ndarray],
        goal: Goal,
        k: FrameKnobs,
        prev: Optional[AttackVector],
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        return self.g._min_tamper(states, goal, k, prev, trust)


class MinimizeMixin(FalseStateMixin):
    """Find the support of an attack window that tampers the fewest devices [WU26 eq. 12]."""

    def min_tamper(
        self,
        states: list[np.ndarray],
        goal: Goal,
        k: FrameKnobs,
        prev: Optional[AttackVector] = None,
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """The cheapest support found for the window of `states` (one [N, 4] true state per snapshot)
        and `goal` (one attack design per snapshot), or None when the goal's buses have no area. `prev`
        is the attack vector of the frame before the window (None when that frame is benign): At's
        stealth bound measures its first step from it [D7]. `trust` is the defender's trusted-PMU
        schedule [WU26 eqs. 26-32]. The search runs with BLAS on one thread (`_one_blas_thread`)."""
        if trust is not None:  # refused before the search runs: every trusted bus carries a PMU
            TrustablePmus(trust.buses, frozenset(self.meters.pmu), self.C)
        with _one_blas_thread():
            return self.support_strategy(goal, k).find(states, goal, k, prev, trust)

    def support_strategy(self, goal: Optional[Goal], k: Optional[FrameKnobs]) -> SupportStrategy:
        """[WU26]'s row reduction (`rref.RrefSupport`) for a flow goal when `k.support_method` asks for
        it, else the candidate search (`SearchSupport`)."""
        rref = getattr(k, "support_method", None) == SupportMethod.RREF.value
        if rref and getattr(goal, "kind", None) == "flow":
            from .rref import RrefSupport

            return RrefSupport(self)
        return SearchSupport(self)

    def window_area(self, states: list[np.ndarray], goal: Goal, k: FrameKnobs) -> Optional[np.ndarray]:
        """The attacker's area for the window [WU26] Sec. III-A: `k.area` when given (the paper's own),
        else `attack_area` around the goal's buses (`k.hops`, or the rules with `k.area_rule`)."""
        if k.area is not None:
            return self.explicit_area(np.asarray(k.area, np.int64))
        seeds, _, _ = self._goal_seeds(goal)
        lines = list(cast(FlowGoal, goal).lines) if goal.kind == "flow" else []
        return self.attack_area(seeds, k.hops, k.area_rule, (lines, np.stack(states)))

    def _min_tamper(
        self,
        states: list[np.ndarray],
        goal: Goal,
        k: FrameKnobs,
        prev: Optional[AttackVector],
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """`min_tamper`'s candidate loop (module docstring), on whatever BLAS threads the caller set."""
        area = self.window_area(states, goal, k)
        if area is None:
            return None
        _, starts, must_hold = self._goal_seeds(goal)
        starts = _starts_in_area(starts, area, must_hold)
        window = _Window(self, states, goal, k, prev=prev, trust=trust)
        later = (S for S in self._supports(starts, area, must_hold) if not np.array_equal(S, area))
        best, evaluated = _cheapest(window, itertools.chain([area], later), area, k.min_budget)
        if best is None:  # no candidate meets every constraint: say so, keep the area
            return MinimizerResult(area, -1, -1, evaluated)
        devices, channels = window.counts(best[0])
        return MinimizerResult(best[1], devices, channels, evaluated)

    def _goal_seeds(self, goal: Goal) -> tuple[np.ndarray, list[frozenset[int]], Optional[frozenset[int]]]:
        """(the buses the area grows from, the candidate supports' starting sets, the buses a candidate
        must hold one of, or None). A load goal acts through its targeted load buses, all of them in
        every support. A flow goal acts on each of its branches through either end, so every choice of
        one end per branch starts its own candidates, and a support must hold a free injection (an
        attackable load or a generator, the injections the flow goal frees)."""
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
    ) -> Optional[np.ndarray]:
        """The false state of snapshot t on support S that meets the goal there inside the operating
        limits, or None. Unknowns: |V| and theta of S, every other voltage held true. A load goal is the
        square local power flow (the targeted loads take their new values, every other bus of S keeps
        its true injection); a flow goal frees the attackable loads and generators of S and holds the
        rest (`solve_flow_local`, [WU26 eqs. 13-25])."""
        if goal.kind == "flow":
            return self._flow_goal_state(cast(FlowGoal, goal), t, Xt, S, k)
        return self._load_goal_state(cast(LoadGoal, goal), t, Xt, S, k)

    def _flow_goal_state(
        self, goal: FlowGoal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> Optional[np.ndarray]:
        Xa, _, dload = self.solve_flow_local(Xt, S, goal.lines, goal.targets_at(t), k.limits, k.load_cap)
        if Xa is None:
            return None
        if k.limits is not None:  # every generator the attack moves, on S's edge too [D16]
            gen = generator_output(Xt, self.load_base, self.gen_base)
            if not within_limits(Xa, Xt, gen, dload, k.limits, self.touched_buses(S)):
                return None
        return Xa

    def _load_goal_state(
        self, goal: LoadGoal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> Optional[np.ndarray]:
        design = goal.designs[t]
        Lp_true, Lq = self.true_load(Xt), self.true_reactive_load(Xt)
        Lp = Lp_true.copy()
        Lp[design.targets] *= design.mult
        Xa = self.solve_local(Xt, S, Lp, Lq)
        if Xa is None:
            return None
        if k.limits is not None and not self._within_limits(Xa, Xt, Lp - Lp_true, k.limits, S):
            return None
        return Xa

    def _supports(
        self, starts: list[frozenset[int]], area: np.ndarray, must_hold: Optional[frozenset[int]] = None
    ) -> Iterator[np.ndarray]:
        """Every candidate support in the area, smallest first: the goal's buses grown one adjacent area
        bus at a time (so every added bus joins a goal bus through the support), each closed over the
        zero-injection buses of its boundary as the area is (a bus known to inject nothing cannot absorb
        the change, so it moves with the support), each set once."""
        allowed = {int(b) for b in area}
        adj = self._area_adjacency(allowed)
        heap: list[tuple[int, tuple[int, ...]]] = []
        seen: set[frozenset[int]] = set()
        for s in starts:
            _push(heap, seen, self._zero_closed(s))
        while heap:
            _, key = heapq.heappop(heap)
            S = frozenset(key)
            if must_hold is None or S & must_hold:  # a flow goal's support needs a free injection
                yield np.array(key, dtype=np.int64)
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
        """S with every zero-injection bus of its boundary taken in, as the area is grown."""
        grown, _ = self._grow_over_zero_injection(np.array(sorted(S), dtype=np.int64))
        return frozenset(int(b) for b in grown)


def _cheapest(
    window: _Window, candidates: Iterator[np.ndarray], area: np.ndarray, budget: int
) -> tuple[Optional[tuple[_Cost, np.ndarray]], int]:
    """(The cheapest of at most `budget` candidates with its cost, or None; the candidates solved). It
    stops early at one device on one channel, the least an attack tampers: the candidates after the area
    come smallest first, so none later beats it."""
    best: Optional[tuple[_Cost, np.ndarray]] = None
    evaluated = 0
    for S in itertools.islice(candidates, budget):
        evaluated += 1
        cost = window.cost(S, None if best is None else best[0])  # None unless S beats the best
        best = best if cost is None else (cost, S)
        if best is not None and best[0][:2] == (1, 1) and best[1] is not area:
            break
    return best, evaluated


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
    """One attack window evaluated for candidate supports: the true states, their noise thresholds and
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
        # model has no increment constraint: its noise only decides which changes the l0 count
        # ignores, so a flow goal (Am) is never bounded [D11]. `stealth_bound` off is for analysis.
        self.g, self.states, self.goal, self.k = g, states, goal, k
        self.stealth_bound = stealth_bound and goal.kind == "load"
        self.channels_first = k.objective == CostUnit.CHANNELS.value  # what the cost minimizes first
        # the buses a trusted PMU pins at each snapshot [WU26 eqs. 29-31]
        self.pinned = [frozenset[int]() if trust is None else trust.pinned(t) for t in range(len(states))]
        self._near: Optional[tuple[bytes, _Near]] = None  # the last support's branches (`near`)
        self.node_m, self.edge_m = g.meter_masks()
        self.pmu = np.zeros(g.C, bool)
        self.pmu[sorted(g.meters.pmu)] = True
        # a bus angle is a PMU channel: a SCADA voltmeter reads |V| only [D10]
        self.node_m = self.node_m.copy()
        self.node_m[~self.pmu, NODE.theta] = 0
        self.flows = g.clean_flows_from_states(np.stack(states))  # [T, E, 2], unmetered zeroed
        # the PMU branch-current channels [WU26 eqs. 19-20]: their mask and noise scale (None without)
        self.i_m = g.current_mask()
        self.currents, self.i_sigma = self._currents(g, states, goal)
        self.sigma = self._sigmas(g, states, goal)
        self.zero = np.array(sorted({int(b) for b in g.zero_inj} - {g.slack_bus}), dtype=np.int64)
        self.prev = self._before(prev)

    def _sigmas(
        self, g: MinimizeMixin, states: list[np.ndarray], goal: Goal
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """What a change must exceed to count as tampering at each snapshot, and for At also the most a
        channel may move between snapshots: [WU26]'s own noise for the overload attack, the l0 threshold
        only [D8] [D11]; the meters' rated accuracy for At [D7]."""
        if goal.kind == "flow":
            paper = paper_sigma(self.node_m.shape, self.edge_m.shape, self.pmu, g._base_mva)
            return [paper for _ in states]
        return [accuracy_sigma(X, F, g.SD, POWER_NOISE_FLOOR_MW) for X, F in zip(states, self.flows)]

    def _before(self, prev: Optional[AttackVector]) -> AttackVector:
        """The attack vector of the frame before the window: zero when it is benign; a bare (node, edge)
        pair is an attack vector without currents."""
        no_i = None if self.i_m is None else np.zeros(self.i_m.shape)
        if prev is None:
            return AttackVector(np.zeros(self.node_m.shape), np.zeros(self.edge_m.shape), no_i)
        p = AttackVector(*prev)
        cur = no_i if p.current is None else np.asarray(p.current, float)
        return AttackVector(np.asarray(p.node, float), np.asarray(p.edge, float), cur)

    @staticmethod
    def _currents(
        g: MinimizeMixin, states: list[np.ndarray], goal: Goal
    ) -> tuple[Optional[np.ndarray], Optional[list[np.ndarray]]]:
        """(The true PMU branch currents per snapshot [T, E, 4]; their noise scale per snapshot: [WU26]'s
        0.01 pu for the overload attack [D8], the PMU accuracy class at the true currents for At [D7]),
        both None without currents in the meter plan."""
        if g.current_mask() is None:
            return None, None
        true = g.currents_from_states(np.stack(states))
        if goal.kind == "flow":
            return true, [paper_current_sigma((g.E, 4)) for _ in true]
        return true, [current_sigma(i, PMU_CURRENT_CLASS) for i in true]

    def cost(self, S: np.ndarray, beat: Optional[_Cost]) -> Optional[_Cost]:
        """The cost of support S held for the window: the union of the tampered devices and channels over
        every snapshot [WU26 eq. 28]. None when S is infeasible at a snapshot, moves no device beyond its
        noise over the window (not an attack), or cannot beat `beat` (stopped as soon as its first count
        so far exceeds it)."""
        if self._zero_on_boundary(S):
            return None  # never for a candidate of `_supports`, which takes such a bus in
        tampered = self._tampered(S, beat)
        if tampered is None:
            return None
        union, devices = tampered
        if not devices:
            return None  # within noise at every snapshot: no effect, so not an attack
        cost = self.rank(devices, _channel_count(union), len(S))
        return cost if beat is None or cost < beat else None

    def support_at(self, t: int, S: np.ndarray) -> np.ndarray:
        """The buses of S free at snapshot t: S less the buses of the PMUs trusted by then ([WU26] eq. 29
        holds their deviation at zero, so they keep their true voltage)."""
        pinned = self.pinned[t]
        return S if not pinned else S[~np.isin(S, sorted(pinned))]

    def _tampered(self, S: np.ndarray, beat: Optional[_Cost]) -> Optional[tuple[_Channels, int]]:
        """The union of tampered channels (node, flow, PMU current masks) on S over the window and its
        device count, or None when S fails at a snapshot or its first count so far exceeds `beat`'s."""
        union: _Channels = (np.zeros(self.node_m.shape, bool), np.zeros(self.edge_m.shape, bool), None)
        devices, prev = 0, self.prev
        for t in range(len(self.states)):
            free = self.support_at(t, S)
            if not len(free) or (self.pinned[t] and self._zero_on_boundary(free)):
                return None
            moved = self._snapshot(t, free, prev)
            if moved is None:
                return None
            node, edge, current, prev = moved
            union = _union(union, (node, edge, current))
            devices = self._devices(union)
            first = _channel_count(union) if self.channels_first else devices
            if beat is not None and first > beat[0]:
                return None
        return union, devices

    def rank(self, devices: int, channels: int, size: int) -> _Cost:
        """A cost in the order the search compares it: the objective first (`FrameKnobs.objective`),
        the other count next, the support size last."""
        return (channels, devices, size) if self.channels_first else (devices, channels, size)

    def counts(self, cost: _Cost) -> tuple[int, int]:
        """(devices, channels) of a cost `rank` built."""
        return (cost[1], cost[0]) if self.channels_first else (cost[0], cost[1])

    def _devices(self, union: _Channels) -> int:
        """How many devices hold a channel of the union (a PMU branch current joins the PMU of the bus
        at its end) [D1]."""
        ei = self.g.ei
        return len(tampered_devices(union[0], union[1], self.pmu, ei[0], union[2], ei[1]))

    def _zero_on_boundary(self, S: np.ndarray) -> bool:
        """Whether a zero-injection bus sits on S's boundary (it could not absorb the changed power)."""
        return bool(np.isin(self.g._boundary(S)[1], self.zero).any())

    def _snapshot(
        self, t: int, S: np.ndarray, prev: AttackVector
    ) -> Optional[tuple[np.ndarray, np.ndarray, Optional[np.ndarray], AttackVector]]:
        """Snapshot t on support S: (the node, flow and PMU current channels moved beyond their noise,
        and this snapshot's attack vector), or None when S has no false state here or (At) breaks the
        stealth bound against the previous snapshot's attack vector `prev`."""
        Xa = self.g.goal_state(self.goal, t, self.states[t], S, self.k)
        if Xa is None:
            return None
        a_node = np.asarray(Xa, float) - np.asarray(self.states[t], float)
        a_edge, a_cur = self._branch_attack(t, Xa, self.near(S))
        sig_node, sig_edge = self.sigma[t]
        if self.stealth_bound:  # [D7]: no metered channel moves more than its rated accuracy per step
            scale = self.k.stealth_scale
            step = tampered_channels(
                a_node - prev.node,
                a_edge - prev.edge,
                scale * sig_node,
                scale * sig_edge,
                self.node_m,
                self.edge_m,
            )
            if step[0].any() or step[1].any() or self._current_step(t, a_cur, prev, scale):
                return None
        node, edge = tampered_channels(a_node, a_edge, sig_node, sig_edge, self.node_m, self.edge_m)
        return node, edge, self._over(t, a_cur, 1.0), AttackVector(a_node, a_edge, a_cur)

    def near(self, S: np.ndarray) -> _Near:
        """`_Near` of support S, kept for the last support (a candidate is solved at every snapshot of
        the window in turn)."""
        key = np.asarray(S, np.int64).tobytes()
        if self._near is None or self._near[0] != key:
            self._near = (key, _near_branches(self.g, S))
        return self._near[1]

    def _branch_attack(self, t: int, Xa: np.ndarray, near: _Near) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """The attack vector of false state Xa at snapshot t on the flow channels [E, 2] and the PMU
        branch currents [E, 4] (None without currents): only a branch with an end in the support sees a
        false voltage, so only the support's branches (`near`) are evaluated and every other branch's
        entry is zero."""
        g = self.g
        lut = g._ppc_row[np.arange(g.C)]
        V = np.zeros(g._n_ppc_buses, complex)
        V[lut] = complex_voltages(Xa[:, NODE.v], Xa[:, NODE.theta])
        (Yf, f_cols), (Yt, t_cols) = near.yf, near.yt
        If = Yf @ V[f_cols]
        Sf = V[near.from_ppc] * np.conj(If) * g._base_mva
        flows = np.stack([np.real(Sf), np.imag(Sf)], axis=1)
        flows[~near.metered] = (
            0.0  # an unmetered flow is zero on both sides, as `clean_flows_from_states` has it
        )
        a_edge = np.zeros(self.edge_m.shape)
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
        """Whether the attack's step on a PMU branch-current channel breaks the stealth bound [D7]."""
        if a_cur is None or prev.current is None:
            return False
        over = self._over(t, a_cur - prev.current, scale)
        return bool(over is not None and over.any())


def _channel_count(union: _Channels) -> int:
    """How many channels (node, flow, PMU current) the union holds."""
    return sum(int(m.sum()) for m in union if m is not None)


def _union(a: _Channels, b: _Channels) -> _Channels:
    """The channels tampered in either of a and b (a PMU current part is None without currents)."""
    current = b[2] if a[2] is None else (a[2] if b[2] is None else a[2] | b[2])
    return a[0] | b[0], a[1] | b[1], current
