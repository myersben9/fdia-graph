"""The cheapest support of an attack window [WU26 eq. 12].

[WU26] minimizes the tampered measurements, a change under a meter's noise not counted, subject to the
AC model and the operating limits. Here every tampered reading comes from a local false state, so the
support S (the buses whose voltages the false state frees) sets which meters move. The candidates are
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
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, cast

import numpy as np

from ...constants.channels import NodeChannel
from ...formulas.attacks import tampered_channels, tampered_devices, within_limits
from ...formulas.network import _row_block, complex_voltages
from ...formulas.noise import paper_current_sigma, paper_sigma
from ...models.choices import CostUnit, SupportMethod
from ...models.config import TrustSchedule
from ...models.frames import AttackVector, FlowGoal, FrameKnobs, LoadGoal, MinimizerResult
from ...models.inputs import TrustablePmus
from ..base import Part

if TYPE_CHECKING:
    from threadpoolctl import ThreadpoolController

    from ..core import FdiaGenerator

# (devices, channels, support size): the objective, then its tie-breaks; with `FrameKnobs.objective`
# "channels" the first two swap (`_Window.rank`)
_Cost = tuple[int, int, int]
# the tampered node, flow and PMU current channels (masks)
_Channels = tuple[np.ndarray, np.ndarray, np.ndarray]
Goal = Union[LoadGoal, FlowGoal]  # what a window must realize: the loads of At, the flow of Am


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
        k: FrameKnobs,
        prev: Optional[AttackVector] = None,
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """The cheapest support found for the window of `states` (one [N, 4] true state per snapshot) and
        `goal`, or None when the goal's buses have no area. `prev` is the attack vector of the frame before
        the window (At's stealth bound starts from it); `trust` the trusted-PMU schedule [WU26 eqs. 26-32].
        With `k.support_method` "rref" a flow goal takes its support from `rref.RrefSupport` instead."""
        if trust is not None:  # refused before the search runs: every trusted bus carries a PMU
            TrustablePmus(trust.buses, frozenset(self.g.pmu_buses.tolist()), self.g.C)
        with _one_blas_thread():
            if k.support_method == SupportMethod.RREF.value and goal.kind == "flow":
                from .rref import RrefSupport

                return RrefSupport(self.g).rref_support(states, cast(FlowGoal, goal), k, prev, trust)
            return self._min_tamper(states, goal, k, prev, trust)

    def window_area(self, states: list[np.ndarray], goal: Goal, k: FrameKnobs) -> Optional[np.ndarray]:
        """The attacker's area for the window [WU26] Sec. III-A: `k.area` when the caller gives one,
        else `attack_area` around the goal's buses (`k.hops`, or the rules with `k.area_rule`)."""
        if k.area is not None:
            return self.g.area.explicit_area(np.asarray(k.area, np.int64))
        seeds, _, _ = self._goal_seeds(goal)
        lines = list(cast(FlowGoal, goal).lines) if goal.kind == "flow" else []
        return self.g.area.attack_area(seeds, k.hops, k.area_rule, (lines, np.stack(states)))

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
        window = _Window(self.g, states, goal, k, prev=prev, trust=trust)
        later = (S for S in self._supports(starts, area, must_hold) if not np.array_equal(S, area))
        best, evaluated = _cheapest(window, itertools.chain([area], later), area, k.min_budget)
        if best is None:  # no candidate meets every constraint: say so, keep the area
            return MinimizerResult(area, -1, -1, evaluated)
        devices, channels = window.counts(best[0])
        return MinimizerResult(best[1], devices, channels, evaluated)

    def _goal_seeds(self, goal: Goal) -> tuple[np.ndarray, list[frozenset[int]], Optional[frozenset[int]]]:
        """(the buses the area grows from, the candidates' starting sets, the buses a candidate must hold
        one of, or None). A load goal starts from its load buses. A flow goal acts through either end of
        each branch, so each choice of ends is a start, and a support must hold a free injection."""
        if goal.kind == "load":
            buses = np.unique(self.g.load_bus[cast(LoadGoal, goal).designs[0].targets])
            return buses, [frozenset(int(b) for b in buses)], None
        lines = list(cast(FlowGoal, goal).lines)
        ends = np.unique(self.g.ei[:, lines])
        ends_of = [self.g.ei[:, ln] for ln in lines]  # one end of each target line per choice
        choices = [frozenset(int(b) for b in combo) for combo in itertools.product(*ends_of)]
        starts = list(dict.fromkeys(choices))  # each set once, in a fixed order
        return ends, starts, frozenset(int(b) for b in self.g.false_state.free_injection_buses())

    def goal_state(
        self, goal: Goal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> Optional[np.ndarray]:
        """The false state of snapshot t on support S that meets the goal inside the operating limits, or
        None. Unknowns: |V| and theta of S, every other voltage held true. A load goal is the square local
        power flow; a flow goal frees S's loads and generators (`solve_flow_local`, [WU26 eqs. 13-25])."""
        if goal.kind == "flow":
            return self._flow_goal_state(cast(FlowGoal, goal), t, Xt, S, k)
        return self._load_goal_state(cast(LoadGoal, goal), t, Xt, S, k)

    def _flow_goal_state(
        self, goal: FlowGoal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> Optional[np.ndarray]:
        Xa, _, dload = self.g.false_state.solve_flow_local(
            Xt, S, goal.lines, goal.targets_at(t), k.limits, k.load_cap
        )
        if Xa is None:
            return None
        if k.limits is not None:  # every generator the attack moves, on S's edge too
            gen = self.g.split.generation(Xt)
            if not within_limits(Xa, Xt, gen, dload, k.limits, self.g.false_state.touched_buses(S)):
                return None
        return Xa

    def _load_goal_state(
        self, goal: LoadGoal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> Optional[np.ndarray]:
        design = goal.designs[t]
        Lp_true, Lq = self.g.scan_loads.true_load(Xt), self.g.scan_loads.true_reactive_load(Xt)
        Lp = Lp_true.copy()
        Lp[design.targets] *= design.mult
        Xa = self.g.false_state.solve_local(Xt, S, Lp, Lq)
        if Xa is None:
            return None
        if k.limits is not None and not self.g.false_state.state_within_limits(
            Xa, Xt, Lp - Lp_true, k.limits, S
        ):
            return None
        return Xa

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
            S = frozenset(key)
            if must_hold is None or S & must_hold:  # a flow goal's support needs a free injection
                yield np.array(key, dtype=np.int64)
            for v in sorted({n for b in S for n in adj[b]} - S):
                child = self.zero_closed(S | {v})
                if child <= allowed:
                    _push(heap, seen, child)

    def _area_adjacency(self, allowed: set[int]) -> dict[int, set[int]]:
        """The branches between buses of the area, as neighbour sets."""
        adj: dict[int, set[int]] = {b: set() for b in allowed}
        for a, b in self.g.ei.T:
            if int(a) in allowed and int(b) in allowed:
                adj[int(a)].add(int(b))
                adj[int(b)].add(int(a))
        return adj

    def zero_closed(self, S: frozenset[int]) -> frozenset[int]:
        """S with every zero-injection bus of its boundary taken in, as the area is grown."""
        grown, _ = self.g.area.grow_over_zero_injection(np.array(sorted(S), dtype=np.int64))
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
    from buses, and whether each is metered."""

    lines: np.ndarray
    yf: tuple[np.ndarray, np.ndarray]
    yt: tuple[np.ndarray, np.ndarray]
    from_bus: np.ndarray
    metered: np.ndarray


def _near_branches(g: FdiaGenerator, S: np.ndarray) -> _Near:
    """`_Near` of support S: the branches with an end in S."""
    lines = np.flatnonzero(np.isin(g.ei, S).any(axis=0))
    yf, yt = _row_block(g.net.yf, lines), _row_block(g.net.yt, lines)
    return _Near(lines, yf, yt, g.case.f[lines], g.flow_metered[lines])


class _Window:
    """One attack window evaluated for candidate supports: the true states, their noise thresholds and
    meter masks computed once, the goal's design per snapshot."""

    def __init__(
        self,
        g: FdiaGenerator,
        states: list[np.ndarray],
        goal: Goal,
        k: FrameKnobs,
        stealth_bound: bool = True,
        prev: Optional[AttackVector] = None,
        trust: Optional[TrustSchedule] = None,
    ) -> None:
        # The between-snapshot stealth bound is At's alone (a sub-noise ramp is what At is). [WU26]'s
        # model has no increment constraint: its noise only decides which changes the l0 count
        # ignores, so a flow goal (Am) is never bounded. `stealth_bound` off is for analysis.
        self.g, self.states, self.goal, self.k = g, states, goal, k
        self.stealth_bound = stealth_bound and goal.kind == "load"
        self.channels_first = k.objective == CostUnit.CHANNELS.value  # what the cost minimizes first
        # the buses a trusted PMU pins at each snapshot [WU26 eqs. 29-31]
        self.pinned = [frozenset[int]() if trust is None else trust.pinned(t) for t in range(len(states))]
        self._near: Optional[tuple[bytes, _Near]] = None  # the last support's branches (`near`)
        self.node_m, self.edge_m = g.masks.node, g.masks.edge
        self.pmu = np.zeros(g.C, bool)
        self.pmu[g.pmu_buses] = True
        # a bus angle is a PMU channel: a SCADA voltmeter reads |V| only
        self.node_m = self.node_m.copy()
        self.node_m[~self.pmu, NodeChannel.THETA] = 0
        self.flows = g.emitter.clean_flows_from_states(np.stack(states))  # [T, E, 2], unmetered zeroed
        # the PMU branch-current channels [WU26 eqs. 19-20]: their mask, true values and noise scale
        self.i_m: np.ndarray = g.masks.current
        self.currents: np.ndarray = g.emitter.currents_from_states(np.stack(states))
        self.sigma, self.i_sigma = self._sigmas(g, states, goal)
        self.zero = np.array(sorted({int(b) for b in g.zero_inj} - {g.slack_bus}), dtype=np.int64)
        self.prev = self._before(prev)

    def _sigmas(
        self, g: FdiaGenerator, states: list[np.ndarray], goal: Goal
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
        """What a change must exceed to count as tampering at each snapshot, and for At also the most a
        channel may move between snapshots, as (the node and flow layers, the PMU current layer) per
        snapshot: [WU26]'s own noise for the overload attack (0.01 pu on a current), the l0 threshold
        only; the meters' rated accuracy at the true readings for At (`NoiseModel.sigmas`)."""
        if goal.kind == "flow":
            paper = paper_sigma(self.node_m.shape, self.edge_m.shape, self.pmu, g.case.base_mva)
            return [paper for _ in states], [paper_current_sigma((g.E, 4)) for _ in states]
        exact = (np.asarray(X, np.float64) for X in states)
        true = (g.plan.gather(X, F, I) for X, F, I in zip(exact, self.flows, self.currents))
        rated = [g.emitter.layers(g.noise.sigmas(z)) for z in true]
        return [(s.node, s.edge) for s in rated], [s.current for s in rated]

    def _before(self, prev: Optional[AttackVector]) -> AttackVector:
        """The attack vector of the frame before the window: zero when it is benign."""
        if prev is None:
            shapes = (self.node_m.shape, self.edge_m.shape, self.i_m.shape)
            return AttackVector(*(np.zeros(shape) for shape in shapes))
        return AttackVector(*(np.asarray(part, float) for part in prev))

    def cost(self, S: np.ndarray, beat: Optional[_Cost]) -> Optional[_Cost]:
        """The cost of support S held for the window: the union of the tampered devices and channels over
        every snapshot [WU26 eq. 28]. None when S is infeasible at a snapshot, moves no device beyond its
        noise over the window (not an attack), or cannot beat `beat`."""
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
        shapes = (self.node_m.shape, self.edge_m.shape, self.i_m.shape)
        union: _Channels = (np.zeros(shapes[0], bool), np.zeros(shapes[1], bool), np.zeros(shapes[2], bool))
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
        at its end)."""
        ei = self.g.ei
        return len(tampered_devices(union[0], union[1], self.pmu, ei[0], union[2], ei[1]))

    def _zero_on_boundary(self, S: np.ndarray) -> bool:
        """Whether a zero-injection bus sits on S's boundary (it could not absorb the changed power)."""
        return bool(np.isin(self.g.area.boundary(S)[1], self.zero).any())

    def _snapshot(
        self, t: int, S: np.ndarray, prev: AttackVector
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, AttackVector]]:
        """Snapshot t on support S: (the node, flow and PMU current channels moved beyond their noise,
        and this snapshot's attack vector), or None when S has no false state here or (At) breaks the
        stealth bound against the previous snapshot's attack vector `prev`."""
        Xa = self.g.minimizer.goal_state(self.goal, t, self.states[t], S, self.k)
        if Xa is None:
            return None
        a_node = np.asarray(Xa, float) - np.asarray(self.states[t], float)
        a_edge, a_cur = self._branch_attack(t, Xa, self.near(S))
        sig_node, sig_edge = self.sigma[t]
        if self.stealth_bound:  # no metered channel moves more than its rated accuracy per step
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
        """`_Near` of support S, cached for the last support."""
        key = np.asarray(S, np.int64).tobytes()
        if self._near is None or self._near[0] != key:
            self._near = (key, _near_branches(self.g, S))
        return self._near[1]

    def _branch_attack(self, t: int, Xa: np.ndarray, near: _Near) -> tuple[np.ndarray, np.ndarray]:
        """The attack vector of false state Xa at snapshot t on the flow channels [E, 2] and the PMU
        branch currents [E, 4]. Only a branch with an end in the support sees a
        false voltage, so only those (`near`) are evaluated; every other entry is zero."""
        V = complex_voltages(Xa[:, NodeChannel.V], Xa[:, NodeChannel.THETA])
        (Yf, f_cols), (Yt, t_cols) = near.yf, near.yt
        If = Yf @ V[f_cols]
        Sf = V[near.from_bus] * np.conj(If) * self.g.case.base_mva
        flows = np.stack([np.real(Sf), np.imag(Sf)], axis=1)
        # an unmetered flow is zero on both sides, as `clean_flows_from_states` has it
        flows[~near.metered] = 0.0
        a_edge = np.zeros(self.edge_m.shape)
        a_edge[near.lines] = flows - self.flows[t][near.lines]
        It = Yt @ V[t_cols]
        cur = np.stack([np.real(If), np.imag(If), np.real(It), np.imag(It)], axis=1) * self.i_m[near.lines]
        a_cur = np.zeros(self.i_m.shape)
        a_cur[near.lines] = cur - self.currents[t][near.lines]
        return a_edge, a_cur

    def _over(self, t: int, a_cur: np.ndarray, scale: float) -> np.ndarray:
        """The PMU branch-current channels an attack-vector part moves beyond `scale` times their noise
        at snapshot t, metered ones only."""
        return (np.abs(a_cur) > scale * self.i_sigma[t]) & (self.i_m > 0)

    def _current_step(self, t: int, a_cur: np.ndarray, prev: AttackVector, scale: float) -> bool:
        """Whether the attack's step on a PMU branch-current channel breaks the stealth bound."""
        return bool(self._over(t, a_cur - prev.current, scale).any())


def _channel_count(union: _Channels) -> int:
    """How many channels (node, flow, PMU current) the union holds."""
    return sum(int(m.sum()) for m in union)


def _union(a: _Channels, b: _Channels) -> _Channels:
    """The channels tampered in either of a and b (node, flow, PMU current)."""
    return a[0] | b[0], a[1] | b[1], a[2] | b[2]
