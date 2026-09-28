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
meets the goal there on its noiseless readings h(x^a), stays inside the operating limits, and moves
no metered channel by more than its accuracy-class sigma from one snapshot to the next (the
stealth bound, the
first snapshot measured from no attack). Its cost is the number
of devices (`formulas.attacks.tampered_devices`) with a channel moved beyond its accuracy-class
sigma (`formulas.noise.accuracy_sigma`) at some
snapshot of the window, the union [WU26] counts over its window.

The region itself is solved first: the episode was accepted on it, so when the region meets the
stealth bound a feasible support exists and bounds the rest. The other candidates follow in order
of size; the search stops when every candidate is solved (the optimum over supports held for the
window), when the best cost reaches the devices the goal forces above noise whatever the support
(the only bound pruned with, the SCADA terminals of the targeted buses at a snapshot whose step
exceeds their noise), or at `k.min_budget` solved candidates (the best found, no claim), and records
which (`MinimizerResult.proven`). A candidate is dropped early only when its own exact union so far
already exceeds the best. When no held support meets every constraint, the result says so
(`devices = -1`) and the episode runs on the region as without the search.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from typing import Optional

import numpy as np

from ...formulas.attacks import tampered_channels, tampered_devices
from ...formulas.noise import accuracy_sigma
from ...models.frames import FrameKnobs, LoadGoal, MinimizerResult
from ...models.grid import NODE
from ..base import POWER_NOISE_FLOOR_MW
from .false_state import FalseStateMixin

_Cost = tuple[int, int, int]  # (devices, channels, support size): the objective, then its tie-breaks
_Best = tuple[_Cost, np.ndarray]  # a solved support and its cost


class MinimizeMixin(FalseStateMixin):
    """Find the support of an attack window that tampers the fewest devices [WU26, eq. 12]."""

    def min_tamper(
        self, states: list[np.ndarray], goal: LoadGoal, k: FrameKnobs
    ) -> Optional[MinimizerResult]:
        """The fewest-tamper support for the window of `states` (one [N, 4] true state per snapshot)
        and `goal` (one attack design per snapshot), or None when the goal's buses have no area."""
        goal_buses = np.unique(self.load_bus[goal.designs[0].targets])
        area = self.local_region(goal_buses, k.hops)
        if area is None:
            return None
        window = _Window(self, states, goal, k)
        region = window.cost(np.asarray(area), None)
        best = None if region is None else (region, np.asarray(area))
        lower = window.lower_bound()
        best, evaluated, exhausted = self._search(
            window, self._supports(goal_buses, area), (best, lower), area, k
        )
        if best is None:  # no support held for the window meets every constraint: say so, keep the region
            return MinimizerResult(np.asarray(area), -1, -1, False, evaluated, lower)
        (devices, channels, _), support = best
        return MinimizerResult(support, devices, channels, exhausted or devices <= lower, evaluated, lower)

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
            cost = window.cost(S, best[0] if best is not None else None)
            if cost is not None and (best is None or cost < best[0]):
                best = (cost, S)
        return best, evaluated, True

    def goal_state(
        self, goal: LoadGoal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> Optional[np.ndarray]:
        """The false state of snapshot t on support S that meets the goal there (unknowns: |V| and theta of
        S, every other voltage held true), inside the operating limits, or None. Each goal kind has its
        own solve: a load goal is the square local power flow of `stealthy_state` (the targeted loads
        take their new values, every other bus of S keeps its true injection)."""
        solve = {"load": self._load_goal_state}[goal.kind]
        return solve(goal, t, Xt, S, k)

    def _load_goal_state(
        self, goal: LoadGoal, t: int, Xt: np.ndarray, S: np.ndarray, k: FrameKnobs
    ) -> Optional[np.ndarray]:
        return self.stealthy_state(Xt, goal.designs[t]._replace(interior=S), k)

    def _supports(self, goal_buses: np.ndarray, area: np.ndarray) -> Iterator[np.ndarray]:
        """Every candidate support in the area, smallest first: the goal's buses, grown one adjacent area
        bus at a time (so every added bus joins a goal bus through the support), each closed over the
        zero-injection buses of its boundary exactly as `local_region` grows the region (a bus known to
        inject nothing cannot absorb the change, so it moves with the support), each set once. The
        slack is never in a support: the area excludes it."""
        allowed = {int(b) for b in area}
        adj = self._area_adjacency(allowed)
        start = self._zero_closed(frozenset(int(b) for b in goal_buses))
        heap: list[tuple[int, tuple[int, ...]]] = [(len(start), tuple(sorted(start)))]
        seen = {start}
        while heap:
            _, key = heapq.heappop(heap)
            S = frozenset(key)
            yield np.array(key, dtype=np.int64)
            for v in sorted({n for b in S for n in adj[b]} - S):
                child = self._zero_closed(S | {v})
                if child not in seen and child <= allowed:
                    seen.add(child)
                    heapq.heappush(heap, (len(child), tuple(sorted(child))))

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


class _Window:
    """One attack window evaluated for candidate supports: the true states, their accuracy sigmas and
    meter masks computed once, the goal's design per snapshot."""

    def __init__(
        self,
        g: MinimizeMixin,
        states: list[np.ndarray],
        goal: LoadGoal,
        k: FrameKnobs,
        stealth_bound: bool = True,
    ) -> None:
        # `stealth_bound` off only for analysis (the generator always applies it)
        self.g, self.states, self.goal, self.k, self.stealth_bound = g, states, goal, k, stealth_bound
        self.node_m, self.edge_m = g.meter_masks()
        self.pmu = np.zeros(g.C, bool)
        self.pmu[sorted(g.meters.pmu)] = True
        flows = g.clean_flows_from_states(np.stack(states))  # [T, E, 2], unmetered zeroed
        # the meters' rated accuracy: what a change must exceed to count, and the most a channel may move
        # between snapshots (the emitter's per-scan jitter is smaller, and is not a detection threshold)
        self.sigma = [accuracy_sigma(X, F, g.SD, POWER_NOISE_FLOOR_MW) for X, F in zip(states, flows)]
        zero = {int(b) for b in g.zero_inj} - {g.slack_bus}
        self.zero = np.array(sorted(zero), dtype=np.int64)

    def lower_bound(self) -> int:
        """Devices every support tampers, the only bound the search prunes with: the SCADA terminal of a
        target bus whose injected active power the goal fixes (the load the attacker pretends) and whose
        designed step exceeds that injection meter's accuracy-class sigma at some snapshot. The goal
        leaves a bus's reactive load unchanged, so only the active channel is forced."""
        g, C = self.g, self.g.C
        power = np.zeros(C, bool)
        for t, design in enumerate(self.goal.designs):
            Lp = g.true_load(self.states[t])
            dload = np.zeros(C)
            np.add.at(dload, g.load_bus[design.targets], Lp[design.targets] * (np.asarray(design.mult) - 1.0))
            sig_node, _ = self.sigma[t]
            power |= (np.abs(dload) > sig_node[:, NODE.p_inj]) & (self.node_m[:, NODE.p_inj] > 0)
        node = np.zeros((C, 4), bool)
        node[:, NODE.p_inj] = power
        return len(tampered_devices(node, np.zeros(self.edge_m.shape, bool), self.pmu, g.ei[0]))

    def cost(self, S: np.ndarray, beat: Optional[_Cost]) -> Optional[_Cost]:
        """The cost of support S over the window, or None when S is infeasible at a snapshot or cannot
        beat `beat` (stopped as soon as its devices so far exceed it)."""
        g = self.g
        boundary = np.setdiff1d(np.unique(g._live_edges()[:, np.isin(g._live_edges(), S).any(axis=0)]), S)
        if np.isin(boundary, self.zero).any():
            return None  # never for a candidate of `_supports`, which takes such a bus in; a guard for others
        devices: set[int] = set()
        channels: set[tuple[int, int, int]] = set()
        prev = (np.zeros(self.node_m.shape), np.zeros(self.edge_m.shape))  # before the window: no attack
        for t in range(len(self.goal.designs)):
            moved = self._snapshot(t, S, prev)
            if moved is None:
                return None
            node, edge, prev = moved
            devices |= set(tampered_devices(node, edge, self.pmu, g.ei[0]).tolist())
            channels |= {(0, int(i), int(j)) for i, j in zip(*np.nonzero(node))}
            channels |= {(1, int(i), int(j)) for i, j in zip(*np.nonzero(edge))}
            if beat is not None and len(devices) > beat[0]:
                return None
        cost = (len(devices), len(channels), len(S))
        return cost if beat is None or cost < beat else None

    def _snapshot(
        self, t: int, S: np.ndarray, prev: tuple[np.ndarray, np.ndarray]
    ) -> Optional[tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]]:
        """Snapshot t on support S: (the channels moved beyond their accuracy-class sigma, node and flow,
        and this snapshot's attack vector), or None when S has no false state here or breaks the stealth
        bound against the previous snapshot's attack vector `prev`."""
        g = self.g
        Xa = g.goal_state(self.goal, t, self.states[t], S, self.k)
        if Xa is None:
            return None
        a_node, a_edge = g._attack_vector(Xa, self.states[t])
        sig_node, sig_edge = self.sigma[t]
        # the stealth bound: no metered channel moves more than its accuracy-class sigma between snapshots
        step = tampered_channels(
            a_node - prev[0], a_edge - prev[1], sig_node, sig_edge, self.node_m, self.edge_m
        )
        if self.stealth_bound and (step[0].any() or step[1].any()):
            return None
        node, edge = tampered_channels(a_node, a_edge, sig_node, sig_edge, self.node_m, self.edge_m)
        return node, edge, (a_node, a_edge)
