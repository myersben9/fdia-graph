"""What one candidate support costs over a window (`AttackWindow.cost`).

The cost is the devices and channels the support's false states tamper over the whole window [WU26
eq. 12], counted against the meters' own noise: the accuracy class of each meter at its true reading,
which is the noise the file's readings carry (`NoiseModel.sigmas`). A support has no cost when its
false state fails at a snapshot, or when its step between snapshots is larger than that noise where a
stealth bound holds. It calls `FalseState.goal_state` for each snapshot, `Emitter` for the true flows
and currents, `AttackArea.boundary` and `formulas.attacks` for the counts."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np

from ...constants.channels import NodeChannel
from ...formulas.attacks import tampered_channels, tampered_devices
from ...formulas.network import _row_block, complex_voltages
from ...models.choices import CostUnit
from ...models.config import TrustSchedule
from ...models.frames import AttackVector, FrameKnobs, Goal, MeterLayers, SupportCost

if TYPE_CHECKING:
    from ..core import FdiaGenerator


class _Near(NamedTuple):
    """The branches with an end in a support (`lines`) and what their readings need: their from-end
    and to-end Yf and Yt rows cut to the columns they touch (`_row_block`: block, columns), their
    from buses, and whether each is metered."""

    lines: np.ndarray
    yf: tuple[np.ndarray, np.ndarray]
    yt: tuple[np.ndarray, np.ndarray]
    from_bus: np.ndarray
    metered: np.ndarray


def _near_branches(g: FdiaGenerator, support: np.ndarray) -> _Near:
    """`_Near` of a support: the branches with an end in the support."""
    lines = np.flatnonzero(np.isin(g.edge_index, support).any(axis=0))
    yf, yt = _row_block(g.net.yf, lines), _row_block(g.net.yt, lines)
    return _Near(lines, yf, yt, g.case.f[lines], g.flow_metered[lines])


class AttackWindow:
    """One attack window evaluated for candidate supports: the true states, their noise thresholds and
    meter masks computed once, the goal's design per snapshot."""

    def __init__(
        self,
        g: FdiaGenerator,
        states: list[np.ndarray],
        goal: Goal,
        knobs: FrameKnobs,
        stealth_bound: bool = True,
        prev: Optional[AttackVector] = None,
        trust: Optional[TrustSchedule] = None,
    ) -> None:
        # The between-snapshot stealth bound is At's alone (a sub-noise ramp is what At is). [WU26]'s
        # model has no increment constraint: noise only decides which changes the l0 count ignores
        # [WU26, Sec. V-A], so a flow goal (Am) is never bounded. `stealth_bound` off is for analysis.
        self.g, self.states, self.goal, self.knobs = g, states, goal, knobs
        self.stealth_bound = stealth_bound and goal.kind == "load"
        self.channels_first = knobs.objective == CostUnit.CHANNELS.value  # what the cost minimizes first
        # the buses a trusted PMU pins at each snapshot [WU26 eqs. 29-31]
        self.pinned = [frozenset[int]() if trust is None else trust.pinned(t) for t in range(len(states))]
        self._near: Optional[tuple[bytes, _Near]] = None  # the last support's branches (`near`)
        self.node_m, self.edge_m = g.masks.node, g.masks.edge
        self.pmu = np.zeros(g.n_bus, bool)
        self.pmu[g.pmu_buses] = True
        # a bus angle is a PMU channel: a SCADA voltmeter reads |V| only
        self.node_m = self.node_m.copy()
        self.node_m[~self.pmu, NodeChannel.THETA] = 0
        self.flows = g.emitter.clean_flows_from_states(np.stack(states))  # [T, E, 2], unmetered zeroed
        # the PMU branch-current channels [WU26 eqs. 19-20]: their mask, true values and noise scale
        self.i_m: np.ndarray = g.masks.current
        self.currents: np.ndarray = g.emitter.currents_from_states(np.stack(states))
        self.sigma, self.i_sigma = self._sigmas(g, states)
        self.zero = np.array(sorted({int(b) for b in g.zero_inj} - {g.slack_bus}), dtype=np.int64)
        self.prev = self._before(prev)

    def _sigmas(
        self, g: FdiaGenerator, states: list[np.ndarray]
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
        """What a change must exceed to count as tampering at each snapshot, and for At also the most a
        channel may move between snapshots, as (the node and flow layers, the PMU current layer) per
        snapshot: the meters' rated accuracy at the true readings (`NoiseModel.sigmas`), for both
        families. It is the noise the readings of the file carry, so a change under it is one no
        reader of the file can tell from noise."""
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

    @property
    def least(self) -> tuple[int, int]:
        """The fewest (devices, channels) a support of this goal can cost. A ramp may stay under the
        noise of every meter, which is zero of each; an overload has to move at least one."""
        return (1, 1) if self.goal.kind == "flow" else (0, 0)

    def cost(self, support: np.ndarray, beat: Optional[SupportCost]) -> Optional[SupportCost]:
        """The cost of a support held for the window: the union of the tampered devices and channels over
        every snapshot [WU26 eq. 28]. None when the support is infeasible at a snapshot, breaks the
        stealth bound, cannot beat `beat`, or carries a flow goal and moves no device beyond noise."""
        tampered = None if self._zero_on_boundary(support) else self._tampered(support, beat)
        if tampered is None:  # a zero-injection boundary: never for a candidate of `Minimizer._supports`
            return None
        union, devices = tampered
        # A change under the noise is not tampering [WU26, Sec. V-A]. A ramp that stays under the noise
        # of every meter is the cheapest ramp there is, and costs nothing. A flow goal that moves no
        # meter beyond its noise shows no overload, so it is no attack.
        if devices < self.least[0]:
            return None
        cost = SupportCost(devices, _channel_count(union), len(support))
        first = self.channels_first
        return cost if beat is None or cost.key(first) < beat.key(first) else None

    def holds(self, support: np.ndarray) -> bool:
        """Whether the goal has a false state on `support` at every snapshot, inside the operating
        limits and, for a ramp, inside the stealth bound: what a ramp run on a given support must
        keep, whatever it costs."""
        return not self._zero_on_boundary(support) and self._tampered(support, None) is not None

    def support_at(self, t: int, support: np.ndarray) -> np.ndarray:
        """The buses of the support free at snapshot t: the support less the buses of the PMUs trusted by then
        ([WU26] eq. 29 holds their deviation at zero, so they keep their true voltage)."""
        pinned = self.pinned[t]
        return support if not pinned else support[~np.isin(support, sorted(pinned))]

    def _tampered(
        self, support: np.ndarray, beat: Optional[SupportCost]
    ) -> Optional[tuple[MeterLayers, int]]:
        """The union of tampered channels (node, flow, PMU current masks) on the support over the window and
        its device count, or None when the support fails at a snapshot or its first count so far exceeds
        `beat`'s."""
        union = MeterLayers(*(np.zeros(mask.shape, bool) for mask in (self.node_m, self.edge_m, self.i_m)))
        devices, prev = 0, self.prev
        for t in range(len(self.states)):
            free = self.support_at(t, support)
            if not len(free) or (self.pinned[t] and self._zero_on_boundary(free)):
                return None
            moved = self._snapshot(t, free, prev)
            if moved is None:
                return None
            beyond_noise, prev = moved
            union = _union(union, beyond_noise)
            devices = self._devices(union)
            first = _channel_count(union) if self.channels_first else devices
            if beat is not None and first > beat.key(self.channels_first)[0]:
                return None
        return union, devices

    def _devices(self, union: MeterLayers) -> int:
        """How many devices hold a channel of the union (a PMU branch current joins the PMU of the bus
        at its end)."""
        from_bus, to_bus = self.g.edge_index
        return len(tampered_devices(union.node, union.edge, self.pmu, from_bus, union.current, to_bus))

    def _zero_on_boundary(self, support: np.ndarray) -> bool:
        """Whether a zero-injection bus sits on the support's boundary (it could not absorb the changed
        power)."""
        return bool(np.isin(self.g.area.boundary(support)[1], self.zero).any())

    def _snapshot(
        self, t: int, support: np.ndarray, prev: AttackVector
    ) -> Optional[tuple[MeterLayers, AttackVector]]:
        """Snapshot t on a support: (the node, flow and PMU current channels moved beyond their noise, as
        masks, and this snapshot's attack vector), or None when the support has no false state here or (At)
        breaks the stealth bound against the previous snapshot's attack vector `prev`."""
        x_false = self.g.false_state.goal_state(self.goal, t, self.states[t], support, self.knobs)
        if x_false is None:
            return None
        a_node = np.asarray(x_false, float) - np.asarray(self.states[t], float)
        a_edge, a_cur = self._branch_attack(t, x_false, self.near(support))
        sig_node, sig_edge = self.sigma[t]
        if self.stealth_bound:  # no metered channel moves more than its rated accuracy per step
            scale = self.knobs.stealth_scale
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
        return MeterLayers(node, edge, self._over(t, a_cur, 1.0)), AttackVector(a_node, a_edge, a_cur)

    def near(self, support: np.ndarray) -> _Near:
        """`_Near` of a support, cached for the last one."""
        key = np.asarray(support, np.int64).tobytes()
        if self._near is None or self._near[0] != key:
            self._near = (key, _near_branches(self.g, support))
        return self._near[1]

    def _branch_attack(self, t: int, x_false: np.ndarray, near: _Near) -> tuple[np.ndarray, np.ndarray]:
        """The attack vector of false state x_false at snapshot t on the flow channels [E, 2] and the PMU
        branch currents [E, 4]. Only a branch with an end in the support sees a
        false voltage, so only those (`near`) are evaluated; every other entry is zero."""
        V = complex_voltages(x_false[:, NodeChannel.V], x_false[:, NodeChannel.THETA])
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


def _channel_count(union: MeterLayers) -> int:
    """How many channels (node, flow, PMU current) the union holds."""
    return sum(int(m.sum()) for m in union)


def _union(a: MeterLayers, b: MeterLayers) -> MeterLayers:
    """The channels tampered in either of a and b (node, flow, PMU current)."""
    return MeterLayers(a.node | b.node, a.edge | b.edge, a.current | b.current)
