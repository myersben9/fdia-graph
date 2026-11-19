"""The local false state of a stealthy attack: the attacker's area re-solved under false loads with
the boundary held true, and checked against the operating limits. `frames.py` writes its attack
vector onto a scan."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Optional, Union

import numpy as np

from ...constants.channels import NodeChannel
from ...formulas.attacks import operating_limits, within_limits
from ...formulas.network import (
    _dense,
    _flow_blocks,
    _FlowBlocks,
    bus_injections,
    complex_voltages,
    local_ac_solve,
    local_flow_solve,
)
from ...models.frames import AttackDesign, FrameKnobs, OperatingLimits
from ..base import Part


def _voltages_out_of_range(
    x_false: np.ndarray, x_true: np.ndarray, S: np.ndarray, limits: OperatingLimits, held: dict[int, float]
) -> dict[int, float]:
    """The buses of S whose false |V| left its limit [WU26 eq. 21], each with the limit to hold it at.
    A bus already outside its limit is bounded by its true value, as in `within_limits`."""
    lo, hi = (
        np.minimum(limits.v_lo, x_true[:, NodeChannel.V]),
        np.maximum(limits.v_hi, x_true[:, NodeChannel.V]),
    )
    out: dict[int, float] = {}
    for b in np.asarray(S, int):
        v = x_false[b, NodeChannel.V]
        if int(b) not in held and (v < lo[b] or v > hi[b]):
            out[int(b)] = float(np.clip(v, lo[b], hi[b]))
    return out


def _clip_generator(
    b: int,
    gen: np.ndarray,
    dS: complex,
    box: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    pins: tuple[tuple[dict[int, float], dict[int, float]], tuple[dict[int, float], dict[int, float]]],
) -> None:
    """Record in `pins`[1] each of generator b's components (P, Q) that the change dS drives outside
    its limits `box` [WU26 eqs. 22-23] and that `pins`[0] has not pinned yet, with the change that
    puts it at the nearest limit (generation positive)."""
    pinned, out = pins
    for c, moved in enumerate((np.real(dS), np.imag(dS))):
        value = gen[c] + moved
        clipped = float(np.clip(value, box[c][0][b], box[c][1][b]))
        if b not in pinned[c] and clipped != value:
            out[c][b] = clipped - gen[c]


class _FlowRegion(NamedTuple):
    """One support S at one true state: what every solve of `solve_flow_local` on it shares. `free` are
    S's free injection buses and `rest` the others; `goal` = (branches, targets in MVA); `V` the true
    voltages; `touched` S with its edge and `S_true` their true injections (per unit);
    `gen` [N, 2] and `load` [N] the true generator output and load (MW)."""

    S: np.ndarray
    free: np.ndarray
    rest: np.ndarray
    goal: tuple[np.ndarray, np.ndarray]
    V: np.ndarray
    touched: np.ndarray
    S_true: np.ndarray
    blocks: _FlowBlocks
    gen: np.ndarray
    load: np.ndarray


class FalseState(Part):
    """Solve and check the false state an attack design implies, and the meters it moves."""

    def operating_limits(self, X: np.ndarray) -> OperatingLimits:
        """The constraints every false state must satisfy [WU26 eqs. 21-23]: the case's bus voltage
        limits, and its generator limits widened to what the pool X ran each generator over."""
        return operating_limits(self.g.v_case, self.g.p_lim, self.g.q_lim, self.g.split.generation(X))

    def solve_local(
        self, x_true: np.ndarray, interior: np.ndarray, p_load: np.ndarray, q_load: np.ndarray
    ) -> Optional[np.ndarray]:
        """The local false state [N, 4] [WU26]: the interior re-solved under the false loads `p_load`,
        `q_load`
        (per load-table position, MW/MVAr), every other voltage held true. The injections of the interior
        and its boundary follow the false voltages. None when the local power flow does not converge."""
        n_bus = self.g.n_bus
        x_false = np.array(x_true, np.float64, copy=True)
        Pinj, Qinj = x_false[:, NodeChannel.P_INJ].copy(), x_false[:, NodeChannel.Q_INJ].copy()
        # the pool's injection is load-positive: every load element at a bus minus that bus's
        # generation, held at this scan's dispatch (the attacker moves loads only)
        load, qload = np.zeros(n_bus), np.zeros(n_bus)
        np.add.at(load, self.g.load_bus, p_load)
        np.add.at(qload, self.g.load_bus, q_load)
        buses = np.unique(self.g.load_bus)
        Pinj[buses] = load[buses] - self.g.scan_loads.scan_generation(x_true)[buses]
        Qinj[buses] = qload[buses] - self.g.scan_loads.scan_reactive_generation(x_true)[buses]
        Vc = complex_voltages(x_false[:, NodeChannel.V], x_false[:, NodeChannel.THETA])
        base_mva = self.g.case.base_mva
        target = -(Pinj[interior] + 1j * Qinj[interior]) / base_mva  # generation-positive, per unit
        v_false = local_ac_solve(self._dense_admittances()[0], Vc, interior, target)
        if v_false is None:
            return None
        x_false[interior, NodeChannel.V] = np.abs(v_false[interior])
        x_false[interior, NodeChannel.THETA] = np.degrees(np.angle(v_false[interior]))
        touched = np.union1d(interior, self.g.area.boundary(interior)[1])  # and its boundary
        S = bus_injections(v_false, self.g.net.ybus, base_mva)[touched]
        x_false[touched, NodeChannel.P_INJ] = -S.real
        x_false[touched, NodeChannel.Q_INJ] = -S.imag
        return x_false

    def free_load_buses(self) -> np.ndarray:
        """The buses whose load an attacker may pretend in a flow goal (a load off every generator bus,
        under the load cap), sorted."""
        return np.unique(self.g.load_bus[self.g.stealthy_pos])

    def generator_buses(self) -> np.ndarray:
        """The grid's own buses with a generator (zero-MW condensers included), the slack excluded: their
        output is an injection measurement the overload attacker may tamper [WU26 eqs. 13-14], within
        eqs. 22-23."""
        cached = getattr(self, "_generator_buses", None)
        if cached is None:  # asked once per candidate solve: built once, read-only
            cached = np.setdiff1d(np.flatnonzero(self.g.has_gen & self.g.case.own_bus), [self.g.slack_bus])
            cached.flags.writeable = False
            self._generator_buses = cached
        return cached

    def free_injection_buses(self) -> np.ndarray:
        """The buses whose injection a flow-goal attacker may change: the attackable loads and the
        generators. A zero-injection bus is held at zero (ours, not the paper's): nothing connects there."""
        cached = getattr(self, "_free_injection_buses", None)
        if cached is None:  # asked once per candidate solve: built once, read-only
            cached = np.union1d(self.free_load_buses(), self.generator_buses())
            cached.flags.writeable = False
            self._free_injection_buses = cached
        return cached

    def solve_flow_local(
        self,
        x_true: np.ndarray,
        S: np.ndarray,
        line: Union[int, Sequence[int]],
        target_mva: Union[float, Sequence[float]],
        limits: Optional[OperatingLimits] = None,
        load_cap: Optional[float] = None,
    ) -> tuple[Optional[np.ndarray], bool, np.ndarray]:
        """The false state of a flow goal on support S [WU26 eqs. 24-25]: each goal branch's from-end
        apparent flow reaches `target_mva`; only S's voltages and free injections (and its edge's) move.
        Returns (state [N, 4] or None, converged, pretended load change [N] MW, zero at generator buses)."""
        # With `limits`, each generator the attack changes stays inside [WU26 eqs. 22-23] and each voltage
        # of S inside eq. 21; with `load_cap`, a load bus shows at most that share of its true load. A
        # component (P or Q) the solve drives past its bound is pinned there and the solve repeated (an
        # active set). The state is None when the solve fails, a bound cannot be met, or S has no free bus.
        dload = np.zeros(self.g.n_bus)
        free = np.intersect1d(S, self.free_injection_buses())
        if len(free) == 0:
            return None, True, dload
        region = self._flow_region(x_true, (S, free), line, target_mva)
        edge = np.setdiff1d(region.touched, S)
        # bus -> its pinned change per component (P in MW, Q in MVAr, generation positive)
        pinned: tuple[dict[int, float], dict[int, float]] = ({}, {})
        held_v: dict[int, float] = {}  # bus -> the voltage limit its magnitude is held at, pu
        for _ in range(3 * (len(S) + len(edge)) + 1):
            solved = self._flow_solve_pinned(x_true, region, (pinned, held_v))
            if solved is None:
                return None, False, dload
            x_false, dS = solved
            over = self._out_of_bounds(region, dS, (limits, load_cap), pinned)
            v_over = {} if limits is None else _voltages_out_of_range(x_false, x_true, S, limits, held_v)
            if not over[0] and not over[1] and not v_over:
                # a load bus's change is its pretended load
                loads = np.setdiff1d(free, self.generator_buses())
                dload[loads] = -np.real(dS[loads])
                return x_false, True, dload
            for component, changes in zip(pinned, over):
                component.update(changes)
            held_v.update(v_over)
        return None, True, dload

    def touched_buses(self, S: np.ndarray) -> np.ndarray:
        """S and its edge: the buses whose reported injection a false state on S can change."""
        return np.union1d(S, self.g.area.boundary(S)[1])

    def _flow_region(
        self,
        x_true: np.ndarray,
        support: tuple[np.ndarray, np.ndarray],
        line: Union[int, Sequence[int]],
        target_mva: Union[float, Sequence[float]],
    ) -> _FlowRegion:
        """What every solve on support S at the true state x_true shares; `support` = (S, its free buses)."""
        S, free = support
        lines = np.atleast_1d(np.asarray(line, np.int64))
        V = complex_voltages(x_true[:, NodeChannel.V], x_true[:, NodeChannel.THETA])
        touched, blocks = self._support_blocks(S, lines)
        # per unit, the model's injections at the true voltages
        S_true = bus_injections(V, blocks.Yb)[touched]
        return _FlowRegion(
            S,
            free,
            np.setdiff1d(S, free),
            (lines, np.atleast_1d(np.asarray(target_mva, float))),
            V,
            touched,
            S_true,
            blocks,
            self.g.split.generation(x_true),
            self.g.split.load(x_true),
        )

    def _support_blocks(self, S: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, _FlowBlocks]:
        """The parts of a flow region that depend on the support and the goal branches alone: S with
        its edge, and the solve's blocks. A search solves one support at every snapshot of its window in
        turn, so the last one is cached."""
        key = (np.asarray(S, np.int64).tobytes(), lines.tobytes())
        cached = getattr(self, "_last_support_blocks", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        Yb, Yf = self._dense_admittances()
        touched = self.touched_buses(S)  # only S and its edge change injection
        value = (touched, _flow_blocks(Yb, Yf[lines], S, touched))
        self._last_support_blocks = (key, value)
        return value

    def _flow_solve_pinned(
        self,
        x_true: np.ndarray,
        region: _FlowRegion,
        held: tuple[tuple[dict[int, float], dict[int, float]], dict[int, float]],
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """One flow solve on the region's support. `held` = ((bus -> pinned P change, bus -> pinned Q
        change), bus -> held |V|). Returns (the false state [N, 4], the injection change per bus [N],
        generation positive, MVA), or None when the solve fails."""
        n_bus, S, touched, base_mva = self.g.n_bus, region.S, region.touched, self.g.case.base_mva
        lines, targets = region.goal
        (pin_p, pin_q), held_v = held
        both = region.rest  # the buses of S that hold their injection
        fixed = np.union1d(both, np.array(sorted({*pin_p, *pin_q}), np.int64))
        change = np.array([complex(pin_p.get(int(b), 0.0), pin_q.get(int(b), 0.0)) for b in fixed])
        S_held = region.S_true[np.searchsorted(touched, fixed)] + change / base_mva
        # a held bus of S holds both components, a pinned bus only the pinned one(s)
        whole = set(both.tolist())
        ints = [int(b) for b in fixed]
        hold = np.array([(b in whole or b in pin_p, b in whole or b in pin_q) for b in ints])
        vb = np.array(sorted(held_v), np.int64)
        Yb, Yf = self._dense_admittances()
        v_false = local_flow_solve(
            Yb,
            Yf[lines],
            self.g.case.f[lines],
            region.V,
            S,
            fixed,
            S_held,
            targets / base_mva,
            vm_fixed=(vb, np.array([held_v[int(b)] for b in vb])),
            hold=hold.reshape(-1, 2),
            blocks=region.blocks,
        )
        if v_false is None:
            return None
        x_false = np.array(x_true, np.float64, copy=True)
        x_false[S, NodeChannel.V] = np.abs(v_false[S])
        x_false[S, NodeChannel.THETA] = np.degrees(np.angle(v_false[S]))
        dS = np.zeros(n_bus, complex)
        dS[touched] = (bus_injections(v_false, Yb)[touched] - region.S_true) * base_mva
        x_false[touched, NodeChannel.P_INJ] -= np.real(dS[touched])  # the stored injection is load positive
        x_false[touched, NodeChannel.Q_INJ] -= np.imag(dS[touched])
        return x_false, dS

    def _out_of_bounds(
        self,
        region: _FlowRegion,
        dS: np.ndarray,
        bounds: tuple[Optional[OperatingLimits], Optional[float]],
        pinned: tuple[dict[int, float], dict[int, float]],
    ) -> tuple[dict[int, float], dict[int, float]]:
        """The components of a solve past their bounds over `region.touched`, each with the change that
        puts it at the bound (MW or MVAr, generation positive): a generator's P or Q outside [WU26 eqs.
        22-23], a load's active change past `load_cap`. `bounds` = (limits, load_cap), either None."""
        limits, load_cap = bounds
        out: tuple[dict[int, float], dict[int, float]] = ({}, {})
        if limits is not None:
            box = ((limits.p_lo, limits.p_hi), (limits.q_lo, limits.q_hi))
            for b in np.intersect1d(region.touched, self.generator_buses()):
                _clip_generator(int(b), region.gen[b], complex(dS[b]), box, (pinned, out))
        if load_cap is not None:
            out[0].update(self._loads_past_cap(region, dS, load_cap, pinned[0]))
        return out

    def _loads_past_cap(
        self, region: _FlowRegion, dS: np.ndarray, load_cap: float, pinned: dict[int, float]
    ) -> dict[int, float]:
        """The load buses (no generator, not the slack) whose reported active load the solve changes
        by more than `load_cap` times their true load [YUA11], each with the injection change
        (generation positive, MW) that puts the load change at the cap."""
        load = region.load
        loaded = np.intersect1d(region.touched, np.flatnonzero(self.g.split.base_load[:, 0] != 0))
        buses = np.setdiff1d(loaded, self.generator_buses())
        out: dict[int, float] = {}
        for b in np.setdiff1d(buses, [self.g.slack_bus]):
            b = int(b)
            cap = load_cap * abs(float(load[b]))
            dD = -float(np.real(dS[b]))  # the load change the reading shows, MW
            if b not in pinned and abs(dD) > cap:
                out[b] = -float(np.clip(dD, -cap, cap))
        return out

    def _dense_admittances(self) -> tuple[np.ndarray, np.ndarray]:
        """Ybus and Yf as dense arrays, built once (the flow solve indexes rows and multiplies often)."""
        cached = getattr(self, "_dense_Y", None)
        if cached is None:
            cached = (_dense(self.g.net.ybus), _dense(self.g.net.yf))
            self._dense_Y = cached
        return cached

    def stealthy_state(
        self, x_true: np.ndarray, design: AttackDesign, knobs: FrameKnobs
    ) -> Optional[np.ndarray]:
        """The local false state of the design (its targets scaled by its multiplier, re-solved on its
        interior, or on the region around its targets), or None when the local power flow has no solution
        or the state breaks the operating limits [WU26 eqs. 21-23]. Spends no random draw."""
        placed = self.with_region(design, knobs)
        if placed is None or placed.interior is None:
            return None
        interior = placed.interior
        p_load_true = self.g.scan_loads.true_load(x_true)  # this scan's active load per load element
        q_load = self.g.scan_loads.true_reactive_load(x_true)
        p_load = p_load_true.copy()
        p_load[design.targets] *= design.mult
        x_false = self.solve_local(x_true, interior, p_load, q_load)
        if x_false is None:
            return None
        if knobs.limits is not None and not self.state_within_limits(
            x_false, x_true, p_load - p_load_true, knobs.limits, interior
        ):
            return None
        return x_false

    def with_region(self, design: AttackDesign, knobs: FrameKnobs) -> Optional[AttackDesign]:
        """The design with its interior filled in: the held one, else the region within `knobs.hops`
        branches of its targets; None when the targets have no such region."""
        if design.interior is not None:
            return design
        interior = self.g.area.local_region(self.g.load_bus[design.targets], knobs.hops)
        return None if interior is None else design._replace(interior=interior)

    def is_feasible(self, x_true: np.ndarray, design: AttackDesign, knobs: FrameKnobs) -> bool:
        """Whether the design has a stealthy state on `x_true`: the onset test of an episode (no random
        draw)."""
        return self.stealthy_state(x_true, design, knobs) is not None

    def state_within_limits(
        self,
        x_false: np.ndarray,
        x_true: np.ndarray,
        load_delta_pos: np.ndarray,
        limits: OperatingLimits,
        interior: np.ndarray,
    ) -> bool:
        """[WU26 eqs. 21-23] on a false state: `load_delta_pos` is the pretended load change per
        load-table position (MW), summed per bus for buses carrying several loads; the generator
        limits apply to the generators of `interior`, the attacked subnetwork."""
        dload = np.zeros(self.g.n_bus)
        np.add.at(dload, self.g.load_bus, load_delta_pos)
        return within_limits(x_false, x_true, self.g.split.generation(x_true), dload, limits, interior)
