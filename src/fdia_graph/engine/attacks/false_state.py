"""The local false state of a stealthy attack: the attacker's area re-solved under false loads with
the boundary held true, checked against the operating limits, and the attack vector it writes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Optional, Union

import numpy as np

from ...formulas.attacks import bus_load, generator_output, operating_limits, within_limits
from ...formulas.network import (
    _dense,
    _flow_blocks,
    _FlowBlocks,
    bus_injections,
    complex_voltages,
    local_ac_solve,
    local_flow_solve,
)
from ...models.frames import AttackDesign, Frame, FrameKnobs, OperatingLimits, Scan
from ...models.grid import NODE
from .area import AreaMixin


def _voltages_out_of_range(
    Xa: np.ndarray, Xt: np.ndarray, S: np.ndarray, limits: OperatingLimits, held: dict[int, float]
) -> dict[int, float]:
    """The buses of S whose false |V| left its limit (21) (the bound at a bus already outside it is
    its true value, as `within_limits` has it), each with the limit to hold it at."""
    lo, hi = np.minimum(limits.v_lo, Xt[:, NODE.v]), np.maximum(limits.v_hi, Xt[:, NODE.v])
    out: dict[int, float] = {}
    for b in np.asarray(S, int):
        v = Xa[b, NODE.v]
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
    its limits `box` [WU26, eqs. 22-23] and that `pins`[0] has not pinned yet, with the change that
    puts it at the nearest limit (generation positive)."""
    pinned, out = pins
    for c, moved in enumerate((np.real(dS), np.imag(dS))):
        value = gen[c] + moved
        clipped = float(np.clip(value, box[c][0][b], box[c][1][b]))
        if b not in pinned[c] and clipped != value:
            out[c][b] = clipped - gen[c]


class _FlowRegion(NamedTuple):
    """One support S at one true state, what every solve of `solve_flow_local`'s active set on it
    shares: S, its free injection buses and the rest of S, the goal (branches, targets in MVA), the
    true voltages (ppc order), S with its edge (`touched`, the buses whose injection the solve moves)
    and their true injections (per unit), the solve's blocks, every bus's true generator output
    [N, 2] and true load [N] (MW)."""

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


class FalseStateMixin(AreaMixin):
    """Solve and check the false state an attack design implies, and the meters it moves."""

    def operating_limits(self, X: np.ndarray) -> OperatingLimits:
        """The constraints every false state of this system must satisfy [WU26, eqs. 21-23]: the
        case's bus voltage limits verbatim and its generator limits widened to what the pool X ran
        each generator over (formulas.attacks.operating_limits)."""
        return operating_limits(self.v_case, self.p_lim, self.q_lim, X, (self.load_base, self.gen_base))

    def solve_local(
        self, Xt: np.ndarray, interior: np.ndarray, Lp: np.ndarray, Lq: np.ndarray
    ) -> Optional[np.ndarray]:
        """The local attacker's false state [WU26]: the interior buses re-solved under the false loads
        `Lp`, `Lq` (per load-table position, MW/MVAr) with every other voltage held at its true value.
        Returns the false state [N, 4] in the pool's columns, with the injections of the interior and
        boundary buses recomputed from the false voltages (the meters the attack must write), or None
        when the local power flow does not converge."""
        C = self.C
        Xa = np.array(Xt, np.float64, copy=True)
        Pinj, Qinj = Xa[:, NODE.p_inj].copy(), Xa[:, NODE.q_inj].copy()
        # the pool's injection is load-positive: every load element at a bus minus that bus's
        # generation, held at this scan's dispatch (the attacker moves loads only)
        load, qload = np.zeros(C), np.zeros(C)
        np.add.at(load, self.load_bus, Lp)
        np.add.at(qload, self.load_bus, Lq)
        buses = np.unique(self.load_bus)
        Pinj[buses] = load[buses] - self.scan_generation(Xt)[buses]
        Qinj[buses] = qload[buses] - self.scan_reactive_generation(Xt)[buses]
        lut = self._ppc_row[np.arange(C)]
        Vc = np.zeros(self._n_ppc_buses, complex)
        Vc[lut] = complex_voltages(Xa[:, NODE.v], Xa[:, NODE.theta])
        target = -(Pinj[interior] + 1j * Qinj[interior]) / self._base_mva  # generation-positive, per unit
        Vf = local_ac_solve(self._dense_admittances()[0], Vc, lut[interior], target)
        if Vf is None:
            return None
        Xa[interior, NODE.v] = np.abs(Vf[lut[interior]])
        Xa[interior, NODE.theta] = np.degrees(np.angle(Vf[lut[interior]]))
        touched = np.union1d(interior, self._boundary(interior)[1])  # and its boundary
        S = bus_injections(Vf, self._Ybus, self._base_mva)[lut[touched]]
        Xa[touched, NODE.p_inj] = -S.real
        Xa[touched, NODE.q_inj] = -S.imag
        return Xa

    def free_load_buses(self) -> np.ndarray:
        """The buses whose load an attacker may pretend in a flow goal: the buses of the stealthy load
        positions (a load off every generator bus, under the load cap), sorted."""
        return np.unique(self.load_bus[self.stealthy_pos])

    def generator_buses(self) -> np.ndarray:
        """The buses with a generator (zero-MW condensers included), the slack excluded: their output
        is an injection measurement the overload attacker may tamper [WU26, eqs. 13-14], bounded by
        the generator limits (22)-(23)."""
        cached = getattr(self, "_generator_buses", None)
        if cached is None:  # asked once per candidate solve: built once, read-only
            cached = np.setdiff1d(np.flatnonzero(self.has_gen), [self.slack_bus])
            cached.flags.writeable = False
            self._generator_buses = cached
        return cached

    def free_injection_buses(self) -> np.ndarray:
        """The buses whose injection a flow-goal attacker may change: the attackable loads and the
        generators (the plan's D14). A zero-injection bus is held at zero: nothing is connected there
        (a rule of ours, not the paper's)."""
        cached = getattr(self, "_free_injection_buses", None)
        if cached is None:  # asked once per candidate solve: built once, read-only
            cached = np.union1d(self.free_load_buses(), self.generator_buses())
            cached.flags.writeable = False
            self._free_injection_buses = cached
        return cached

    def solve_flow_local(
        self,
        Xt: np.ndarray,
        S: np.ndarray,
        line: Union[int, Sequence[int]],
        target_mva: Union[float, Sequence[float]],
        limits: Optional[OperatingLimits] = None,
        load_cap: Optional[float] = None,
    ) -> tuple[Optional[np.ndarray], bool, np.ndarray]:
        """The false state of a flow goal on support S [WU26, eqs. 24-25]: the voltages of S move, the
        free injections of S (attackable loads and generators, `free_injection_buses`) move, every
        other bus of S keeps its true injection (a zero-injection bus at zero), and each goal branch's
        from-end apparent flow reaches its target (`line` and `target_mva` one each, or one per
        branch; `formulas.network.local_flow_solve`). The injections of S's edge buses move with the
        false voltages. With `limits`, every generator whose reported output the attack changes, in
        S or on its edge, stays inside its limits (22)-(23), and every voltage of S inside (21); with
        `load_cap` tau, every load bus whose reported injection it changes shows at most tau times
        its true load (D16). A component (P or Q) the solve drives past its bound is pinned at the
        bound, the other left free, and the solve repeated (an active set), so the least-norm state
        is sought among those that keep every bound. Returns (the false state [N, 4] with the
        injections of S and its edge moved by exactly the change the false voltages cause, whether
        the solve converged, the pretended load change per load bus of S in MW [N], zero at a
        generator bus, whose change is its output's and is checked against the generator limits by
        `within_limits`); the state is None when the solve fails, a bound cannot be met, or S holds
        no free injection."""
        dload = np.zeros(self.C)
        free = np.intersect1d(S, self.free_injection_buses())
        if len(free) == 0:
            return None, True, dload
        region = self._flow_region(Xt, (S, free), line, target_mva)
        edge = np.setdiff1d(region.touched, S)
        # bus -> its pinned change per component (P in MW, Q in MVAr, generation positive)
        pinned: tuple[dict[int, float], dict[int, float]] = ({}, {})
        held_v: dict[int, float] = {}  # bus -> the voltage limit its magnitude is held at, pu
        for _ in range(3 * (len(S) + len(edge)) + 1):
            solved = self._flow_solve_pinned(Xt, region, (pinned, held_v))
            if solved is None:
                return None, False, dload
            Xa, dS = solved
            over = self._out_of_bounds(region, dS, (limits, load_cap), pinned)
            v_over = {} if limits is None else _voltages_out_of_range(Xa, Xt, S, limits, held_v)
            if not over[0] and not over[1] and not v_over:
                # a load bus's change is its pretended load
                loads = np.setdiff1d(free, self.generator_buses())
                dload[loads] = -np.real(dS[loads])
                return Xa, True, dload
            for component, changes in zip(pinned, over):
                component.update(changes)
            held_v.update(v_over)
        return None, True, dload

    def touched_buses(self, S: np.ndarray) -> np.ndarray:
        """S and its edge: the buses whose reported injection a false state on S can change."""
        return np.union1d(S, self._boundary(S)[1])

    def _flow_region(
        self,
        Xt: np.ndarray,
        support: tuple[np.ndarray, np.ndarray],
        line: Union[int, Sequence[int]],
        target_mva: Union[float, Sequence[float]],
    ) -> _FlowRegion:
        """What every solve of the active set on support S (`support` = (S, its free injection buses))
        at the true state Xt shares, built once."""
        S, free = support
        lut = self._ppc_row[np.arange(self.C)]
        lines = np.atleast_1d(np.asarray(line, np.int64))
        V = np.zeros(self._n_ppc_buses, complex)
        V[lut] = complex_voltages(Xt[:, NODE.v], Xt[:, NODE.theta])
        touched, blocks = self._support_blocks(S, lines)
        # per unit, the model's injections at the true voltages
        S_true = bus_injections(V, blocks.Yb)[lut[touched]]
        return _FlowRegion(
            S,
            free,
            np.setdiff1d(S, free),
            (lines, np.atleast_1d(np.asarray(target_mva, float))),
            V,
            touched,
            S_true,
            blocks,
            generator_output(Xt, self.load_base, self.gen_base),
            bus_load(Xt, self.load_base, self.gen_base),
        )

    def _support_blocks(self, S: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, _FlowBlocks]:
        """The parts of a flow region that depend on the support and the goal branches alone: S with
        its edge and the solve's blocks (`_flow_blocks`, with every bus of S and its edge a row a solve
        may hold, since the active set pins edge generators and loads). A search solves one support at
        every snapshot of its window in turn, so the last one is kept."""
        key = (np.asarray(S, np.int64).tobytes(), lines.tobytes())
        cached = getattr(self, "_last_support_blocks", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        lut = self._ppc_row[np.arange(self.C)]
        Yb, Yf = self._dense_admittances()
        touched = self.touched_buses(S)  # only S and its edge change injection
        value = (touched, _flow_blocks(Yb, Yf[lines], lut[S], lut[touched]))
        self._last_support_blocks = (key, value)
        return value

    def _flow_solve_pinned(
        self,
        Xt: np.ndarray,
        region: _FlowRegion,
        held: tuple[tuple[dict[int, float], dict[int, float]], dict[int, float]],
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """One flow solve on the region's support with its free injections, the non-free buses of the
        support held, the pinned components (in the support or on its edge) each at its true value
        plus the pinned change, and the held voltage magnitudes (`held` = ((bus -> P change, bus -> Q
        change), bus -> |V|)): (the false state [N, 4], the injection change per bus [N], generation
        positive, MVA), or None when the solve fails."""
        C, lut, S, touched = self.C, self._ppc_row[np.arange(self.C)], region.S, region.touched
        lines, targets = region.goal
        (pin_p, pin_q), held_v = held
        both = region.rest  # the buses of S that hold their injection
        fixed = np.union1d(both, np.array(sorted({*pin_p, *pin_q}), np.int64))
        change = np.array([complex(pin_p.get(int(b), 0.0), pin_q.get(int(b), 0.0)) for b in fixed])
        S_held = region.S_true[np.searchsorted(touched, fixed)] + change / self._base_mva
        # a held bus of S holds both components, a pinned bus only the pinned one(s)
        whole = set(both.tolist())
        hold = np.array(
            [(int(b) in whole or int(b) in pin_p, int(b) in whole or int(b) in pin_q) for b in fixed]
        )
        vb = np.array(sorted(held_v), np.int64)
        Yb, Yf = self._dense_admittances()
        Vf = local_flow_solve(
            Yb,
            Yf[lines],
            np.asarray(self._from_bus_ppc)[lines],
            region.V,
            lut[S],
            lut[fixed],
            S_held,
            targets / self._base_mva,
            vm_fixed=(lut[vb], np.array([held_v[int(b)] for b in vb])),
            hold=hold.reshape(-1, 2),
            blocks=region.blocks,
        )
        if Vf is None:
            return None
        Xa = np.array(Xt, np.float64, copy=True)
        Xa[S, NODE.v] = np.abs(Vf[lut[S]])
        Xa[S, NODE.theta] = np.degrees(np.angle(Vf[lut[S]]))
        dS = np.zeros(C, complex)
        dS[touched] = (bus_injections(Vf, Yb)[lut[touched]] - region.S_true) * self._base_mva
        Xa[touched, NODE.p_inj] -= np.real(dS[touched])  # the stored injection is load positive
        Xa[touched, NODE.q_inj] -= np.imag(dS[touched])
        return Xa, dS

    def _out_of_bounds(
        self,
        region: _FlowRegion,
        dS: np.ndarray,
        bounds: tuple[Optional[OperatingLimits], Optional[float]],
        pinned: tuple[dict[int, float], dict[int, float]],
    ) -> tuple[dict[int, float], dict[int, float]]:
        """The components of a solve past their bounds, each with the change that puts it at the
        bound (MW or MVAr, generation positive): a generator's P or Q outside its limits (22)-(23),
        in S or on its edge (the slack excluded, its output the balance); a load bus's active change
        beyond `load_cap` times its true load (D16), both over the region's support and its edge
        (`region.touched`). `bounds` = (limits, load_cap), either None."""
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
        buses = np.setdiff1d(
            np.intersect1d(region.touched, np.flatnonzero(self.load_base[:, 0] != 0)), self.generator_buses()
        )
        out: dict[int, float] = {}
        for b in np.setdiff1d(buses, [self.slack_bus]):
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
            cached = (_dense(self._Ybus), _dense(self._Yf))
            self._dense_Y = cached
        return cached

    def frame_from_state(self, Xt: np.ndarray, Xa: np.ndarray, buses: np.ndarray, dev: np.ndarray) -> Frame:
        """The emitted frame of a local false state `Xa` on the true state `Xt`: the true scan (the
        benign twin and every meter's noise draw) plus the attack vector a = h(x_false) - h(x_true) on
        the meters it moves, labelled at `buses` with their designed magnitudes `dev`."""
        scan = self.emit_from_state(Xt)  # the true scan: the benign twin, and the draw every meter keeps
        bnx, bex = scan.node_x, scan.edge_x
        a_node, a_edge = self._attack_vector(Xa, Xt)
        moved = changed_meters(a_node, a_edge, scan)
        nx, ex = bnx.copy(), bex.copy()
        nx[moved[0]] += a_node[moved[0]]
        ex[moved[1]] += a_edge[moved[1]]
        tamper = (nx != bnx, ex != bex)  # the meters whose stored float32 reading changed, no fewer, no more
        y = np.zeros(self.C, np.uint8)
        y[buses] = 1
        ix, itamper = self._currents_with_attack(scan, Xa, Xt)
        return Frame(
            nx,
            scan.node_m,
            ex,
            scan.edge_m,
            y,
            1,
            buses,
            dev,
            bnx,
            bex,
            tamper,
            i_x=ix,
            i_m=scan.i_m,
            benign_i_x=scan.i_x,
            i_tamper=itamper,
        )

    def _currents_with_attack(
        self, scan: Scan, Xa: np.ndarray, Xt: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """The PMU branch-current readings of a stealthy frame [WU26, eqs. 19-20]: the true scan's
        currents plus the attack vector on them, and the current channels written; (None, None)
        without currents in the meter plan."""
        a = self._current_attack(Xa, Xt)
        if a is None or scan.i_x is None or scan.i_m is None:
            return None, None
        moved = (np.abs(a) > 1e-9) & (scan.i_m > 0)
        ix = scan.i_x.copy()
        ix[moved] += a[moved].astype(np.float32)
        return ix, ix != scan.i_x

    def stealthy_state(self, Xt: np.ndarray, design: AttackDesign, k: FrameKnobs) -> Optional[np.ndarray]:
        """The local false state of the design (its targets scaled by its multiplier, re-solved on its
        interior), or None when the local power flow has no solution or the state breaks the operating
        limits [WU26, eqs. 21-23]. A design without an interior is solved on the region around its
        targets. Spends no random draw, so an episode can test its design at onset and redraw."""
        placed = self.with_region(design, k)
        if placed is None or placed.interior is None:
            return None
        interior = placed.interior
        Lp = self.true_load(Xt)  # this scan's active load per load element
        Lq = self.true_reactive_load(Xt)
        Lp_true, Lp = Lp, Lp.copy()
        Lp[design.targets] *= design.mult
        Xa = self.solve_local(Xt, interior, Lp, Lq)
        if Xa is None:
            return None
        if k.limits is not None and not self._within_limits(Xa, Xt, Lp - Lp_true, k.limits, interior):
            return None
        return Xa

    def with_region(self, design: AttackDesign, k: FrameKnobs) -> Optional[AttackDesign]:
        """The design with its interior filled in: the held one, else the region within `k.hops`
        branches of its targets; None when the targets have no such region."""
        if design.interior is not None:
            return design
        interior = self.local_region(self.load_bus[design.targets], k.hops)
        return None if interior is None else design._replace(interior=interior)

    def is_feasible(self, Xt: np.ndarray, design: AttackDesign, k: FrameKnobs) -> bool:
        """Whether the design has a stealthy state on `Xt`, on its interior or the region around its
        targets; the onset test of an episode (no random draw)."""
        return self.stealthy_state(Xt, design, k) is not None

    def _within_limits(
        self,
        Xa: np.ndarray,
        Xt: np.ndarray,
        load_delta_pos: np.ndarray,
        limits: OperatingLimits,
        interior: np.ndarray,
    ) -> bool:
        """[WU26, eqs. 21-23] on a false state: `load_delta_pos` is the pretended load change per
        load-table position (MW), summed per bus for buses carrying several loads; the generator
        limits apply to the generators of `interior`, the attacked subnetwork."""
        dload = np.zeros(self.C)
        np.add.at(dload, self.load_bus, load_delta_pos)
        gen = generator_output(Xt, self.load_base, self.gen_base)
        return within_limits(Xa, Xt, gen, dload, limits, interior)

    def _attack_vector(self, Xa: np.ndarray, Xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The attack vector a = h(x_false) - h(x_true) [WU26] per node channel [N, 4] and per flow
        channel [E, 2], in the scan's physical units: the noiseless reading of the false state minus
        that of the true state (unmetered flows zero on both sides)."""
        flows = self.clean_flows_from_states(np.stack([Xa, Xt]))  # [2, E, 2], unmetered zeroed
        return (np.asarray(Xa, float) - np.asarray(Xt, float)).astype(np.float32), flows[0] - flows[1]

    def _current_attack(self, Xa: np.ndarray, Xt: np.ndarray) -> Optional[np.ndarray]:
        """The attack vector on the PMU branch-current channels [E, 4] (per unit), zero where no PMU
        reads that end; None without currents in the meter plan. The currents of a false state follow
        from its voltages, so the attacker must write them too [WU26, eqs. 19-20]."""
        if self.current_mask() is None:
            return None
        cur = self.currents_from_states(np.stack([Xa, Xt]))
        return cur[0] - cur[1]


def changed_meters(
    a_node: np.ndarray, a_edge: np.ndarray, scan: Scan, tol: float = 1e-7
) -> tuple[np.ndarray, np.ndarray]:
    """The metered channels the attack vector moves (the tamper set): the interior's and the
    boundary's injections and voltages and every flow on a branch touching the interior."""
    node = (np.abs(a_node) > tol) & (scan.node_m > 0)
    edge = (np.abs(a_edge) > tol) & (scan.edge_m > 0)
    return node, edge
