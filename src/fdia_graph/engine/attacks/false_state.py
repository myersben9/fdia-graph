"""The local false state of a stealthy attack: the attacker's area re-solved under false loads with
the boundary held true, checked against the operating limits, and the attack vector it writes."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...formulas.attacks import generator_output, operating_limits, within_limits
from ...formulas.network import bus_injections, complex_voltages, local_ac_solve, subnetwork
from ...models.frames import AttackDesign, FrameKnobs, OperatingLimits, Scan
from ...models.grid import NODE
from .area import AreaMixin


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
        Vf = local_ac_solve(self._Ybus, Vc, lut[interior], target)
        if Vf is None:
            return None
        Xa[interior, NODE.v] = np.abs(Vf[lut[interior]])
        Xa[interior, NODE.theta] = np.degrees(np.angle(Vf[lut[interior]]))
        touched = np.union1d(interior, subnetwork(self._live_edges(), interior, 0, C)[1])  # and its boundary
        S = bus_injections(Vf, self._Ybus, self._base_mva)[lut[touched]]
        Xa[touched, NODE.p_inj] = -S.real
        Xa[touched, NODE.q_inj] = -S.imag
        return Xa

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


def changed_meters(
    a_node: np.ndarray, a_edge: np.ndarray, scan: Scan, tol: float = 1e-7
) -> tuple[np.ndarray, np.ndarray]:
    """The metered channels the attack vector moves (the tamper set): the interior's and the
    boundary's injections and voltages and every flow on a branch touching the interior."""
    node = (np.abs(a_node) > tol) & (scan.node_m > 0)
    edge = (np.abs(a_edge) > tol) & (scan.edge_m > 0)
    return node, edge
