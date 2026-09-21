"""AC power-flow re-solve: recompute a state under new loads with generation pinned to true dispatch."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..formulas.attacks import operating_limits
from ..formulas.network import bus_injections, complex_voltages, local_ac_solve, subnetwork
from ..models.frames import (  # noqa: F401  re-exported: defined here before the models package
    OperatingLimits,
    ResolvedPool,
)
from ..models.grid import NODE
from .base import GridBase


class PhysicsMixin(GridBase):
    """Re-solve the grid under attacked/redistributed loads. Mixed into FdiaGenerator."""

    def resolve_states(self, X: np.ndarray) -> ResolvedPool:
        """Re-solve a pool of operating points [T,N,4] under THIS generator's topology.

        A stored state carries the injections AND the voltages the INTACT network produced. Under a
        contingency the same loads give a different state, so emitting an intact state through a
        post-contingency Ybus would fabricate measurements satisfying no power flow. Re-solving holds the
        loads and reconstructed dispatch at the base-case values and changes only the topology (else a
        shifted load profile would confound topology with load level).

        Returns (Xnew [T,N,4], ok [T] bool). Non-converged or non-finite rows are left as-is and flagged
        False (not dropped), so the caller can intersect converged sets across scenarios on one timestamp axis.
        """
        X = np.asarray(X, dtype=np.float64)
        out = X.copy()
        ok = np.zeros(len(X), bool)
        for t in range(len(X)):
            Xt = X[t]  # [N,4] = [|V|, Pinj, Qinj, theta]
            # Base active load per element = stored injection + generation folded onto that bus.
            Lp = Xt[self.load_bus, 1] + self.load_genP
            Lq = Xt[self.load_bus, 2].copy()
            # Lp_true==Lp: alpha=1 no-op re-solve. Reproduces the stored state on the intact topology (the
            # pinning check); yields the post-contingency state on a contingency topology.
            net = self.solve(Lp, Lq, Xt=Xt, Lp_true=Lp)
            if net is None:
                continue
            s = self.state_from_net(net)
            if not np.isfinite(s).all():
                continue
            out[t] = s
            ok[t] = True
        return ResolvedPool(out, ok)

    def _pin_generation(self, net: Any, Lp: np.ndarray, base_load: np.ndarray, Xt: np.ndarray) -> None:
        """Hold every generator at the TRUE dispatch of the unattacked state and spread the attack's net
        load change across generators in proportion to dispatch (AGC-like).

        Otherwise the re-solve leaves gens at base setpoints and dumps the load change onto the slack,
        so even a zero-attack re-solve drifts far from the true state (a residual that is NOT the
        attack). Each bus's true generation is reconstructed from the stored injection
        (net.load = Lfull + folded gen, so gen = Lfull - Pinj_true, split across co-located gens),
        and the spread keeps the counterfactual generation-balanced: its footprint is the attacked
        loads plus a small spread, not a single-bus slack spike. Voltage setpoints and the slack
        reference are pinned to the true state too.
        """
        Lfull = np.zeros(self.C)  # total true load per bus (bus-indexed)
        for val, b in zip(base_load, self.load_bus):
            Lfull[int(b)] += val
        Pinj_true = Xt[:, NODE.p_inj]  # Xt = [|V|, Pinj, Qinj, theta]
        gbus = net.gen["bus"].values
        ncnt: dict[int, int] = {}
        for b in gbus:
            ncnt[int(b)] = ncnt.get(int(b), 0) + 1
        gp = np.array([(Lfull[int(b)] - Pinj_true[int(b)]) / ncnt[int(b)] for b in gbus], float)
        dL = float(np.sum(Lp) - np.sum(base_load))  # net extra load introduced by the attack
        tot = gp.sum()
        if tot > 0 and dL != 0.0:
            gp = gp + dL * (gp / tot)
        net.gen["p_mw"] = gp
        net.gen["vm_pu"] = [Xt[int(b), 0] for b in gbus]  # hold each gen at its true voltage setpoint
        sb = net.ext_grid["bus"].values  # pin the slack reference to the true voltage and angle
        net.ext_grid["vm_pu"] = [Xt[int(b), 0] for b in sb]
        net.ext_grid["va_degree"] = [Xt[int(b), 3] for b in sb]

    def local_region(self, seeds: np.ndarray, hops: int) -> Optional[np.ndarray]:
        """The attacker's interior around `seeds` [WU26]: the buses within `hops` branches, never the
        slack (the angle reference the estimator pins, so its voltage stays true), grown to take in
        any zero-injection bus on the boundary (a boundary bus absorbs the changed power, and a bus
        known to inject nothing cannot), and shrunk in reach until a boundary of fixed-voltage buses
        exists at all. None when even the seeds alone leave no boundary."""
        live = self._live_edges()
        for h in range(hops, -1, -1):
            interior, _ = subnetwork(live, seeds, h, self.C)
            interior, boundary = self._grow_over_zero_injection(interior[interior != self.slack_bus])
            if len(interior) and len(boundary):
                return interior
        return None

    def _live_edges(self) -> np.ndarray:
        """The edge index without the branches out of service: an opened line (an N-1 contingency)
        is not a hop and its far bus is not a boundary."""
        status = self.branch.status
        return self.ei if status is None else self.ei[:, status > 0]

    def _grow_over_zero_injection(self, interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The interior with every zero-injection bus of its boundary taken in (repeated until the
        boundary holds none), and that boundary; the slack stays out."""
        zero = {int(b) for b in self.zero_inj} - {self.slack_bus}
        live = self._live_edges()
        interior, boundary = subnetwork(live, interior, 0, self.C)
        while len(boundary):
            grow = [int(b) for b in boundary if int(b) in zero]
            if not grow:
                break
            interior, boundary = subnetwork(live, np.union1d(interior, grow), 0, self.C)
        return interior, boundary

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
        for pos, b in enumerate(
            self.load_bus
        ):  # the pool's injection is load-positive: load minus generation
            Pinj[b] = Lp[pos] - self.load_genP[pos]
            Qinj[b] = Lq[pos]
        lut = self._lut[np.arange(C)]
        Vc = np.zeros(self._nppc, complex)
        Vc[lut] = complex_voltages(Xa[:, NODE.v], Xa[:, NODE.theta])
        target = -(Pinj[interior] + 1j * Qinj[interior]) / self._bMVA  # generation-positive, per unit
        Vf = local_ac_solve(self._Ybus, Vc, lut[interior], target)
        if Vf is None:
            return None
        Xa[interior, NODE.v] = np.abs(Vf[lut[interior]])
        Xa[interior, NODE.theta] = np.degrees(np.angle(Vf[lut[interior]]))
        touched = np.union1d(interior, subnetwork(self._live_edges(), interior, 0, C)[1])  # and its boundary
        S = bus_injections(Vf, self._Ybus, self._bMVA)[lut[touched]]
        Xa[touched, NODE.p_inj] = -S.real
        Xa[touched, NODE.q_inj] = -S.imag
        return Xa

    def solve(
        self,
        Lp: np.ndarray,
        Lq: np.ndarray,
        Xt: Optional[np.ndarray] = None,
        Lp_true: Optional[np.ndarray] = None,
    ) -> Optional[Any]:
        # Set new load P/Q on the reusable net and re-run AC power flow. Returns the solved net, or None on
        # non-convergence (attacks can push loads into non-convergent regions — caller skips those).
        net = self._solvenet
        net.load["p_mw"] = Lp
        net.load["q_mvar"] = Lq
        # Pin generation to the TRUE dispatch. Otherwise the re-solve leaves gens at base setpoints and dumps
        # the load change onto the slack, so even a zero-attack re-solve drifts far from the true state (a
        # residual that is NOT the attack). Reconstruct each bus's true gen from the stored injection, hold it
        # at the UNATTACKED dispatch, and let the slack (plus AGC spread below) absorb the delta.
        if Xt is not None:
            self._pin_generation(net, Lp, Lp_true if Lp_true is not None else Lp, Xt)
        try:
            self.pp.runpp(net)
            return net
        except Exception:
            return None
