"""AC power-flow re-solve: recompute a state under new loads with generation pinned to true dispatch.

The attacker's local false state (its area, the local solve, the operating limits) is in
`engine.attacks`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..formulas.attacks import bus_load, element_loads, generator_output
from ..models.frames import (  # noqa: F401  re-exported: defined here before the models package
    OperatingLimits,
    ResolvedPool,
)
from ..models.grid import NODE
from .base import GridBase

if TYPE_CHECKING:
    from .pp_types import PandapowerNet


class PhysicsMixin(GridBase):
    """Re-solve the grid under new loads, and read a scan's load and generation. Mixed into FdiaGenerator."""

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
            Lp = self.true_load(Xt)  # this scan's active load per load element
            Lq = self.true_reactive_load(Xt)
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

    def true_load(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's active load per load element [n_loads] (MW), `formulas.attacks.bus_load`: the
        stored injection plus the co-located generation at this scan's scale, not the base case's.
        Not the load at the slack bus (see `bus_load`), which no attack targets."""
        return element_loads(bus_load(Xt, self.load_base, self.gen_base), self.load_bus, self.load_p0)

    def true_reactive_load(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's reactive load per load element [n_loads] (MVAr): the bus's stored reactive
        injection plus its generators' reactive output (`scan_reactive_generation`), split over the
        bus's load elements by base reactive share (`formulas.attacks.element_loads`)."""
        return element_loads(
            Xt[:, NODE.q_inj] + self.scan_reactive_generation(Xt), self.load_bus, self.load_q0
        )

    def scan_reactive_generation(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's generator reactive output per bus [N] (MVAr), `formulas.attacks.generator_output`,
        zero at a bus with no generator (a zero-MW condenser counts as one)."""
        q = generator_output(Xt, self.load_base, self.gen_base)[:, 1]
        return np.where(self.has_gen, q, 0.0)

    def scan_generation(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's active generation per bus [N] (MW), `formulas.attacks.generator_output`;
        zero where the bus has no generator."""
        return generator_output(Xt, self.load_base, self.gen_base)[:, 0]

    def _pin_generation(
        self, net: PandapowerNet, Lp: np.ndarray, base_load: np.ndarray, Xt: np.ndarray
    ) -> None:
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

    def solve(
        self,
        Lp: np.ndarray,
        Lq: np.ndarray,
        Xt: Optional[np.ndarray] = None,
        Lp_true: Optional[np.ndarray] = None,
    ) -> Optional[PandapowerNet]:
        # Set new load P/Q on the reusable net and re-run AC power flow. Returns the solved net, or None on
        # non-convergence (attacks can push loads into non-convergent regions — caller skips those).
        net = self._solve_net
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
