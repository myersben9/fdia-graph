"""AC power-flow re-solve: recompute a state under new loads with generation pinned to true dispatch."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from ._base import GridBase


class PhysicsMixin(GridBase):
    """Re-solve the grid under attacked/redistributed loads. Mixed into FdiaGenerator."""

    def resolve_states(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
            Xt = X[t]
            # Base active load per element = stored injection + generation folded onto that bus.
            Lp = Xt[self.load_bus, 0] + self.load_genP
            Lq = Xt[self.load_bus, 1].copy()
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
        return out, ok

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
            base_load = Lp_true if Lp_true is not None else Lp  # unattacked load -> the fixed dispatch
            Lfull = np.zeros(self.C)  # total true load per bus (bus-indexed)
            for val, b in zip(base_load, self.load_bus):
                Lfull[int(b)] += val
            Pinj_true = Xt[:, 0]
            gbus = net.gen["bus"].values
            ncnt = {}
            for b in gbus:
                ncnt[int(b)] = ncnt.get(int(b), 0) + 1
            # gen bus injection reproduced: net.load(=Lfull+foldedgen) - gen = Xt[b,0]; split across co-located gens
            gp = np.array([(Lfull[int(b)] - Pinj_true[int(b)]) / ncnt[int(b)] for b in gbus], float)
            # Spread the attack's net load change across gens (AGC-like, proportional to dispatch) instead of
            # onto the slack alone: keeps a plausible, generation-balanced (stealthy) counterfactual whose
            # footprint is the attacked loads plus a small spread, not a single-bus slack spike.
            dL = float(np.sum(Lp) - np.sum(base_load))  # net extra load introduced by the attack
            tot = gp.sum()
            if tot > 0 and dL != 0.0:
                gp = gp + dL * (gp / tot)
            net.gen["p_mw"] = gp
            net.gen["vm_pu"] = [Xt[int(b), 2] for b in gbus]  # hold each gen at its true voltage setpoint
            sb = net.ext_grid["bus"].values  # pin slack reference to true voltage/angle
            net.ext_grid["vm_pu"] = [Xt[int(b), 2] for b in sb]
            net.ext_grid["va_degree"] = [Xt[int(b), 3] for b in sb]
        try:
            self.pp.runpp(net)
            return net
        except Exception:
            return None
