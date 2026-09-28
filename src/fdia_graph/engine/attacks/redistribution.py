"""The load redistribution behind `Al` and `Am`: the target lines ranked once by how far a
load-conserving redistribution can steer them, and the redistribution drawn for one attack."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...models.frames import Redistribution
from .false_state import FalseStateMixin


class RedistributionMixin(FalseStateMixin):
    """Choose a target line and the load-conserving redistribution that steers it [DAT26, WU26]."""

    def _lra_for_line(
        self,
        L: int,
        Lp: np.ndarray,
        rel: float,
        K: int,
        rand: bool = False,
        floor: float = 0.02,
        allowed: Optional[np.ndarray] = None,
    ) -> Optional[Redistribution]:
        # Load Redistribution Attack for target line L: a load-injection delta that is LOAD-CONSERVING (total
        # unchanged -> looks like normal re-dispatch), PER-BUS BOUNDED (|delta_b| <= rel*|Lp_b|), and steers
        # line-L flow via PTDF. rand=True picks buses from the top-2K high-PTDF candidates (varies per record,
        # not memorizable); rand=False is deterministic ranking.
        pl = self._ptdf_load_buses[L]
        cap = rel * np.abs(Lp)
        score = np.abs(pl) * cap  # pl = line-L PTDF row over load buses

        # Raise load on the positive PTDF side, drop it on the negative side: the line-L flow change
        # -sum(PTDF * delta) is negative, so the line reads lighter in the false state (lra_delta then
        # orients it against the base flow). Al and Am are stealthy families, so only their target set
        # moves: active loads, never the slack, never a load on a generator bus [BOY22].
        ok = self._stealthy_mask if allowed is None else (self._stealthy_mask & allowed)
        pos = self._pick_side(np.where((pl > 0) & ok)[0], score, K, rand)
        neg = self._pick_side(np.where((pl < 0) & ok)[0], score, K, rand)
        if len(pos) == 0 or len(neg) == 0:
            return None
        # Both sides scale to a common `budget` (MW moved) so net load change = 0. Always moving the max budget
        # pins the smaller side at cap and piles Al at 20%; instead draw the budget at random within the range
        # keeping both sides' deviation in [floor, rel] (spreads Al across the band, still load-conserving).
        ps, ns = cap[pos].sum(), cap[neg].sum()
        if min(ps, ns) <= 0:
            return None
        lo = (floor / rel) * max(ps, ns)  # smallest budget keeping the larger side above the floor
        hi = min(ps, ns)  # largest budget within the per-bus caps
        if lo >= hi:
            return None  # line too lopsided for a plausible in-band budget -> reject
        budget = float(self.rng.uniform(lo, hi))
        up = cap[pos] * (budget / ps)
        dn = cap[neg] * (budget / ns)
        d = np.zeros_like(Lp)
        d[pos] = up
        d[neg] = -dn
        # Return (delta, attacked-bus indices, achieved line-L flow change = -sum(PTDF*delta)).
        return Redistribution(d, np.r_[pos, neg], float(-np.sum(pl * d)))

    def _pick_side(self, side: np.ndarray, score: np.ndarray, K: int, rand: bool) -> np.ndarray:
        """Rank one PTDF-sign side by score and keep the strongest K, or a random K of the top 2K
        when randomized (varies per record, not memorizable)."""
        side = side[np.argsort(-score[side])]
        if len(side) == 0:
            return side
        top = side[: 2 * K]
        k = min(K, len(top))
        return self.rng.choice(top, k, replace=False) if rand else top[:k]

    def _pick_lra_target(self, rel: float, K: int, n_targets: int = 15) -> None:
        # Rank lines by achievable conserving-redistribution flow change; keep top-`n_targets` as a target
        # POOL. Varying the target per attack diversifies the attacked-bus set so LRA is not trivially
        # memorizable. Evaluate on base-case loads once, up front.
        bl = self.base.load.p_mw.values
        # Skip the outaged line explicitly: its PTDF row is zero (ranks last anyway) but its base-case flow is
        # NaN, and a NaN reaching self._line_flow_sign would poison every LRA delta on that line.
        pot = [
            (L, self._lra_for_line(L, bl, rel, K)) for L in range(self.n_lines) if L != self.contingency.pos
        ]
        pot = [(L, r) for L, r in pot if r is not None]
        pot.sort(key=lambda x: -abs(x[1].line_flow_change))  # most attackable lines first
        self._target_lines = [L for L, _ in pot[: min(n_targets, len(pot))]]
        # Sign of each candidate's base flow (fallback +1): lra_delta orients the redistribution against
        # it, so the target line's |flow| drops in the false state and a real overload reads lighter.
        self._line_flow_sign = {
            L: (float(np.sign(self.base.res_line.p_from_mw.values[L])) or 1.0) for L in self._target_lines
        }
        # default/primary target = most attackable line; -1 when no line admits a redistribution
        self._primary_target_line = self._target_lines[0] if self._target_lines else -1

    def lra_delta(
        self, Lp: np.ndarray, rel: float, K: int, floor: float = 0.02, hops: int = 2
    ) -> Redistribution:
        """A load redistribution steering a random target line, confined to the attacker's
        subnetwork within `hops` branches of the line [WU26]: only loads inside it move."""
        if not self._target_lines:  # no line admits a redistribution: nothing to attack
            return Redistribution(np.zeros_like(Lp), np.array([], int), 0.0, -1, None)
        L = int(self.rng.choice(self._target_lines))  # random target line per attack
        interior = self.local_region(self.ei[:, L], hops)
        if interior is None:
            return Redistribution(np.zeros_like(Lp), np.array([], int), 0.0, L, None)
        allowed = np.isin(self.load_bus, interior)
        r = self._lra_for_line(L, Lp, rel, K, rand=True, floor=floor, allowed=allowed)
        # Apply the base-flow sign so the redistribution LOWERS the target line's |flow| in the false state
        # (a real overload reads lighter, the Al attack); no feasible delta -> zero delta and an empty
        # attacked-bus set (the record stays effectively benign).
        if r is None:
            return Redistribution(np.zeros_like(Lp), np.array([], int), 0.0, L, interior)
        # The flow change carries the same sign so the model describes the redistribution it holds.
        return Redistribution(
            r.delta * self._line_flow_sign[L],
            r.buses,
            r.line_flow_change * self._line_flow_sign[L],
            L,
            interior,
        )
