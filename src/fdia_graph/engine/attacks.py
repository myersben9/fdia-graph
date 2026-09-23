"""Attack construction: meter-level corruption (Ad/As/Ar) and load-redistribution (Al) deltas."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.frames import Redistribution  # noqa: F401  re-exported: defined here before the models package
from .base import GridBase


class AttackMixin(GridBase):
    """Build the attacks that tamper measurements or redistribute load. Mixed into FdiaGenerator."""

    def corrupt(
        self,
        nx: np.ndarray,
        ex: np.ndarray,
        atk: np.ndarray,
        kind: str,
        replay: Optional[np.ndarray],
        floor: float = 0.02,
        cap: float = 0.20,
    ) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        """Measurement-level attacks (the BDD-detectable contrast families): tamper the already-emitted
        measurements at the attacked buses `atk` and their incident branches WITHOUT respecting the
        power-flow physics, which is why bad-data detection catches them [DAT26].

        The plausibility band [floor, cap] keeps each tamper above the noise floor (not a within-noise
        no-op) and below the literature cap. Returns (nx, ex, weak, mags): `weak` flags a record whose
        realized change fell inside the band's floor (only Ar can, since it replays the grid) so the
        caller can reject and redraw; `mags` is the realized per-bus |delta| / |base| on the P/Q
        injection channels. The family is decided once; each family's draws happen bus by bus, then
        branch by branch, in the order released shards were built with.
        """
        inc = [e for e in range(self.E) if self.ei[0, e] in atk or self.ei[1, e] in atk]
        if kind == "Ad":
            mags = self._corrupt_bias(nx, ex, atk, inc, floor, cap)
            return nx, ex, False, np.array(mags, float)
        if kind == "As":
            mags = self._corrupt_scaling(nx, ex, atk, inc, floor, cap)
            return nx, ex, False, np.array(mags, float)
        if kind == "Ar" and replay is not None:
            mags, weak = self._corrupt_replay(nx, atk, replay, floor, cap)
            return nx, ex, weak, np.array(mags, float)
        return nx, ex, False, np.zeros(0, float)  # Ar with nothing to replay yet: untouched

    def _band_shift(self, cur: np.ndarray, floor: float, cap: float) -> np.ndarray:
        """An additive perturbation with per-channel |delta| / |cur| drawn UNIFORMLY over [floor, cap]
        and a random sign. An in-band draw (rather than clipping a big Gaussian) keeps Ad spread
        across the band instead of piled at the cap."""
        base = np.abs(cur) + 1e-6
        rel = self.rng.uniform(floor, cap, cur.shape)
        sign = np.where(self.rng.random(cur.shape) < 0.5, -1.0, 1.0)
        return sign * rel * base

    def _corrupt_bias(self, nx, ex, atk, inc, floor, cap) -> list[float]:
        """Ad: an in-band additive shift on P/Q and a small |V| shift at each attacked bus, then an
        in-band shift on the flows of every incident branch."""
        mags = []
        for b in atk:
            base = np.abs(nx[b, 1:3]) + 1e-6
            sh = self._band_shift(nx[b, 1:3], floor, cap)
            nx[b, 1:3] += sh
            nx[b, 0] += self.rng.normal(0, 0.02)
            mags.append(float(np.max(np.abs(sh) / base)))
        for e in inc:
            ex[e] += self._band_shift(ex[e], floor, cap)
        return mags

    def _corrupt_scaling(self, nx, ex, atk, inc, floor, cap) -> list[float]:
        """As: a multiplicative gain inside the band on P/Q at each attacked bus and on each incident flow."""
        mags = []
        for b in atk:
            gain = self.rng.uniform(1.0 + floor, 1.0 + cap)
            nx[b, 1:3] *= gain
            mags.append(abs(gain - 1.0))
        for e in inc:
            ex[e] *= self.rng.uniform(1.0 + floor, 1.0 + cap)
        return mags

    def _corrupt_replay(self, nx, atk, replay, floor, cap) -> tuple[list[float], bool]:
        """Ar: replace each attacked bus's reading with an earlier benign scan's; weak when the realized
        change leaves the plausibility band."""
        mags, weak = [], False
        for b in atk:
            base = np.abs(nx[b, 1:3]) + 1e-6
            cur = nx[b, 1:3].copy()
            nx[b, :] = replay[b, :]
            m = float(np.max(np.abs(nx[b, 1:3] - cur) / base))
            mags.append(m)
            if m < floor or m > cap:
                weak = True
        return mags, weak

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
        # orients it against the base flow). Restrict to ATTACKABLE (active-load) buses so a reactive-only bus is never redistributed onto / labelled.
        ok = self._attackable_mask if allowed is None else (self._attackable_mask & allowed)
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
        self._primary_target_line = self._target_lines[0]  # default/primary target = most attackable line

    def lra_delta(
        self, Lp: np.ndarray, rel: float, K: int, floor: float = 0.02, hops: int = 2
    ) -> Redistribution:
        """A load redistribution steering a random target line, confined to the attacker's
        subnetwork within `hops` branches of the line [WU26]: only loads inside it move."""
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
