"""Attack construction: meter-level corruption (Ad/As/Ar) and load-redistribution (Al) deltas."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

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
    ) -> Tuple[np.ndarray, np.ndarray, bool, np.ndarray]:
        # Measurement-level attacks (BDD-DETECTABLE contrast families): perturb already-emitted measurements
        # at attacked buses `atk` and incident branches WITHOUT respecting power-flow physics — which is why
        # bad-data detection catches them. Plausibility band [floor, cap] keeps each tamper above the noise
        # floor (not a within-noise no-op) and below the literature cap (realistic FDIA). `weak` flags a
        # record whose realized change fell inside the floor (only Ar can, since it replays the grid) so
        # make() can reject and redraw it.
        inc = [e for e in range(self.E) if self.ei[0, e] in atk or self.ei[1, e] in atk]
        weak = False
        mags = []  # mags = realized per-bus |delta|/|base| on the P/Q injection channels

        def band_shift(cur: np.ndarray) -> np.ndarray:
            # additive perturbation with per-channel |delta|/|cur| drawn UNIFORMLY over [floor, cap], random
            # sign. In-band draw (vs clipping a big Gaussian) keeps Ad spread across the band, not piled at cap.
            base = np.abs(cur) + 1e-6
            rel = self.rng.uniform(floor, cap, cur.shape)
            sign = np.where(self.rng.random(cur.shape) < 0.5, -1.0, 1.0)
            return sign * rel * base

        for b in atk:
            base = np.abs(nx[b, 1:3]) + 1e-6
            if kind == "Ad":
                sh = band_shift(nx[b, 1:3])
                nx[b, 1:3] += sh
                nx[b, 0] += self.rng.normal(0, 0.02)
                mags.append(float(np.max(np.abs(sh) / base)))
            elif kind == "As":
                gain = self.rng.uniform(1.0 + floor, 1.0 + cap)  # gain inside the plausibility band
                nx[b, 1:3] *= gain
                mags.append(abs(gain - 1.0))
            elif kind == "Ar" and replay is not None:
                cur = nx[b, 1:3].copy()
                nx[b, :] = replay[b, :]
                m = float(np.max(np.abs(nx[b, 1:3] - cur) / base))
                mags.append(m)
                if m < floor or m > cap:
                    weak = True  # replay outside the plausibility band -> reject
        for e in inc:
            if kind == "Ad":
                ex[e] += band_shift(ex[e])
            elif kind == "As":
                ex[e] *= self.rng.uniform(1.0 + floor, 1.0 + cap)
        return nx, ex, weak, np.array(mags, float)

    # ---- LRA (Yuan et al. 2011) target line + delta ----
    def _lra_for_line(
        self, L: int, Lp: np.ndarray, rel: float, K: int, rand: bool = False, floor: float = 0.02
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        # Load Redistribution Attack for target line L: a load-injection delta that is LOAD-CONSERVING (total
        # unchanged -> looks like normal re-dispatch), PER-BUS BOUNDED (|delta_b| <= rel*|Lp_b|), and steers
        # line-L flow via PTDF. rand=True picks buses from the top-2K high-PTDF candidates (varies per record,
        # not memorizable); rand=False is deterministic ranking.
        pl = self._ptdf_lb[L]
        cap = rel * np.abs(Lp)
        score = np.abs(pl) * cap  # pl = line-L PTDF row over load buses

        def pick(side: np.ndarray) -> np.ndarray:
            # Rank one PTDF-sign side by score, keep strongest K (or random K of top-2K if randomized).
            side = side[np.argsort(-score[side])]
            if len(side) == 0:
                return side
            top = side[: 2 * K]
            k = min(K, len(top))
            return self.rng.choice(top, k, replace=False) if rand else top[:k]

        # Raise load on the positive PTDF side, drop on the negative side, to push flow up on line L.
        # Restrict to ATTACKABLE (active-load) buses so a reactive-only bus is never redistributed onto / labelled.
        pos = pick(np.where((pl > 0) & self._attackable_mask)[0])
        neg = pick(np.where((pl < 0) & self._attackable_mask)[0])
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
        return d, np.r_[pos, neg], float(-np.sum(pl * d))

    def _pick_lra_target(self, rel: float, K: int, n_targets: int = 15) -> None:
        # Rank lines by achievable conserving-redistribution flow change; keep top-`n_targets` as a target
        # POOL. Varying the target per attack diversifies the attacked-bus set so LRA is not trivially
        # memorizable. Evaluate on base-case loads once, up front.
        bl = self.base.load.p_mw.values
        # Skip the outaged line explicitly: its PTDF row is zero (ranks last anyway) but its base-case flow is
        # NaN, and a NaN reaching self._sgn would poison every LRA delta on that line.
        pot = [(L, self._lra_for_line(L, bl, rel, K)) for L in range(self.nl) if L != self.outage_pos]
        pot = [(L, r) for L, r in pot if r is not None]
        pot.sort(key=lambda x: -abs(x[1][2]))  # most attackable lines first
        self._Lcands = [L for L, _ in pot[: min(n_targets, len(pot))]]
        # Sign of each candidate's base flow (fallback +1) so the attack WORSENS existing loading (masks a real
        # overload rather than relieving it).
        self._sgn = {L: (float(np.sign(self.base.res_line.p_from_mw.values[L])) or 1.0) for L in self._Lcands}
        self._Ltgt = self._Lcands[0]  # default/primary target = most attackable line

    def lra_delta(
        self, Lp: np.ndarray, rel: float, K: int, floor: float = 0.02
    ) -> Tuple[np.ndarray, np.ndarray]:
        L = int(self.rng.choice(self._Lcands))  # random target line per attack
        r = self._lra_for_line(
            L, Lp, rel, K, rand=True, floor=floor
        )  # + randomized bus subset -> not memorizable
        # Apply the base-flow sign so redistribution masks (not relieves) the overload; no feasible delta ->
        # zero delta and empty attacked-bus set (record stays effectively benign).
        return (r[0] * self._sgn[L], r[1]) if r is not None else (np.zeros_like(Lp), np.array([], int))
