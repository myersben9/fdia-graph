"""The attacker's area: the subnetwork a stealthy attack re-solves, and the boundary it holds true.

[WU26] Section III-A (p. 654) chooses the attacker's subnetwork A = (N_a, E_a) by four principles:
1) it holds at least one compromised PMU or SCADA channel for voltage or injection data, 2) its buses
form a connected region, 3) it favours areas whose load buses are significant or whose injection
changes spread across the network, 4) it favours stable load and generation and good observability.
`rule_area` applies them (rules 1 and 2 as checks, 3 and 4 as a score, ours); a caller that knows the
paper's own area passes it (`FrameKnobs.area`), and the default is the buses within `hops`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...formulas.attacks import bus_load
from ...formulas.network import subnetwork
from ...models.frames import AreaScore
from ..base import GridBase

# the regions `rule_area` chooses among: the buses within this many hops of the goal (ours)
AREA_HOPS = (1, 2, 3, 4)
# the score's price per bus, as a share of the system (ours): rules 3 and 4 each grow with the area,
# so without it the largest region always wins; at 0.5 the rule picks IEEE-118's 3-hop region around
# [WU26]'s lines 84-85 and 99-100, the closest to its Fig. 9 area (experiment `minlp.area_rules`)
AREA_SIZE_PENALTY = 0.5


def area_score_total(s: AreaScore) -> float:
    """The score `rule_area` ranks candidates by: rules 3 and 4's terms less the size charge (ours)."""
    return s.load_share + s.spread + s.stability + s.observability - AREA_SIZE_PENALTY * s.size


class AreaMixin(GridBase):
    """Where a stealthy attack acts: the interior buses it may move and the boundary around them."""

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

    def _live_adjacency(self) -> list[frozenset[int]]:
        """Every bus's neighbours over the live branches, built once per branch status: the searches
        ask for boundaries thousands of times per window, and a pass over every edge each time cost
        more than the rest of `_grow_over_zero_injection`."""
        status = self.branch.status
        cached = getattr(self, "_adjacency_cache", None)
        if cached is None or cached[0] is not status:
            adj: list[set[int]] = [set() for _ in range(self.C)]
            for a, b in self._live_edges().T:
                adj[int(a)].add(int(b))
                adj[int(b)].add(int(a))
            cached = (status, [frozenset(n) for n in adj])
            self._adjacency_cache = cached
        return cached[1]

    def _boundary(self, interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(the interior sorted, its boundary: the buses one live branch outside it, sorted), what
        `subnetwork(self._live_edges(), interior, 0, self.C)` returns, from the cached adjacency."""
        adj = self._live_adjacency()
        inside = {int(b) for b in interior}
        ring = {j for i in inside for j in adj[i]} - inside
        return np.array(sorted(inside), int), np.array(sorted(ring), int)

    def _grow_over_zero_injection(self, interior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The interior with every zero-injection bus of its boundary taken in (repeated until the
        boundary holds none), and that boundary; the slack stays out."""
        zero = {int(b) for b in self.zero_inj} - {self.slack_bus}
        interior, boundary = self._boundary(interior)
        while len(boundary):
            grow = [int(b) for b in boundary if int(b) in zero]
            if not grow:
                break
            interior, boundary = self._boundary(np.union1d(interior, grow))
        return interior, boundary

    def attack_area(
        self,
        seeds: np.ndarray,
        hops: int,
        rule: str = "hops",
        context: Optional[tuple[list[int], np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """The attacker's area for a goal acting through `seeds`: the buses within `hops` (`local_region`,
        the default), or with `rule` "rules" the region [WU26]'s Section III-A principles pick
        (`rule_area`; `context` = (goal lines, the window's states))."""
        if rule == "rules" and context is not None:
            return self.rule_area(seeds, *context)
        return self.local_region(seeds, hops)

    def explicit_area(self, buses: np.ndarray) -> Optional[np.ndarray]:
        """An area given bus by bus (the paper's, `FrameKnobs.area`): sorted, the slack out, grown over
        the zero-injection buses of its boundary as every region is; None when no boundary is left."""
        inside = np.setdiff1d(np.asarray(buses, np.int64), [self.slack_bus])
        interior, boundary = self._grow_over_zero_injection(inside)
        return interior if len(interior) and len(boundary) else None

    def rule_area(self, seeds: np.ndarray, lines: list[int], states: np.ndarray) -> Optional[np.ndarray]:
        """[D18] The best-scoring region around `seeds` among `AREA_HOPS` that passes rules 1 and 2 (ours, from
        [WU26] p. 654): None when none does."""
        scored = []
        for h in AREA_HOPS:
            area = self.local_region(seeds, h)
            if area is not None and not any(np.array_equal(area, s.buses) for s in scored):
                scored.append(self.area_score(area, lines, states))
        ok = [s for s in scored if s.observable and s.connected]
        return max(ok, key=area_score_total).buses if ok else None

    def area_score(self, buses: np.ndarray, lines: list[int], states: np.ndarray) -> AreaScore:
        """`AreaScore` of the area `buses` for a goal on `lines` (none for a load goal) over the window's
        true `states` [T, N, 4]."""
        node_m, _ = self.meter_masks()
        loads = np.abs(bus_load(np.asarray(states, float), self.load_base, self.gen_base))  # [T, N] MW
        mean, sd = loads.mean(0), loads.std(0)
        total = float(mean.sum())
        loaded = mean[buses] > 1e-6
        cv = sd[buses][loaded] / mean[buses][loaded]
        per_bus = node_m.sum(1)
        return AreaScore(
            buses,
            bool(node_m[buses].any()),
            self._connected(buses),
            float(mean[buses].sum() / total) if total > 0 else 0.0,
            min(1.0, float(np.abs(self._ptdf()[lines][:, buses]).mean())) if lines else 0.0,
            float(1.0 / (1.0 + cv.mean())) if len(cv) else 0.0,
            float(min(1.0, per_bus[buses].mean() / max(per_bus.mean(), 1e-9) / 2)),
            len(buses) / self.C,
        )

    def _connected(self, buses: np.ndarray) -> bool:
        """Whether `buses` form one connected region over live branches."""
        inside = {int(b) for b in buses}
        if not inside:
            return False
        adj = self._live_adjacency()
        seen: set[int] = set()
        stack = [next(iter(inside))]
        while stack:
            b = stack.pop()
            if b not in seen:
                seen.add(b)
                stack += [n for n in adj[b] if n in inside and n not in seen]
        return seen == inside

    def _ptdf(self) -> np.ndarray:
        """The DC power transfer distribution factors [E, N] to injections at each bus, slack-referenced,
        from the branch reactances; built once."""
        cached = getattr(self, "_ptdf_cache", None)
        if cached is None:
            A = np.zeros((self.E, self.C))
            A[np.arange(self.E), self.ei[0]] = 1.0
            A[np.arange(self.E), self.ei[1]] = -1.0
            Bf = A / np.asarray(self.branch.x, float)[:, None]
            keep = np.setdiff1d(np.arange(self.C), [self.slack_bus])
            cached = np.zeros((self.E, self.C))
            cached[:, keep] = Bf[:, keep] @ np.linalg.pinv((A.T @ Bf)[np.ix_(keep, keep)])
            self._ptdf_cache = cached
        return cached
