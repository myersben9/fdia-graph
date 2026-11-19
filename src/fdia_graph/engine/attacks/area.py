"""Given seed buses, the attacker's area and the boundary of any set of buses.

The default area is `hop_area`: the buses within `hops` branches of the seeds, the slack out, grown
over the zero-injection buses of its boundary. A caller that knows the area passes it (`FrameKnobs.area`,
`explicit_area`). `rule_area` instead scores the hop areas by the four principles of [WU26] Section
III-A (p. 654): 1) the area holds a compromised voltage or injection channel, 2) its buses are
connected, 3) its loads are significant or its injection changes spread, 4) its load and generation
are stable and well observed. Rules 1 and 2 are checks, 3 and 4 a score (ours). This part calls
`formulas.network` (`subnetwork`, `flow_sensitivities`) and the generator's `LoadSplit`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...constants.channels import NodeChannel
from ...formulas.network import complex_voltages, flow_sensitivities, subnetwork
from ...models.frames import AreaScore
from ..base import Part

# the areas `rule_area` chooses among: the buses within this many hops of the goal (ours)
AREA_HOPS = (1, 2, 3, 4)
# the score's price per bus, as a share of the system (ours): rules 3 and 4 each grow with the area,
# so without it the largest area always wins; at 0.5 the rule picks IEEE-118's 3-hop area around
# [WU26]'s lines 84-85 and 99-100, the closest to its Fig. 9 area
AREA_SIZE_PENALTY = 0.5


def area_score_total(s: AreaScore) -> float:
    """The score `rule_area` ranks candidates by: rules 3 and 4's terms less the size charge (ours)."""
    return s.load_share + s.spread + s.stability + s.observability - AREA_SIZE_PENALTY * s.size


class AttackArea(Part):
    """Where a stealthy attack acts: the buses it may move and the boundary around them."""

    def hop_area(self, seeds: np.ndarray, hops: int) -> Optional[np.ndarray]:
        """The attacker's area around `seeds` [WU26]: the buses within `hops` branches, never the slack
        (the estimator pins it), grown over zero-injection boundary buses (they cannot absorb the changed
        power), with `hops` reduced until a boundary exists. None when the seeds alone leave no boundary."""
        for h in range(hops, -1, -1):
            area, _ = subnetwork(self.g.edge_index, seeds, h, self.g.n_bus)
            area, boundary = self.grow_over_zero_injection(area[area != self.g.slack_bus])
            if len(area) and len(boundary):
                return area
        return None

    def _adjacency(self) -> list[frozenset[int]]:
        """Every bus's neighbours, built once: the searches ask for boundaries thousands of times per
        window."""
        cached = getattr(self, "_adjacency_cache", None)
        if cached is None:
            adj: list[set[int]] = [set() for _ in range(self.g.n_bus)]
            for a, b in self.g.edge_index.T:
                adj[int(a)].add(int(b))
                adj[int(b)].add(int(a))
            cached = [frozenset(n) for n in adj]
            self._adjacency_cache = cached
        return cached

    def boundary(self, buses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(`buses` sorted, their boundary: the buses one branch outside them, sorted). `buses` is an
        area or a support."""
        adj = self._adjacency()
        inside = {int(b) for b in buses}
        outside = {j for i in inside for j in adj[i]} - inside
        return np.array(sorted(inside), int), np.array(sorted(outside), int)

    def grow_over_zero_injection(self, buses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`buses` (an area or a support) with every zero-injection bus of its boundary taken in
        (repeated until the boundary holds none), and that boundary; the slack stays out."""
        zero = {int(b) for b in self.g.zero_inj} - {self.g.slack_bus}
        grown, boundary = self.boundary(buses)
        while len(boundary):
            grow = [int(b) for b in boundary if int(b) in zero]
            if not grow:
                break
            grown, boundary = self.boundary(np.union1d(grown, grow))
        return grown, boundary

    def attack_area(
        self,
        seeds: np.ndarray,
        hops: int,
        rule: str = "hops",
        context: Optional[tuple[list[int], np.ndarray]] = None,
    ) -> Optional[np.ndarray]:
        """The attacker's area for a goal acting through `seeds`: the buses within `hops` (`hop_area`,
        the default), or with `rule` "rules" the area [WU26]'s Section III-A principles pick
        (`rule_area`; `context` = (goal lines, the window's states))."""
        if rule == "rules" and context is not None:
            return self.rule_area(seeds, *context)
        return self.hop_area(seeds, hops)

    def explicit_area(self, buses: np.ndarray) -> Optional[np.ndarray]:
        """An area the caller gives bus by bus (`FrameKnobs.area`): sorted, the slack out, grown over
        the zero-injection buses of its boundary as every area is; None when no boundary is left."""
        inside = np.setdiff1d(np.asarray(buses, np.int64), [self.g.slack_bus])
        area, boundary = self.grow_over_zero_injection(inside)
        return area if len(area) and len(boundary) else None

    def rule_area(self, seeds: np.ndarray, lines: list[int], states: np.ndarray) -> Optional[np.ndarray]:
        """The best-scoring area around `seeds` among `AREA_HOPS` that passes rules 1 and 2 (ours, from
        [WU26] p. 654): None when none does."""
        scored = []
        spread = self._spread(states) if lines else None  # one solve for every candidate
        for h in AREA_HOPS:
            area = self.hop_area(seeds, h)
            if area is not None and not any(np.array_equal(area, s.buses) for s in scored):
                scored.append(self.area_score(area, lines, states, spread))
        ok = [s for s in scored if s.observable and s.connected]
        return max(ok, key=area_score_total).buses if ok else None

    def area_score(
        self, buses: np.ndarray, lines: list[int], states: np.ndarray, spread: Optional[np.ndarray] = None
    ) -> AreaScore:
        """`AreaScore` of the area `buses` for a goal on `lines` (none for a load goal) over the window's
        true `states` [T, N, 4]; `spread` is `_spread(states)` when the caller already has it."""
        if lines and spread is None:
            spread = self._spread(states)
        node_m = self.g.masks.node
        loads = np.abs(self.g.split.load(np.asarray(states, float)))  # [T, N] MW
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
            min(1.0, float(np.abs(spread[lines][:, buses]).mean())) if spread is not None else 0.0,
            float(1.0 / (1.0 + cv.mean())) if len(cv) else 0.0,
            float(min(1.0, per_bus[buses].mean() / max(per_bus.mean(), 1e-9) / 2)),
            len(buses) / self.g.n_bus,
        )

    def _connected(self, buses: np.ndarray) -> bool:
        """Whether `buses` form one connected region."""
        inside = {int(b) for b in buses}
        if not inside:
            return False
        adj = self._adjacency()
        seen: set[int] = set()
        stack = [next(iter(inside))]
        while stack:
            b = stack.pop()
            if b not in seen:
                seen.add(b)
                stack += [n for n in adj[b] if n in inside and n not in seen]
        return seen == inside

    def _spread(self, states: np.ndarray) -> np.ndarray:
        """Each branch's active flow change per MW injected at each bus [E, N], at the window's first
        true state (`formulas.network.flow_sensitivities`, the AC power flow's own sensitivities)."""
        first = np.asarray(states, float)[0]
        V = complex_voltages(first[:, NodeChannel.V], first[:, NodeChannel.THETA])
        return flow_sensitivities(V, self.g.net.ybus, self.g.net.yf, self.g.case.f, self.g.slack_bus)
