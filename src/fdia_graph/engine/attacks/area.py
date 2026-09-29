"""The attacker's area: the subnetwork a stealthy attack re-solves, and the boundary it holds true."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...formulas.network import subnetwork
from ..base import GridBase


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
