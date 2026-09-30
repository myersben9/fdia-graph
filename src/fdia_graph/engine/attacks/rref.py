"""[WU26]'s own construction of the attack's support: row reduction of the transposed attack-area
Jacobian with column exchanges.

The paper names the method twice. For the attacker (p. 655): "the attacker can utilize the transpose of
the measurement matrix h, applying multiple elementary row transformations and column exchanges to find
the optimal attack vector solution [25]" ([YAN17]). For the defender's Solution 1 (p. 657, steps 1-5):
the Jacobian transposed, reduced to row echelon form by row swaps, scaling and row additions, the row with
the fewest nonzeros found, its nonzero columns moved to the end with a tracking matrix, the reduction
repeated until that row stabilizes, and the row mapped back through the tracking matrix as the attack
vector (`formulas.trust.sparsest_rows`).

Here, at every snapshot of the window, the Jacobian is that of the attackable measurements of the attack
area (the hybrid meter plan: metered injections and flows, |V| everywhere metered, angles and branch
currents at the PMUs) with respect to |V| and angle of the area's buses, less the buses of the PMUs
trusted by then (eqs. 27, 29: a trusted PMU's own |V| and angle rows are secure, so its bus holds its
true voltage). Its rows are scaled by [WU26]'s noise so a channel's size is in units of its own noise.
Each row of the reduction is an attack vector h c; its state change c, found by least squares, names the
buses it moves. The support of a snapshot is the sparsest rows that together move every target line's
flow (a row moves a line when the linearized flow change is not zero), and the window's support is the
union over the snapshots [WU26, eq. 28]. The magnitudes come from the AC local flow solve on that support
(`FalseStateMixin.solve_flow_local`), so the goal, the operating limits (21)-(23) and the D16 bounds hold
as in the search. When the support cannot reach the goal, it grows by the next sparsest rows of each
snapshot until it can: a fallback of ours, since the paper gives the row reduction only for the sparsest
attack. The result is never `proven`: row reduction finds a sparse attack, not the sparsest.
"""

from __future__ import annotations

from typing import Optional, cast

import numpy as np

from ...formulas.network import branch_currents, branch_flows, bus_injections, complex_voltages
from ...formulas.trust import sparsest_rows
from ...models.choices import SupportMethod
from ...models.config import TrustSchedule
from ...models.frames import AttackVector, FlowGoal, FrameKnobs, MinimizerResult
from ...models.grid import NODE
from .minimize import Goal, MinimizeMixin, _Window

V_STEP = 1e-6  # pu: the finite-difference step of |V| in the attack-area Jacobian
THETA_STEP = 1e-5  # degrees: the step of the angle
ZERO = 1e-9  # a Jacobian entry or state change this small relative to the largest is structurally zero


class RrefMixin(MinimizeMixin):
    """The overload attack's support by [WU26]'s row reduction (`OverloadSettings.support_method`
    "rref"), beside the fewest-tamper search."""

    def _min_tamper(
        self,
        states: list[np.ndarray],
        goal: Goal,
        k: FrameKnobs,
        prev: Optional[AttackVector],
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """The row reduction for a flow goal when `k.support_method` asks for it, else the search."""
        if goal.kind == "flow" and k.support_method == SupportMethod.RREF.value:
            return self.rref_support(states, cast(FlowGoal, goal), k, prev, trust)
        return super()._min_tamper(states, goal, k, prev, trust)

    def rref_support(
        self,
        states: list[np.ndarray],
        goal: FlowGoal,
        k: FrameKnobs,
        prev: Optional[AttackVector] = None,
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """The support [WU26]'s row reduction gives the window of `states` and the flow `goal` (the union
        of each snapshot's sparsest goal-moving rows, grown by the next sparsest until the AC solve
        reaches the goal), costed as the search costs a support; None when the goal has no area.
        `evaluated` counts the supports solved, and `devices` is -1 when none reaches the goal."""
        seeds, _, _ = self._goal_seeds(goal)
        region = self.local_region(seeds, k.hops)
        if region is None:
            return None
        area = np.asarray(region)
        window = _Window(self, states, goal, k, prev=prev, trust=trust)
        ladders = [self._rref_ladder(window, t, area) for t in range(len(states))]
        levels = max((len(ladder) for ladder in ladders), default=0)
        tried: set[bytes] = set()
        for level in range(levels):
            S = self._rref_level(ladders, level, area)
            if S.tobytes() in tried:
                continue
            tried.add(S.tobytes())
            cost = window.cost(S, None)
            window.unsolved += 0 if window.converged else 1
            if cost is not None:
                return MinimizerResult(
                    S, cost[0], cost[1], False, len(tried), window.lower_bound(), window.unsolved
                )
        return MinimizerResult(area, -1, -1, False, len(tried), window.lower_bound(), window.unsolved)

    def _rref_level(self, ladders: list[list[np.ndarray]], level: int, area: np.ndarray) -> np.ndarray:
        """The window's support at a growth level: every snapshot's first `level + 1` rungs joined,
        closed over the zero-injection buses of its boundary as a search candidate is, kept in the area."""
        buses = {int(b) for ladder in ladders for rung in ladder[: level + 1] for b in rung}
        closed = self._zero_closed(frozenset(buses)) if buses else frozenset()
        return np.array(sorted(closed & {int(b) for b in area}), dtype=np.int64)

    def _rref_ladder(self, window: _Window, t: int, area: np.ndarray) -> list[np.ndarray]:
        """Snapshot t's supports in the order the row reduction ranks them: first the buses of the
        sparsest attack that moves each target line (the column exchanges chase, for each line, the
        sparsest row whose state change moves its flow; restricting the chased row to the goal is ours,
        the paper chases the sparsest row outright), then the buses of each remaining row of the last
        reduction, sparsest first. Empty when some target line moves with no free bus (every bus that
        could move it trusted): the snapshot then adds nothing, and the AC solve decides the window."""
        free = np.array([int(b) for b in area if int(b) not in window.pinned[t]], dtype=np.int64)
        if not len(free):
            return []
        H, G = self._area_jacobian(window, t, free)
        if not H.size:
            return []
        tol = ZERO * float(np.abs(H).max())
        first: set[int] = set()
        rows = np.zeros((0, H.shape[0]))
        for line in range(G.shape[0]):  # the sparsest attack that moves each target line
            rows = sparsest_rows(
                H.T, tol, eligible=lambda a, line=line: line in _moved_lines(G, _state_change(H, a))
            )
            if not len(rows):  # no attack on the free buses moves this line: nothing to rank here
                return []
            first |= {int(b) for b in _moved_buses(free, _state_change(H, rows[0]))}
        rest = [_moved_buses(free, _state_change(H, a)) for a in rows[1:]]
        return [np.array(sorted(first), dtype=np.int64), *rest]

    def _area_jacobian(self, window: _Window, t: int, free: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(H [m, 2F], G [L, 2F]) at snapshot t by forward differences in |V| and angle of the `free` buses
        (columns: every free bus's |V|, then its angle): H the attackable channels each over its noise,
        the rows no free bus moves dropped; G the apparent from-end flow (MVA) of every target line."""
        X = np.asarray(window.states[t], np.float64)
        F = len(free)
        batch = np.repeat(X[None], 1 + 2 * F, axis=0)
        batch[1 + np.arange(F), free, NODE.v] += V_STEP
        batch[1 + F + np.arange(F), free, NODE.theta] += THETA_STEP
        steps = np.r_[np.full(F, V_STEP), np.full(F, THETA_STEP)]
        z = self._scaled_channels(window, t, batch)
        H = ((z[1:] - z[0]) / steps[:, None]).T
        H = H[np.abs(H).max(axis=1) > 0]
        flows = np.abs(self.all_flows_from_states(batch)[:, list(cast(FlowGoal, window.goal).lines)])
        G = ((flows[1:] - flows[0]) / steps[:, None]).T
        return H, G

    def _scaled_channels(self, window: _Window, t: int, batch: np.ndarray) -> np.ndarray:
        """Every attackable channel of each state in `batch` [B, N, 4] over its [WU26] noise, [B, m]:
        the metered node channels (an angle only at a PMU), the metered flows, the PMU branch currents."""
        lut = self._ppc_row[np.arange(batch.shape[1])]
        Vc = np.zeros((len(batch), self._n_ppc_buses), complex)
        Vc[:, lut] = complex_voltages(batch[:, :, NODE.v], batch[:, :, NODE.theta])
        S = bus_injections(Vc, self._Ybus, self._base_mva)[:, lut]
        node = batch.copy()
        node[:, :, NODE.p_inj], node[:, :, NODE.q_inj] = S.real, S.imag
        Sf = branch_flows(Vc, self._Yf, self._from_bus_ppc, self._base_mva)
        edge = np.stack([Sf.real, Sf.imag], axis=2)
        sig_node, sig_edge = window.sigma[t]
        parts = [
            (node / _positive(sig_node))[:, window.node_m > 0],
            (edge / _positive(sig_edge))[:, window.edge_m > 0],
        ]
        if window.i_m is not None and window.i_sigma is not None:
            cur = branch_currents(Vc, self._Yf, self._Yt) / _positive(window.i_sigma[t])
            parts.append(cur[:, window.i_m > 0])
        return np.concatenate(parts, axis=1)


def _positive(sigma: np.ndarray) -> np.ndarray:
    """A noise scale safe to divide by: a zero scale (a channel that is never read) counts as one."""
    sigma = np.asarray(sigma, np.float64)
    return np.where(sigma > 0, sigma, 1.0)


def _state_change(H: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The state change c behind the attack vector a = H c (least squares; a lies in H's range)."""
    return np.linalg.lstsq(H, a, rcond=None)[0]


def _moved_lines(G: np.ndarray, c: np.ndarray) -> set[int]:
    """The target lines whose linearized flow the state change c moves."""
    dS = np.abs(G @ c)
    scale = float(np.abs(G).max() * np.abs(c).max()) if G.size and c.size else 0.0
    return {int(i) for i in np.flatnonzero(dS > ZERO * scale)} if scale > 0 else set()


def _moved_buses(free: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The free buses whose |V| or angle the state change c moves (columns: |V| of every free bus, then
    its angle)."""
    F = len(free)
    size = np.abs(c)
    moved = (size[:F] > ZERO * size.max()) | (size[F:] > ZERO * size.max())
    return free[moved]
