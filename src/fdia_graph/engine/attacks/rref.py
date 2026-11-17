"""[WU26]'s construction of the attack's support: row reduction of the transposed area Jacobian.

The method is [WU26] p. 655 (the attacker, after [YAN17]) and p. 657, Solution 1 steps 1-5 (the
defender); the reduction is `formulas.trust.sparsest_rows`. At each snapshot the Jacobian is that of the
area's attackable measurements with respect to |V| and angle of the area's buses, less the buses of the
PMUs trusted by then [WU26 eqs. 27, 29], each row scaled by its channel's noise. Each reduced row is an
attack vector H c; its state change c names the buses it moves. A snapshot's support is the sparsest
rows that together move every target line's flow; the window's is their union [WU26 eq. 28]. The
magnitudes come from the AC solve on that support (`FalseStateMixin.solve_flow_local`). A support that
cannot reach the goal grows by the next sparsest rows (ours: the paper gives only the sparsest attack).
Row reduction finds a sparse attack, not necessarily the sparsest."""

from __future__ import annotations

from typing import Optional, cast

import numpy as np

from ...constants.channels import CurrentChannel, NodeChannel
from ...constants.numerics import JACOBIAN_THETA_STEP, JACOBIAN_V_STEP, RELATIVE_ZERO
from ...formulas.network import branch_currents, branch_flows, bus_injections, complex_voltages
from ...formulas.trust import sparsest_rows
from ...models.config import TrustSchedule
from ...models.frames import AttackVector, FlowGoal, FrameKnobs, MinimizerResult
from .minimize import MinimizeMixin, _Window


class RrefSupport:
    """The overload attack's support by [WU26]'s row reduction (`OverloadSettings.support_method`
    "rref"), which `MinimizeMixin.min_tamper` takes for a flow goal in place of its candidate search.
    It holds the generator it reads the grid from."""

    def __init__(self, g: MinimizeMixin) -> None:
        self.g = g

    def rref_support(
        self,
        states: list[np.ndarray],
        goal: FlowGoal,
        k: FrameKnobs,
        prev: Optional[AttackVector] = None,
        trust: Optional[TrustSchedule] = None,
    ) -> Optional[MinimizerResult]:
        """The support the row reduction gives the window of `states` and the flow `goal`, grown until
        the AC solve reaches the goal; None when the goal has no area. `evaluated` counts the supports
        solved, and `devices` is -1 when none reaches the goal."""
        region = self.g.window_area(states, goal, k)
        if region is None:
            return None
        area = np.asarray(region)
        window = _Window(self.g, states, goal, k, prev=prev, trust=trust)
        ladders = [self.ladder(window, t, area) for t in range(len(states))]
        levels = max((len(ladder) for ladder in ladders), default=0)
        tried: set[bytes] = set()
        for level in range(levels):
            S = self.level(ladders, level, area)
            if S.tobytes() in tried:
                continue
            tried.add(S.tobytes())
            cost = window.cost(S, None)
            if cost is not None:
                devices, channels = window.counts(cost)
                return MinimizerResult(S, devices, channels, len(tried))
        return MinimizerResult(area, -1, -1, len(tried))

    def level(self, ladders: list[list[np.ndarray]], level: int, area: np.ndarray) -> np.ndarray:
        """The window's support at a growth level: every snapshot's first `level + 1` rungs joined,
        closed over the zero-injection buses of its boundary as a search candidate is, kept in the area."""
        buses = {int(b) for ladder in ladders for rung in ladder[: level + 1] for b in rung}
        closed = self.g._zero_closed(frozenset(buses)) if buses else frozenset()
        return np.array(sorted(closed & {int(b) for b in area}), dtype=np.int64)

    def ladder(self, window: _Window, t: int, area: np.ndarray) -> list[np.ndarray]:
        """Snapshot t's supports, in the reduction's order: first the buses of the sparsest attack that
        moves each target line (ours: the paper chases the sparsest row outright), then the buses of each
        remaining row, sparsest first. Empty when no free bus moves some target line."""
        free = np.array([int(b) for b in area if int(b) not in window.pinned[t]], dtype=np.int64)
        if not len(free):
            return []
        H, G = self._area_jacobian(window, t, free)
        if not H.size:
            return []
        tol = RELATIVE_ZERO * float(np.abs(H).max())
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
        """(H, G) of `area_jacobian`, without the rows' devices."""
        H, G, _ = self.area_jacobian(window, t, free)
        return H, G

    def area_jacobian(
        self, window: _Window, t: int, free: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(H [m, 2F], G [L, 2F], devices [m]) at snapshot t, by forward differences over the `free` buses
        (columns: |V| of every free bus, then every angle): H the attackable channels over their noise (rows
        no free bus moves dropped), G the apparent from-end flow (MVA) of each target line, and the device
        of each row's channel."""
        X = np.asarray(window.states[t], np.float64)
        F = len(free)
        batch = np.repeat(X[None], 1 + 2 * F, axis=0)
        batch[1 + np.arange(F), free, NodeChannel.V] += JACOBIAN_V_STEP
        batch[1 + F + np.arange(F), free, NodeChannel.THETA] += JACOBIAN_THETA_STEP
        steps = np.r_[np.full(F, JACOBIAN_V_STEP), np.full(F, JACOBIAN_THETA_STEP)]
        z = self._scaled_channels(window, t, batch)
        H = ((z[1:] - z[0]) / steps[:, None]).T
        kept = np.abs(H).max(axis=1) > 0
        flows = np.abs(self.g.all_flows_from_states(batch)[:, list(cast(FlowGoal, window.goal).lines)])
        G = ((flows[1:] - flows[0]) / steps[:, None]).T
        return H[kept], G, self.channel_devices(window)[kept]

    def channel_devices(self, window: _Window) -> np.ndarray:
        """The device of every channel `_scaled_channels` lists, in its order (the SCADA terminal of bus b
        is b, its PMU N + b): |V| and angle belong to the bus's PMU when it has one, injections and flows
        to its SCADA terminal, a branch current to the PMU at that end."""
        N, pmu, ei = self.g.C, window.pmu, self.g.ei
        b, c = np.nonzero(window.node_m > 0)
        at_pmu = np.isin(c, (NodeChannel.V, NodeChannel.THETA)) & pmu[b]
        parts = [np.where(at_pmu, N + b, b), ei[0, np.nonzero(window.edge_m > 0)[0]]]
        if window.i_m is not None and window.i_sigma is not None:
            e, ce = np.nonzero(window.i_m > 0)
            parts.append(
                N
                + np.where(np.isin(ce, (CurrentChannel.RE_FROM, CurrentChannel.IM_FROM)), ei[0, e], ei[1, e])
            )
        return np.concatenate(parts).astype(np.int64)

    def _scaled_channels(self, window: _Window, t: int, batch: np.ndarray) -> np.ndarray:
        """Every attackable channel of each state in `batch` [B, N, 4] over its [WU26] noise, [B, m]:
        the metered node channels (an angle only at a PMU), the metered flows, the PMU branch currents."""
        lut = self.g._ppc_row[np.arange(batch.shape[1])]
        Vc = np.zeros((len(batch), self.g._n_ppc_buses), complex)
        Vc[:, lut] = complex_voltages(batch[:, :, NodeChannel.V], batch[:, :, NodeChannel.THETA])
        S = bus_injections(Vc, self.g._Ybus, self.g._base_mva)[:, lut]
        node = batch.copy()
        node[:, :, NodeChannel.P_INJ], node[:, :, NodeChannel.Q_INJ] = S.real, S.imag
        Sf = branch_flows(Vc, self.g._Yf, self.g._from_bus_ppc, self.g._base_mva)
        edge = np.stack([Sf.real, Sf.imag], axis=2)
        sig_node, sig_edge = window.sigma[t]
        parts = [
            (node / _positive(sig_node))[:, window.node_m > 0],
            (edge / _positive(sig_edge))[:, window.edge_m > 0],
        ]
        if window.i_m is not None and window.i_sigma is not None:
            cur = branch_currents(Vc, self.g._Yf, self.g._Yt) / _positive(window.i_sigma[t])
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
    return {int(i) for i in np.flatnonzero(dS > RELATIVE_ZERO * scale)} if scale > 0 else set()


def _moved_buses(free: np.ndarray, c: np.ndarray) -> np.ndarray:
    """The free buses whose |V| or angle the state change c moves (c: |V| of every free bus, then every
    angle)."""
    F = len(free)
    size = np.abs(c)
    moved = (size[:F] > RELATIVE_ZERO * size.max()) | (size[F:] > RELATIVE_ZERO * size.max())
    return free[moved]
