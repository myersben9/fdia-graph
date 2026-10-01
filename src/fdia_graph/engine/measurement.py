"""Measurement emission: turn a grid state into meter readings (the measurement function h(x))."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..formulas.network import branch_currents, branch_flows, complex_voltages
from ..formulas.noise import biased_current, current_sigma, jitter_sigma
from ..models.grid import CURRENT, EDGE, NODE
from .base import POWER_NOISE_FLOOR_MW, GridBase
from .records import Scan

# the jitter stream's key beside (seed, t): one stream per timeline frame, apart from every other draw
JITTER_STREAM = 0x4A17

if TYPE_CHECKING:
    pass


# the order of a bus's node draws: |V|, the angle, then P and Q
_NODE_DRAW_ORDER = (NODE.v, NODE.theta, NODE.p_inj, NODE.q_inj)


class MeasurementMixin(GridBase):
    """Emit measurement graphs from a state or a solved net. Mixed into FdiaGenerator."""

    # Draw one zero-mean Gaussian noise sample with std `s` (the meter-noise primitive).
    def _jitter_stream(self) -> np.random.Generator:
        """The stream an emission draws its jitter from. With a `scan_key` t (set by the timeline walk),
        its own stream seeded by (seed, t): frame t's noise is then the same whatever was drawn
        before it, so the benign frames of two timelines on one seed and pool that place different
        attacks are identical, and every emission of t (an attacked frame's true scan, any retry)
        draws the same noise as its benign twin. Without a key, the generator's own stream."""
        if self.scan_key is None:
            return self.rng
        return np.random.default_rng([self.seed, JITTER_STREAM, self.scan_key])

    def meter_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """The meter masks of every scan, drawn from the plan with no random draw: node [N, 4] (|V| at
        the voltage-meter and PMU buses, the angle at the PMU buses under the hybrid meter model and
        at every voltage-meter bus under the v0.8.3 one, P and Q at the injection and zero-injection
        buses) and flow [E, 2] (both channels of a metered branch)."""
        C, plan = self.C, self.meters
        buses = np.arange(C)
        pmu = np.isin(buses, list(plan.pmu))
        volt = pmu | np.isin(buses, list(plan.vbus))
        nm = np.zeros((C, 4), np.uint8)
        nm[volt, NODE.v] = 1
        # a SCADA voltmeter cannot read an angle; the v0.8.3 plan wrote one at every voltmeter bus
        nm[volt, NODE.theta] = pmu[volt] | (not plan.angle_at_pmu_only)
        nm[
            np.isin(buses, list(plan.inj)) | np.isin(buses, list(self.zero_inj)), NODE.p_inj : NODE.q_inj + 1
        ] = 1
        em = np.zeros((self.E, 2), np.uint8)
        em[np.asarray(plan.flow, bool)] = 1
        return nm, em

    def current_mask(self) -> Optional[np.ndarray]:
        """The PMU branch-current channels [E, 4] (`CURRENT` columns), 1 where a PMU sits at that end
        of an in-service branch [WU26 eqs. 19-20]; None when the meter plan has no currents."""
        plan = self.meters
        if not plan.pmu_currents:
            return None
        pmu = np.zeros(self.C, bool)
        pmu[sorted(plan.pmu)] = True
        live = np.ones(self.E, bool) if self.branch.status is None else np.asarray(self.branch.status) > 0
        cm = np.zeros((self.E, 4), np.uint8)
        from_end, to_end = pmu[self.ei[0]] & live, pmu[self.ei[1]] & live
        cm[from_end, CURRENT.re_from] = cm[from_end, CURRENT.im_from] = 1
        cm[to_end, CURRENT.re_to] = cm[to_end, CURRENT.im_to] = 1
        return cm

    def emit_from_state(self, X: np.ndarray) -> Scan:
        # Emit a measurement graph DIRECTLY from a stored state X (no re-solve): exact 0-error flows before
        # meter noise. X columns = [|V|, Pinj, Qinj, angle], the one column order used everywhere.
        C, bias = self.C, self.bias
        self._jitter = self._jitter_stream()
        V, Pi, Qi, TH = (X[..., i] for i in NODE)  # numpy views of the four columns, no copy
        # The complex bus-voltage phasors in ppc ordering, then the exact from-end flows in MW and MVAr:
        # one physics primitive (formulas.network) shared with the loader and the estimator.
        Vc = np.zeros(self._n_ppc_buses, complex)
        Vc[self._ppc_row[np.arange(C)]] = complex_voltages(V, TH)
        Sf = branch_flows(Vc, self._Yf, self._from_bus_ppc, self._base_mva)
        # Each reading = true + constant per-meter bias + per-scan jitter, the jitter's std from the one
        # noise rule the fewest-tamper objective also counts against (formulas.noise.jitter_sigma).
        # V-mag/flow biases relative, V/angle biases absolute. va bias/jitter are radians -> degrees to
        # match TH. The draw order (bus by bus, then branch by branch) is the released files' order.
        sig, sig_f = jitter_sigma(X, np.stack([Sf.real, Sf.imag], axis=1), self.SDj, POWER_NOISE_FLOOR_MW)
        nm, em = self.meter_masks()
        # node readings: within a bus the draws go |V|, angle, P, Q (the released files' order)
        jit = self._jitter_in_order(nm, sig, _NODE_DRAW_ORDER)
        on = nm.astype(bool)
        nx = np.zeros((C, 4), np.float32)
        nx[:, NODE.v] = np.where(on[:, NODE.v], V + bias.v + jit[:, NODE.v], 0.0)
        nx[:, NODE.theta] = np.where(on[:, NODE.theta], TH + np.degrees(bias.va) + jit[:, NODE.theta], 0.0)
        # Injection/zero-injection buses emit P/Q: relative bias + jitter (+small floor so ~0 injection
        # still gets a nonzero std).
        nx[:, NODE.p_inj] = np.where(on[:, NODE.p_inj], Pi * (1.0 + bias.pi) + jit[:, NODE.p_inj], 0.0)
        nx[:, NODE.q_inj] = np.where(on[:, NODE.q_inj], Qi * (1.0 + bias.qi) + jit[:, NODE.q_inj], 0.0)
        # Edge buffers: cols [P_from, Q_from], zero where no flow meter; relative bias + jitter on P and Q.
        jit_f = self._jitter_in_order(em, sig_f, (EDGE.p_from, EDGE.q_from))
        fon = em.astype(bool)
        ex = np.zeros((self.E, 2), np.float32)
        ex[:, EDGE.p_from] = np.where(
            fon[:, EDGE.p_from], Sf.real * (1.0 + bias.pf) + jit_f[:, EDGE.p_from], 0.0
        )
        ex[:, EDGE.q_from] = np.where(
            fon[:, EDGE.q_from], Sf.imag * (1.0 + bias.qf) + jit_f[:, EDGE.q_from], 0.0
        )
        cm = self.current_mask()
        if cm is None:  # the v0.8.3 meter model: no currents, and no draw after the flows
            return Scan(nx, nm, ex, em)
        return Scan(nx, nm, ex, em, self._emit_currents(Vc, cm), cm)

    def _emit_currents(self, Vc: np.ndarray, cm: np.ndarray) -> np.ndarray:
        """The PMU branch-current readings [E, 4] of one scan (hybrid meters): the exact phasor at each
        metered end plus its relative bias times the end's phasor magnitude and a per-scan jitter,
        both on the C37.118 scale of `current_sigma` [C37118] (`biased_current`), drawn branch by
        branch and column by column after the flows; zero where no PMU reads that end."""
        true = branch_currents(Vc, self._Yf, self._Yt)
        sig = current_sigma(true, self._i_jitter)
        bias = self.bias.i
        assert bias is not None, "a hybrid-meter generator draws the current biases at construction"
        biased = biased_current(true, bias)
        jit = self._jitter_in_order(cm, sig, tuple(CURRENT))
        return np.where(cm.astype(bool), biased + jit, 0.0).astype(np.float32)

    def _jitter_in_order(self, mask: np.ndarray, sig: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
        """Per-scan jitter for every metered channel of `mask` [K, c], zero elsewhere: one vector draw
        from the emission's stream, row by row and, within a row, in the column `order`, the same
        sequence of normals the per-channel draws took."""
        cols = list(order)
        on = mask[:, cols].astype(bool)
        draws = np.zeros(on.shape)
        draws[on] = self._jitter.normal(0.0, sig[:, cols][on])
        out = np.zeros(mask.shape)
        out[:, cols] = draws
        return out

    def clean_flows_from_states(self, X: np.ndarray) -> np.ndarray:
        # Batched, noiseless sibling of emit_from_state's Sf: exact from-end branch flows for a whole stack of
        # states in one matmul (the per-frame version above is O(T) sparse multiplies). Used to build the clean
        # SE-target edge layer shared by the single-timestamp shards and the streams, so both go through this one
        # physics primitive instead of re-deriving Ybus flows in the loader/generator.
        # X is [T, N, 4] = [|V|, Pinj, Qinj, angle]; only |V| (col 0) and angle (col 3) enter.
        # Returns [T, E, 2] = [P_from MW, Q_from MVAr], unmetered branches zeroed to match emit()'s flow mask.
        Sf = self.all_flows_from_states(X)
        ec = np.stack([Sf.real, Sf.imag], axis=2).astype(np.float32)
        ec[:, ~np.asarray(self.meters.flow, bool), :] = 0.0
        return ec

    def all_flows_from_states(self, X: np.ndarray) -> np.ndarray:
        """The exact from-end complex flow [T, E] (MW + j MVAr) of every branch, metered or not, for a
        stack of states [T, N, 4] (`clean_flows_from_states` masks it to the metered branches)."""
        X = np.asarray(X, float)
        Vc = np.zeros((len(X), self._n_ppc_buses), complex)
        Vc[:, self._ppc_row[np.arange(X.shape[1])]] = complex_voltages(X[:, :, NODE.v], X[:, :, NODE.theta])
        return branch_flows(Vc, self._Yf, self._from_bus_ppc, self._base_mva)

    def currents_from_states(self, X: np.ndarray) -> np.ndarray:
        """The exact PMU branch-current channels [T, E, 4] (`CURRENT` columns, per unit) of a stack of
        states [T, N, 4], zero where no PMU reads that end (the batched, noiseless sibling of
        `_emit_currents`); all zero when the meter plan has no currents."""
        X = np.asarray(X, float)
        cm = self.current_mask()
        if cm is None:
            return np.zeros((len(X), self.E, 4))
        Vc = np.zeros((len(X), self._n_ppc_buses), complex)
        Vc[:, self._ppc_row[np.arange(X.shape[1])]] = complex_voltages(X[:, :, NODE.v], X[:, :, NODE.theta])
        return branch_currents(Vc, self._Yf, self._Yt) * cm
