"""Measurement emission: turn a grid state into meter readings (the measurement function h(x))."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..constants.channels import NODE_DRAW_ORDER, CurrentChannel, EdgeChannel, NodeChannel
from ..constants.meters import POWER_NOISE_FLOOR_MW
from ..constants.streams import JITTER_STREAM
from ..formulas.network import branch_currents, branch_flows, complex_voltages
from ..formulas.noise import biased_current, current_sigma, jitter_sigma
from ..models.frames import Scan
from .base import GridBase

if TYPE_CHECKING:
    pass


class MeasurementMixin(GridBase):
    """Emit measurement graphs from a state or a solved net. Mixed into FdiaGenerator."""

    def _jitter_stream(self) -> np.random.Generator:
        """The stream an emission draws its jitter from. With a `scan_key` t it is seeded by (seed, t), so
        frame t's noise does not depend on earlier draws: an attacked frame's true scan matches its benign
        twin, whatever attacks the timeline places. Without a key, the generator's own stream."""
        if self.scan_key is None:
            return self.rng
        return np.random.default_rng([self.seed, JITTER_STREAM, self.scan_key])

    def meter_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """The meter masks of every scan, from the plan with no random draw: node [N, 4] (|V| at voltmeter
        and PMU buses; the angle at PMU buses under the hybrid model, at every voltmeter bus under the v0.8.3
        one; P and Q at injection and zero-injection buses) and flow [E, 2]."""
        C, plan = self.C, self.meters
        buses = np.arange(C)
        pmu = np.isin(buses, list(plan.pmu))
        volt = pmu | np.isin(buses, list(plan.vbus))
        nm = np.zeros((C, 4), np.uint8)
        nm[volt, NodeChannel.V] = 1
        # a SCADA voltmeter cannot read an angle; the v0.8.3 plan wrote one at every voltmeter bus
        nm[volt, NodeChannel.THETA] = pmu[volt] | (not plan.angle_at_pmu_only)
        injection = np.isin(buses, list(plan.inj)) | np.isin(buses, list(self.zero_inj))
        nm[injection, NodeChannel.P_INJ : NodeChannel.Q_INJ + 1] = 1
        em = np.zeros((self.E, 2), np.uint8)
        em[np.asarray(plan.flow, bool)] = 1
        return nm, em

    def current_mask(self) -> Optional[np.ndarray]:
        """The PMU branch-current channels [E, 4] (`CurrentChannel` columns), 1 where a PMU sits at that end
        of an in-service branch [WU26 eqs. 19-20]; None when the meter plan has no currents."""
        plan = self.meters
        if not plan.pmu_currents:
            return None
        pmu = np.zeros(self.C, bool)
        pmu[sorted(plan.pmu)] = True
        live = np.ones(self.E, bool) if self.branch.status is None else np.asarray(self.branch.status) > 0
        cm = np.zeros((self.E, 4), np.uint8)
        from_end, to_end = pmu[self.ei[0]] & live, pmu[self.ei[1]] & live
        cm[from_end, CurrentChannel.RE_FROM] = cm[from_end, CurrentChannel.IM_FROM] = 1
        cm[to_end, CurrentChannel.RE_TO] = cm[to_end, CurrentChannel.IM_TO] = 1
        return cm

    def emit_from_state(self, X: np.ndarray) -> Scan:
        # The readings of one stored state X ([N, 4] = |V|, Pinj, Qinj, angle), with no re-solve: the flows
        # are exact before meter noise.
        C, bias, N = self.C, self.bias, NodeChannel
        self._jitter = self._jitter_stream()
        V, Pi, Qi, TH = (X[..., i] for i in NodeChannel)
        # voltage phasors in ppc ordering, then the exact from-end flows (MW and MVAr)
        Vc = np.zeros(self._n_ppc_buses, complex)
        Vc[self._ppc_row[np.arange(C)]] = complex_voltages(V, TH)
        Sf = branch_flows(Vc, self._Yf, self._from_bus_ppc, self._base_mva)
        # Each reading = true + constant per-meter bias + per-scan jitter (formulas.noise.jitter_sigma, the
        # rule the fewest-tamper objective also counts against). |V| and angle biases are absolute, power
        # biases relative; the angle bias is radians, converted to degrees to match TH.
        sig, sig_f = jitter_sigma(X, np.stack([Sf.real, Sf.imag], axis=1), self.SDj, POWER_NOISE_FLOOR_MW)
        nm, em = self.meter_masks()
        # draw order, the released files': bus by bus (|V|, angle, P, Q within a bus), then branch by branch
        jit = self._jitter_in_order(nm, sig, NODE_DRAW_ORDER)
        on = nm.astype(bool)
        nx = np.zeros((C, 4), np.float32)
        nx[:, N.V] = np.where(on[:, N.V], V + bias.v + jit[:, N.V], 0.0)
        nx[:, N.THETA] = np.where(on[:, N.THETA], TH + np.degrees(bias.va) + jit[:, N.THETA], 0.0)
        nx[:, N.P_INJ] = np.where(on[:, N.P_INJ], Pi * (1.0 + bias.pi) + jit[:, N.P_INJ], 0.0)
        nx[:, N.Q_INJ] = np.where(on[:, N.Q_INJ], Qi * (1.0 + bias.qi) + jit[:, N.Q_INJ], 0.0)
        P, Q = EdgeChannel.P_FROM, EdgeChannel.Q_FROM
        jit_f = self._jitter_in_order(em, sig_f, (P, Q))
        fon = em.astype(bool)
        ex = np.zeros((self.E, 2), np.float32)
        ex[:, P] = np.where(fon[:, P], Sf.real * (1.0 + bias.pf) + jit_f[:, P], 0.0)
        ex[:, Q] = np.where(fon[:, Q], Sf.imag * (1.0 + bias.qf) + jit_f[:, Q], 0.0)
        cm = self.current_mask()
        if cm is None:  # the v0.8.3 meter model: no currents, and no draw after the flows
            return Scan(nx, nm, ex, em)
        return Scan(nx, nm, ex, em, self._emit_currents(Vc, cm), cm)

    def _emit_currents(self, Vc: np.ndarray, cm: np.ndarray) -> np.ndarray:
        """The PMU branch-current readings [E, 4] of one scan (hybrid meters): the exact phasor plus a
        relative bias and a per-scan jitter on the C37.118 scale of `current_sigma` [C37118]. Drawn branch
        by branch, column by column, after the flows; zero where no PMU reads that end."""
        true = branch_currents(Vc, self._Yf, self._Yt)
        sig = current_sigma(true, self._i_jitter)
        bias = self.bias.i
        assert bias is not None, "a hybrid-meter generator draws the current biases at construction"
        biased = biased_current(true, bias)
        jit = self._jitter_in_order(cm, sig, tuple(CurrentChannel))
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
        # The exact from-end flows [T, E, 2] (P MW, Q MVAr) of a stack of states [T, N, 4], with no noise:
        # the clean edge layer an estimator is scored against. Unmetered branches are zero.
        Sf = self.all_flows_from_states(X)
        ec = np.stack([Sf.real, Sf.imag], axis=2).astype(np.float32)
        ec[:, ~np.asarray(self.meters.flow, bool), :] = 0.0
        return ec

    def all_flows_from_states(self, X: np.ndarray) -> np.ndarray:
        """The exact from-end complex flow [T, E] (MW + j MVAr) of every branch, metered or not, for a
        stack of states [T, N, 4] (`clean_flows_from_states` masks it to the metered branches)."""
        X = np.asarray(X, float)
        Vc = np.zeros((len(X), self._n_ppc_buses), complex)
        Vc[:, self._ppc_row[np.arange(X.shape[1])]] = complex_voltages(
            X[:, :, NodeChannel.V], X[:, :, NodeChannel.THETA]
        )
        return branch_flows(Vc, self._Yf, self._from_bus_ppc, self._base_mva)

    def currents_from_states(self, X: np.ndarray) -> np.ndarray:
        """The exact PMU branch-current channels [T, E, 4] (`CurrentChannel` columns, per unit) of a stack of
        states [T, N, 4], zero where no PMU reads that end; all zero when the plan has no currents."""
        X = np.asarray(X, float)
        cm = self.current_mask()
        if cm is None:
            return np.zeros((len(X), self.E, 4))
        Vc = np.zeros((len(X), self._n_ppc_buses), complex)
        Vc[:, self._ppc_row[np.arange(X.shape[1])]] = complex_voltages(
            X[:, :, NodeChannel.V], X[:, :, NodeChannel.THETA]
        )
        return branch_currents(Vc, self._Yf, self._Yt) * cm
