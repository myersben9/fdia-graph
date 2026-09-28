"""Measurement emission: turn a grid state into meter readings (the measurement function h(x))."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..formulas.network import branch_flows, complex_voltages
from ..formulas.noise import jitter_sigma
from ..models.grid import EDGE, NODE
from .base import POWER_NOISE_FLOOR_MW, GridBase
from .records import Scan

if TYPE_CHECKING:
    from .pp_types import PandapowerNet


class MeasurementMixin(GridBase):
    """Emit measurement graphs from a state or a solved net. Mixed into FdiaGenerator."""

    # Draw one zero-mean Gaussian noise sample with std `s` (the meter-noise primitive).
    def _draw_noise(self, s: float) -> float:
        return self.rng.normal(0, s)

    def meter_masks(self) -> tuple[np.ndarray, np.ndarray]:
        """The meter masks of every scan, drawn from the plan with no random draw: node [N, 4] (|V|
        and angle at the voltage-meter and PMU buses, P and Q at the injection and zero-injection
        buses) and flow [E, 2] (both channels of a metered branch)."""
        C, plan = self.C, self.meters
        nm = np.zeros((C, 4), np.uint8)
        for b in range(C):
            if b in plan.vbus or b in plan.pmu:  # |V| and angle observed at the same buses
                nm[b, NODE.v] = 1
                nm[b, NODE.theta] = 1
            if b in plan.inj or b in self.zero_inj:
                nm[b, NODE.p_inj : NODE.q_inj + 1] = 1
        em = np.zeros((self.E, 2), np.uint8)
        em[np.asarray(plan.flow, bool)] = 1
        return nm, em

    def emit_from_state(self, X: np.ndarray) -> Scan:
        # Emit a measurement graph DIRECTLY from a stored state X (no re-solve): exact 0-error flows before
        # meter noise. X columns = [|V|, Pinj, Qinj, angle], the one column order used everywhere.
        C, bias = self.C, self.bias
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
        nx = np.zeros((C, 4), np.float32)
        for b in range(C):
            if nm[b, NODE.v]:
                nx[b, NODE.v] = V[b] + bias.v[b] + self._draw_noise(sig[b, NODE.v])
                nx[b, NODE.theta] = TH[b] + np.degrees(bias.va[b]) + self._draw_noise(sig[b, NODE.theta])
            # Injection/zero-injection buses emit P/Q: relative bias + jitter (+small floor so ~0 injection
            # still gets a nonzero std).
            if nm[b, NODE.p_inj]:
                nx[b, NODE.p_inj] = Pi[b] * (1.0 + bias.pi[b]) + self._draw_noise(sig[b, NODE.p_inj])
                nx[b, NODE.q_inj] = Qi[b] * (1.0 + bias.qi[b]) + self._draw_noise(sig[b, NODE.q_inj])
        # Edge buffers: cols [P_from, Q_from], zero where no flow meter.
        ex = np.zeros((self.E, 2), np.float32)
        for e in range(self.E):
            if em[e, EDGE.p_from]:  # metered branch flow: relative bias + jitter on P and Q
                ex[e, EDGE.p_from] = Sf.real[e] * (1.0 + bias.pf[e]) + self._draw_noise(sig_f[e, EDGE.p_from])
                ex[e, EDGE.q_from] = Sf.imag[e] * (1.0 + bias.qf[e]) + self._draw_noise(sig_f[e, EDGE.q_from])
        return Scan(nx, nm, ex, em)

    def clean_flows_from_states(self, X: np.ndarray) -> np.ndarray:
        # Batched, noiseless sibling of emit_from_state's Sf: exact from-end branch flows for a whole stack of
        # states in one matmul (the per-frame version above is O(T) sparse multiplies). Used to build the clean
        # SE-target edge layer shared by the single-timestamp shards and the streams, so both go through this one
        # physics primitive instead of re-deriving Ybus flows in the loader/generator.
        # X is [T, N, 4] = [|V|, Pinj, Qinj, angle]; only |V| (col 0) and angle (col 3) enter.
        # Returns [T, E, 2] = [P_from MW, Q_from MVAr], unmetered branches zeroed to match emit()'s flow mask.
        X = np.asarray(X, float)
        C = X.shape[1]
        Vc = np.zeros((len(X), self._n_ppc_buses), complex)
        Vc[:, self._ppc_row[np.arange(C)]] = complex_voltages(X[:, :, NODE.v], X[:, :, NODE.theta])
        Sf = branch_flows(Vc, self._Yf, self._from_bus_ppc, self._base_mva)
        ec = np.stack([Sf.real, Sf.imag], axis=2).astype(np.float32)
        ec[:, ~np.asarray(self.meters.flow, bool), :] = 0.0
        return ec

    def state_from_net(self, net: PandapowerNet) -> np.ndarray:
        # Pull operating state [N,4]=[|V|, Pinj, Qinj, theta] from a SOLVED net, matching the stored pool.
        Pi = net.res_bus.p_mw.values.copy()
        Qi = net.res_bus.q_mvar.values.copy()
        # Shunts live in res_bus; subtract shunt draw so Pi/Qi reflect gen/load injection only (matching the
        # stored states and emit_from_state).
        for i in net.shunt.index:
            b = net.shunt.at[i, "bus"]
            Pi[b] -= net.res_shunt.p_mw[i]
            Qi[b] -= net.res_shunt.q_mvar[i]
        V = net.res_bus.vm_pu.values
        TH = net.res_bus.va_degree.values
        return np.column_stack([V, Pi, Qi, TH])  # [N,4] = [|V|, Pinj, Qinj, theta]

    def emit(self, net: PandapowerNet) -> Scan:
        # Emit from a SOLVED net (re-solving attacks) by routing its state through emit_from_state, so
        # attacked and benign samples use the IDENTICAL measurement path. Emitting flows from res_line here
        # (while benign uses the Ybus identity) left a ~7 MW systematic benign-vs-attack offset; sharing one
        # path removes it, so an alpha=1 no-op re-solve matches benign.
        return self.emit_from_state(self.state_from_net(net))
