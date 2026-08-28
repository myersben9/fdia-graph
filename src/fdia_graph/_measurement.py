"""Measurement emission: turn a grid state into meter readings (the measurement function h(x))."""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ._base import GridBase


class MeasurementMixin(GridBase):
    """Emit measurement graphs from a state or a solved net. Mixed into FdiaGenerator."""

    # Draw one zero-mean Gaussian noise sample with std `s` (the meter-noise primitive).
    def _n(self, s: float) -> float: return self.rng.normal(0, s)

    def emit_from_state(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Emit a measurement graph DIRECTLY from a stored state X (no re-solve): exact 0-error flows before
        # meter noise. X columns = [Pinj, Qinj, |V|, angle].
        C, SD, M = self.C, self.SD, self.M
        Pi, Qi, V, TH = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        # Rebuild the complex bus-voltage phasor vector in ppc ordering: V * e^{j*theta}.
        Vc = np.zeros(self._nppc, complex)
        for b in range(C):
            Vc[self._lut[b]] = V[b]*np.exp(1j*np.deg2rad(TH[b]))
        # Exact from-end flow via Sf = V_from*conj(Yf@V), scaled to physical units (Sf.real=MW, Sf.imag=MVAr).
        Sf = Vc[self._fb]*np.conj(self._Yf@Vc)*self._bMVA
        # Node buffers: cols [|V|, P_inj, Q_inj, angle]; mask=1 where metered.
        nx = np.zeros((C, 4), np.float32)
        nm = np.zeros((C, 4), np.uint8)
        # Each reading = true + constant per-meter bias + per-scan jitter. V-mag/flow biases relative, V/angle
        # biases absolute. va bias/jitter are radians -> degrees to match TH.
        SDj = self.SDj
        for b in range(C):
            if b in M["vbus"] or b in M["pmu"]:     # |V| and angle observed at the same buses
                nx[b, 0] = V[b] + self.bias_v[b] + self._n(SDj["v"])
                nm[b, 0] = 1
                nx[b, 3] = TH[b] + np.degrees(self.bias_va[b]) + self._n(np.degrees(SDj["va"]))
                nm[b, 3] = 1
            # Injection/zero-injection buses emit P/Q: relative bias + jitter (+small floor so ~0 injection
            # still gets a nonzero std).
            if b in M["inj"] or b in self.zero_inj:
                nx[b, 1] = Pi[b]*(1.0+self.bias_pi[b]) + self._n(abs(Pi[b])*SDj["pi"]+1e-3)
                nx[b, 2] = Qi[b]*(1.0+self.bias_qi[b]) + self._n(abs(Qi[b])*SDj["qi"]+1e-3)
                nm[b, 1:3] = 1
        # Edge buffers: cols [P_from, Q_from]; mask=1 where a flow meter exists.
        ex = np.zeros((self.E, 2), np.float32)
        em = np.zeros((self.E, 2), np.uint8)
        for e in range(self.E):
            if self.flow_meter[e]:                  # metered branch flow: relative bias + jitter on P and Q
                ex[e, 0] = Sf.real[e]*(1.0+self.bias_pf[e]) + self._n(abs(Sf.real[e])*SDj["pf"]+1e-3)
                ex[e, 1] = Sf.imag[e]*(1.0+self.bias_qf[e]) + self._n(abs(Sf.imag[e])*SDj["qf"]+1e-3)
                em[e] = 1
        return nx, nm, ex, em

    def state_from_net(self, net: Any) -> np.ndarray:
        # Pull operating state [N,4]=[Pinj, Qinj, |V|, theta] from a SOLVED net, matching the stored pool.
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
        return np.column_stack([Pi, Qi, V, TH])              # [N,4] = [Pinj, Qinj, |V|, theta]

    def emit(self, net: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Emit from a SOLVED net (re-solving attacks) by routing its state through emit_from_state, so
        # attacked and benign samples use the IDENTICAL measurement path. Emitting flows from res_line here
        # (while benign uses the Ybus identity) left a ~7 MW systematic benign-vs-attack offset; sharing one
        # path removes it, so an alpha=1 no-op re-solve matches benign.
        return self.emit_from_state(self.state_from_net(net))
