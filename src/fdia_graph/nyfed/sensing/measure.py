"""The measurement function: the meter readings a grid state gives, and their derivatives."""

from __future__ import annotations

from dataclasses import fields

import numpy as np

from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.sensing.meters import IA, KINDS, PF, QF, VA, I, MeterPlan, P, Q, V


# Derivatives are taken for every bus and branch end, then the metered ones are gathered; an unmetered
# branch can carry no current (an external branch between buses at equal voltage), giving NaN there.
def _magnitude(w: np.ndarray, dw: np.ndarray) -> np.ndarray:
    """d|w| from the derivatives dw [K, X] of complex values w [K]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.real(np.conj(w)[:, None] * dw) / np.abs(w)[:, None]


def _angle(w: np.ndarray, dw: np.ndarray) -> np.ndarray:
    """d angle(w) in degrees from the derivatives dw [K, X] of complex values w [K]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.degrees(np.imag(np.conj(w)[:, None] * dw) / (np.abs(w) ** 2)[:, None])


class Measurement:
    """Holds a Network and a MeterPlan. `h` gives the flat reading vector z of the plan for a state
    (`vm` p.u., `va` degrees; |V| p.u., angles degrees, powers MW and MVAr, currents p.u.), exactly, with
    no noise and no re-solve."""

    def __init__(self, net: Network, plan: MeterPlan):
        self.net, self.plan = net, plan
        self._rows = {k: np.flatnonzero(plan.kind == k) for k in range(len(KINDS))}

    def _gather(self, out: np.ndarray, bus: dict, branch: dict) -> np.ndarray:
        """Fill each meter row of `out` from the per-bus values `bus[kind]` (last axis buses) or the
        per-branch from and to end values `branch[kind]` (last axis branches)."""
        plan = self.plan
        for k, rows in self._rows.items():
            if k in bus:
                out[..., rows] = bus[k][..., plan.bus[rows]]
            else:
                at_f, at_t = branch[k]
                br = plan.branch[rows]
                out[..., rows] = np.where(plan.end[rows] == 0, at_f[..., br], at_t[..., br])
        return out

    def h(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        """z [..., M] for states vm, va [..., N]."""
        p, q = self.net.injections(vm, va)
        pf, qf, pt, qt = self.net.flows(vm, va)
        i_f, i_t = self.net.currents(vm, va)
        branch = {
            PF: (pf, pt),
            QF: (qf, qt),
            I: (np.abs(i_f), np.abs(i_t)),
            IA: (np.angle(i_f, deg=True), np.angle(i_t, deg=True)),
        }
        z = np.empty(np.shape(vm)[:-1] + (len(self.plan.kind),))
        return self._gather(z, {V: vm, VA: va, P: p, Q: q}, branch)

    def jacobian(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        """dz/dx [M, 2N] at one state, x = (vm [N] p.u., va [N] degrees)."""
        net, case = self.net, self.net.case
        v = vm * np.exp(1j * np.deg2rad(va))
        dv = np.hstack([np.diag(v / vm), np.diag(1j * v * np.pi / 180.0)])  # dV/dvm, dV/dva
        i_bus, i_f, i_t = net.ybus @ v, net.yf @ v, net.yt @ v
        d_bus, d_f, d_t = net.ybus @ dv, net.yf @ dv, net.yt @ dv
        # S = V conj(I) at the buses and at each branch end
        ds = (np.conj(i_bus)[:, None] * dv + v[:, None] * np.conj(d_bus)) * case.base_mva
        ds_f = (np.conj(i_f)[:, None] * dv[case.f] + v[case.f, None] * np.conj(d_f)) * case.base_mva
        ds_t = (np.conj(i_t)[:, None] * dv[case.t] + v[case.t, None] * np.conj(d_t)) * case.base_mva
        bus = {V: _magnitude(v, dv), VA: _angle(v, dv), P: ds.real, Q: ds.imag}
        branch = {
            PF: (ds_f.real, ds_t.real),
            QF: (ds_f.imag, ds_t.imag),
            I: (_magnitude(i_f, d_f), _magnitude(i_t, d_t)),
            IA: (_angle(i_f, d_f), _angle(i_t, d_t)),
        }
        # rows index the first axis here, so move the meter axis last for the gather and back
        bus = {k: d.T for k, d in bus.items()}
        branch = {k: (a.T, b.T) for k, (a, b) in branch.items()}
        return self._gather(np.empty((2 * len(v), len(self.plan.kind))), bus, branch).T

    def for_client(self, c: int) -> Measurement:
        """The Measurement of client `c`'s meters only."""
        keep = self.plan.owner == c
        return Measurement(
            self.net, MeterPlan(**{f.name: getattr(self.plan, f.name)[keep] for f in fields(MeterPlan)})
        )
