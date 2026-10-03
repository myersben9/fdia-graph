"""[WU26]'s attack and defense as the paper states them (docs/wu26/README.md, "Faithful reproduction"):
the meter plan, the multi-snapshot attack and the incremental trusted-PMU defense, on a pandapower case.

    net = wu26_network("case118", pmus=[76, 78, 80, ...])     # buses as the paper numbers them
    states = wu26_snapshots(net.case, 10, seed=123)            # p. 658's data recipe
    base = solve_window(net, states, area, targets, Wu26Attack(rho=1.5, tau=0.05))
    freeze = incremental_freeze(order, slots, base)            # trust, eqs. 29-30 (E1)
    defended = solve_window(net, states, area, targets, attack, freeze)
    base.devices_any, defended.devices_any                     # the cost of Fig. 4 and Fig. 12

The pieces, each from the paper or labelled ours:

- The meter plan [E17] ("wu26"; [29] Sec. 2.2: S^n = [P^n, Q^n, P^{nn'}, Q^{nn'}, V^n, delta^n]): a SCADA device
  at every bus reads its injection and the flow of every incident branch at its own end, so every branch
  is metered at both ends; a PMU at a PMU bus reads |V|, the angle and the current of every incident
  branch at its end [WU26 eqs. 17-20]; no SCADA voltmeter. A device is everything of one type at a node.
- The attack [WU26 eqs. 12-21, 24-25] [E16]: a false state of the area's buses per snapshot, its boundary held at the
  true state ([37] eq. 12), each target line's apparent flow at its overload by the window's end and never
  falling between snapshots (eqs. 24-25 as written [E19]; no per-snapshot target, unlike the generator's
  D9 ramp), the voltages within limits (eq. 21). The objective is eq. (12) read literally, the norm of the SUM over
  the window, ||sum_t a_t||, with the l0 relaxed to l1 ([37] p. 1898), plus `tau` times the per-snapshot
  l1 so the intermediate snapshots stay bounded (ours). The whole window is one IPOPT problem, since the
  sum couples the snapshots. Changes below the paper's noise (0.03 pu SCADA, 0.01 pu PMU, p. 658) count
  as untampered ([37] p. 1898, [WU26] p. 659).
- The defense [WU26 eqs. 29-30] [E1]: a PMU trusted at snapshot s keeps the attack offset its bus had at
  snapshot s-1 from then on ("accumulating attack vectors", p. 659; eq. 30's h_t-1), its currents
  staying untrusted (eq. 27).
- The cost [E3]: the devices tampered beyond noise at any snapshot (Fig. 4's and Fig. 12's count), with eq.
  (33)'s l0 of the net vector over channels also reported.

Ours, where the paper is silent [E18]: the overload as `rho` times each line's true flow (no rating is
stated), the voltage band 0.94 to 1.06 pu, the per-snapshot trust region (`dv`, `da`) that keeps the
solve on the stealthy branch of the flow constraint, and `tau`. Generator limits (22)-(23) are not
imposed: on the cases' buses mixing load and generation they bound the true state itself.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Optional, cast

import numpy as np

from ..models.config import Wu26Attack
from ..models.errors import NoSuchBranch, SnapshotDrawFailed
from ..models.frames import WindowAttack

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from ..engine.pp_types import PandapowerNet

SIGMA_SCADA = 0.03  # pu, [WU26] p. 658
SIGMA_PMU = 0.01  # pu, [WU26] p. 658
_EPS = 1e-6  # smoothing of |x| in the l1 terms, sqrt(x^2 + eps)

# meter kinds
_P, _Q, _PF, _QF, _V, _TH, _IR, _II = range(8)

Freeze = dict[int, dict[int, tuple[float, float]]]  # snapshot -> {bus: (dVm, dVa)} held offsets


class Wu26Network:
    """A case's network and its [WU26] meter plan (`wu26_network`). Buses and branches in pandapower's
    (ppc) order; each meter row has a kind, the bus it sits at, the branch and end it reads (-1 for a
    node meter), its device (type 0 SCADA, 1 PMU, at a bus) and its noise sigma."""

    def __init__(
        self,
        case: PandapowerNet,
        admittance: tuple[csr_matrix, csr_matrix, csr_matrix],
        branch: np.ndarray,
        pmus: tuple[int, ...],
        rows: list[tuple[int, ...]],
    ) -> None:
        from pandapower.pypower.idx_brch import F_BUS, T_BUS

        self.case = case  # the solved pandapower net
        self.Ybus, self.Yf, self.Yt = admittance
        self.branch = branch
        self.f = branch[:, F_BUS].real.astype(int)
        self.t = branch[:, T_BUS].real.astype(int)
        self.pmus = pmus
        self.kind, self.bus, self.br, self.end, self.dev_type = (np.array(c) for c in zip(*rows))
        self.sigma = np.where(self.dev_type == 0, SIGMA_SCADA, SIGMA_PMU)

    @property
    def n(self) -> int:
        return self.Ybus.shape[0]

    def devices(self, rows: np.ndarray) -> set[tuple[str, int]]:
        """The devices ("SCADA"/"PMU", bus as the paper numbers it) holding the meter `rows`."""
        names = ("SCADA", "PMU")
        return {(names[self.dev_type[r]], int(self.bus[r]) + 1) for r in rows}

    def branch_of(self, a: int, b: int) -> tuple[int, int]:
        """The branch between buses a and b (paper numbering) and the end at bus a (0 from, 1 to)."""
        for k in range(len(self.f)):
            if (self.f[k], self.t[k]) == (a - 1, b - 1):
                return k, 0
            if (self.f[k], self.t[k]) == (b - 1, a - 1):
                return k, 1
        raise NoSuchBranch(f"no branch joins buses {a} and {b}")

    def measure(self, Vm: np.ndarray, Va: np.ndarray) -> np.ndarray:
        """Every meter's reading at the state (per unit; angles in radians)."""
        V = Vm * np.exp(1j * Va)
        S = V * np.conj(self.Ybus @ V)
        If, It = self.Yf @ V, self.Yt @ V
        Sbr = np.stack([V[self.f] * np.conj(If), V[self.t] * np.conj(It)])
        Ibr = np.stack([If, It])
        z = np.empty(len(self.kind))
        k = self.kind
        z[k == _P], z[k == _Q] = S.real[self.bus[k == _P]], S.imag[self.bus[k == _Q]]
        for code, part in ((_PF, np.real), (_QF, np.imag)):
            m = k == code
            z[m] = part(Sbr[self.end[m], self.br[m]])
        for code, part in ((_IR, np.real), (_II, np.imag)):
            m = k == code
            z[m] = part(Ibr[self.end[m], self.br[m]])
        z[k == _V], z[k == _TH] = Vm[self.bus[k == _V]], Va[self.bus[k == _TH]]
        return z

    def jacobian(self, Vm: np.ndarray, Va: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """d(readings)/d[Vm, Va] at the buses `cols`: dense, meters x 2 len(cols) (pandapower's
        dSbus_dV, dSbr_dV, dIbr_dV)."""
        from pandapower.pypower.dIbr_dV import dIbr_dV
        from pandapower.pypower.dSbr_dV import dSbr_dV
        from pandapower.pypower.dSbus_dV import dSbus_dV

        V = Vm * np.exp(1j * Va)
        dS_dVm, dS_dVa = dSbus_dV(self.Ybus, V)
        dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, _, _ = dSbr_dV(self.branch, self.Yf, self.Yt, V)
        dIf_dVa, dIf_dVm, dIt_dVa, dIt_dVm, _, _ = dIbr_dV(self.branch, self.Yf, self.Yt, V)

        sub = partial(_columns, cols=cols)
        inj = np.hstack([sub(dS_dVm), sub(dS_dVa)])
        sbr = np.stack([np.hstack([sub(dSf_dVm), sub(dSf_dVa)]), np.hstack([sub(dSt_dVm), sub(dSt_dVa)])])
        ibr = np.stack([np.hstack([sub(dIf_dVm), sub(dIf_dVa)]), np.hstack([sub(dIt_dVm), sub(dIt_dVa)])])
        J = np.zeros((len(self.kind), 2 * len(cols)))
        k = self.kind
        J[k == _P], J[k == _Q] = np.real(inj)[self.bus[k == _P]], np.imag(inj)[self.bus[k == _Q]]
        for code, part, src in (
            (_PF, np.real, sbr),
            (_QF, np.imag, sbr),
            (_IR, np.real, ibr),
            (_II, np.imag, ibr),
        ):
            m = k == code
            J[m] = part(src[self.end[m], self.br[m]])
        pos = {int(b): i for i, b in enumerate(cols)}
        for code, off in ((_V, 0), (_TH, len(cols))):
            for r in np.flatnonzero(k == code):
                if int(self.bus[r]) in pos:
                    J[r, off + pos[int(self.bus[r])]] = 1.0
        return J

    def flow(self, Vm: np.ndarray, Va: np.ndarray, k: int, end: int) -> complex:
        """The complex power branch k carries at its end `end` (0 from, 1 to)."""
        V = Vm * np.exp(1j * Va)
        bus = self.f[k] if end == 0 else self.t[k]
        Y = self.Yf if end == 0 else self.Yt
        return complex(V[bus] * np.conj((Y[k] @ V).item()))

    def flow_rows(self, k: int, end: int) -> tuple[int, int]:
        """The SCADA meter rows of branch k's P and Q flow at its end `end`."""
        rows = np.flatnonzero((self.br == k) & (self.end == end) & (self.dev_type == 0))
        return int(rows[self.kind[rows] == _PF][0]), int(rows[self.kind[rows] == _QF][0])


def _columns(M: csr_matrix, cols: np.ndarray) -> np.ndarray:
    """The columns `cols` of a sparse derivative matrix, dense."""
    return np.asarray(M[:, cols].toarray())


def wu26_network(case: str, pmus: Sequence[int]) -> Wu26Network:
    """The pandapower case `case` (e.g. "case14", "case118") with the [WU26] meter plan and PMUs at the
    buses `pmus` (as the paper numbers them, 1-based)."""
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.pypower.idx_brch import F_BUS, T_BUS
    from pandapower.pypower.makeYbus import makeYbus

    net = getattr(pn, case)()
    pp.runpp(net)
    ppc = net._ppc
    Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    branch = ppc["branch"]
    f, t = branch[:, F_BUS].real.astype(int), branch[:, T_BUS].real.astype(int)
    pmu = tuple(sorted(int(b) - 1 for b in pmus))
    return Wu26Network(
        net, (Ybus.tocsr(), Yf.tocsr(), Yt.tocsr()), branch, pmu, _meter_rows(Ybus.shape[0], f, t, pmu)
    )


def _meter_rows(n: int, f: np.ndarray, t: np.ndarray, pmus: tuple[int, ...]) -> list[tuple[int, ...]]:
    """(kind, bus, branch, end, device type) per meter: [29] Sec. 2.2's S^n at every bus, and the PMUs."""
    rows: list[tuple[int, ...]] = []
    incident: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for k, (a, b) in enumerate(zip(f, t)):
        incident[int(a)].append((k, 0))
        incident[int(b)].append((k, 1))
    for i in range(n):
        rows += [(_P, i, -1, -1, 0), (_Q, i, -1, -1, 0)]
        for k, e in incident[i]:
            rows += [(_PF, i, k, e, 0), (_QF, i, k, e, 0)]
    for i in pmus:
        rows += [(_V, i, -1, -1, 1), (_TH, i, -1, -1, 1)]
        for k, e in incident[i]:
            rows += [(_IR, i, k, e, 1), (_II, i, k, e, 1)]
    return rows


def wu26_snapshots(
    case: PandapowerNet, count: int, seed: int, tries: int = 10
) -> list[tuple[np.ndarray, np.ndarray]]:
    """[WU26] p. 658's steady-state samples: each snapshot scales every load by a system level drawn from
    U[0.9, 1] times (1 + N(0, 0.03)) per load, and solves the power flow; (|V|, angle in radians) per
    snapshot. A non-converging draw is redrawn, up to `tries` times `count` draws in all
    (`SnapshotDrawFailed` past that)."""
    import pandapower as pp

    rng = np.random.default_rng(seed)
    p0, q0 = case.load.p_mw.values.copy(), case.load.q_mvar.values.copy()
    states: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(tries * count):
        if len(states) == count:
            return states
        net = copy.deepcopy(case)
        scale = rng.uniform(0.9, 1.0) * (1.0 + rng.normal(0.0, 0.03, size=len(p0)))
        net.load.p_mw, net.load.q_mvar = p0 * scale, q0 * scale
        try:
            pp.runpp(net)
        except Exception:  # noqa: BLE001 (pandapower raises its own convergence errors)
            continue
        states.append((net.res_bus.vm_pu.values.copy(), np.radians(net.res_bus.va_degree.values)))
    if len(states) < count:
        raise SnapshotDrawFailed(f"{len(states)} of {count} snapshots converged in {tries * count} draws")
    return states


def solve_window(
    net: Wu26Network,
    states: list[tuple[np.ndarray, np.ndarray]],
    area: Sequence[int],
    targets: Sequence[tuple[int, int]],
    attack: Wu26Attack,
    freeze: Optional[Freeze] = None,
) -> WindowAttack:
    """[WU26 eq. 12]'s attack on the window `states` over the `area` buses (0-based), overloading the
    `targets` (bus pairs as the paper numbers them), with the trusted buses' offsets `freeze` held."""
    from cyipopt import minimize_ipopt

    problem = _WindowProblem(net, states, np.asarray(sorted(area)), targets, attack, freeze or {})
    res = minimize_ipopt(
        problem.objective,
        problem.x0,
        jac=problem.gradient,
        constraints=problem.constraints(),
        bounds=problem.bounds,
        options={
            "print_level": 0,
            "max_iter": attack.max_iter,
            "tol": 1e-6,
            "acceptable_tol": 1e-5,
            "acceptable_iter": 10,
            "mu_strategy": "adaptive",
        },
    )
    return problem.result(res.x, int(res.status))


class _WindowProblem:
    """Eq. (12)'s window problem: variables [Vm, Va] of the area buses per snapshot."""

    def __init__(
        self,
        net: Wu26Network,
        states: list[tuple[np.ndarray, np.ndarray]],
        area: np.ndarray,
        targets: Sequence[tuple[int, int]],
        attack: Wu26Attack,
        freeze: Freeze,
    ) -> None:
        self.net, self.states, self.area, self.attack, self.freeze = net, states, area, attack, freeze
        self.T, self.nb = len(states), len(area)
        self.lines = [net.branch_of(a, b) for a, b in targets]
        self.flow_rows = [net.flow_rows(k, e) for k, e in self.lines]
        self.z_true = [net.measure(Vm, Va) for Vm, Va in states]
        # eq. (25)'s S_max: rho times each line's true flow at the window's end (ours; no rating is stated)
        end_Vm, end_Va = states[-1]
        self.smax = [abs(net.flow(end_Vm, end_Va, k, e)) * attack.rho for k, e in self.lines]
        self.bounds, self.x0 = self._bounds()
        self._cache: dict[int, tuple[tuple[int, bytes], np.ndarray, Optional[np.ndarray]]] = {}

    def _bounds(self) -> tuple[list[tuple[float, float]], np.ndarray]:
        """The per-snapshot trust region around the true state, the voltage band (eq. 21), and each
        trusted bus fixed at its true state plus its held offset (E1)."""
        a, lo, hi = self.attack, [], []
        x0 = np.empty(2 * self.nb * self.T)
        for t, (Vm, Va) in enumerate(self.states):
            held = self.freeze.get(t, {})
            vm_lo = np.maximum(a.vmin, Vm[self.area] - a.dv)
            vm_hi = np.minimum(a.vmax, Vm[self.area] + a.dv)
            va_lo, va_hi = Va[self.area] - a.da, Va[self.area] + a.da
            for i, b in enumerate(self.area):
                if int(b) in held:
                    dvm, dva = held[int(b)]
                    vm_lo[i] = vm_hi[i] = min(max(Vm[b] + dvm, a.vmin), a.vmax)
                    va_lo[i] = va_hi[i] = Va[b] + dva
            lo += [*vm_lo, *va_lo]
            hi += [*vm_hi, *va_hi]
            x0[self._slice(t)] = np.concatenate(
                [np.clip(Vm[self.area], vm_lo, vm_hi), np.clip(Va[self.area], va_lo, va_hi)]
            )
        return list(zip(lo, hi)), x0

    def _slice(self, t: int) -> slice:
        return slice(2 * self.nb * t, 2 * self.nb * (t + 1))

    def _state(self, x: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        Vm, Va = self.states[t][0].copy(), self.states[t][1].copy()
        xt = x[self._slice(t)]
        Vm[self.area], Va[self.area] = xt[: self.nb], xt[self.nb :]
        return Vm, Va

    def _eval(self, x: np.ndarray, t: int, jac: bool = False) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Snapshot t's readings (and their Jacobian), cached on its slice of x: IPOPT asks for the
        objective, its gradient and every goal constraint at the same point."""
        key = (t, x[self._slice(t)].tobytes())
        hit = self._cache.get(t)
        if hit is None or hit[0] != key or (jac and hit[2] is None):
            Vm, Va = self._state(x, t)
            J = self.net.jacobian(Vm, Va, self.area) if jac else None
            hit = (key, self.net.measure(Vm, Va), J)
            self._cache[t] = hit
        return hit[1], hit[2]

    def _dz(self, x: np.ndarray) -> np.ndarray:
        return np.stack([self._eval(x, t)[0] - self.z_true[t] for t in range(self.T)])

    def objective(self, x: np.ndarray) -> float:
        """l1(sum_t a_t) + tau sum_t l1(a_t), both in noise units, smoothed."""
        u = self._dz(x) / self.net.sigma
        net = u.sum(axis=0)
        return float(np.sqrt(net**2 + _EPS).sum() + self.attack.tau * np.sqrt(u**2 + _EPS).sum())

    def gradient(self, x: np.ndarray) -> np.ndarray:
        u = self._dz(x) / self.net.sigma
        net = u.sum(axis=0)
        g_net = net / np.sqrt(net**2 + _EPS)
        g = np.empty_like(x)
        for t in range(self.T):
            w = (g_net + self.attack.tau * u[t] / np.sqrt(u[t] ** 2 + _EPS)) / self.net.sigma
            g[self._slice(t)] = w @ cast(np.ndarray, self._eval(x, t, jac=True)[1])
        return g

    def constraints(self) -> list[dict[str, object]]:
        """Eqs. (24)-(25) as written [E19]: each target's flow at the window's end at least its S_max, and
        the displayed flow never falling from one snapshot to the next (S_t = S_t-1 + dS_t, dS_t >= 0);
        no per-snapshot target. Squared magnitudes, as IPOPT's inequalities (>= 0)."""
        lines = range(len(self.lines))
        end = [
            {"type": "ineq", "fun": partial(self._end, i), "jac": partial(self._end_grad, i)} for i in lines
        ]
        rise = [
            {"type": "ineq", "fun": partial(self._rise, t, i), "jac": partial(self._rise_grad, t, i)}
            for t in range(1, self.T)
            for i in lines
        ]
        return end + rise

    def _flow2(self, t: int, i: int, x: np.ndarray) -> float:
        """Snapshot t's squared apparent flow on target line i, as its SCADA meter reads it."""
        p_row, q_row = self.flow_rows[i]
        z = self._eval(x, t)[0]
        return float(z[p_row] ** 2 + z[q_row] ** 2)

    def _flow2_grad(self, t: int, i: int, x: np.ndarray) -> np.ndarray:
        """The gradient of `_flow2` on snapshot t's block."""
        p_row, q_row = self.flow_rows[i]
        z, J = self._eval(x, t, jac=True)
        J = cast(np.ndarray, J)
        return 2 * z[p_row] * J[p_row] + 2 * z[q_row] * J[q_row]

    def _end(self, i: int, x: np.ndarray) -> float:
        return self._flow2(self.T - 1, i, x) - self.smax[i] ** 2

    def _end_grad(self, i: int, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        out[self._slice(self.T - 1)] = self._flow2_grad(self.T - 1, i, x)
        return out

    def _rise(self, t: int, i: int, x: np.ndarray) -> float:
        return self._flow2(t, i, x) - self._flow2(t - 1, i, x)

    def _rise_grad(self, t: int, i: int, x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        out[self._slice(t)] = self._flow2_grad(t, i, x)
        out[self._slice(t - 1)] = -self._flow2_grad(t - 1, i, x)
        return out

    def result(self, x: np.ndarray, status: int) -> WindowAttack:
        dz = self._dz(x)
        offsets = np.empty((self.T, self.nb, 2))
        for t in range(self.T):
            xt = x[self._slice(t)]
            offsets[t, :, 0] = xt[: self.nb] - self.states[t][0][self.area]
            offsets[t, :, 1] = xt[self.nb :] - self.states[t][1][self.area]
        lines = range(len(self.lines))
        feasible = all(self._end(i, x) > -1e-4 for i in lines) and all(
            self._rise(t, i, x) > -1e-4 for t in range(1, self.T) for i in lines
        )
        net = self.net
        area = tuple(int(b) for b in self.area)
        return WindowAttack(dz, offsets, net.sigma, net.dev_type, net.bus, area, feasible, status)


def incremental_freeze(order: Sequence[int], slots: Sequence[int], undefended: WindowAttack) -> Freeze:
    """[WU26 eqs. 29-30] read incrementally (E1): the PMU at bus `order[i]` (0-based) trusted from snapshot
    `slots[i]` (0-based) keeps the offset its bus had in the undefended attack at snapshot slots[i] - 1
    (none before the window's first snapshot) for every later snapshot."""
    pos = {b: i for i, b in enumerate(undefended.area)}
    freeze: Freeze = {t: {} for t in range(len(undefended.dz))}
    for b, s in zip(order, slots):
        held: tuple[float, float] = (0.0, 0.0)
        if s > 0 and b in pos:
            held = (float(undefended.offsets[s - 1, pos[b], 0]), float(undefended.offsets[s - 1, pos[b], 1]))
        for t in range(s, len(undefended.dz)):
            freeze[t][int(b)] = held
    return freeze
