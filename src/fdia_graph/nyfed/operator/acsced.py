"""The operator's real-time dispatch as an AC optimal power flow (`AcSced`), scan by scan, beside the DC `DcSced`.

Each scan the operator solves one AC OPF (`OperatorOpf`) at the loads it reads: the true loads plus what the attack
changed in its New York injection meters (the external areas, which it does not meter, at their basepoint). The units
move within their limits and five-minute ramps from its previous dispatch; nuclear holds its output, hydro and the
external units hold theirs under the pool's quadratic penalty (N39), so their schedules stand; every interface's AC flow,
corrected by what the operator read after its last dispatch (the closed loop), stays within its limit or pays for the
excess; load may be shed where shedding relieves an interface (`SHED_FACTOR` of its AC sensitivity) after `SHED_AFTER`
scans over; and the reference bus's balance may be left short or over at a price. Two objectives: "movement" moves the
least generation (the least cost breaking ties), "cost" clears at the least linear cost with each MW moved charged
`MOVE_USD` and the shortage curve and energy cap of `DcSced`, and prices every bus at its AC balance dual (the AC LMP).

On a scan with nothing to answer (the loads read are the true ones, nothing to correct, no dispatch to unwind, every
interface within its limit) either objective's answer is the AC point itself, since the operator solves the problem the
pool solved (`PoolFrames`), so it is returned without a solve; prices are then only where the scan is solved.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from fdia_graph.errors import ConfigError
from fdia_graph.nyfed.grid.case import REF, Case
from fdia_graph.nyfed.grid.interfaces import Interface
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.day import PoolFrames
from fdia_graph.nyfed.operator.sced import (
    FRAME_HOURS,
    HELD,
    MOVE_USD,
    NEEDED_MW,
    SHED_FACTOR,
    SHED_WEIGHT,
    SHORTAGE,
    SLACK_WEIGHT,
    Step,
    loss_factors,
)
from fdia_graph.nyfed.scenario.solver import OperatorOpf, OpfSolution

AC_OBJECTIVES = ("movement", "cost")
ANCHOR_WEIGHT = (
    1000.0  # per MW^2: the pool's penalty holding hydro and external units to their schedule (N39)
)
TIE_BREAK = 1e-3  # the least-movement dispatch's weight on cost, so a dollar never outweighs a megawatt moved
SCALE = 1e-3  # the objective's scale for Ipopt
# per p.u.^2: units hold their bus's voltage at the AC point's schedule, as the true grid's power flow holds it, until
# their reactive range binds; without it the least-movement dispatch's voltages are free, and its model flows part
# from the true grid's by what the voltages carry
V_WEIGHT = 1e7
BOUND_TOL = 1e-4
BINDING = 1e-6  # an interface row's dual (objective units per MW) or excess (MW) above which it binds  # MW the pool's bounds widen by, so its own point stays inside past the solver's rounding


def _jacobian(case: Case, net: Network, vm: np.ndarray, va: np.ndarray):
    """The power flow Jacobian at a state, units' buses PV and the others PQ: (jac, angle buses, magnitude buses,
    dS/dva, dS/dvm, reference)."""
    n = len(case.bus_id)
    ref = int(np.flatnonzero(case.bus_type == REF)[0])
    v = vm * np.exp(1j * np.deg2rad(va))
    y = net.ybus.tocsc()
    i_bus, dv = y @ v, sp.diags(v)
    unit_v = sp.diags(v / np.abs(v))
    ds_dva = 1j * dv @ np.conj(sp.diags(i_bus) - y @ dv)
    ds_dvm = dv @ np.conj(y @ unit_v) + np.conj(sp.diags(i_bus)) @ unit_v
    pq = np.ones(n, bool)
    pq[case.unit_bus] = False
    pq[ref] = False
    ang, mag = np.flatnonzero(np.arange(n) != ref), np.flatnonzero(pq)
    jac = sp.vstack(
        [
            sp.hstack([ds_dva.real[ang][:, ang], ds_dvm.real[ang][:, mag]]),
            sp.hstack([ds_dva.imag[mag][:, ang], ds_dvm.imag[mag][:, mag]]),
        ]
    ).tocsc()
    return jac, ang, mag, ds_dva, ds_dvm, ref


def ac_shift(case: Case, vm: np.ndarray, va: np.ndarray, interfaces: list[Interface]) -> np.ndarray:
    """`[I, N]` each interface's AC flow change per MW injected at each bus and taken at the reference bus, at the AC
    state (`vm` p.u., `va` degrees), the units' buses holding their voltage: the power flow Jacobian, transposed,
    solved against each interface flow's gradient. Zero at the reference bus."""
    net = Network(case)
    jac, ang, mag, _, _, _ = _jacobian(case, net, vm, va)
    n = len(case.bus_id)
    th = np.deg2rad(va)
    out = np.zeros((len(interfaces), n))
    yf = net.yf.tocsr()
    lu = spl.splu(jac.T.tocsc())
    for k, it in enumerate(interfaces):
        g_th, g_vm = np.zeros(n), np.zeros(n)
        for l, s in zip(it.branches, it.signs):
            f, t = int(case.f[l]), int(case.t[l])
            gff, gft, bft = yf[l, f].real, yf[l, t].real, yf[l, t].imag
            d = th[f] - th[t]
            dth = vm[f] * vm[t] * (-gft * np.sin(d) + bft * np.cos(d))
            g_th[f] += s * dth
            g_th[t] -= s * dth
            g_vm[f] += s * (2 * vm[f] * gff + vm[t] * (gft * np.cos(d) + bft * np.sin(d)))
            g_vm[t] += s * vm[f] * (gft * np.cos(d) + bft * np.sin(d))
        w = lu.solve(np.r_[g_th[ang], g_vm[mag]])
        out[k, ang] = w[: len(ang)]
    return out


def ac_power_flow(
    case: Case, net: Network, vm: np.ndarray, va: np.ndarray, p: np.ndarray, q: np.ndarray, tol: float = 1e-8
) -> tuple[np.ndarray, np.ndarray] | None:
    """Newton's method for the AC power flow on `net` from the state (`vm` p.u., `va` degrees): every bus but the
    reference holds its injection `p [N]` (MW) and every bus without a unit its `q [N]` (MVAr), the units' buses hold
    their voltage. Returns (vm, va) or None when it does not converge in 20 iterations (an outage that islands)."""
    base = case.base_mva
    vm, va = vm.copy(), np.deg2rad(va).copy()
    for _ in range(20):
        jac, ang, mag, _, _, _ = _jacobian(case, net, vm, np.rad2deg(va))
        v = vm * np.exp(1j * va)
        s = v * np.conj(net.ybus @ v)
        mis = np.r_[s.real[ang] - p[ang] / base, s.imag[mag] - q[mag] / base]
        if np.abs(mis).max() < tol:
            return vm, np.rad2deg(va)
        try:
            dx = spl.spsolve(jac, -mis)
        except RuntimeError:
            return None
        if not np.isfinite(dx).all():
            return None
        va[ang] += dx[: len(ang)]
        vm[mag] += dx[len(ang) :]
    return None


def open_branch(case: Case, k: int) -> Case:
    """`case` with branch `k` open: no series admittance and no charging."""
    x = case.x.copy()
    r, b = case.r.copy(), case.b.copy()
    x[k], r[k], b[k] = 1e12, 0.0, 0.0
    return replace(case, x=x, r=r, b=b)


class AcSced:
    """The operator's AC dispatch over one day's frames: the case, the pool basepoint `unit_p [T, G]`, the units'
    `cost [T, G]` ($/MWh), the `interfaces` it enforces, the true AC state `vm`, `va [T, N]` (the AC points), the true
    bus loads `pd`, `qd [T, N]` (MW, MVAr, `TrueGrid`'s), the load it may shed `bus_load [T, N]`, the quick-start units
    and the `objective` ("movement" or "cost"). `step` solves one scan."""

    def __init__(
        self,
        case: Case,
        unit_p: np.ndarray,
        cost: np.ndarray,
        interfaces: list[Interface],
        state: tuple[np.ndarray, np.ndarray],
        loads: tuple[np.ndarray, np.ndarray],
        frames: PoolFrames,
        bus_load: np.ndarray | None = None,
        quick_start: np.ndarray | None = None,
        objective: str = "movement",
        solve_quiet: bool = False,
    ):
        if objective not in AC_OBJECTIVES:
            raise ConfigError(f"objective must be one of {AC_OBJECTIVES}, not {objective!r}")
        self.case, self.unit_p, self.cost, self.objective = case, unit_p, cost, objective
        self.interfaces = interfaces
        self.vm, self.va = state
        self.pd, self.qd = loads
        self.frames = frames
        self.opf = OperatorOpf(case, [(it.branches, it.signs) for it in interfaces])
        fuel = case.fuel.astype(str)
        self.free = ~np.isin(fuel, HELD)
        self.nuclear = fuel == "Nuclear"
        self.ramp = np.where(np.isfinite(case.ramp), case.ramp * 5.0, np.inf)  # the pool's, MW per scan
        self.quick = np.zeros(len(self.free), bool) if quick_start is None else np.asarray(quick_start, bool)
        self.quick_ramp = np.where(self.quick, np.maximum(self.ramp, case.pmax / 2.0), self.ramp)  # N64
        self.bus_load = np.zeros_like(self.pd) if bus_load is None else bus_load
        net = Network(case)
        self.flow0 = np.array(
            [it.model_flow(net.flows(self.vm, self.va)[0]) for it in interfaces]
        ).T  # [T, I]
        ub = self.opf.unit_buses
        # each unit bus's reactive range: its units' together, as the pool's dispatch limits them (N38)
        self.q_lo = np.array([np.nan_to_num(case.qmin[case.unit_bus == b]).sum() for b in ub])
        self.q_hi = np.array([np.nan_to_num(case.qmax[case.unit_bus == b]).sum() for b in ub])
        self.beta = loss_factors(case, self.vm, self.va)
        self._last: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._shift: dict[int, np.ndarray] = {}
        self.solves, self.seconds = 0, 0.0
        self.solve_quiet = solve_quiet
        self.v_w = np.zeros(len(case.bus_id))
        self.v_w[case.unit_bus] = V_WEIGHT  # the units' buses regulate
        self.solved_at: set[int] = set()  # the scans solved, not answered by the basepoint

    def _bounds(
        self, t: int, prev: np.ndarray | None, commit: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every unit's output bounds at scan `t` (MW), its movement weight and its anchor (MW), as the pool bounded
        the frame (`PoolFrames`): free units within the pool's bounds clipped to their ramp from the previous dispatch as
        the pool clips them (from the pool's previous frame where the run has no previous dispatch, held at the
        basepoint at the day's first frame), nuclear at its schedule, hydro and external units within their bounds under
        the anchor. The least-cost dispatch anchors them to their schedule, as the pool did; the least-movement one to
        the basepoint, so the anchors never pull it. With `commit` a quick-start unit the AC point has off may start,
        half its range per scan (N64); without, it stays off. `BOUND_TOL` admits the basepoint past the solver's
        rounding."""
        f, p0 = self.frames, self.unit_p[t]
        lo, hi = f.p_lo[t].copy(), f.p_hi[t].copy()
        if prev is not None:
            was = self.unit_p[t - 1] + prev
        elif t > 0 and np.isfinite(self.unit_p[t - 1]).all():
            was = self.unit_p[t - 1]
        else:
            was = p0
        off = self.quick & (f.p_hi[t] <= 0.0)
        if commit:
            hi = np.where(off, np.nan_to_num(self.case.pmax), hi)
        ramp = np.where(off & commit, self.quick_ramp, self.ramp)
        if was is p0:  # no history at all: the basepoint is the only point the ramp is known to reach
            ramp = np.zeros_like(ramp)
        lo = np.where(self.free, np.clip(lo, was - ramp, was + ramp), lo) - BOUND_TOL
        hi = np.where(self.free, np.clip(hi, was - ramp, was + ramp), hi) + BOUND_TOL
        weight = np.where(self.free, 1.0 if self.objective == "movement" else MOVE_USD, 0.0)
        anchor = f.sp[t] if self.objective == "cost" else p0
        return lo, hi, weight, anchor

    def _sensitivity(self, t: int) -> np.ndarray:
        if t not in self._shift:
            self._shift[t] = ac_shift(self.case, self.vm[t], self.va[t], self.interfaces)
        return self._shift[t]

    def _solve(
        self, t, load_p, load_q, corr, limit, prev, shed_ok, enforce: bool, commit: bool
    ) -> OpfSolution | None:
        import time

        c = self.case
        lo, hi, move_w, anchor = self._bounds(t, prev, commit)
        on = np.isfinite(limit) & enforce
        lim = np.where(on, limit, np.nan)
        may = (
            (np.broadcast_to(np.asarray(shed_ok, bool), on.shape) & on)
            if enforce
            else np.zeros(len(on), bool)
        )
        shed_ub = np.zeros(len(c.bus_id))
        if may.any():
            sens = self._sensitivity(t)
            ok = (sens[may] <= SHED_FACTOR).any(axis=0) & (self.bus_load[t] > 0)
            shed_ub = np.where(ok, self.bus_load[t], 0.0)
        qratio = np.divide(load_q, load_p, out=np.zeros_like(load_p), where=load_p > 1e-6)
        if self.objective == "cost":
            seg_w = tuple(w for w, _ in SHORTAGE)
            seg_p = tuple(p for _, p in SHORTAGE)
            # load left unmatched only where nothing else serves it: the pool had no such slack, and its anchored units
            # can price energy above the offer cap where their schedules bind (the AC LMP is then that marginal value)
            w_cost, bal_w = 1.0, SLACK_WEIGHT
        else:
            seg_w, seg_p = (np.inf,) * 3, (SLACK_WEIGHT,) * 3
            w_cost, bal_w = TIE_BREAK, SLACK_WEIGHT
        start = self._last if self._last is not None else (self.vm[t], self.va[t], self.unit_p[t])
        anchor_w = np.where(~self.free & ~self.nuclear, ANCHOR_WEIGHT, 0.0)
        clock = time.perf_counter()
        sol = self.opf.solve(
            load_p,
            load_q,
            p0=self.unit_p[t],
            p_lo=lo,
            p_hi=hi,
            q_lo=self.q_lo,
            q_hi=self.q_hi,
            anchor=anchor,
            v_sched=self.vm[t],
            v_w=self.v_w,
            cost=self.cost[t],
            move_w=move_w,
            anchor_w=anchor_w,
            shed_ub=shed_ub,
            shed_w=SHED_WEIGHT,
            qratio=qratio,
            corr=corr,
            lim=lim,
            seg_width=seg_w,
            seg_price=seg_p,
            w_cost=w_cost,
            bal_w=bal_w,
            scale=SCALE,
            start=start,
        )
        self.solves += 1
        self.seconds += time.perf_counter() - clock
        if sol is None and self._last is not None:  # once more from the AC point itself
            self._last = None
            return self._solve(t, load_p, load_q, corr, limit, prev, shed_ok, enforce, commit)
        if sol is not None:
            self._last = (sol.vm, sol.va, sol.unit_p)
        return sol

    def _basepoint(self, t: int) -> Step:
        G, N, I = len(self.free), len(self.case.bus_id), len(self.interfaces)
        return Step(
            np.zeros(G),
            np.zeros(N),
            np.zeros(I),
            0.0,
            0.0,
            np.full(I, np.nan),
            float("nan"),
            flow=self.flow0[t].copy(),
            price=None,
        )

    def step(
        self,
        t: int,
        load_p: np.ndarray,
        load_q: np.ndarray,
        corr: np.ndarray,
        limit: np.ndarray,
        prev: np.ndarray | None,
        shed_ok: bool | np.ndarray = True,
        force: bool = False,
    ) -> Step:
        """Scan `t`: the bus loads the operator reads `load_p`, `load_q [N]`, each interface's reading correction
        `corr [I]` (MW: what it read after its last dispatch beyond its own model, plus any what-if push) against
        `limit [I]` (NaN unenforced), and `prev [G]` its deviation at the scan before (None at the first). A quiet scan
        (the true loads, nothing to correct or unwind, every interface within its limit) returns the basepoint without
        a solve, as either objective's answer there is the AC point the pool solved; `force` (or `solve_quiet`) solves it,
        for its prices. Raises RuntimeError where Ipopt finds no dispatch."""
        on = np.isfinite(limit)
        quiet = (
            not (force or self.solve_quiet)
            and np.array_equal(load_p, self.pd[t])
            and np.array_equal(load_q, self.qd[t])
            and np.abs(np.nan_to_num(corr)).max(initial=0.0) < 1e-9
            and (prev is None or np.abs(prev).max(initial=0.0) < 1e-9)
            and bool((self.flow0[t][on] <= limit[on]).all())
        )
        if quiet:
            self._last = None
            return self._basepoint(t)
        self.solved_at.add(t)
        commit = self.objective == "movement"
        sol = self._solve(t, load_p, load_q, corr, limit, prev, shed_ok, True, commit)
        if self.objective == "cost" and sol is not None:
            short = float(np.nansum(sol.over)) + float(sol.shed.sum()) + abs(sol.unmatched)
            if short > NEEDED_MW:  # the committed units leave a constraint unmet: start the quick-start units
                commit = True
                sol = self._solve(t, load_p, load_q, corr, limit, prev, shed_ok, True, commit)
        if sol is None:
            raise RuntimeError(f"scan {t}: the AC dispatch found no optimum")
        dev = sol.unit_p - self.unit_p[t]
        dev[np.abs(dev) < 1e-7] = 0.0
        relief = 0.0
        # the dispatch without interface limits differs only where an interface row binds: elsewhere it is this one
        binding = bool(
            (np.abs(np.nan_to_num(sol.congestion)) > BINDING).any() or np.nansum(sol.over) > BINDING
        )
        if np.abs(dev).sum() > 1e-6 and on.any() and binding:
            # the same dispatch with the same commitment, only without the interface limits; it starts no scan after it
            last = self._last
            free = self._solve(t, load_p, load_q, corr, limit, prev, shed_ok, False, commit)
            self._last = last
            if free is not None:
                relief = float(self.cost[t] @ (sol.unit_p - free.unit_p)) * FRAME_HOURS
        unrelieved = np.where(on, sol.over, np.nan)
        if self.objective == "cost":
            # the balance dual reads per p.u. of the objective; per MW and per dollar it is the bus's AC LMP
            price = sol.price / SCALE
            congestion = np.where(on, sol.congestion / SCALE, np.nan)
            energy = float(price[self.opf.ref])
        else:
            price, congestion, energy = None, np.where(on, 0.0, np.nan), float("nan")
        return Step(
            dev, sol.shed, unrelieved, sol.unmatched, relief, congestion, energy, flow=sol.flow, price=price
        )

    def delivered(self, t: int, dev: np.ndarray, shed: np.ndarray) -> float:
        """What the change `dev [G]` and the shed `shed [N]` deliver at the reference bus at scan `t` (MW), weighted by
        the AC delivery factors of the true state (`loss_factors`)."""
        return float(self.beta[t, self.case.unit_bus] @ dev + self.beta[t] @ shed)
