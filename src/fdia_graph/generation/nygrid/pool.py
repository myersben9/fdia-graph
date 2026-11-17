"""The pool: one AC operating state per 5-minute frame, from a ramp-constrained real-time dispatch."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from fdia_graph.constants.newyork import HQ_BUS, SolveKind
from fdia_graph.constants.time import HALF_HOUR
from fdia_graph.errors import DataConditionError, NotFitted
from fdia_graph.generation.nygrid.case import Case
from fdia_graph.generation.nygrid.network import Network

if TYPE_CHECKING:  # the pool record is read without pandapower, so the solver is only a type here
    from fdia_graph.generation.nygrid.dispatch import Dispatch
    from fdia_graph.generation.nygrid.loads import LoadSeries
    from fdia_graph.generation.nygrid.solver import AcSolver


@dataclass(frozen=True)
class Pool:
    """Per frame `time [T]` (NYISO's local clock): the state `vm [T, N]` p.u. and `va [T, N]` degrees, the
    net bus injections `p`, `q [T, N]` (MW, MVAr) the admittances give for it (`Network.injections`, so
    shunts belong to the network), `solve_kind [T]` and each unit's dispatch basepoint `unit_p [T, G]` MW
    (None when the file has none). A frame that is not SOLVED keeps its place on the 5-minute time axis with
    NaN state. Branch flows are `Network(case).flows(vm, va)`."""

    time: np.ndarray
    vm: np.ndarray
    va: np.ndarray
    p: np.ndarray
    q: np.ndarray
    solve_kind: np.ndarray
    unit_p: np.ndarray | None = None


class PoolBuilder:
    """Solves every frame of `loads` as NYISO's real-time dispatch would, in time order: a least-cost AC OPF
    (`AcSolver.dispatch`, at `dispatch.unit_cost`) on the frame's bus loads less NYgrid's netted renewables
    `renew [H, N]`, then an AC power flow at the OPF's basepoints. Thermal units move within their limits and
    within `ramp [G]` (MW per frame, inf where unbounded) of the previous basepoint; nuclear, hydro and
    external units follow NYgrid's hourly AC points (`AcSolver.project`), since hydro is energy limited and
    interchange is scheduled. Unit Q stays within `q_cap [G]` (MVAr) or the solver's `Compensation`, under
    which the pool's `q` includes the banks' output `shunt_q`."""

    def __init__(
        self,
        solver: AcSolver,
        loads: LoadSeries,
        dispatch: Dispatch,
        renew: np.ndarray,
        ramp: np.ndarray,
        q_cap: np.ndarray | None = None,
    ):
        self.solver, self.loads, self.dispatch, self.renew = solver, loads, dispatch, renew
        self.ramp, self.q_cap = np.asarray(ramp, float), q_cap
        self.hour_points: dict[int, dict | None] = {}  # projected hours
        self.pool: Pool | None = None
        self.seconds = 0.0
        T = len(loads.time)
        self.slack_mw = np.full(T, np.nan)  # reference bus output in the power flow at the OPF basepoints
        self.cost = np.full(T, np.nan)  # $/h of the frame's dispatch
        self.ramp_bound = np.full(T, np.nan)  # share of the free ramp-limited units held at a ramp limit
        self.q_fallback = np.zeros(T, bool)  # solved only with the case's reactive limits
        self.shunt_q = np.full((T, len(solver.case.bus_id)), np.nan)  # MVAr each bank injects (Compensation)
        fuel = solver.case.fuel
        self._held = np.isin(fuel, ("Nuclear", "Hydro", "Import"))  # follow NYgrid's hourly output
        self._fixed = fuel == "Nuclear"  # exactly
        # Hydro and external units hold the schedule under a quadratic penalty (`dispatch(anchor=...)`): held
        # exactly, the external areas cannot cover their own losses and the morning pickup is infeasible.
        self._anchored = self._held & ~self._fixed  # under the penalty
        self.anchor_dev = np.full(T, np.nan)  # MW, the anchored units' total |output - schedule|
        self.final: tuple[np.ndarray, ...] | None = None  # last solved (unit_p, unit_q, vm, va)

    def _brackets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """For every frame, the hour before, the hour after and the weight of the hour after."""
        mid = (self.dispatch.hour.astype("datetime64[s]") + HALF_HOUR).astype(np.int64).astype(float)
        x = self.loads.time.astype("datetime64[s]").astype(np.int64).astype(float)
        pos = np.interp(x, mid, np.arange(len(mid)))  # clamps to the first and last hour at the ends
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(mid) - 1)
        return lo, hi, pos - lo

    def _project(self, h: int) -> dict | None:
        if h not in self.hour_points:
            d = self.dispatch
            self.hour_points[h] = self.solver.project(
                d.pd[h], d.qd[h], d.unit_p[h], d.unit_on[h], d.hvdc_p[h]
            )
        return self.hour_points[h]

    def _limits(self, lo: int, hi: int, w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """A frame's lower and upper output per unit (MW) before the ramp, and the setpoints interpolated between
        the two hours around it (each hourly point sits at its half hour). A unit at zero in both hours is off,
        at zero in one may run from zero to Pmax, else Pmin to Pmax widened to admit the setpoint."""
        case, d = self.solver.case, self.dispatch
        a, b = d.unit_p[lo], d.unit_p[hi]
        sp = (1 - w) * a + w * b
        pa, pb = self.hour_points.get(lo), self.hour_points.get(hi)
        if pa is not None and pb is not None:
            sp = np.where(self._held, (1 - w) * pa["unit_p"] + w * pb["unit_p"], sp)
        both, either = (a != 0) & (b != 0), (a != 0) | (b != 0)
        p_lo = np.where(both, np.minimum(case.pmin, sp), np.where(either, np.minimum(0.0, sp), 0.0))
        p_hi = np.where(either, np.maximum(case.pmax, sp), 0.0)
        # external units 0 to 1.5 times the setpoint (updateOpCond.m L360); the HQ import keeps its
        # -1100 to 1500 MW widened to the schedule, as NYgrid appends it after that line (L367-377)
        imp = (case.fuel == "Import") & (case.bus_id[case.unit_bus] != HQ_BUS) & (sp > 0)
        p_lo[imp], p_hi[imp] = 0.0, 1.5 * sp[imp]
        p_lo[self._fixed] = p_hi[self._fixed] = sp[self._fixed]
        return p_lo, p_hi, sp

    def build(self, start: tuple[np.ndarray, ...] | None = None) -> Pool:
        """Dispatch and solve every frame in time order, from `start` = (unit_p, unit_q, vm, va), the `final` of
        the build before it, or else from the AC point of the first frame's hour. A frame infeasible under the
        capped unit Q is solved again with the case's limits and counted in `q_fallback`."""
        clock = time.perf_counter()
        case, d, s = self.solver.case, self.dispatch, self.solver
        lo, hi, w = self._brackets()
        T, n, g = len(self.loads.time), len(case.bus_id), len(case.unit_bus)
        vm, va, p, q, unit_p = (np.full((T, k), np.nan) for k in (n, n, n, n, g))
        kind = np.full(T, SolveKind.FAILED, dtype=np.int8)
        held, fixed, anchored = self._held, self._fixed, self._anchored
        # every hourly AC point first, as projecting moves the solver's state
        for h in np.unique(np.r_[lo, hi]):
            self._project(int(h))
        if start is not None:
            prev, prev_q = np.array(start[0], float), np.array(start[1], float)
            s.vm, s.va = np.array(start[2], float), np.array(start[3], float)
        else:
            first = self._project(int(lo[0]))
            prev = None if first is None else first["unit_p"].copy()
            prev_q = None if first is None else first["unit_q"].copy()
            if first is not None:
                s.vm, s.va = first["vm"].copy(), first["va"].copy()
        if prev is None or prev_q is None:  # the first hour's AC point seeds the dispatch
            raise DataConditionError(
                f"frame {int(lo[0])}: the span's first hour has no AC point to start from"
            )
        q_lo, q_hi = (case.qmin, case.qmax) if self.q_cap is None else (-self.q_cap, self.q_cap)
        comp = s.compensation
        if comp is not None:  # the sensitivity's unit reactive limits
            q_lo, q_hi = comp.q_lo, comp.q_hi
        capped = self.q_cap is not None or comp is not None
        for f in range(T):
            pd_f = self.loads.p[f] - ((1 - w[f]) * self.renew[lo[f]] + w[f] * self.renew[hi[f]])
            if np.isnan(pd_f).any() or np.isnan(self.loads.q[f]).any():
                kind[f] = SolveKind.NO_LOAD
                continue
            if prev is None:
                kind[f] = SolveKind.NO_HOUR
                continue
            base_lo, base_hi, sp = self._limits(lo[f], hi[f], w[f])
            p_lo = np.clip(base_lo, prev - self.ramp, prev + self.ramp)
            p_hi = np.clip(base_hi, prev - self.ramp, prev + self.ramp)
            p_lo[held], p_hi[held] = base_lo[held], base_hi[held]  # no ramp: the schedule moves them
            p_lo[fixed], p_hi[fixed] = sp[fixed], sp[fixed]
            anchor = np.where(anchored, sp, np.nan)
            # every unit stays in service for Q, as in NYgrid's case (GEN_STATUS 1): without the reactive
            # support of units at zero output, evening peaks are infeasible at the 0.9 p.u. voltage floor
            # (535 frames in March and August)
            on = np.ones(len(prev), bool)
            hvdc_p = (1 - w[f]) * d.hvdc_p[lo[f]] + w[f] * d.hvdc_p[hi[f]]
            cost = (1 - w[f]) * d.unit_cost[lo[f]] + w[f] * d.unit_cost[hi[f]]
            qd_f, x0 = self.loads.q[f], (prev, prev_q)  # x0: the solve starts at the previous frame
            r = s.dispatch(pd_f, qd_f, cost, p_lo, p_hi, q_lo, q_hi, on, hvdc_p, init=x0, anchor=anchor)
            if r is None and capped:
                self.q_fallback[f] = True
                q_min, q_max = case.qmin, case.qmax  # the units' whole reactive range
                r = s.dispatch(pd_f, qd_f, cost, p_lo, p_hi, q_min, q_max, on, hvdc_p, init=x0, anchor=anchor)
            if r is None:
                continue
            pf = s.power_flow(
                pd_f,
                self.loads.q[f] - r["shunt_q"],  # a bank's output as a reactive injection at its bus
                r["unit_p"],
                on,
                r["vm"][case.unit_bus],
                hvdc_p,
                start=(r["vm"], r["va"]),
            )
            if pf is None:
                continue
            kind[f] = SolveKind.SOLVED
            vm[f], va[f] = pf["vm"], pf["va"]
            p[f], q[f] = s.network.injections(pf["vm"], pf["va"])
            unit_p[f] = r["unit_p"]
            self.shunt_q[f] = r["shunt_q"]
            self.slack_mw[f] = float(s.net.res_ext_grid.p_mw.sum())
            self.cost[f] = r["cost"]
            self.anchor_dev[f] = float(np.abs(r["unit_p"] - sp)[anchored].sum())
            free = on & np.isfinite(self.ramp) & (p_hi - p_lo > 1e-6)
            at_ramp = free & (
                ((p_lo > base_lo + 1e-6) & (r["unit_p"] < p_lo + 1e-3))
                | ((p_hi < base_hi - 1e-6) & (r["unit_p"] > p_hi - 1e-3))
            )
            self.ramp_bound[f] = at_ramp.sum() / max(free.sum(), 1)
            prev, prev_q = r["unit_p"], r["unit_q"]
            self.final = (prev.copy(), prev_q.copy(), pf["vm"].copy(), pf["va"].copy())
        self.seconds = time.perf_counter() - clock
        self.pool = Pool(time=self.loads.time, vm=vm, va=va, p=p, q=q, solve_kind=kind, unit_p=unit_p)
        return self.pool

    def report(self) -> dict:
        """The last build in numbers: counts per solve kind, the voltage range, the reference bus output, the
        dispatch cost, the share of units at a ramp limit, the `q_fallback` frames and the time per frame."""
        pool, case = self.pool, self.solver.case
        if pool is None:
            raise NotFitted("report() describes the last build: call build() first")
        ok = pool.solve_kind == SolveKind.SOLVED
        kinds = {
            name: int((pool.solve_kind == k).sum())
            for name, k in (
                ("solved", SolveKind.SOLVED),
                ("failed", SolveKind.FAILED),
                ("no_hour", SolveKind.NO_HOUR),
                ("no_load", SolveKind.NO_LOAD),
            )
        }
        slack = np.abs(self.slack_mw[ok])
        any_ok = bool(ok.any())
        failed = [str(self.dispatch.hour[h]) for h, r in sorted(self.hour_points.items()) if r is None]
        return {
            "frames": len(pool.time),
            **kinds,
            "solved_pct": round(100 * float(ok.mean()), 3),
            "no_load_times": [str(t) for t in pool.time[pool.solve_kind == SolveKind.NO_LOAD]],
            "hours_projected": len(self.hour_points),
            "hours_failed": failed,
            "vm_min": float(np.nanmin(pool.vm[ok])) if any_ok else None,
            "vm_max": float(np.nanmax(pool.vm[ok])) if any_ok else None,
            "ny_vm_min": float(np.nanmin(pool.vm[ok][:, case.is_ny])) if any_ok else None,
            "ny_vm_max": float(np.nanmax(pool.vm[ok][:, case.is_ny])) if any_ok else None,
            "slack_mw_median": float(np.median(slack)) if any_ok else None,
            "slack_mw_p95": float(np.quantile(slack, 0.95)) if any_ok else None,
            "slack_mw_max": float(slack.max()) if any_ok else None,
            "cost_median": float(np.median(self.cost[ok])) if any_ok else None,
            "anchor_dev_mw_median": float(np.median(self.anchor_dev[ok])) if any_ok else None,
            "anchor_dev_mw_max": float(np.max(self.anchor_dev[ok])) if any_ok else None,
            "ramp_bound_share": float(np.mean(self.ramp_bound[ok])) if any_ok else None,
            "q_fallback_frames": int(self.q_fallback.sum()),
            "seconds": round(self.seconds, 1),
            "ms_per_frame": round(1000 * self.seconds / max(len(pool.time), 1), 2),
        }


def peak_ratings(case: Case, pool: Pool, factor: float = 1.0, chunk: int = 8760) -> Case:
    """`case` with every unrated branch (rating 0) rated at `factor` times its peak apparent flow (MVA, the
    larger end) over the pool's solved frames; at 1.0 a line's busiest scan of the year sits at its rating.
    A public rating is kept. Frames go in chunks so a year of flows is never in memory at once."""
    net = Network(case)
    solved = np.flatnonzero(pool.solve_kind == SolveKind.SOLVED)
    peak = np.zeros(len(case.f))
    for i in range(0, len(solved), chunk):
        k = solved[i : i + chunk]
        pf, qf, pt, qt = net.flows(pool.vm[k], pool.va[k])
        peak = np.maximum(peak, np.maximum(np.hypot(pf, qf), np.hypot(pt, qt)).max(axis=0))
    rating = np.where(case.rating > 0, case.rating, factor * peak)
    return replace(case, rating=rating)
