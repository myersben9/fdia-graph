"""The attacker's false states from a local AC optimal power flow (N35): the support search and the per-scan
OPF every OPF attack shares. `OpfFdia` is their base class; a subclass adds the goal and `design`
(`interface.InterfaceFdia`, the one the release builds).

Search. Candidate supports (the buses whose voltage moves) grow from the goal's end buses inside the attacker's
area, smallest first (`_candidates`). Each is solved once at the last rise scan with every static bound
(`_static`), and a support that passes is solved over the window scan by scan with the per-scan bounds
(`_sequential`). The subclass keeps the window that tampers the fewest meters ([1] eq. 12).

Each scan's OPF frees the support's voltages (every voltage outside is held true) under the AC power flow at
every bus of the support and its edge. Its objective is the pretended injection change beyond each injection
meter's noise, with a small quadratic term that keeps sub-noise changes from wandering.

Bounds, per scan and free bus (pretended change dP MW, dQ MVAr against the true injection):
- a load: |dP| <= cap P and |dQ| <= cap max(|Q|, 0.2 P) (D16, N24, N28); over the exit its active change moves
  per scan by at most the bus's step bound (N30);
- a generator: its reported output (the units' own, `gen_output`, N42) inside its limits, moving per scan by at
  most its five-minute ramp or the true step where the dispatch moves faster (N33), in total within its
  basepoint band `gen_band` (N33); its reactive output within `q_lo`, `q_hi` (N34) and, where `q_ramp` is
  given, that much per scan; load at the bus moves within the load cap;
- with `loads_only` (N43) only loads are pretended;
- every other New York bus of the support and its edge keeps its true injection; a bus outside New York on
  the edge absorbs, unless it has no unit and no injection (N44).

With `exit_scans` (N30) the last scan carries no pretended change. A scan's OPF cannot see the scans after it,
so each per-scan bounded change is also kept within what the remaining scans can still reach: the static
solution at the last rise scan, and zero at the last scan. Solved by Ipopt through Pyomo.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from fdia_graph.constants.meters import MeterKind
from fdia_graph.engine.attacks.attacker import Attacker
from fdia_graph.engine.attacks.family import AttackFamily
from fdia_graph.engine.solvers import ipopt
from fdia_graph.errors import ConfigError

SUPPORT_BUDGET = 32  # candidate supports tried per search


@dataclass
class _Window:
    """What every OPF of one window shares: the true states and injections, the noise of each meter, the hour
    of each scan and the count of rise scans."""

    onset: int
    vm: np.ndarray
    th: np.ndarray
    p: np.ndarray
    q: np.ndarray
    load: np.ndarray
    sig: np.ndarray
    hours: np.ndarray | None
    rise: int
    gen: np.ndarray | None = None  # [L, N] the units' output at each bus (MW), None: the net injection


@dataclass
class _OpfModel:
    """One support's OPF, built by `_model` and filled per scan by `_set_scan`: the Pyomo `model`, the buses whose
    voltage it frees (`interior`), the buses whose injection it may pretend (`free`), every bus whose injection it
    holds or moves (`rows`), the buses beyond the interior held at their true voltage (`outside`), and each free bus's
    P and Q injection meter (`meter[(bus, 0 or 1)]`, -1 for none). An interface goal adds the inner branches the
    support leaves at their true flow (`still`, branch and sign) and the meter of each objective term
    (`objective_meters`)."""

    model: Any
    interior: np.ndarray
    free: np.ndarray
    rows: np.ndarray
    outside: list[int]
    meter: dict[tuple[int, int], int]
    still: list[tuple[int, float]] = field(default_factory=list)
    objective_meters: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class _Prepared:
    """A window ready for the support search: the shared `window`, the attacker's `area`, the buses the search grows
    its supports from (`start`), and the buses whose injection it may pretend (`free`)."""

    window: _Window
    area: np.ndarray
    start: frozenset[int]
    free: set[int]


class OpfFdia(AttackFamily):
    """The base class of the attacks whose false states come from a local AC OPF: it holds the `Attacker` and
    the bounds, searches the supports (`_candidates`) and solves one support's window (`_static`, then
    `_sequential`). A subclass gives `design` and the goal, by extending `_model` and `_set_scan`.

    Settings: `hops` for the attacker's area, `load_cap` the plausibility cap of a pretended load (0.1, N28),
    `budget` candidate supports per search, `windows_solved` how many statically feasible supports get the
    full window solve, `family` the name written on the episode, and `exit_scans` the scans at the end of a
    window spent unwinding (N30, 0 for none), which needs `step_bound [24, N]` (each bus's natural five-minute
    step by hour, MW, `natural_steps`) and `frame_hour [T]` (the clock hour of every frame an onset indexes).
    The generator bounds (N33, N34): `gen_ramp [N]` each bus's five-minute ramp (MW, inf unbounded), `gen_band
    [N]` the largest deviation from the basepoint (MW), `q_lo`, `q_hi [N]` the reactive capability (MVAr,
    default the case's limits), `q_ramp [N]` the reactive move per scan (MVAr, default unbounded), `gen_output
    [T, N]` the units' output at each bus (MW, N42), `loads_only` (N43) and `gen_v_tol` (N75)."""

    def __init__(
        self,
        attacker: Attacker,
        *,
        gen_ramp: np.ndarray,
        gen_band: np.ndarray | None = None,
        q_lo: np.ndarray | None = None,
        q_hi: np.ndarray | None = None,
        q_ramp: np.ndarray | None = None,
        windows_solved: int = 3,
        gen_output: np.ndarray | None = None,
        loads_only: bool = False,
        gen_v_tol: float | None = None,
        hops: int = 3,
        load_cap: float = 0.1,
        budget: int = SUPPORT_BUDGET,
        step_bound: np.ndarray | None = None,
        frame_hour: np.ndarray | None = None,
        family: str = "Am",
        exit_scans: int = 0,
    ):
        if exit_scans and step_bound is None:
            raise ConfigError("an exit (exit_scans) unwinds within the step bound, so it needs step_bound")
        if step_bound is not None and frame_hour is None:
            raise ConfigError("a step bound needs the clock hour of every frame (frame_hour)")
        self.att = attacker
        self.hops, self.load_cap, self.budget = hops, load_cap, budget
        self.step_bound = None if step_bound is None else np.asarray(step_bound, float)
        self.frame_hour = None if frame_hour is None else np.asarray(frame_hour, int)
        self.family, self.exit_scans = family, exit_scans
        n = len(attacker.owner)
        self.gen_output = None if gen_output is None else np.asarray(gen_output, float)  # [T, N] MW, N42
        self.gen_ramp = np.asarray(gen_ramp, float)
        self.gen_band = np.full(n, np.inf) if gen_band is None else np.asarray(gen_band, float)
        # N75: a generator bus regulates its voltage to a schedule the operator knows, so its false voltage may
        # leave the true one by at most this much (pu); None leaves only the bus voltage limits
        self.gen_v_tol = gen_v_tol
        self.q_lo = attacker.q_lo if q_lo is None else np.asarray(q_lo, float)
        self.q_hi = attacker.q_hi if q_hi is None else np.asarray(q_hi, float)
        self.q_ramp = np.full(n, np.inf) if q_ramp is None else np.asarray(q_ramp, float)
        self.windows_solved = windows_solved
        self.loads_only = loads_only  # N43: no pretended generator output, loads alone move
        self._solver = None
        self.solver_errors = 0  # solves Ipopt stopped abnormally on, each read as an infeasible support

    def _free(self, vm: np.ndarray, va: np.ndarray, buses: np.ndarray) -> np.ndarray:
        """The buses of `buses` whose injection may be pretended: the attackable loads alone with
        `loads_only` (N43), the loads and the generators otherwise. A zero-injection bus never is."""
        if self.loads_only:
            return self.att.load_buses(vm, va, buses)
        return self.att.free_injection_buses(vm, va, buses)

    # ------------------------------------------------------------------ the support search
    def _candidates(self, area: np.ndarray, start: frozenset[int], free: set[int]) -> Iterator[np.ndarray]:
        """Supports inside the area, smallest first: the `start` buses grown one adjacent area bus at a time,
        each support holding a free injection (a goal needs one to pretend). At most `budget` of them."""
        allowed = set(int(b) for b in area)
        adj = self._area_adjacency(allowed)
        heap: list[tuple[int, tuple[int, ...]]] = []
        seen: set[frozenset[int]] = set()
        if start:
            heapq.heappush(heap, (len(start), tuple(sorted(start))))
            seen.add(start)
        yielded = 0
        while heap and yielded < self.budget:
            _, key = heapq.heappop(heap)
            s = frozenset(key)
            if s & free:  # a goal needs a free injection to pretend
                yielded += 1
                yield np.array(key, int)
            for v in sorted({n for b in s for n in adj[b]} - s):
                child = s | {v}
                if child not in seen and child <= allowed:
                    seen.add(child)
                    heapq.heappush(heap, (len(child), tuple(sorted(child))))

    def _area_adjacency(self, allowed: set[int]) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = {b: set() for b in allowed}
        f, t = self.att.net.case.f, self.att.net.case.t
        for a, b in zip(f.tolist(), t.tolist()):
            if a in allowed and b in allowed:
                adj[a].add(b)
                adj[b].add(a)
        return adj

    # ------------------------------------------------------------------ the OPF on one support
    def _model(self, w: _Window, support: np.ndarray):
        """The Pyomo model of one support (its parameters set per scan by `_set_scan`) and its sets. It has no
        goal: the subclass adds its own constraint."""
        import pyomo.environ as pe

        att, base = self.att, self.att.base_mva
        interior = att.absorb(w.vm[0], np.rad2deg(w.th[0]), np.asarray(support, int))
        interior = interior[interior != att.slack]
        touched = np.union1d(interior, att.boundary(interior))
        free = self._free(w.vm[0], np.rad2deg(w.th[0]), touched)
        free = free[free != att.slack]
        held = np.setdiff1d(touched[att._held_bus(w.vm[0], np.rad2deg(w.th[0]))[touched]], free)  # N44
        rows = np.r_[free, held].astype(int)
        inside = set(interior.tolist())
        outside = sorted({int(j) for b in rows for j in np.flatnonzero(att.ybus[b])} - inside)
        plan = att.meas.plan
        meter = {}
        for kind, comp in ((MeterKind.P, 0), (MeterKind.Q, 1)):
            for b in free.tolist():
                j = np.flatnonzero((plan.kind == kind) & (plan.bus == b))
                meter[(b, comp)] = int(j[0]) if len(j) else -1

        m = pe.ConcreteModel()
        m.I, m.O = pe.Set(initialize=interior.tolist()), pe.Set(initialize=outside)
        m.F, m.R = pe.Set(initialize=free.tolist()), pe.Set(initialize=rows.tolist())
        m.vo, m.to = pe.Param(m.O, mutable=True, initialize=1.0), pe.Param(m.O, mutable=True, initialize=0.0)
        m.pt, m.qt = pe.Param(m.R, mutable=True, initialize=0.0), pe.Param(m.R, mutable=True, initialize=0.0)
        m.sp, m.sq = pe.Param(m.F, mutable=True, initialize=0.0), pe.Param(m.F, mutable=True, initialize=0.0)
        m.vm, m.th = pe.Var(m.I, initialize=1.0), pe.Var(m.I, initialize=0.0)
        m.dp, m.dq = pe.Var(m.F, initialize=0.0), pe.Var(m.F, initialize=0.0)
        m.up, m.uq = pe.Var(m.F, within=pe.NonNegativeReals), pe.Var(m.F, within=pe.NonNegativeReals)

        def V(b):
            return (m.vm[b], m.th[b]) if b in inside else (m.vo[b], m.to[b])

        def power(b, y_row, cols):
            vb, tb = V(b)
            p = q = 0
            for j in cols:
                vj, tj = V(j)
                d = tb - tj
                p += vb * vj * (y_row[j].real * pe.cos(d) + y_row[j].imag * pe.sin(d))
                q += vb * vj * (y_row[j].real * pe.sin(d) - y_row[j].imag * pe.cos(d))
            return p * base, q * base

        m.c = pe.ConstraintList()
        free_set = set(free.tolist())
        for b in rows.tolist():
            p, q = power(b, att.ybus[b], np.flatnonzero(att.ybus[b]).tolist())
            m.c.add(p == m.pt[b] + (m.dp[b] if b in free_set else 0.0))
            m.c.add(q == m.qt[b] + (m.dq[b] if b in free_set else 0.0))
        for b in free.tolist():
            m.c.add(m.up[b] >= m.dp[b] - m.sp[b])
            m.c.add(m.up[b] >= -m.dp[b] - m.sp[b])
            m.c.add(m.uq[b] >= m.dq[b] - m.sq[b])
            m.c.add(m.uq[b] >= -m.dq[b] - m.sq[b])
        m.obj = pe.Objective(
            expr=sum(m.up[b] + m.uq[b] + 1e-4 * (m.dp[b] ** 2 + m.dq[b] ** 2) for b in free.tolist())
        )
        return _OpfModel(m, interior, free, rows, outside, meter)

    def _set_scan(self, w, ctx: _OpfModel, k, prev, static, start=None, end=None) -> bool:
        """Set scan k's parameters, bounds and starting point; False when some bound box is empty."""
        m, interior, free, rows, outside, meter = (
            ctx.model,
            ctx.interior,
            ctx.free,
            ctx.rows,
            ctx.outside,
            ctx.meter,
        )
        case = self.att.net.case
        for b in outside:
            m.vo[b], m.to[b] = float(w.vm[k, b]), float(w.th[k, b])
        for b in rows.tolist():
            m.pt[b], m.qt[b] = float(w.p[k, b]), float(w.q[k, b])
        for b in interior.tolist():
            lo_v, hi_v = min(case.vmin[b], w.vm[k, b]), max(case.vmax[b], w.vm[k, b])
            if self.gen_v_tol is not None and self.att.gen_bus[b]:  # N75: held near its voltage schedule
                lo_v, hi_v = max(lo_v, w.vm[k, b] - self.gen_v_tol), min(hi_v, w.vm[k, b] + self.gen_v_tol)
            m.vm[b].setlb(lo_v)
            m.vm[b].setub(hi_v)
            dv, dt = (0.0, 0.0) if start is None else start[b]
            m.vm[b].value = float(np.clip(w.vm[k, b] + dv, m.vm[b].lb, m.vm[b].ub))
            m.th[b].value = float(w.th[k, b] + dt)
        for b in free.tolist():
            lo_p, hi_p, lo_q, hi_q = self._box(w, k, b, prev[b], static, end)
            if lo_p > hi_p + 1e-6 or lo_q > hi_q + 1e-6:  # Ipopt relaxes its bounds by about 1e-8
                return False
            lo_p, hi_p, lo_q, hi_q = min(lo_p, hi_p), max(lo_p, hi_p), min(lo_q, hi_q), max(lo_q, hi_q)
            m.dp[b].setlb(lo_p)
            m.dp[b].setub(hi_p)
            m.dq[b].setlb(lo_q)
            m.dq[b].setub(hi_q)
            m.dp[b].value = float(np.clip(prev[b][0], lo_p, hi_p))
            m.dq[b].value = float(np.clip(prev[b][1], lo_q, hi_q))
            m.sp[b] = float(w.sig[k, meter[(b, 0)]]) if meter[(b, 0)] >= 0 else 0.0
            m.sq[b] = float(w.sig[k, meter[(b, 1)]]) if meter[(b, 1)] >= 0 else 0.0
        return True

    def _run(self, m) -> bool:
        from pyomo.common.errors import ApplicationError

        if self._solver is None:  # raises SolverUnavailable where Ipopt cannot run at all
            self._solver = ipopt(tol=1e-7, acceptable_tol=1e-5, max_iter=600, print_level=0)
        try:
            r = self._solver.solve(m, load_solutions=False)
        except ApplicationError:  # Ipopt started but stopped abnormally on this model: no point, counted
            self.solver_errors += 1
            return False
        if str(r.solver.termination_condition) not in ("optimal", "locallyOptimal"):
            return False
        m.solutions.load_from(r)
        return True

    def _static(self, w: _Window, support: np.ndarray) -> dict | None:
        """Whether the support carries the full overload at the last rise scan inside every static bound:
        each free bus's (dP, dQ) there, or None."""
        import pyomo.environ as pe

        ctx = self._model(w, support)
        m, free = ctx.model, ctx.free.tolist()
        prev = {b: (0.0, 0.0) for b in free}
        if not (self._set_scan(w, ctx, w.rise - 1, prev, static=True) and self._run(m)):
            return None
        return {b: (float(pe.value(m.dp[b])), float(pe.value(m.dq[b]))) for b in free}

    def _sequential(self, w: _Window, support: np.ndarray, end: dict):
        """The window scan by scan on one support, each scan bounded against the one before and started
        from its false deviation. Returns the false (vm [L, N], va [L, N] degrees) and the free buses
        whose injection moves, or None when a scan has no feasible point. `end` is `_static`'s solution,
        the change each ramp-bounded output must still reach at the last rise scan."""
        import pyomo.environ as pe

        ctx = self._model(w, support)
        m, interior, free = ctx.model, ctx.interior, ctx.free
        L = w.vm.shape[0]
        vm_f, th_f = w.vm.copy(), w.th.copy()
        prev = {b: (0.0, 0.0) for b in free.tolist()}
        used = np.zeros(len(free))
        start = None
        for k in range(L):
            if not self._set_scan(w, ctx, k, prev, static=False, start=start, end=end) or not self._run(m):
                return None
            for i, b in enumerate(free.tolist()):
                prev[b] = (float(pe.value(m.dp[b])), float(pe.value(m.dq[b])))
                used[i] = max(used[i], abs(prev[b][0]) + abs(prev[b][1]))
            for b in interior.tolist():
                vm_f[k, b], th_f[k, b] = pe.value(m.vm[b]), pe.value(m.th[b])
            start = {b: (vm_f[k, b] - w.vm[k, b], th_f[k, b] - w.th[k, b]) for b in interior.tolist()}
        return vm_f, np.rad2deg(th_f), free[used > 1e-3]

    def _box(
        self, w: _Window, k: int, b: int, prev: tuple[float, float], static: bool, end: dict | None = None
    ):
        """The bounds of bus b's pretended change (dP MW, dQ MVAr) at scan k, given the previous scan's;
        `static` leaves out the per-scan ramp and step bounds and what the remaining scans can reach."""
        lo_p, hi_p, lo_q, hi_q = self._own_box(w, k, b, prev, static, end)
        if (
            not static and self.exit_scans and k == w.vm.shape[0] - 1
        ):  # N30: nothing pretended at the last scan
            lo_p, hi_p, lo_q, hi_q = max(lo_p, 0.0), min(hi_p, 0.0), max(lo_q, 0.0), min(hi_q, 0.0)
        return lo_p, hi_p, lo_q, hi_q

    def _own_box(self, w, k, b, prev, static, end):
        att, last = self.att, w.vm.shape[0] - 1
        if att.gen_bus[b]:
            units = w.p if w.gen is None else w.gen  # the units' output, or the injection standing for it
            lo_p = min(att.p_lo[b], units[k, b]) - units[k, b]
            hi_p = max(att.p_hi[b], units[k, b]) - units[k, b]
            lo_q = min(self.q_lo[b], w.q[k, b]) - w.q[k, b]
            hi_q = max(self.q_hi[b], w.q[k, b]) - w.q[k, b]
            band = self.gen_band[b]
            lo_p, hi_p = max(lo_p, -band), min(hi_p, band)
            # the reported output moves per scan within the ramp, or the true step where it is larger
            for comp, ramp, true in ((0, self.gen_ramp[b], units), (1, self.q_ramp[b], w.q)):
                if not np.isfinite(ramp):
                    continue
                reach = np.maximum(ramp, np.abs(np.diff(true[:, b])))  # reach[j - 1]: scan j's largest move
                if static:  # only as far from the true output at the onset as the scans up to k can move it
                    goals = [(true[0, b], ramp + reach[:k].sum())]
                else:
                    step = 0.0 if k == 0 else true[k, b] - true[k - 1, b]
                    width = max(ramp, abs(step))
                    c = prev[comp] - step
                    if comp == 0:
                        lo_p, hi_p = max(lo_p, c - width), min(hi_p, c + width)
                    else:
                        lo_q, hi_q = max(lo_q, c - width), min(hi_q, c + width)
                    # still able to reach the static solution at the last rise scan
                    goals = (
                        [(true[w.rise - 1, b] + end[b][comp], reach[k : w.rise - 1].sum())]
                        if end is not None and k < w.rise
                        else []
                    )
                if self.exit_scans:  # and the true output at the last scan
                    goals.append((true[last, b], reach[k:last].sum()))
                for out, r in goals:
                    lo, hi = out - r - true[k, b], out + r - true[k, b]
                    if comp == 0:
                        lo_p, hi_p = max(lo_p, lo), min(hi_p, hi)
                    else:
                        lo_q, hi_q = max(lo_q, lo), min(hi_q, hi)
            cap = self.load_cap * max(units[k, b] - w.p[k, b], 0.0)  # the load at the bus (N42)
            return lo_p - cap, hi_p + cap, lo_q, hi_q
        load = max(w.load[k, b], 0.0)
        cap, cap_q = self.load_cap * load, self.load_cap * max(abs(w.q[k, b]), 0.2 * load)
        lo_p, hi_p, lo_q, hi_q = -cap, cap, -cap_q, cap_q
        if self.step_bound is None or not self.exit_scans:
            return lo_p, hi_p, lo_q, hi_q
        if not static and k >= w.rise:  # the exit moves within the bus's step bound
            s = self.step_bound[w.hours[k], b]
            if np.isfinite(s):
                lo_p, hi_p = max(lo_p, prev[0] - s), min(hi_p, prev[0] + s)
        span = np.arange(max(k + 1, w.rise), last + 1)  # within reach of zero at the last scan
        r = float(self.step_bound[w.hours[span], b].sum())
        if np.isfinite(r):
            lo_p, hi_p = max(lo_p, -r), min(hi_p, r)
        return lo_p, hi_p, lo_q, hi_q
