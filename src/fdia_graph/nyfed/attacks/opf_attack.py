"""The overload attacks with the attacker's false states from a local optimal power flow (N35, ours).

[1] has the attacker run a local OPF over its subnetwork and then solve an l0 problem for the attack
vector. Here the l0 part is MsFdia's search over supports (the sets of buses whose voltage moves,
smallest first, grown from the target's ends), since the meters a change moves are set by which
voltages move, and each candidate support is tested by an AC OPF: first one scan at the full overload
with every static bound (fast), then, for a support that passes, the window scan by scan with the
per-scan bounds. The support whose window tampers the fewest meters is kept (eq. 12).

Each scan's OPF has the support's voltages as variables (every voltage outside held true), the AC power
flow at every bus of the support and its edge, the goal on the target lines, and as objective the total
pretended injection change beyond each injection meter's noise (a change under the noise counts as
none, as in eq. 12) with a small quadratic term that keeps sub-noise changes from wandering.

Bounds, per scan k and free bus b (MW and MVAr, pretended change dP, dQ against the true injection):
- a load: |dP| <= cap P and |dQ| <= cap max(|Q|, 0.2 P) (D16, N24, N28), and for Ap the active step
  |dP_k - dP_k-1| <= kappa s_b(hour) (N22);
- a generator: its reported output inside its limits (22)-(23), the output being the units' own
  (`gen_output`, N42) where the bus also carries load, that load moving within the load cap as at a load
  bus (before N42 the net injection stood for the output, which load at the bus pushes under the units'
  minimum, so a loaded generator bus could never report less); per scan the reported output moves by at
  most its five-minute ramp, or the true step where the dispatch itself moves faster (N33, every family,
  the Am rise included); in total |dP| <= its basepoint band `gen_band` (N33, the operator knows its
  instruction); its reactive output within `q_lo`, `q_hi` (N34), unbounded per scan unless `q_ramp` is
  given (the exciter moves it within seconds);
- with `loads_only` (N43) no generator bus is pretended at all, only loads above 1 MW (a zero-injection
  bus never is);
- every other New York bus of the support and its edge keeps its true injection; a bus outside New
  York on the edge carries no meter and absorbs, as in `Attacker`, unless it has no unit and no
  injection (N44, bus 140), which is held.

The goal rises to the rating at the last rise scan and, with `exit_scans` (N30), falls back to the true
flow in equal steps to no pretended change at all at the last scan. A scan's OPF cannot see the scans
after it, so each per-scan bounded change is also kept within what its remaining scans can still reach:
the static solution at the last rise scan, and zero at the last scan. Without this the first scans
drift where the ramps cannot bring them back. Solved by Ipopt through Pyomo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdia_graph.nyfed.attacks.attacker import Attacker
from fdia_graph.nyfed.attacks.episode import Episode, Window
from fdia_graph.nyfed.attacks.msfdia import MsFdia
from fdia_graph.nyfed.solvers import ipopt

P_KIND, Q_KIND = 2, 3  # nyfed.sensing.meters.KINDS "p", "q": the injection meters a pretended change moves


@dataclass
class _Window:
    """What every OPF of one window shares: the true states and injections, the noise of each meter,
    the hour of each scan and the goal [L, n_targets]."""

    onset: int
    vm: np.ndarray
    th: np.ndarray
    p: np.ndarray
    q: np.ndarray
    load: np.ndarray
    sig: np.ndarray
    hours: np.ndarray | None
    targets: np.ndarray
    goal: np.ndarray
    rise: int
    gen: np.ndarray | None = None  # [L, N] the units' output at each bus (MW), None: the net injection


class OpfFdia(MsFdia):
    """Am, or Ap with a step bound, whose false states come from a local AC OPF on each candidate
    support. Takes MsFdia's target choice (eligible lines, N26 order, D17 sets), support search and
    settings, plus the generator bounds of N33/N34: `gen_ramp [N]` each bus's five-minute ramp (MW, inf
    unbounded), `gen_band [N]` the largest total deviation from the basepoint (MW, inf unbounded),
    `q_lo`, `q_hi [N]` the reactive capability (MVAr, default the case's limits) and `q_ramp [N]` the
    reactive move per scan (MVAr, default unbounded). `windows_solved` caps how many statically feasible
    supports get the full window solve (3)."""

    def __init__(
        self,
        attacker: Attacker,
        ratings: np.ndarray,
        *,
        gen_ramp: np.ndarray,
        gen_band: np.ndarray | None = None,
        q_lo: np.ndarray | None = None,
        q_hi: np.ndarray | None = None,
        q_ramp: np.ndarray | None = None,
        windows_solved: int = 3,
        gen_output: np.ndarray | None = None,
        loads_only: bool = False,
        **kwargs,
    ):
        super().__init__(attacker, ratings, **kwargs)
        n = len(attacker.owner)
        self.gen_output = None if gen_output is None else np.asarray(gen_output, float)  # [T, N] MW, N42
        self.gen_ramp = np.asarray(gen_ramp, float)
        self.gen_band = np.full(n, np.inf) if gen_band is None else np.asarray(gen_band, float)
        # N75: a generator bus regulates its voltage to a schedule the operator knows, so its false voltage may
        # leave the true one by at most this much (pu); None leaves only the bus voltage limits
        self.gen_v_tol: float | None = None
        self.q_lo = attacker.q_lo if q_lo is None else np.asarray(q_lo, float)
        self.q_hi = attacker.q_hi if q_hi is None else np.asarray(q_hi, float)
        self.q_ramp = np.full(n, np.inf) if q_ramp is None else np.asarray(q_ramp, float)
        self.windows_solved = windows_solved
        self.loads_only = loads_only  # N43: no pretended generator output, loads alone move
        self._solver = None
        self.solver_errors = 0  # solves Ipopt stopped abnormally on, each read as an infeasible support

    def reachable(self, window: Window) -> bool:
        """Whether some target set of the window has a support that carries the full overload at the last
        rise scan inside every static bound (no ramp, step or exit): the necessary condition, fast."""
        onset, vm, va = window
        lines = self._eligible(vm, va)
        if len(lines) == 0:
            return False
        rise = vm.shape[0] - self.exit_scans
        for targets in self._target_sets(self._order(lines, vm[:rise], va[:rise], np.random.default_rng(0))):
            prep = self._prepare(onset, vm, va, np.asarray(targets, int))
            if prep is None:
                continue
            w, area, start, free = prep
            if any(self._static(w, s) is not None for s in self._candidates(area, start, free)):
                return True
        return False

    def _free(self, vm: np.ndarray, va: np.ndarray, buses: np.ndarray) -> np.ndarray:
        """The buses of `buses` whose injection may be pretended: the attackable loads alone with
        `loads_only` (N43), the loads and the generators otherwise. A zero-injection bus never is."""
        if self.loads_only:
            return self.att.load_buses(vm, va, buses)
        return self.att.free_injection_buses(vm, va, buses)

    def _prepare(self, onset, vm, va, targets):
        att, case = self.att, self.att.net.case
        length = vm.shape[0]
        rise = length - self.exit_scans
        area = att.area(int(case.f[targets[0]]), self.hops)
        if area is None or rise < 1:
            return None
        targets = np.array([t for t in targets if {int(case.f[t]), int(case.t[t])} & set(area.tolist())], int)
        if len(targets) == 0:
            return None
        pf, qf, _, _ = att.net.flows(vm, va)
        true_flow = np.hypot(pf[:, targets], qf[:, targets])
        gap = self.ratings[targets][None, :] - true_flow[rise - 1][None, :]
        share = np.r_[
            np.arange(1, rise + 1) / rise, 1 - np.arange(1, self.exit_scans + 1) / max(self.exit_scans, 1)
        ]
        goal = true_flow + share[:, None] * gap
        goal[rise - 1] = self.ratings[targets]
        p, q = att.net.injections(vm, va)
        hours = None if self.frame_hour is None else self.frame_hour[onset : onset + length]
        gen = None if self.gen_output is None else self.gen_output[onset : onset + length]
        w = _Window(
            onset,
            vm,
            np.deg2rad(va),
            p,
            q,
            att.bus_load(vm, va),
            att.sigma(att.meas.h(vm, va)),
            hours,
            targets,
            goal,
            rise,
            gen,
        )
        free = set(self._free(vm[0], va[0], area).tolist())
        start = frozenset(
            int(b) for b in np.unique(np.r_[case.f[targets], case.t[targets]]) if int(b) in set(area.tolist())
        )
        return (w, area, start, free) if free else None

    def _attack(self, onset: int, vm: np.ndarray, va: np.ndarray, targets: np.ndarray) -> Episode | None:
        att = self.att
        prep = self._prepare(onset, vm, va, targets)
        if prep is None:
            return None
        w, area, start, free = prep
        best_cost, best, solved = None, None, 0
        for support in self._candidates(area, start, free):
            if solved >= self.windows_solved:
                break
            end = self._static(w, support)
            if end is None:
                continue
            solved += 1
            res = self._sequential(w, support, end)
            if res is None:
                continue
            vm_f, va_f, buses = res
            a = att.attack_vector(vm, va, vm_f, va_f)
            tam = att.tampered(vm, va, a)
            if self.exit_scans and tam[-1].any():  # N30: unwound by the last scan
                continue
            cost = int(tam.any(axis=0).sum())
            if cost >= 1 and (best_cost is None or cost < best_cost):
                best_cost, best = cost, (a, buses, tam)
        if best is None:
            return None
        a, buses, tam = best
        clients = np.unique(att.meas.plan.owner[np.flatnonzero(tam.any(axis=0))])
        return Episode(self.family, clients, onset, vm.shape[0], buses, int(w.targets[0]), a)

    # ------------------------------------------------------------------ the OPF on one support
    def _model(self, w: _Window, support: np.ndarray):
        """The Pyomo model of one support (its parameters set per scan by `_set_scan`) and its sets."""
        import pyomo.environ as pe

        att, case, base = self.att, self.att.net.case, self.att.base_mva
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
        for kind, comp in ((P_KIND, 0), (Q_KIND, 1)):
            for b in free.tolist():
                j = np.flatnonzero((plan.kind == kind) & (plan.bus == b))
                meter[(b, comp)] = int(j[0]) if len(j) else -1

        m = pe.ConcreteModel()
        m.I, m.O = pe.Set(initialize=interior.tolist()), pe.Set(initialize=outside)
        m.F, m.R = pe.Set(initialize=free.tolist()), pe.Set(initialize=rows.tolist())
        m.vo, m.to = pe.Param(m.O, mutable=True, initialize=1.0), pe.Param(m.O, mutable=True, initialize=0.0)
        m.pt, m.qt = pe.Param(m.R, mutable=True, initialize=0.0), pe.Param(m.R, mutable=True, initialize=0.0)
        m.sp, m.sq = pe.Param(m.F, mutable=True, initialize=0.0), pe.Param(m.F, mutable=True, initialize=0.0)
        m.goal = pe.Param(range(len(w.targets)), mutable=True, initialize=0.0)
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
        for t_i, ln in enumerate(w.targets.tolist()):
            f, t = int(case.f[ln]), int(case.t[ln])
            p, q = power(f, att.yf[ln], [f, t])
            m.c.add(p**2 + q**2 == m.goal[t_i] ** 2)
        for b in free.tolist():
            m.c.add(m.up[b] >= m.dp[b] - m.sp[b])
            m.c.add(m.up[b] >= -m.dp[b] - m.sp[b])
            m.c.add(m.uq[b] >= m.dq[b] - m.sq[b])
            m.c.add(m.uq[b] >= -m.dq[b] - m.sq[b])
        m.obj = pe.Objective(
            expr=sum(m.up[b] + m.uq[b] + 1e-4 * (m.dp[b] ** 2 + m.dq[b] ** 2) for b in free.tolist())
        )
        return m, interior, free, rows, outside, meter

    def _set_scan(self, w, ctx, k, prev, static, start=None, end=None) -> bool:
        """Set scan k's parameters, bounds and starting point; False when some bound box is empty."""
        m, interior, free, rows, outside, meter = ctx
        case = self.att.net.case
        for b in outside:
            m.vo[b], m.to[b] = float(w.vm[k, b]), float(w.th[k, b])
        for b in rows.tolist():
            m.pt[b], m.qt[b] = float(w.p[k, b]), float(w.q[k, b])
        for t_i in range(len(w.targets)):
            m.goal[t_i] = float(w.goal[k, t_i])
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
        m, free = ctx[0], ctx[2].tolist()
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
        m, interior, free, *_ = ctx
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
        if not static and self.step_bound is not None and (self.bound_rise or k >= w.rise):  # Ap, every exit
            s = self.kappa * self.step_bound[w.hours[k], b]
            if np.isfinite(s):
                lo_p, hi_p = max(lo_p, prev[0] - s), min(hi_p, prev[0] + s)
        if self.step_bound is not None:  # within reach of zero at the onset (Ap) and at the last scan (exit)
            spans = [np.arange(0, k + 1)] if static and self.bound_rise else []
            if self.exit_scans:
                spans.append(np.arange(k + 1, last + 1))
            for span in spans:
                span = span[bool(self.bound_rise) | (span >= w.rise)]
                r = float(self.kappa * self.step_bound[w.hours[span], b].sum())
                if np.isfinite(r):
                    lo_p, hi_p = max(lo_p, -r), min(hi_p, r)
        return lo_p, hi_p, lo_q, hi_q
