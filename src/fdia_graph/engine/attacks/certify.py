"""Certifying the fewest-tamper attack [WU26, eq. 12] by convex relaxation
(docs/plans/RELAX_CERTIFIER_PLAN.md).

The fewest-tamper search returns a feasible exact-AC attack, so its device count is an upper bound
on the optimum. `certify` also solves a mixed-integer second-order-cone relaxation of the same
problem over the same area: one binary per device, Jabr's W = V V^H with the rank-one condition
dropped, the goal, the voltage limits and the zero injections, and for an overload goal the bounds
the search holds its flow solve to (the plan's D14 and D16): every generator the attack can move
inside its limits (22)-(23) and every load bus it can move within `load_cap` of its true load. Every attack the search can return
is a point of the relaxation, so the relaxation's optimum is a lower bound; when it meets the
search's count, the search's attack is globally optimal over the area.

An analysis tool, never on the generation path. Needs cvxpy and SCIP:
pip install "fdia-graph[certify]".
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import time
from typing import Optional, cast

import numpy as np

from ...formulas.attacks import LIMIT_TOL_PQ, LIMIT_TOL_V, bus_load, generator_output
from ...formulas.network import _dense, bus_injections, complex_voltages
from ...formulas.relax import big_m, cone_gap, roundoff_slack, sector_cuts, voltage_box
from ...models.choices import CertifyVerdict
from ...models.config import CertifyOptions
from ...models.frames import (
    AttackVector,
    BoundClaim,
    Certificate,
    FlowGoal,
    FrameKnobs,
    LoadGoal,
    MinimizerResult,
)
from ...models.grid import CURRENT, NODE
from ...models.inputs import CertifiableLimits
from . import relax_cuts
from .minimize import Goal, MinimizeMixin, _Window

GOAL_TOL_MVA = 1e-3  # how far the search's solve may leave a goal line's flow from its target
POWER_TOL_MW = 1e-3  # how far a held or designed injection may move in the search's solve


def _require_solver() -> None:
    """Fail early, with the install line, when cvxpy or SCIP is missing."""
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError('The certifier needs cvxpy and SCIP: pip install "fdia-graph[certify]"') from e
    if "SCIP" not in cp.installed_solvers():
        raise ImportError('The certifier needs the SCIP solver: pip install "fdia-graph[certify]"')


def _bounding_scip():
    """cvxpy's SCIP interface that also keeps SCIP's status and dual bound: the bound is what makes
    a solve stopped at its time limit still a valid (weaker) lower bound."""
    from cvxpy.reductions.solvers.conic_solvers.scip_conif import SCIP

    class _BoundingScip(SCIP):
        status = "not solved"
        dual_bound = math.nan

        def name(self) -> str:
            return "SCIP_DUAL_BOUND"

        def _solve(self, model, variables, constraints, data, dims):
            out = super()._solve(model, variables, constraints, data, dims)
            self.status = str(model.getStatus())
            self.dual_bound = float(model.getDualbound())
            return out

    return _BoundingScip()


class _Layout:
    """The variables of one snapshot in one vector x = [W_ii of the area buses, Re and Im of W_ij of
    each connected pair of area buses, Re and Im of V_i of the area buses, r_i >= |V_i - V_i^true|]
    (ppc indices), and the linear maps from x to the meters' readings. Buses outside the area keep
    their true voltage."""

    def __init__(self, area: np.ndarray, Y: np.ndarray) -> None:
        self.area = area
        self.pos = {int(p): a for a, p in enumerate(area)}
        linked = np.abs(Y[np.ix_(area, area)]) > 0
        a, b = np.nonzero(np.triu(linked | linked.T, 1))
        lo, hi = np.minimum(area[a], area[b]), np.maximum(area[a], area[b])
        self.pair_a = np.array([self.pos[int(p)] for p in lo], dtype=np.int64)  # the lower ppc end
        self.pair_b = np.array([self.pos[int(q)] for q in hi], dtype=np.int64)
        self.pair = {(int(p), int(q)): k for k, (p, q) in enumerate(zip(lo, hi))}
        na, npair = len(area), len(lo)
        self.w = np.arange(na)
        self.re = na + np.arange(npair)
        self.im = na + npair + np.arange(npair)
        self.e = na + 2 * npair + np.arange(na)
        self.f = 2 * na + 2 * npair + np.arange(na)
        self.r = 3 * na + 2 * npair + np.arange(na)
        self.n = 4 * na + 2 * npair

    def point(self, V: np.ndarray, V0: Optional[np.ndarray] = None) -> np.ndarray:
        """x of the rank-one W of the voltages V (ppc order), with r = |V - V0| (0 without V0)."""
        va = V[self.area]
        x = np.zeros(self.n)
        x[self.w] = np.abs(va) ** 2
        cross = va[self.pair_a] * np.conj(va[self.pair_b])
        x[self.re], x[self.im] = np.real(cross), np.imag(cross)
        x[self.e], x[self.f] = np.real(va), np.imag(va)
        if V0 is not None:
            x[self.r] = np.abs(va - V0[self.area])
        return x

    def w_rows(self, i: int, j: int, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The linear part (rows of Re and Im) of W_ij = V_i conj(V_j) in x; zero when both ends are
        fixed (W_ij is then a constant)."""
        re, im = np.zeros(self.n), np.zeros(self.n)
        pi, pj = self.pos.get(i), self.pos.get(j)
        if pi is not None and pj is not None:
            self._area_pair(i, j, pi, re, im)
            return re, im
        if pi is not None:
            return self._left(pi, V[j])
        if pj is not None:
            return self._right(pj, V[i])
        return re, im

    def _left(self, pi: int, c: complex) -> tuple[np.ndarray, np.ndarray]:
        """V_i conj(c) with V_i = e + jf a variable: (e a + f b) + j (f a - e b), c = a + jb."""
        re, im = np.zeros(self.n), np.zeros(self.n)
        re[[self.e[pi], self.f[pi]]] = c.real, c.imag
        im[[self.f[pi], self.e[pi]]] = c.real, -c.imag
        return re, im

    def _right(self, pj: int, c: complex) -> tuple[np.ndarray, np.ndarray]:
        """c conj(V_j) with V_j = e + jf a variable: (a e + b f) + j (b e - a f), c = a + jb."""
        re, im = np.zeros(self.n), np.zeros(self.n)
        re[[self.e[pj], self.f[pj]]] = c.real, c.imag
        im[[self.e[pj], self.f[pj]]] = c.imag, -c.real
        return re, im

    def residual_rows(self, i: int, j: int, V0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The linear part of X_ij = W_ij - V_i conj(V0_j) - V0_i conj(V_j) + V0_i conj(V0_j), which a
        rank-one W makes (V_i - V0_i) conj(V_j - V0_j); i and j area buses (i = j allowed). It is zero
        at the true point, so X_ij = rows (x - x0)."""
        re, im = np.zeros(self.n), np.zeros(self.n)
        self._area_pair(i, j, self.pos[i], re, im)
        for lr, li in (self._left(self.pos[i], V0[j]), self._right(self.pos[j], V0[i])):
            re -= lr
            im -= li
        return re, im

    def _area_pair(self, i: int, j: int, pi: int, re: np.ndarray, im: np.ndarray) -> None:
        """W_ij with both ends in the area: W_ii, or the pair's variables (conjugated when i > j)."""
        if i == j:
            re[self.w[pi]] = 1.0
            return
        k = self.pair[(min(i, j), max(i, j))]
        re[self.re[k]] = 1.0
        im[self.im[k]] = 1.0 if i < j else -1.0

    def power_rows(self, i: int, Yrow: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The linear part of sum_n conj(Y_n) W_in (an injection with Ybus's row, or a from-end flow
        with Yf's row and i its from bus), per unit, generation positive."""
        re, im = np.zeros(self.n), np.zeros(self.n)
        for n in np.nonzero(Yrow)[0]:
            wr, wi = self.w_rows(i, int(n), V)
            G, B = Yrow[n].real, Yrow[n].imag
            re += G * wr + B * wi
            im += G * wi - B * wr
        return re, im

    def current_rows(self, Yrow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The linear part of sum_n Y_n V_n (a branch-end current with Yf's or Yt's row), per unit."""
        re, im = np.zeros(self.n), np.zeros(self.n)
        for n in np.nonzero(Yrow)[0]:
            a = self.pos.get(int(n))
            if a is not None:
                G, B = Yrow[n].real, Yrow[n].imag
                re[[self.e[a], self.f[a]]] += G, -B
                im[[self.e[a], self.f[a]]] += B, G
        return re, im

    def radius(self, x0: np.ndarray, wlo: np.ndarray, whi: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """Each variable's largest distance from its true value x0 at an AC point inside the box the
        voltage limits give and within rho of the true voltages (|V_i - V0_i| <= rho_i):
        |W_ij| <= sqrt(W_ii,max W_jj,max), |V_i| <= sqrt(W_ii,max), and
        |W_ij - W0_ij| <= rho_i |V_j|max + |V0_i| rho_j."""
        r = np.zeros(self.n)
        r[self.w] = np.maximum(whi - x0[self.w], x0[self.w] - wlo)
        vmax, v0 = np.sqrt(whi), np.hypot(x0[self.e], x0[self.f])
        a, b = self.pair_a, self.pair_b
        box = np.sqrt(whi[a] * whi[b]) + np.hypot(x0[self.re], x0[self.im])
        pair = np.minimum(box, rho[a] * vmax[b] + v0[a] * rho[b])
        r[self.re], r[self.im] = pair, pair
        r[self.e] = np.minimum(vmax + np.abs(x0[self.e]), rho)
        r[self.f] = np.minimum(vmax + np.abs(x0[self.f]), rho)
        r[self.r] = np.minimum(vmax + v0, rho)
        return r


class _Channels:
    """The linear channels of one snapshot the count sees: rows A [m, n] with a = A (x - x0) in the
    scan's units, their noise sigma, device, the previous frame's attack value (At's bound) and the
    float32 roundoff slack of the search's comparison with sigma (`formulas.relax.roundoff_slack`,
    from the true reading the search rounds)."""

    def __init__(self, channels: list[tuple[np.ndarray, float, int, float, float]]) -> None:
        self.A = np.array([c[0] for c in channels]).reshape(len(channels), -1)
        self.sigma = np.array([c[1] for c in channels], float)
        self.device = np.array([c[2] for c in channels], dtype=np.int64)
        self.prev = np.array([c[3] for c in channels], float)
        self.slack = roundoff_slack(self.sigma, np.array([c[4] for c in channels], float))


class _Relaxation:
    """The mixed-integer second-order-cone relaxation of one window's fewest-tamper problem over the
    area (module docstring; docs/plans/RELAX_CERTIFIER_PLAN.md, section 2)."""

    def __init__(self, g: MinimizeMixin, window: _Window, area: np.ndarray) -> None:
        self.g, self.window = g, window
        self.lut = g._ppc_row[np.arange(g.C)]
        Y, Yf = g._dense_admittances()
        self.Y, self.Yf, self.Yt = Y, Yf, _dense(g._Yt)
        self.from_ppc = g._from_bus_ppc
        self.layout = _Layout(self.lut[np.asarray(area, dtype=np.int64)], Y)
        self.area = np.asarray(area, dtype=np.int64)
        # the voltage box comes from the search's own limits: knobs without them are refused
        self.limits = CertifiableLimits(window.k.limits).given
        self.V = [self._voltages(X) for X in window.states]
        self.x0 = [self.layout.point(V) for V in self.V]
        self.box = [self._box(t) for t in range(len(self.V))]
        # how far each area bus's voltage may move from its true value (the box's own reach until
        # bound tightening shrinks it), and the QC variables of the last `build` per snapshot
        self.rho = [np.sqrt(hi) + np.abs(V[self.layout.area]) for (_, hi), V in zip(self.box, self.V)]
        self.aux: dict[int, dict] = {}
        self.cuts: tuple[str, ...] = ()
        self._integer = True
        self.sectors: list[tuple] = []  # the flow goal's (sector binaries, flow) of the last `build`
        # the area buses whose injection the search holds when they are in the support (every one but
        # the free injections of the goal), and their support binaries of the last `build`
        free = g.free_injection_buses() if window.goal.kind == "flow" else self._targets()
        self.held = np.setdiff1d(self.area, free)
        self.in_support = None
        self.lin = [self._linear(t) for t in range(len(self.V))]
        self.devices = np.unique(np.concatenate([c.device for c in self.lin] + [self._nonlinear_devices()]))

    def _radius(self, t: int) -> np.ndarray:
        """The variables' reach at snapshot t (`_Layout.radius`) under its current bounds."""
        return self.layout.radius(self.x0[t], *self.box[t], self.rho[t])

    def _voltages(self, X: np.ndarray) -> np.ndarray:
        V = np.zeros(self.g._n_ppc_buses, complex)
        V[self.lut] = complex_voltages(X[:, NODE.v], X[:, NODE.theta])
        return V

    def _box(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """The W_ii bounds of the area buses at snapshot t ([WU26, eq. 21] as the search applies it)."""
        v = self.window.states[t][self.area, NODE.v]
        return voltage_box(v, self.limits.v_lo[self.area], self.limits.v_hi[self.area], LIMIT_TOL_V)

    def injection(self, t: int, bus: int) -> tuple[np.ndarray, np.ndarray]:
        """Rows of the stored injection's change at `bus` (load positive, MW, MVAr)."""
        i = int(self.lut[bus])
        re, im = self.layout.power_rows(i, self.Y[i], self.V[t])
        return -self.g._base_mva * re, -self.g._base_mva * im

    def flow(self, t: int, line: int) -> tuple[np.ndarray, np.ndarray]:
        """Rows of the from-end flow of `line` (MW, MVAr)."""
        re, im = self.layout.power_rows(int(self.from_ppc[line]), self.Yf[line], self.V[t])
        return self.g._base_mva * re, self.g._base_mva * im

    def _linear(self, t: int) -> _Channels:
        """The metered injections, from-end flows and PMU branch-current channels of snapshot t."""
        w, g = self.window, self.g
        sig_node, sig_edge = w.sigma[t]
        out: list[tuple[np.ndarray, float, int, float, float]] = []
        for b in range(g.C):  # the search rounds only an injection's change: no reading to add
            for col, row in zip((NODE.p_inj, NODE.q_inj), self.injection(t, b)):
                if w.node_m[b, col] > 0:
                    out.append((row, sig_node[b, col], b, w.prev.node[b, col], 0.0))
        for line in range(g.E):  # the search rounds both absolute flows before subtracting
            for col, row in enumerate(self.flow(t, line)):
                if w.edge_m[line, col] > 0:
                    reading = float(w.flows[t][line, col])
                    out.append(
                        (row, sig_edge[line, col], int(g.ei[0, line]), w.prev.edge[line, col], reading)
                    )
        return _Channels(out + self._currents(t))

    def _currents(self, t: int) -> list[tuple[np.ndarray, float, int, float, float]]:
        """The metered PMU branch-current channels of snapshot t (a current joins the PMU of the bus at
        its end)."""
        w, g = self.window, self.g
        if w.i_m is None or w.i_sigma is None:
            return []
        out: list[tuple[np.ndarray, float, int, float, float]] = []
        ends = ((CURRENT.re_from, CURRENT.im_from, self.Yf, 0), (CURRENT.re_to, CURRENT.im_to, self.Yt, 1))
        for line, (c_re, c_im, Ymat, end) in itertools.product(range(g.E), ends):
            for col, row in zip((c_re, c_im), self.layout.current_rows(Ymat[line])):
                if w.i_m[line, col] > 0:
                    before = 0.0 if w.prev.current is None else w.prev.current[line, col]
                    out.append((row, w.i_sigma[t][line, col], g.C + int(g.ei[end, line]), before, 0.0))
        return out

    def _nonlinear_devices(self) -> np.ndarray:
        """The devices of the area's metered |V| and angle channels (a PMU's at a PMU bus)."""
        w, C = self.window, self.g.C
        ids = [C + b if w.pmu[b] else b for b in self.area if w.node_m[b, NODE.v] > 0]
        ids += [C + b for b in self.area if w.node_m[b, NODE.theta] > 0]
        return np.array(ids, dtype=np.int64)

    def build(
        self,
        snapshots: Optional[list[int]] = None,
        integer: bool = True,
        cutoff: Optional[int] = None,
        objective=None,
    ) -> tuple:
        """(the cvxpy problem, x per snapshot [T, n], the device binaries) of the relaxation over
        `snapshots` (all when None), with the cut families in `cuts`. Dropping a snapshot drops its
        constraints, so the bound stays valid. `integer` off relaxes every binary to [0, 1];
        `cutoff` admits at most that many devices; `objective`, a vector over the first kept
        snapshot's x, replaces the device count (bound tightening). The flow goal's sector binaries
        and flows are kept in `sectors`."""
        import cvxpy as cp

        T, n = len(self.V), self.layout.n
        keep = list(range(T)) if snapshots is None else sorted(snapshots)
        X = cp.Variable((T, n))
        b = cp.Variable(len(self.devices), boolean=integer)
        y = cp.Variable(len(self.held), boolean=integer)
        cons = [cp.sum(b) >= 1] + ([] if integer else [b >= 0, b <= 1, y >= 0, y <= 1])
        cons += [] if cutoff is None else [cp.sum(b) <= cutoff]
        self.sectors, self.in_support, self._integer, self.aux = [], y, integer, {}
        for t in keep:
            cons += self._physics(X[t], t) + self._count(X[t], b, t) + self._goal(X[t], t)
            cons += self._residual(X[t], b, t) + self._support(X[t], y, t) + self._cuts(X[t], t)
            cons += self._flow_bounds(X[t], t)
            cons += self._step(X, t) if t == 0 or t - 1 in keep else []
        goal = cp.sum(b) if objective is None else objective @ X[keep[0]]
        return cp.Problem(cp.Minimize(goal), cons), X, b

    def _flow_bounds(self, x, t: int) -> list:
        """An overload goal's bounds on the injections the attack moves, as the search's flow solve
        holds them (`FalseStateMixin.solve_flow_local`, `_out_of_bounds`): the generator limits
        (22)-(23) and the load cap (D16), both over the area and its edge, every bus whose injection
        a support in the area can change; a bus the attack leaves alone meets them at its true
        value, so they hold for every support."""
        if self.window.goal.kind != "flow":
            return []  # the search's load-goal solve holds every non-target injection of S
        buses = self.g.touched_buses(self.area)
        return self._generator_bounds(x, t, buses) + self._load_cap_bounds(x, t, buses)

    def _generator_bounds(self, x, t: int, buses: np.ndarray) -> list:
        """Each generator's implied output P_gen - dP_inj, Q_gen - dQ_inj among `buses` inside its
        limits (22)-(23), widened to its true output as `within_limits` has it."""
        g, lim = self.g, self.window.k.limits
        if lim is None:
            return []
        gen = generator_output(self.window.states[t], g.load_base, g.gen_base)
        cons = []
        for b in np.intersect1d(buses, g.generator_buses()):
            box = ((float(lim.p_lo[b]), float(lim.p_hi[b])), (float(lim.q_lo[b]), float(lim.q_hi[b])))
            for c, row in enumerate(self.injection(t, int(b))):
                change = row @ (x - self.x0[t])  # load positive, so the output is gen - change
                cons += _output_in_box(change, float(gen[b, c]), box[c])
        return cons

    def _load_cap_bounds(self, x, t: int, buses: np.ndarray) -> list:
        """Each load bus among `buses` (no generator, not the slack) showing an active change of at
        most `load_cap` times its true load [YUA11] (D16)."""
        import cvxpy as cp

        g, cap = self.g, self.window.k.load_cap
        if cap is None:
            return []
        load = bus_load(self.window.states[t], g.load_base, g.gen_base)
        loads = np.setdiff1d(
            np.intersect1d(buses, np.flatnonzero(g.load_base[:, 0] != 0)), g.generator_buses()
        )
        cons = []
        for b in np.setdiff1d(loads, [g.slack_bus]):
            P, _ = self.injection(t, int(b))
            cons.append(cp.abs(P @ (x - self.x0[t])) <= cap * abs(float(load[b])) + POWER_TOL_MW)
        return cons

    def _cuts(self, x, t: int) -> list:
        """The cut families in `cuts` (engine/attacks/relax_cuts.py): the wedges of the tightened
        bounds, and the QC relaxation with or without bus angles (cycle)."""
        cons = relax_cuts.angle_cuts(self, x, t) if "bounds" in self.cuts else []
        if "qc" in self.cuts or "cycle" in self.cuts:
            cons += relax_cuts.qc(self, x, t, cycle="cycle" in self.cuts)
        return cons

    def solve(
        self,
        time_limit: float,
        snapshots: Optional[list[int]] = None,
        cutoff: Optional[int] = None,
        feastol: Optional[float] = None,
    ) -> tuple[str, float, Optional[np.ndarray], np.ndarray]:
        """(SCIP's status, its dual bound, the relaxed x per snapshot [T, n] or None, the device
        binaries) of the relaxation over `snapshots` (all when None), at most `cutoff` devices, with
        SCIP's feasibility and integrality tolerance numerics/feastol at `feastol` (None: SCIP's
        default)."""
        from cvxpy.error import SolverError

        prob, X, b = self.build(snapshots, cutoff=cutoff)
        solver = _bounding_scip()
        params: dict[str, float] = {"limits/time": float(time_limit)}
        if feastol is not None:
            params["numerics/feastol"] = float(feastol)
        try:
            prob.solve(solver=solver, scip_params=params)
        except SolverError:  # stopped with no feasible point: SCIP's status and dual bound still hold
            return solver.status, solver.dual_bound, None, np.zeros(len(self.devices))
        x = None if X.value is None else np.asarray(X.value)
        bv = np.zeros(len(self.devices)) if b.value is None else np.asarray(b.value)
        return solver.status, solver.dual_bound, x, bv

    def _physics(self, x, t: int) -> list:
        """The voltage box, the cones |W_ij|^2 <= W_ii W_jj and |V_i|^2 <= W_ii, and the zero
        injections of snapshot t."""
        import cvxpy as cp

        L = self.layout
        wlo, whi = self.box[t]
        cons = [x[L.w] >= wlo, x[L.w] <= whi]
        wa, wb = x[L.w[L.pair_a]], x[L.w[L.pair_b]]
        if len(L.pair_a):
            cons.append(cp.SOC(wa + wb, cp.vstack([2 * x[L.re], 2 * x[L.im], wa - wb]), axis=0))
        w = x[L.w]
        cons.append(cp.SOC(w + 1, cp.vstack([2 * x[L.e], 2 * x[L.f], w - 1]), axis=0))
        for bus in self.window.zero:
            for row in self.injection(t, int(bus)):
                if np.any(row):
                    cons.append(cp.abs(row @ (x - self.x0[t])) <= POWER_TOL_MW)
        return cons

    def _own_caps(self, t: int) -> dict[int, float]:
        """rho per PMU bus: how far V may sit from its true value while the PMU is untampered (its |V|
        within sigma and its angle within sigma); 0 at a bus outside the area, whose V is true."""
        w, area = self.window, set(self.area.tolist())
        sig = w.sigma[t][0]
        out: dict[int, float] = {}
        for bus in np.nonzero(w.pmu)[0]:
            if int(bus) not in area:
                out[int(bus)] = 0.0
            elif w.node_m[bus, NODE.v] > 0 and w.node_m[bus, NODE.theta] > 0:
                dv, dth = (float(s + roundoff_slack(s, 0.0)) for s in sig[bus, [NODE.v, NODE.theta]])
                v0, dth = w.states[t][bus, NODE.v], math.radians(dth)
                out[int(bus)] = max(
                    math.sqrt(v * v + v0 * v0 - 2 * v * v0 * math.cos(dth)) for v in (v0 - dv, v0 + dv)
                )
        return out

    def _caps(self, t: int) -> list[list[tuple[int, float]]]:
        """Per area bus, the (device, rho) pairs that cap |V_i - V_i^true| by rho while the device is
        untampered: the bus's own PMU, and a PMU at a neighbour j reading the current at its end of the
        branch between them (I = y_a V_j + y_b V_i, so |dV_i| <= (|dI| + |y_a| rho_j) / |y_b|)."""
        w, g, own = self.window, self.g, self._own_caps(t)
        caps: list[list[tuple[int, float]]] = [[] for _ in self.area]
        pos = {int(bus): a for a, bus in enumerate(self.area)}
        for bus, rho in own.items():
            if bus in pos:
                caps[pos[bus]].append((g.C + bus, rho))
        if w.i_m is not None and w.i_sigma is not None:
            self._current_caps(t, own, pos, caps)
        return caps

    def _current_caps(
        self, t: int, own: dict[int, float], pos: dict[int, int], caps: list[list[tuple[int, float]]]
    ) -> None:
        """Add the caps of the PMU branch currents to `caps` (`_caps`)."""
        w, g = self.window, self.g
        if w.i_m is None or w.i_sigma is None:
            return
        ends = (
            (0, self.Yf, [CURRENT.re_from, CURRENT.im_from]),
            (1, self.Yt, [CURRENT.re_to, CURRENT.im_to]),
        )
        for line, (end, Ymat, cols) in itertools.product(range(g.E), ends):
            j, i = int(g.ei[end, line]), int(g.ei[1 - end, line])
            if i not in pos or j not in own or not np.all(w.i_m[line, cols] > 0):
                continue
            ya, yb = abs(Ymat[line, self.lut[j]]), abs(Ymat[line, self.lut[i]])
            sig = float(np.hypot(*w.i_sigma[t][line, cols]))
            caps[pos[i]].append((g.C + j, (sig + ya * own[j]) / yb))

    def _targets(self) -> np.ndarray:
        """The buses of At's targeted loads (the same at every snapshot)."""
        designs = cast(LoadGoal, self.window.goal).designs
        return np.unique(np.concatenate([self.g.load_bus[d.targets] for d in designs]))

    def _support(self, x, y, t: int) -> list:
        """The search's support rule on the held buses, one binary y_i per bus for the window: a bus out
        of the support keeps its true voltage (|dV_i| <= R_i y_i), a bus in it keeps its true injection
        (|dS_i| <= M_i (1 - y_i)). A free injection bus may move either way."""
        import cvxpy as cp

        L, x0 = self.layout, self.x0[t]
        radius = self._radius(t)
        pos = {int(bus): a for a, bus in enumerate(self.area)}
        cons = []
        for k, bus in enumerate(self.held):
            a = pos[int(bus)]
            cons.append(x[L.r[a]] <= radius[L.r[a]] * y[k])
            for row in self.injection(t, int(bus)):
                M = float(np.abs(row) @ radius)
                cons.append(cp.abs(row @ (x - x0)) <= POWER_TOL_MW + M * (1 - y[k]))
        return cons

    def _residual(self, x, b, t: int) -> list:
        """Tie W to V where the meters pin V: r_i >= |V_i - V_i^true|, capped while a device is
        untampered (`_caps`), |X_ij| <= min over caps of rho_i r_j + (R_i - rho_i) R_j b_d (with R_i the
        most |dV_i| can be), X_ii <= rho_i^2 while the cap's device is untampered, and the 2x2 minors of
        X; all valid since a rank-one W has X = dV dV^H."""
        import cvxpy as cp

        L, V0, x0 = self.layout, self.V[t], self.x0[t]
        R = self._radius(t)[L.r]
        dev = {int(d): k for k, d in enumerate(self.devices)}
        caps = [[(b[dev[d]], rho) for d, rho in c if d in dev] for c in self._caps(t)]
        cons = [cp.norm(cp.vstack([x[L.e] - x0[L.e], x[L.f] - x0[L.f]]), axis=0) <= x[L.r], x[L.r] <= R]
        diag = [L.residual_rows(int(p), int(p), V0)[0] @ (x - x0) for p in L.area]
        cons += self._residual_caps(x, R, caps, diag)
        pairs = [(a, a) for a in range(len(self.area))] + list(zip(L.pair_a.tolist(), L.pair_b.tolist()))
        for a, c in pairs:
            cons += self._residual_pair(x, t, (a, c), (R, caps, diag))
        return cons

    def _residual_caps(self, x, R: np.ndarray, caps: list, diag: list) -> list:
        """r_i <= rho while a cap's device is untampered, and X_ii = |dV_i|^2 <= rho^2 then (R^2 always)."""
        L, cons = self.layout, []
        for a, c in enumerate(caps):
            cons += [x[L.r[a]] <= rho + (R[a] - rho) * d for d, rho in c]
            cons += [diag[a] <= R[a] ** 2] + [diag[a] <= rho**2 + (R[a] ** 2 - rho**2) * d for d, rho in c]
        return cons

    def _residual_pair(self, x, t: int, ends: tuple[int, int], held: tuple) -> list:
        """`_residual`'s cuts on X_ij of area buses a, c = `ends` (a = c for X_ii), with
        `held` = (R, caps, diag)."""
        import cvxpy as cp

        L, V0, x0 = self.layout, self.V[t], self.x0[t]
        (a, c), (R, caps, diag) = ends, held
        re, im = L.residual_rows(int(L.area[a]), int(L.area[c]), V0)
        size = cp.norm(cp.hstack([re @ (x - x0), im @ (x - x0)]))
        cons = [size <= R[a] * x[L.r[c]], size <= R[c] * x[L.r[a]]]
        cons += [size <= rho * x[L.r[c]] + (R[a] - rho) * R[c] * d for d, rho in caps[a]]
        if a != c:  # X = dV dV^H is positive semidefinite: |X_ij|^2 <= X_ii X_jj
            cons += [size <= rho * x[L.r[a]] + (R[c] - rho) * R[a] * d for d, rho in caps[c]]
            minor = cp.hstack([2 * re @ (x - x0), 2 * im @ (x - x0), diag[a] - diag[c]])
            cons.append(cp.norm(minor) <= diag[a] + diag[c])
        return cons

    def _count(self, x, b, t: int) -> list:
        """The big-M links: a device whose binary is 0 keeps every channel of snapshot t within noise."""
        import cvxpy as cp

        ch, dev = self.lin[t], np.searchsorted(self.devices, self.lin[t].device)
        wlo, whi = self.box[t]
        M = big_m(ch.A, self._radius(t), ch.sigma)
        cons = []
        if len(ch.sigma):
            cons.append(cp.abs(ch.A @ (x - self.x0[t])) <= ch.sigma + ch.slack + cp.multiply(M, b[dev]))
        return cons + self._voltage_count(x, b, t)

    def _voltage_count(self, x, b, t: int) -> list:
        """|V| stays in [v - sigma, v + sigma] and the angle within sigma of its true value unless its
        device is tampered: W_ii in the squared band, V_i in the wedge around the true angle, each
        sigma widened by its float32 roundoff slack (`roundoff_slack`)."""
        w, L = self.window, self.layout
        sig = w.sigma[t][0]
        wlo, whi = self.box[t]
        cons = []
        for a, bus in enumerate(self.area):
            v0, th0 = w.states[t][bus, NODE.v], math.radians(w.states[t][bus, NODE.theta])
            if w.node_m[bus, NODE.v] > 0:
                d = b[int(np.searchsorted(self.devices, self.g.C + bus if w.pmu[bus] else bus))]
                sv = float(sig[bus, NODE.v] + roundoff_slack(sig[bus, NODE.v], 0.0))
                hi, lo = (v0 + sv) ** 2, max(v0 - sv, 0.0) ** 2
                cons += [x[L.w[a]] <= hi + max(whi[a] - hi, 0.0) * d]
                cons += [x[L.w[a]] >= lo - max(lo - wlo[a], 0.0) * d]
            if w.node_m[bus, NODE.theta] > 0:
                d = b[int(np.searchsorted(self.devices, self.g.C + bus))]
                sth = float(sig[bus, NODE.theta] + roundoff_slack(sig[bus, NODE.theta], 0.0))
                cons += self._wedge(x, a, th0, math.radians(sth), (whi[a], d))
        return cons

    def _wedge(self, x, a: int, th0: float, delta: float, cap: tuple) -> list:
        """|Im(V e^{-j th0})| <= tan(delta) Re(V e^{-j th0}) + M d: V's angle within delta of th0."""
        import cvxpy as cp

        L, (whi, d) = self.layout, cap
        u = math.cos(th0) * x[L.e[a]] + math.sin(th0) * x[L.f[a]]
        p = -math.sin(th0) * x[L.e[a]] + math.cos(th0) * x[L.f[a]]
        M = math.sqrt(whi) * (1.0 + math.tan(delta))
        return [cp.abs(p) <= math.tan(delta) * u + M * d]

    def _goal(self, x, t: int) -> list:
        """The goal at snapshot t: each goal line's apparent flow on its target (Am), or the designed
        load change on the targeted buses (At)."""
        if self.window.goal.kind == "flow":
            goal = cast(FlowGoal, self.window.goal)
            cons = []
            for line, target in zip(goal.lines, goal.targets_at(t)):
                cons += self._flow_target(x, t, int(line), float(target))
            return cons
        return self._load_target(x, t)

    def _flow_target(self, x, t: int, line: int, target: float) -> list:
        """|S_l| = target, relaxed: |S_l| <= target is a cone, |S_l| >= target the sector disjunction."""
        import cvxpy as cp

        P, Q = self.flow(t, line)
        s_true = self._true_flow(t, line)
        s = cp.hstack([P @ (x - self.x0[t]) + s_true[0], Q @ (x - self.x0[t]) + s_true[1]])
        U, cos_k = sector_cuts()
        z = cp.Variable(len(U), boolean=self._integer)
        hi, lo = target + GOAL_TOL_MVA, max(target - GOAL_TOL_MVA, 0.0)
        self.sectors.append((z, s))
        cons = [cp.norm(s) <= hi, cp.sum(z) == 1, U @ s >= lo * cos_k - (hi + lo * cos_k) * (1 - z)]
        return cons + ([] if self._integer else [z >= 0, z <= 1])

    def _true_flow(self, t: int, line: int) -> tuple[float, float]:
        """The noiseless from-end flow of `line` in the true state (MW, MVAr)."""
        V, f = self.V[t], int(self.from_ppc[line])
        s = V[f] * np.conj(self.Yf[line] @ V) * self.g._base_mva
        return float(s.real), float(s.imag)

    def _load_target(self, x, t: int) -> list:
        """At: each targeted bus's active injection moves by the designed load change, its reactive
        injection not at all (the search solves the targets to exactly these)."""
        import cvxpy as cp

        g, design = self.g, cast(LoadGoal, self.window.goal).designs[t]
        Lp = g.true_load(self.window.states[t])
        dload = np.zeros(g.C)
        np.add.at(dload, g.load_bus[design.targets], Lp[design.targets] * (np.asarray(design.mult) - 1.0))
        cons = []
        for bus in np.unique(g.load_bus[design.targets]):
            P, Q = self.injection(t, int(bus))
            cons.append(cp.abs(P @ (x - self.x0[t]) - dload[bus]) <= POWER_TOL_MW)
            cons.append(cp.abs(Q @ (x - self.x0[t])) <= POWER_TOL_MW)
        return cons

    def _step(self, X, t: int) -> list:
        """At's stealth bound on the linear channels: no metered channel moves more than its rated
        accuracy between snapshots (from the frame before the window at t = 0), widened by the float32
        roundoff of both snapshots' attack values and of the step (`roundoff_slack`)."""
        import cvxpy as cp

        if not self.window.stealth_bound:
            return []
        ch = self.lin[t]
        if not len(ch.sigma):
            return []
        now = ch.A @ (X[t] - self.x0[t])
        before = ch.prev if t == 0 else self.lin[t - 1].A @ (X[t - 1] - self.x0[t - 1])
        scaled = self.window.k.stealth_scale * ch.sigma
        slack = 2 * ch.slack + roundoff_slack(scaled, 0.0)
        return [cp.abs(now - before) <= scaled + slack]

    def mismatch(self, x: np.ndarray, keep: list[int]) -> float:
        """The relaxed point's largest power mismatch (MW or MVAr) over the kept snapshots: each bus's
        injection change as W gives it against the exact change of the relaxed voltages V. Zero when W
        is V V^H on every injection, so the point is an AC state; the cones can be tight pair by pair
        while the cycles of the mesh are not."""
        worst = 0.0
        for t in keep:
            V = self.V[t].copy()
            V[self.layout.area] = x[t][self.layout.e] + 1j * x[t][self.layout.f]
            exact = bus_injections(V, self.Y, self.g._base_mva) - bus_injections(
                self.V[t], self.Y, self.g._base_mva
            )
            for bus in range(self.g.C):
                P, Q = self.injection(t, bus)
                dS = exact[self.lut[bus]]
                worst = max(
                    worst,
                    abs(P @ (x[t] - self.x0[t]) + np.real(dS)),
                    abs(Q @ (x[t] - self.x0[t]) + np.imag(dS)),
                )
        return worst

    def gap(self, x: np.ndarray) -> float:
        """The relaxed point's largest cone slack over the snapshots (0: every cone tight)."""
        L, worst = self.layout, 0.0
        for xt in x:
            w = xt[L.w]
            worst = max(worst, cone_gap(w[L.pair_a], w[L.pair_b], xt[L.re], xt[L.im]))
            worst = max(worst, cone_gap(w, np.ones(len(w)), xt[L.e], xt[L.f]))
        return worst


def _output_in_box(change, output: float, box: tuple[float, float]) -> list:
    """A generator component's implied output `output` - `change` inside `box` widened to `output`
    (the true value), with the search's tolerances; an infinite end adds nothing."""
    lo, hi = min(box[0], output), max(box[1], output)
    cons = []
    if np.isfinite(hi):
        cons.append(change >= output - hi - LIMIT_TOL_PQ - POWER_TOL_MW)
    if np.isfinite(lo):
        cons.append(change <= output - lo + LIMIT_TOL_PQ + POWER_TOL_MW)
    return cons


def certify(
    g: MinimizeMixin,
    states: list[np.ndarray],
    goal: Goal,
    k: FrameKnobs,
    prev: Optional[AttackVector] = None,
    options: Optional[CertifyOptions] = None,
) -> Optional[Certificate]:
    """The fewest-tamper search's attack on the window of `states` and `goal` (as `min_tamper` takes
    them), bounded from below by the relaxation over the same area; None when the goal has no area.
    The cone relaxation alone gives a first bound. With cut families (`CertifyOptions.cuts`), a
    second relaxation admits at most one device fewer than the search's attack: infeasible, at bound
    tightening or at the end, proves the search's count optimal; feasible, its optimum is the bound.
    A certificate is claimed only clear of SCIP's tolerances (`_verdict`): every bound is rounded
    with a margin, an infeasibility must survive a re-solve with loosened tolerances, and a cut
    relaxation whose bound falls below the cone relaxation's (cuts only shrink it) makes the verdict
    "uncertain". The search's forced-device bound is a floor."""
    _require_solver()
    CertifiableLimits(k.limits)  # refused before the search runs: the relaxation needs the limits
    # the window's length joins the options, so a kept snapshot outside it is refused on construction
    opts = dataclasses.replace(CertifyOptions() if options is None else options, window=len(states))
    seeds, _, _ = g._goal_seeds(goal)
    area = g.local_region(seeds, k.hops)
    if area is None:
        return None
    found = g.min_tamper(states, goal, k, prev)
    upper = -1 if found is None else int(found.devices)
    window = _Window(g, states, goal, k, prev=prev)
    keep = [_strongest(g, window)] if opts.snapshots is None else list(opts.snapshots)
    start = time.perf_counter()
    relax = _Relaxation(g, window, np.asarray(area))
    base = solve_claim(relax, keep, (None, upper), opts)
    cut = None
    if opts.cuts and upper >= 1:
        relax = _Relaxation(g, window, np.asarray(area))
        relax.cuts = tuple(opts.cuts)
        cut = solve_claim(relax, keep, (upper - 1, upper), opts, tighten(relax, keep, upper - 1, opts))
    seconds = time.perf_counter() - start
    lower, verdict, reason = _verdict(upper, base, cut, window.lower_bound())
    final = base if cut is None else cut
    support = np.zeros(0, dtype=np.int64) if upper < 0 else np.asarray(cast(MinimizerResult, found).support)
    x = final.x
    tight = (math.nan, math.nan) if x is None else (relax.gap(x[keep]), relax.mismatch(x, keep))
    return Certificate(
        upper,
        lower,
        verdict == CertifyVerdict.CERTIFIED.value,
        final.status,
        seconds,
        *tight,
        np.asarray(area),
        support,
        relax.devices[final.binaries > 0.5],
        verdict,
        reason,
        base.lower,
    )


def tighten(relax: _Relaxation, keep: list[int], cutoff: int, opts: CertifyOptions) -> bool:
    """Bound tightening at `cutoff` devices on the kept snapshots when the relaxation carries the
    "bounds" family (each solve stopped at `opts.tighten_limit`); False when it finds the relaxation
    infeasible, a claim `solve_claim` then has confirmed."""
    if "bounds" not in relax.cuts:
        return True
    return all(relax_cuts.tighten(relax, t, cutoff, opts.tighten_limit) for t in keep)


def solve_claim(
    relax: _Relaxation,
    keep: list[int],
    counts: tuple[Optional[int], int],
    opts: CertifyOptions,
    feasible: bool = True,
) -> BoundClaim:
    """What the relaxation over the kept snapshots proves, `counts` = (the cutoff, at most that many
    devices or None, and the search's count): its dual bound rounded with `opts.bound_margin`, or,
    when it is infeasible (or bound tightening found it so, `feasible` False), the search's count.
    A claim that would certify the search's count, by infeasibility or by an optimum that reaches
    the count, stands only when a re-solve with loosened tolerances certifies it too
    (`_confirm_infeasible`, `_confirm_optimum`)."""
    cutoff, upper = counts
    if feasible:
        status, bound, x, bv = relax.solve(opts.time_limit, keep, cutoff)
        if status != "infeasible":
            claim = BoundClaim(_rounded(bound, opts.bound_margin, upper), status, x, bv, "")
            return _confirm_optimum(relax, keep, counts, opts, claim) if 0 <= upper <= claim.lower else claim
    return _confirm_infeasible(relax, keep, counts, opts)


def _confirm_optimum(
    relax: _Relaxation,
    keep: list[int],
    counts: tuple[Optional[int], int],
    opts: CertifyOptions,
    claim: BoundClaim,
) -> BoundClaim:
    """An optimum that reaches the search's count at SCIP's default tolerances, re-solved with
    numerics/feastol loosened to `opts.robust_feastol`: when the loosened problem's bound still
    reaches the count, `claim` stands; otherwise the loosened bound is the claim, with the doubt
    stated (the loosened problem holds the default one, so its bound is the one clear of the
    tolerances)."""
    cutoff, upper = counts
    status, bound, x, bv = relax.solve(opts.time_limit, keep, cutoff, feastol=opts.robust_feastol)
    if status == "infeasible":  # a larger problem infeasible where the smaller one is not: numerical
        doubt = f"the relaxation is {claim.status} at SCIP's default tolerances but infeasible at feastol {opts.robust_feastol:g}"
        return claim._replace(doubt=doubt)
    loose = _rounded(bound, opts.bound_margin, upper)
    if loose >= upper:
        return claim
    doubt = f"the optimum reaches {claim.lower} at SCIP's default tolerances but {loose} at feastol {opts.robust_feastol:g}"
    return BoundClaim(loose, status, x, bv, doubt)


def _confirm_infeasible(
    relax: _Relaxation, keep: list[int], counts: tuple[Optional[int], int], opts: CertifyOptions
) -> BoundClaim:
    """An infeasibility at SCIP's default tolerances, re-solved with numerics/feastol loosened to
    `opts.robust_feastol`: still infeasible at cutoff c, it proves c + 1 devices (the search's count
    when c is one fewer); feasible, the loosened problem's dual bound is the claim, with the doubt
    stated. Without a cutoff, an infeasible relaxation contradicts the search's own attack, which is
    a point of it."""
    cutoff, upper = counts
    status, bound, x, bv = relax.solve(opts.time_limit, keep, cutoff, feastol=opts.robust_feastol)
    if status == "infeasible":
        doubt = ""
        if cutoff is None and upper >= 0:
            doubt = f"the relaxation admits no attack, though the search's attack of {upper} devices is a point of it"
        proved = upper if cutoff is None else (cutoff + 1 if upper < 0 else min(cutoff + 1, upper))
        return BoundClaim(proved, status, None, np.zeros(len(relax.devices)), doubt)
    doubt = f"infeasible at SCIP's default tolerances but {status} at feastol {opts.robust_feastol:g}"
    return BoundClaim(_rounded(bound, opts.bound_margin, upper), status, x, bv, doubt)


def _rounded(bound: float, margin: float, upper: int) -> int:
    """The device count a relaxation bound proves: ceil(bound - margin), never a plain rounding, so a
    bound just above an integer by SCIP's tolerances is not rounded past it (0 without a bound), and
    never above the search's count, a point of every relaxation."""
    if not math.isfinite(bound):
        return 0
    lower = int(math.ceil(bound - margin))
    return lower if upper < 0 else min(lower, upper)


def _verdict(upper: int, base: BoundClaim, cut: Optional[BoundClaim], forced: int) -> tuple[int, str, str]:
    """(the lower bound, the `CertifyVerdict`, the reason when uncertain) from the cone relaxation's
    claim `base`, the cut relaxation's `cut` (None without cuts) and the search's forced-device bound:
    "uncertain" on any doubt (`_doubts`), with the smaller of the two bounds kept; else "certified"
    when the bound meets the search's count, "gap" when it does not."""
    doubts = _doubts(base, cut)
    lower = base.lower if cut is None else (min(cut.lower, base.lower) if doubts else cut.lower)
    lower = max(lower, forced)
    if doubts:
        return lower, CertifyVerdict.UNCERTAIN.value, "; ".join(doubts)
    if upper >= 0 and lower >= upper:
        return lower, CertifyVerdict.CERTIFIED.value, ""
    return lower, CertifyVerdict.GAP.value, ""


def _doubts(base: BoundClaim, cut: Optional[BoundClaim]) -> list[str]:
    """Why the claims are not clear of SCIP's tolerances: each claim's own doubt (an infeasibility the
    loosened re-solve did not confirm), and a contradiction between the levels. Cuts only shrink the
    relaxation, so a cut bound below the cone bound, both at SCIP's default tolerances, is numerical;
    a doubted claim bounds a larger, loosened problem, so its lower value is doubt, not contradiction."""
    doubts = [c.doubt for c in (base, cut) if c is not None and c.doubt]
    if cut is not None and not doubts and cut.lower < base.lower:
        doubts.append(
            f"the cut relaxation's bound {cut.lower} is below the cone relaxation's {base.lower}, "
            "though cuts only shrink the relaxation"
        )
    return doubts


def _strongest(g: MinimizeMixin, window: _Window) -> int:
    """The snapshot where the goal moves furthest from the true state: the largest gap between a goal
    line's target and its true flow (Am), or the largest designed load multiplier change (At)."""
    if window.goal.kind == "flow":
        goal = cast(FlowGoal, window.goal)
        true = np.hypot(*np.moveaxis(window.flows[:, list(goal.lines)], 2, 0))  # [T, lines]
        targets = np.array([goal.targets_at(t) for t in range(len(window.states))])
        return int(np.argmax(np.max(np.abs(targets - true), axis=1)))
    designs = cast(LoadGoal, window.goal).designs
    return int(np.argmax([np.max(np.abs(np.asarray(d.mult) - 1.0)) for d in designs]))
