"""The attacker: the local subnetwork it re-solves, and the false states its two goals imply.

An attacker knows one connected local subnetwork (the interior) and re-solves it with every bus
outside it held at its true voltage, so only the interior's and its boundary's meters change and the
measurement stays consistent with one AC state [1]. The subnetwork may span utilities: the operator
runs one residual test over every meter, so stealth is global, not per client. The boundary buses
hold their true voltage and absorb the changed power, so their injection moves too. The slack bus (the
estimator's angle reference) never moves.

Two goals re-solve the interior, through one solve (`_local_flow_solve`). The flow goal of Am and Ap
(`false_flow_state`) drives a target branch's apparent flow to a value; the load goal of At
(`false_state`) prescribes a pretended load at chosen buses. In both a New York bus on the interior's
edge whose injection may not change (a zero-injection bus, a load under the floor) first joins the
interior, its voltage free, so the edge moves out to buses the attack may change (`absorb`). The
attackable loads and generators of the interior and its edge are free, every other bus there keeps its
true injection (a zero-injection bus keeps reading zero: the operator meters P and Q at every New York
bus). A bus outside New York carries no meter and absorbs, unless it has no unit and no injection (bus
140, radial off Dunkirk, N44): nothing there could take up a change, so it is held like a New York
zero-injection bus (`_held_bus`). Then the least-norm Gauss-Newton moves the interior voltages; an active set holds each free
injection inside its bounds, generator limits (22)-(23) and the load plausibility cap on P and Q,
pinning one at a bound and releasing it when the solve without the pin leaves it inside. Each step is
halved until the residual drops, so a step too large to absorb fails cleanly instead of diverging.
`attacks` imports only `grid` and `sensing`, so the solve is plain numpy here, never pandapower.

The pattern-aware overload Ap (ours) adds one more box to that active set: each pretended load may
move from the previous scan's by at most the bus's natural five-minute step (`natural_steps`), and a
generator bus's reported output by at most its units' five-minute ramp (`generator_ramps`), so the
attack ramps the way the grid itself can move.

Angle readings (kinds va and ia) wrap at +-180 degrees, so a difference of two of them is wrapped
back to (-180, 180] before it is compared (`wrap_angles`); the stored readings stay raw.
"""

from __future__ import annotations

import numpy as np

from fdia_graph.nyfed.grid.case import REF, Case
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.meters import IA, VA

_BACKTRACK_HALVINGS = 8  # step fractions tried per Newton iteration: 1, 1/2, ... 1/128


def natural_steps(time: np.ndarray, p: np.ndarray, solved: np.ndarray, pct: float = 95.0) -> np.ndarray:
    """Each bus's natural five-minute step by hour of day [24, N] (MW): the `pct` percentile of
    |p(t) - p(t-1)| over every pair of consecutive solved frames, grouped by the clock hour of the later
    frame. `time [T]` the frame times (local clock), `p [T, N]` the net injections (MW, NaN where
    unsolved), `solved [T]` which frames hold a state. A load bus's step is its load's, a generator bus's
    its dispatch's. This is what a pattern-aware attacker is assumed to know of typical load behaviour
    (Ap, ours); an hour with no pair keeps 0."""
    hours = (np.asarray(time).astype("datetime64[s]").astype(np.int64) // 3600) % 24
    ok = np.asarray(solved, bool)
    pair = ok[1:] & ok[:-1]
    step = np.abs(np.diff(np.asarray(p, float), axis=0))[pair]
    hour = hours[1:][pair]
    out = np.zeros((24, p.shape[1]))
    for h in range(24):
        if (hour == h).any():
            out[h] = np.percentile(step[hour == h], pct, axis=0)
    return out


HYDRO_RAMP_PER_MIN = 0.10  # share of nameplate per minute (USBR, Hydro Model Data Project, 2013, Tables 1-2)
IMPORT_RAMP_MW_PER_MIN = (
    300.0 / 15
)  # NY real-time interchange ramp per interface (ISO-NE CROP.31001, 300 MW per 15 min)
BASEPOINT_BAND = 0.03  # share of the upper operating limit (NYISO MST Rate Schedule 3-A, 15.3A.1)
Q_SHARE = 0.312  # |Q| per MW of capability at power factor 0.95 (NYISO Manual 2, 3.6; FERC Order 827)


def _per_bus(case: Case, per_unit: np.ndarray) -> np.ndarray:
    out = np.zeros(len(case.bus_id))
    np.add.at(out, case.unit_bus, per_unit)
    has_unit = np.zeros(len(case.bus_id), bool)
    has_unit[case.unit_bus] = True
    out[~has_unit] = np.inf  # no generator to bound
    return out


def generator_ramps(case: Case, minutes: float = 5.0) -> np.ndarray:
    """Each bus's generator ramp capability over `minutes` [N] (MW), summed over its units, each unit
    bounded on its own (N34): the load-following (AGC) ramp `case.ramp` (MW/min, NYgrid's
    maxRampAgc); a nuclear unit 0, since it does not follow load; a hydro unit with no ramp data 10
    percent of its nameplate per minute; an import 300 MW per 15 minutes; any other unit with no ramp
    data (external units) unbounded. A bus with no unit is inf. Every family bounds a generator bus's
    per-scan change of reported output by this (N33)."""
    ramp = np.asarray(case.ramp, float).copy()
    ramp = np.where((case.fuel == "Hydro") & np.isnan(ramp), HYDRO_RAMP_PER_MIN * case.pmax, ramp)
    ramp = np.where((case.fuel == "Import") & np.isnan(ramp), IMPORT_RAMP_MW_PER_MIN, ramp)
    ramp = np.where(case.fuel == "Nuclear", 0.0, ramp) * minutes
    return _per_bus(case, np.where(np.isnan(ramp), np.inf, ramp))


def generator_band(case: Case) -> np.ndarray:
    """[N] MW: the largest pretended deviation of a generator bus's output from its basepoint (N33), 3
    percent of its units' upper operating limit, NYISO's tolerance for a unit's output against its
    real-time dispatch basepoint; inf at a bus with no unit."""
    return _per_bus(case, BASEPOINT_BAND * np.asarray(case.pmax, float))


def generator_q_capability(case: Case) -> tuple[np.ndarray, np.ndarray]:
    """[N] MVAr each: the reactive capability of a generator bus (N34), plus or minus 0.312 times its
    units' Pmax (power factor 0.95 leading and lagging), replacing NYgrid's placeholder limits of plus
    or minus 9999 MVAr per unit; plus or minus inf at a bus with no unit."""
    q = _per_bus(case, Q_SHARE * np.asarray(case.pmax, float))
    return -q, q


def wrap_angles(d: np.ndarray, kind: np.ndarray) -> np.ndarray:
    """A difference of readings `d [..., M]` with each angle reading's (kinds va and ia, degrees) wrapped
    to (-180, 180]; `kind [M]` the meter kinds. The other readings are returned as they are."""
    d = np.array(d, float, copy=True)
    angle = np.isin(kind, (VA, IA))
    d[..., angle] = -((-d[..., angle] + 180.0) % 360.0 - 180.0)
    return d


def _injection_jacobian(
    y_ri: np.ndarray, v_r: np.ndarray, v_i: np.ndarray, i_r: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """The [2r, 2k] Jacobian of the injections at buses R (inside the interior or on its edge) in the
    interior's [theta_I, |V|_I], from S_r = V_r conj(sum_j Y_rj V_j); `y_ri` are R's rows of Ybus on
    the interior's columns, `i_r = (Ybus V)_R`, and `d[r, i] = 1` where R's bus r is interior bus i."""
    a = 1j * v_r[:, None] * (np.conj(i_r)[:, None] * d - np.conj(y_ri * v_i[None, :]))  # dS/dtheta
    b = (
        v_r[:, None] * np.conj(y_ri * (v_i / np.abs(v_i))[None, :])
        + (np.conj(i_r) * v_r / np.abs(v_r))[:, None] * d
    )
    r, k = y_ri.shape
    j = np.empty((2 * r, 2 * k))
    j[:r, :k], j[:r, k:], j[r:, :k], j[r:, k:] = a.real, b.real, a.imag, b.imag
    return j


def _local_flow_solve(
    ybus: np.ndarray,
    yf: np.ndarray,
    from_bus: np.ndarray,
    v: np.ndarray,
    interior: np.ndarray,
    fixed: np.ndarray,
    s_fixed: np.ndarray,
    target: np.ndarray,
    hold: np.ndarray,
    vm_fixed: tuple[np.ndarray, np.ndarray],
    iters: int = 60,
    tol: float = 1e-9,
) -> np.ndarray | None:
    """The false voltages that drive each goal branch's from-end apparent flow to `target` while the
    buses of `fixed` keep their injection, every bus outside `interior` held true [1] eqs. 24-25. Fewer
    equations than the 2|interior| unknowns, so the least-norm Gauss-Newton step is taken, backtracked
    until the residual drops.

    yf, from_bus : [L, N] goal branches' from-end admittance rows and their from buses
    interior     : buses whose voltage moves (the support)
    fixed        : buses whose injection is held (interior or edge); `s_fixed` their injections (pu),
                   `hold [len(fixed), 2]` which of P and Q each holds
    target       : [L] goal apparent flows (pu); `vm_fixed = (buses, |V|)` held magnitudes (pu)
    """
    v = np.array(v, np.complex128, copy=True)
    i_ = np.asarray(interior, int)
    fx = np.asarray(fixed, int)
    k = len(i_)
    pos = {int(b): p for p, b in enumerate(i_)}
    y_ri = ybus[np.ix_(fx, i_)]
    d = (fx[:, None] == i_[None, :]).astype(float)
    keep = np.concatenate([hold[:, 0], hold[:, 1]]).astype(bool)
    yf_i = yf[:, i_]
    f_at = np.array([pos.get(int(b), -1) for b in from_bus], int)
    vb, vm_val = vm_fixed
    vb_at = np.array([pos[int(b)] for b in vb], int)

    def residual(vv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        yv = ybus @ vv
        mis = s_fixed - (vv * np.conj(yv))[fx]
        inj = np.concatenate([np.real(mis), np.imag(mis)])[keep]
        flow = np.abs(vv[from_bus] * np.conj(yf @ vv)) - target
        vmr = np.abs(vv[vb]) - vm_val
        return np.concatenate([inj, flow, vmr]), yv

    r, yv = residual(v)
    for _ in range(iters):
        norm = float(np.max(np.abs(r)))
        if norm < tol:
            return v
        jinj = _injection_jacobian(y_ri, v[fx], v[i_], yv[fx], d)[keep]
        rows = [-jinj]
        for y_row, yi, fb, at in zip(yf, yf_i, from_bus, f_at):
            iff = y_row @ v
            sf = v[fb] * np.conj(iff)
            dth = -1j * v[fb] * np.conj(yi * v[i_])
            dvm = v[fb] * np.conj(yi * v[i_] / np.abs(v[i_]))
            if at >= 0:
                dth[at] += 1j * sf
                dvm[at] += (v[fb] / abs(v[fb])) * np.conj(iff)
            mag = max(abs(sf), 1e-12)
            rows.append(np.concatenate([np.real(np.conj(sf) * dth), np.real(np.conj(sf) * dvm)])[None] / mag)
        vm_rows = np.zeros((len(vb_at), 2 * k))
        vm_rows[np.arange(len(vb_at)), k + vb_at] = 1.0
        jac = np.vstack(rows + [vm_rows])
        step = np.linalg.lstsq(jac, -r, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            return None
        vm0, va0, accepted = np.abs(v[i_]), np.angle(v[i_]), None
        for _ in range(_BACKTRACK_HALVINGS):
            mag, ang = vm0 + step[k:], va0 + step[:k]
            if np.all(mag > 0):
                vtry = v.copy()
                vtry[i_] = mag * np.exp(1j * ang)
                tried, yv_try = residual(vtry)
                if np.max(np.abs(tried)) < norm:
                    accepted = (vtry, tried, yv_try)
                    break
            step = step / 2
        if accepted is None:
            return None
        v, r, yv = accepted
    return None


class Attacker:
    """Holds a `Network` and a `Measurement`. Finds the local subnetwork of a target (`area`, the
    connected region of [1], which may span utilities), re-solves it under a pretended load change
    (`false_state`, the load goal of At) or a target branch flow (`false_flow_state`, the flow goal of
    Am), and writes the attack vector those states move (`attack_vector`). Meter ownership comes from
    the measurement's plan, so the attacker needs no separate `Ownership`: a bus's client is the owner
    of its meters (a bus with no meters is external, N4), used only to label an episode's owners."""

    def __init__(
        self, net: Network, meas: Measurement, gen_q_share: float = 0.2, hold_external: bool = False
    ):
        self.net, self.meas = net, meas
        # N62: every bus outside New York keeps its true injection, so the change lands on New York's own
        # loads and units and the net interchange stays true up to losses (each tie on its own may still
        # move where one external bus meets several New York buses, so this is a relaxation of holding
        # every tie, and an attack it rules out is ruled out for the stricter form too)
        self.hold_external = hold_external
        # a generator's pretended reactive change stays within this share of its units' Q range (N27,
        # Ben 2026-10-27): without it the cheapest stealthy overload is fake reactive output
        self.gen_q_share = gen_q_share
        case, plan = net.case, meas.plan
        self.ybus = net.ybus.toarray()  # dense: a ~140-bus local solve indexes and multiplies often
        self.yf = net.yf.toarray()
        self.from_bus = case.f
        n = len(case.bus_id)
        # a bus's client from its meters (-1 outside New York)
        owner = np.full(n, -1)
        owner[plan.bus] = plan.owner
        self.owner = owner
        self.slack = int(np.flatnonzero(case.bus_type == REF)[0])
        self.gen_bus = np.zeros(n, bool)
        self.gen_bus[case.unit_bus] = True
        self.base_mva = float(case.base_mva)
        self._adj = net.adjacency
        # per-bus generator capability (MW, MVAr), summed over the units at a bus; widened in the flow
        # solve to include the frame's own output so the true operating point is always feasible
        self.p_lo, self.p_hi, self.q_lo, self.q_hi = (np.zeros(n) for _ in range(4))
        for arr, src in (
            (self.p_lo, case.pmin),
            (self.p_hi, case.pmax),
            (self.q_lo, case.qmin),
            (self.q_hi, case.qmax),
        ):
            np.add.at(arr, case.unit_bus, src)
        # a channel's rated-accuracy standard deviation (the noise floor a change must clear to count)
        self._sigma_rel, self._sigma_abs = plan.sigma_rel, plan.sigma_abs

    def sigma(self, z: np.ndarray) -> np.ndarray:
        """The accuracy-class standard deviation of each meter at readings `z`, the noise a change must
        exceed to count as tampering (`sigma_rel |z| + sigma_abs`, the rated accuracy of M2)."""
        return self._sigma_rel * np.abs(z) + self._sigma_abs

    def _held_bus(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        """[N] bool: the buses whose injection the attack keeps true unless it is free. Every New York
        bus (the operator meters P and Q there), and a bus outside New York with no unit and no injection
        at the state (N44, bus 140 radial off Dunkirk): no meter sees it, but with no load or generation
        nothing there could take up a change, so it is held like a New York zero-injection bus. Every
        other bus outside New York absorbs, unless `hold_external` (N62) holds them all."""
        v = vm * np.exp(1j * np.deg2rad(va))
        dead = ~self.gen_bus & (np.abs(v * np.conj(self.ybus @ v)) < 1e-6)  # pu, the solver's zero
        return (self.owner >= 0) | dead | (self.hold_external & (self.owner < 0))

    def absorb(self, vm: np.ndarray, va: np.ndarray, interior: np.ndarray) -> np.ndarray:
        """The interior grown over its edge's buses whose injection the attack may not change
        (zero-injection buses and loads under the floor; N25, the coordinator's decision; outside New York
        only the dead buses of `_held_bus`, N44): such a bus moves inside, its voltage free and its
        injection still held, so the edge moves out to the next ring, until every edge bus is a free
        injection, the slack, or a bus outside New York that absorbs (no meter sees it and the estimator
        holds it, the limit of the growth)."""
        s = np.asarray(interior, int)
        free = np.zeros(len(self.owner), bool)
        free[self.free_injection_buses(vm, va, np.arange(len(self.owner)))] = True
        held = self._held_bus(vm, va)
        while True:
            edge = self.boundary(s)
            move = edge[held[edge] & ~free[edge] & (edge != self.slack)]
            if self.hold_external:  # N62: a bus outside New York stays on the edge, its injection held by the
                move = move[self.owner[move] >= 0]  # solve, so every tie keeps its true flow
            if not len(move):
                return s
            s = np.union1d(s, move)

    def boundary(self, interior: np.ndarray) -> np.ndarray:
        """The buses one branch outside `interior` (their voltage is held true, their injection
        absorbs the change)."""
        return np.setdiff1d(self.net.neighbors(interior, 1), interior)

    def load_buses(
        self, vm: np.ndarray, va: np.ndarray, buses: np.ndarray | None = None, floor: float = 1.0
    ) -> np.ndarray:
        """The New York buses carrying load above `floor` MW at a state, the slack and generator buses
        out (both trivially detectable to attack [1]); restricted to `buses` when given. Load is read
        from the state, not the case, whose base carries none."""
        load = self.bus_load(vm, va)
        ok = (self.owner >= 0) & ~self.gen_bus & (load > floor)
        ok[self.slack] = False
        if buses is not None:
            keep = np.zeros_like(ok)
            keep[np.asarray(buses, int)] = True
            ok &= keep
        return np.flatnonzero(ok)

    def free_injection_buses(self, vm: np.ndarray, va: np.ndarray, buses: np.ndarray) -> np.ndarray:
        """The buses of `buses` whose injection a flow attacker may change [1]: the attackable loads
        and the generators (the slack out). A zero-injection bus is held at its true injection."""
        gens = np.asarray(buses, int)[self.gen_bus[np.asarray(buses, int)]]
        gens = gens[gens != self.slack]
        if self.hold_external:  # N62: no unit outside New York takes up the change
            gens = gens[self.owner[gens] >= 0]
        return np.union1d(self.load_buses(vm, va, buses), gens)

    def area(self, center: int, hops: int) -> np.ndarray | None:
        """The attacker's local subnetwork around `center` [1]: the buses within `hops` branches, over
        any utility (the subnetwork may span them), the slack excluded, with a non-empty boundary to
        hold at its true voltage. `hops` is reduced until a bus with a boundary remains; None when even
        one hop leaves none (the center isolated, or the slack)."""
        if center == self.slack:
            return None
        for h in range(hops, 0, -1):
            interior = self.net.neighbors([center], h)
            interior = interior[interior != self.slack]
            if self.hold_external:  # N62: no bus outside New York in the area
                interior = interior[self.owner[interior] >= 0]
            if len(interior) and len(self.boundary(interior)):
                return interior
        return None

    def false_state(
        self, vm: np.ndarray, va: np.ndarray, area: np.ndarray, load_change: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The false state a pretended load change implies (the load goal of At): each bus where
        `load_change` is nonzero reads its true injection less that change (extra load, generation
        negative), the other attackable loads and generators of the `area` and its edge absorb the
        rest within the same bounds and active set as the flow goal, and every other bus there, a
        zero-injection bus among them, keeps its true injection; every voltage outside the area is held
        true. Holding the edge leaves more equations than unknowns, so the solve is the least-squares
        Gauss-Newton of the flow goal with no target line. `load_change [N]` is complex, MW + j MVAr,
        nonzero only on `area` buses. Returns the false state (`vm` p.u., `va` degrees) [N], or None
        when no solution inside the bounds is found."""
        res = self._goal_solve(
            vm, va, area, np.zeros(0, int), np.zeros(0), held=-np.asarray(load_change, complex)
        )
        return None if res is None else (res[0], res[1])

    def false_flow_state(
        self,
        vm: np.ndarray,
        va: np.ndarray,
        support: np.ndarray,
        lines: np.ndarray,
        target_mva: np.ndarray,
        load_cap: float = 0.5,
        step: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        start: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The false state that drives the `lines` from-end apparent flows to `target_mva` on the
        `support` [1] eqs. 24-25. A New York bus on the support's edge whose injection may not change (a
        zero-injection bus, a load under the floor) first joins the support, its voltage free, until the
        edge holds only free injections (`absorb`, N25). The attackable loads and generators of the
        support and its edge are free (`free_injection_buses`); every other bus there, a zero-injection
        bus among them, keeps its true injection, since the operator meters P and Q at every New York
        bus; each bus outside the support holds its true voltage. Each free injection change stays inside
        its bounds: a generator's reported output inside its limits (22)-(23), a load's active and
        reactive change within `load_cap` times its true active load and max(|Q|, 0.2 P) (the plausibility
        cap of Yuan, IEEE TSG 2011, ours, not the paper's; on Q by N24), and with `step = (centre [N] complex MVA, bound_p [N] MW, bound_q [N] MVAr)` Ap's box
        (ours): each free injection change within `bound` of `centre`. An active set holds them there:
        an injection past a bound is pinned at it, and a pinned one is released when the solve without
        its pin leaves it inside its bounds. `start = (vm, va)` warm-starts the solve inside the support
        (the previous scan's false state carried onto this one) and the active set with it (an injection
        on a bound there starts pinned on that side), so successive scans stay continuous.
        Returns (`vm` p.u., `va` degrees [N], the pretended active load change per bus [N] MW), or None
        when no solution inside the bounds is found (the support cannot carry the goal)."""
        return self._goal_solve(vm, va, support, lines, target_mva, load_cap, step, start)

    def _goal_solve(
        self,
        vm: np.ndarray,
        va: np.ndarray,
        support: np.ndarray,
        lines: np.ndarray,
        target_mva: np.ndarray,
        load_cap: float = 0.5,
        step: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        start: tuple[np.ndarray, np.ndarray] | None = None,
        held: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """The solve behind both goals: `lines` driven to `target_mva` (the flow goal, none for the load
        goal) with the injection change `held [N]` (complex MVA, generation positive) prescribed where it
        is nonzero (the load goal's pretended loads), the other free injections bounded by the active
        set, every other bus of the support and its edge held at its true injection."""
        case, base = self.net.case, self.base_mva
        v = vm * np.exp(1j * np.deg2rad(va))
        s = self.absorb(vm, va, np.asarray(support, int))  # non-free edge buses join the interior
        edge = self.boundary(s)
        touched = np.union1d(s, edge)
        free = self.free_injection_buses(vm, va, touched)
        if held is None and not len(np.intersect1d(free, s)):  # a flow goal needs a free injection
            return None
        held_bus = np.zeros(0, int) if held is None else np.flatnonzero(held)
        held_set = set(held_bus.tolist())
        held_mva = np.zeros(len(vm), complex) if held is None else held  # read only at held_set's buses
        free = np.setdiff1d(free[free != self.slack], held_bus)
        # every New York bus there that is not free holds its true injection (zero-injection ones too),
        # and so does a dead bus outside New York (N44); any other bus outside New York carries no meter,
        # so on the edge it absorbs the change as before
        rest = np.setdiff1d(touched, np.union1d(free, held_bus))
        rest = rest[self._held_bus(vm, va)[rest]]
        s_true = v * np.conj(self.ybus @ v)  # pu, generation positive, per bus
        gen_p, gen_q = s_true.real * base, s_true.imag * base  # gen output proxy at a generator bus (MW/MVAr)
        true_load = self.bus_load(vm, va)
        lines = np.atleast_1d(np.asarray(lines, int))
        target = np.atleast_1d(np.asarray(target_mva, float)) / base
        v0 = v.copy()
        if start is not None:  # warm start inside the support only; every other voltage stays true
            v0[s] = start[0][s] * np.exp(1j * np.deg2rad(start[1][s]))

        # the bounds of each free injection change (MVA, generation positive): (bus, comp) -> (lo, hi)
        bounds: dict[tuple[int, int], tuple[float, float]] = {}
        for bus in free.tolist():
            if self.gen_bus[bus]:  # limits (22)-(23), widened to the frame's own output
                p_b = (
                    min(self.p_lo[bus], gen_p[bus]) - gen_p[bus],
                    max(self.p_hi[bus], gen_p[bus]) - gen_p[bus],
                )
                q_b = (
                    min(self.q_lo[bus], gen_q[bus]) - gen_q[bus],
                    max(self.q_hi[bus], gen_q[bus]) - gen_q[bus],
                )
                dq = self.gen_q_share * (self.q_hi[bus] - self.q_lo[bus])  # N27
                q_b = (max(q_b[0], -dq), min(q_b[1], dq))
            else:  # load cap [YUA11]
                # P and Q (N24): the reactive cap has a floor of 0.2 times the active load, so a load
                # near unity power factor may still move its Q (Ben, 2026-10-27)
                cap, cap_q = load_cap * true_load[bus], load_cap * max(abs(gen_q[bus]), 0.2 * true_load[bus])
                p_b, q_b = (-cap, cap), (-cap_q, cap_q)
            for comp, (lo, hi) in enumerate((p_b, q_b)):
                if step is not None:  # Ap: within the step box around the centre
                    c = step[0][bus].real if comp == 0 else step[0][bus].imag
                    w = (step[1] if comp == 0 else step[2])[bus]
                    lo, hi = max(lo, c - w), min(hi, c + w)
                    if lo > hi + 1e-9:
                        return None
                bounds[(bus, comp)] = (float(lo), float(hi))

        vlo = np.minimum(case.vmin, vm)
        vhi = np.maximum(case.vmax, vm)
        held_v: dict[int, float] = {}

        def solve(pins: dict[tuple[int, int], float]) -> np.ndarray | None:
            """The flow solve with `rest` held at its true injection and each pin at its value."""
            pinned = {b for b, _ in pins}
            fixed = np.array(sorted(set(rest.tolist()) | pinned | held_set), int)
            whole = set(rest.tolist()) | held_set
            change = np.array(
                [
                    held_mva[int(b)]
                    if int(b) in held_set
                    else pins.get((int(b), 0), 0.0) + 1j * pins.get((int(b), 1), 0.0)
                    for b in fixed
                ],
                complex,
            )
            hold = np.array(
                [
                    [int(b) in whole or (int(b), 0) in pins, int(b) in whole or (int(b), 1) in pins]
                    for b in fixed
                ],
                bool,
            ).reshape(-1, 2)
            vb = np.array(sorted(held_v), int)
            return _local_flow_solve(
                self.ybus,
                self.yf[lines],
                self.from_bus[lines],
                v0,
                s,
                fixed,
                s_true[fixed] + change / base,
                target,
                hold,
                (vb, np.array([held_v[int(b)] for b in vb])),
            )

        def moved(vf: np.ndarray, key: tuple[int, int]) -> float:
            ds = (vf[key[0]] * np.conj(self.ybus[key[0]] @ vf) - s_true[key[0]]) * base
            return float(ds.real if key[1] == 0 else ds.imag)

        pins: dict[tuple[int, int], float] = {}
        if start is not None:  # warm-start the active set too: an injection on or past a bound at the
            for key, (lo, hi) in bounds.items():  # carried-over state starts pinned on that side
                m = moved(v0, key)
                if m <= lo + 1e-3:
                    pins[key] = lo
                elif m >= hi - 1e-3:
                    pins[key] = hi
        releases: dict[tuple[int, int], int] = {}
        for _ in range(4 * len(bounds) + 2 * len(s) + 2):
            vf = solve(pins)
            if vf is None:
                return None
            over = False
            for key, (lo, hi) in bounds.items():  # pin each free injection past a bound at that bound
                if key not in pins:
                    m = moved(vf, key)
                    if m < lo - 1e-6 or m > hi + 1e-6:
                        pins[key], over = float(np.clip(m, lo, hi)), True
            mag = np.abs(vf[s])
            for bus, m in zip(s.tolist(), mag):  # voltage limits (21)
                if bus not in held_v and (m < vlo[bus] - 1e-6 or m > vhi[bus] + 1e-6):
                    held_v[bus], over = float(np.clip(m, vlo[bus], vhi[bus])), True
            if over:
                continue
            # release: a pin whose own unconstrained solve leaves it strictly inside its bounds
            released = False
            for key in sorted(pins):
                if releases.get(key, 0) >= 2:  # no cycling: a pin is released at most twice
                    continue
                trial = solve({k: x for k, x in pins.items() if k != key})
                if trial is None:
                    continue
                lo, hi = bounds[key]
                if lo + 1e-6 < moved(trial, key) < hi - 1e-6:
                    del pins[key]
                    releases[key] = releases.get(key, 0) + 1
                    released = True
                    break
            if released:
                continue
            dload = np.zeros(len(vm))
            loads = np.setdiff1d(free, np.flatnonzero(self.gen_bus))
            dload[loads] = -(((vf * np.conj(self.ybus @ vf)) - s_true)[loads].real * base)
            return np.abs(vf), np.degrees(np.angle(vf)), dload
        return None

    def attack_vector(
        self, vm_t: np.ndarray, va_t: np.ndarray, vm_f: np.ndarray, va_f: np.ndarray
    ) -> np.ndarray:
        """The attack vector a = h(x_false) - h(x_true) [M], in full on every meter it moves, whoever owns
        it: the operator runs one global residual test, so the scan must read as one AC state (global
        stealth). A change within a meter's noise is still written, since zeroing it would break that
        consistency and turn a smooth ramp into steps, but it does not count as tampered (`tampered`, the
        l0 threshold of [1] eq. 12)."""
        return self.meas.h(vm_f, va_f) - self.meas.h(vm_t, va_t)

    def tampered(self, vm: np.ndarray, va: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Which meters an attack moves beyond their noise, per frame [L, M] for true states [L, N] and
        attack vectors [L, M]: the meters [1] eq. 12 counts, the localization labels and their owners.
        An angle change is wrapped first, so a reading crossing +-180 degrees counts by its true move."""
        return np.abs(wrap_angles(a, self.meas.plan.kind)) > self.sigma(self.meas.h(vm, va))

    def bus_load(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        """The active load at every bus (MW) for a state: the net injection is generation minus load,
        and the pool scales a bus's load and generation together, so at a load bus with no generator the
        load is minus the net injection."""
        p, _ = self.net.injections(vm, va)
        return np.where(self.gen_bus, 0.0, -p)
