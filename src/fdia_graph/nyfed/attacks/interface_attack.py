"""The overload attack on a NYISO interface (N48, N49, ours): Am with OpfFdia's false states and every one of
its bounds, the goal moved from one branch's rating to the flow across an interface.

Goal (N49). The attacker strikes an interface that is already close to its limit and closes the last gap
in above-noise steps over the rise. The model's interface flows differ from the real 2019 ones (N48), so
the reported interface flow at the last rise scan is the model's true flow plus that hour's real headroom
(the real limit minus the real flow): the attack adds exactly what the real grid had left. The goal rises
to it in equal steps and, with the exit (N30), falls back to the true flow.

Overshoot (N58, ours). With `overshoot` (MW) above zero the goal is the limit plus that much, so the operator
reads an interface over its limit and has to act: the damaging variant. `max_overshoot` finds the most an
onset allows under every bound, by bisection on the overshoot. At zero the goal is N49's exactly.

Onsets (N48). A window may start where the last rise scan falls in an hour whose real 2019 flow on some
interface is at least `near` of its real limit.

Search. One interface's branches can sit far apart, so the attacker's area is anchored on one interface
branch at a time and grows its supports from that branch's ends (all of the interface's end buses at
once would force supports larger than needed). The interface branches with an end in the support's
interior move with the false voltages; the others keep their true flow. Each statically feasible support
is solved over the window first with the meter objective, which adds to OpfFdia's injection terms the
change beyond noise on every P and Q flow meter of the branches touching the interior and every voltage
meter of the interior (the l1 stand-in for eq. 12's count of meters, of which the injection terms alone
cover only part), and with OpfFdia's objective where Ipopt finds no point with it. The window that tampers
the fewest meters over every anchor is kept (eq. 12).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fdia_graph.nyfed.attacks.attacker import Attacker
from fdia_graph.nyfed.attacks.episode import Episode, Window
from fdia_graph.nyfed.attacks.opf_attack import OpfFdia, _Window
from fdia_graph.nyfed.grid.interfaces import Interface
from fdia_graph.nyfed.sensing.meters import KINDS

KV, KPF, KQF = KINDS.index("v"), KINDS.index("pf"), KINDS.index("qf")


@dataclass
class _IfaceWindow(_Window):
    """`_Window` plus the interface goal: the interface branches the area can move (`inner`, `inner_sign`),
    the true flow of the others [L] (`offset`), the goal [L] and the true reading of every meter [L, M]."""

    inner: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    inner_sign: np.ndarray = field(default_factory=lambda: np.zeros(0))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(0))
    igoal: np.ndarray = field(default_factory=lambda: np.zeros(0))
    h_true: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    _still: list[tuple[int, float]] = field(default_factory=list)  # inner branches the support leaves true
    _meters: list[int] = field(default_factory=list)  # the meter of each objective term


class InterfaceFdia(OpfFdia):
    """Am against `interfaces` (from `hourly_interfaces`, over the same frames as the pool the onsets
    index), with OpfFdia's bounds and settings; `near` the real loading an onset needs (N48, 0.9).
    `windows_solved` (6) caps the statically feasible supports per anchor that get the full window."""

    def __init__(
        self,
        attacker: Attacker,
        interfaces: list[Interface],
        *,
        near: float = 0.9,
        windows_solved: int = 6,
        overshoot: float = 0.0,
        **kwargs,
    ):
        kwargs.setdefault("family", "Am")
        super().__init__(
            attacker, np.zeros(len(attacker.net.case.f)), windows_solved=windows_solved, **kwargs
        )
        self.interfaces, self.near, self.overshoot = interfaces, near, overshoot
        self._yt = attacker.net.yt.toarray()
        self._meter_obj = True

    # ------------------------------------------------------------------ onsets and design
    def _loaded(self, frame: int) -> list[Interface]:
        """The interfaces whose real flow is at least `near` of their real limit at `frame`, the most
        loaded first."""
        out = []
        for it in self.interfaces:
            lim, flow = it.limit[frame], it.flow[frame]
            if np.isfinite(lim) and np.isfinite(flow) and lim > 0 and flow >= self.near * lim:
                out.append((flow / lim, it))
        return [it for _, it in sorted(out, key=lambda x: -x[0])]

    def onset_mask(self, vm: np.ndarray, va: np.ndarray, length: int) -> np.ndarray:
        """[T] whether a window of `length` frames may start at each frame: its last rise scan is in an
        hour of real loading at least `near` on some interface (N48)."""
        T = vm.shape[0]
        if any(len(it.flow) != T for it in self.interfaces):
            raise ValueError("onset_mask needs the states of the frames the interfaces were built over")
        rise = length - self.exit_scans
        loaded = np.zeros(T, bool)
        for it in self.interfaces:
            with np.errstate(invalid="ignore"):
                loaded |= (it.flow >= self.near * it.limit) & (it.limit > 0)
        mask = np.zeros(T, bool)
        if T >= length:
            mask[: T - length + 1] = loaded[rise - 1 : T - length + rise]
        return mask

    def design(self, window: Window, rng: np.random.Generator) -> Episode | None:
        onset, vm, va = window
        rise = vm.shape[0] - self.exit_scans
        loaded = self._loaded(onset + rise - 1)
        if not loaded:
            self.last_miss = "no_target"
        for it in loaded:
            ep = self._attack_interface(onset, vm, va, it)
            if ep is not None:
                return ep
        return None

    def max_overshoot(
        self, window: Window, rng: np.random.Generator, high: float = 800.0, tol: float = 25.0
    ) -> tuple[Episode | None, float]:
        """The damaging variant at its most (N58): the largest overshoot (MW, within `tol`, at most `high`)
        for which the window still has an attack under every bound, with that attack. Bisection, so about
        log2(high / tol) designs. Returns (None, 0.0) where even the limit itself is out of reach."""
        keep = self.overshoot
        try:
            self.overshoot = 0.0
            best = self.design(window, rng)
            if best is None:
                return None, 0.0
            lo, hi, best_mw = 0.0, high, 0.0
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                self.overshoot = mid
                ep = self.design(window, rng)
                if ep is None:
                    hi = mid
                else:
                    lo, best, best_mw = mid, ep, mid
            return best, best_mw
        finally:
            self.overshoot = keep

    def _attack_interface(self, onset: int, vm: np.ndarray, va: np.ndarray, it: Interface) -> Episode | None:
        att = self.att
        best_cost, best = None, None
        for anchor in it.branches.tolist():
            prep = self._prepare_interface(onset, vm, va, it, anchor)
            if prep is None:
                continue
            w, area, start, free = prep
            solved = 0
            for support in self._candidates(area, start, free):
                if solved >= self.windows_solved:
                    break
                self._meter_obj = False  # static feasibility: the bounds alone decide it
                end = self._static(w, support)
                if end is None:
                    continue
                solved += 1
                for meter_obj in (True, False):
                    self._meter_obj = meter_obj
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
                        best_cost, best = cost, (a, buses, tam, anchor)
                    break  # the meter objective's window stands; the fallback is only for its failures
        self._meter_obj = True
        if best is None:
            return None
        a, buses, tam, anchor = best
        clients = np.unique(att.meas.plan.owner[np.flatnonzero(tam.any(axis=0))])
        return Episode(self.family, clients, onset, vm.shape[0], buses, int(anchor), a)

    def _prepare_interface(self, onset, vm, va, it: Interface, anchor: int):
        att, case = self.att, self.att.net.case
        L = vm.shape[0]
        rise = L - self.exit_scans
        area = att.area(int(case.f[anchor]), self.hops)
        if area is None or rise < 1:
            return None
        aset = set(area.tolist())
        inner = np.array(
            [i for i, b in enumerate(it.branches.tolist()) if {int(case.f[b]), int(case.t[b])} & aset], int
        )
        if len(inner) == 0:
            return None
        pf, _, _, _ = att.net.flows(vm, va)
        true = pf[:, it.branches] * it.signs  # [L, n] each branch's flow in the positive direction
        total = true.sum(axis=1)
        frame = onset + rise - 1
        headroom = it.limit[frame] - it.flow[frame]  # N49: what the real grid had left that hour
        if not np.isfinite(headroom) or headroom <= 0:
            return None
        share = np.r_[
            np.arange(1, rise + 1) / rise, 1 - np.arange(1, self.exit_scans + 1) / max(self.exit_scans, 1)
        ]
        goal = total + share * (headroom + self.overshoot)
        p, q = att.net.injections(vm, va)
        hours = None if self.frame_hour is None else self.frame_hour[onset : onset + L]
        gen = None if self.gen_output is None else self.gen_output[onset : onset + L]
        h_true = att.meas.h(vm, va)
        w = _IfaceWindow(
            onset,
            vm,
            np.deg2rad(va),
            p,
            q,
            att.bus_load(vm, va),
            att.sigma(h_true),
            hours,
            np.zeros(0, int),
            np.zeros((L, 0)),
            rise,
            gen,
            inner=it.branches[inner],
            inner_sign=it.signs[inner],
            offset=np.delete(true, inner, axis=1).sum(axis=1),
            igoal=goal,
            h_true=h_true,
        )
        free = set(self._free(vm[0], va[0], area).tolist())
        anchor_ends = {int(case.f[anchor]), int(case.t[anchor])} & aset
        return (w, area, frozenset(anchor_ends), free) if free and anchor_ends else None

    # ------------------------------------------------------------------ the OPF on one support
    def _model(self, w: _IfaceWindow, support: np.ndarray):
        """OpfFdia's model with no branch goal (w.targets is empty), plus the interface goal and, with
        `_meter_obj`, the flow and voltage meter terms of the objective."""
        import pyomo.environ as pe

        ctx = super()._model(w, support)
        m, interior = ctx[0], ctx[1]
        att, case, base = self.att, self.att.net.case, self.att.base_mva
        inside, outside = set(interior.tolist()), set(ctx[4])

        def V(b):
            if b in inside:
                return m.vm[b], m.th[b]
            return (m.vo[b], m.to[b]) if b in outside else None

        def flow(b: int, end: int):
            """The (P, Q) expression (MW, MVAr) at one end of branch b, or None if an end is not in the model."""
            at, y = (int(case.f[b]), att.yf[b]) if end == 0 else (int(case.t[b]), self._yt[b])
            cols = [int(case.f[b]), int(case.t[b])]
            vf, vt = V(cols[0]), V(cols[1])
            if vf is None or vt is None:
                return None
            va_, ta = vf if at == cols[0] else vt
            p = q = 0
            for c, (vc, tc) in zip(cols, (vf, vt)):
                d = ta - tc
                p += va_ * vc * (y[c].real * pe.cos(d) + y[c].imag * pe.sin(d))
                q += va_ * vc * (y[c].real * pe.sin(d) - y[c].imag * pe.cos(d))
            return p * base, q * base

        moving = [
            (b, s)
            for b, s in zip(w.inner.tolist(), w.inner_sign.tolist())
            if int(case.f[b]) in inside or int(case.t[b]) in inside
        ]
        m.fixed_flow = pe.Param(mutable=True, initialize=0.0)  # the inner branches this interior leaves true
        m.igoal = pe.Param(mutable=True, initialize=0.0)
        sent = 0
        for b, s in moving:
            pq = flow(b, 0)
            assert pq is not None  # a moving branch has an interior end, so both its ends are modelled
            sent += s * pq[0]
        m.c.add(sent + m.fixed_flow == m.igoal)
        w._still = [(b, s) for b, s in zip(w.inner.tolist(), w.inner_sign.tolist()) if (b, s) not in moving]

        w._meters = []
        if self._meter_obj:
            plan = att.meas.plan
            terms = []
            for b in range(len(case.f)):
                if int(case.f[b]) not in inside and int(case.t[b]) not in inside:
                    continue
                for end in (0, 1):
                    pq = flow(b, end)
                    if pq is None:
                        continue
                    for kind, ex in ((KPF, pq[0]), (KQF, pq[1])):
                        j = np.flatnonzero((plan.kind == kind) & (plan.branch == b) & (plan.end == end))
                        if len(j):
                            terms.append((int(j[0]), ex))
            for b in inside:
                j = np.flatnonzero((plan.kind == KV) & (plan.bus == b))
                if len(j):
                    terms.append((int(j[0]), m.vm[b]))
            w._meters = [j for j, _ in terms]
            m.MJ = pe.Set(initialize=list(range(len(terms))))
            m.mt = pe.Param(m.MJ, mutable=True, initialize=0.0)
            m.ms = pe.Param(m.MJ, mutable=True, initialize=0.0)
            m.um = pe.Var(m.MJ, within=pe.NonNegativeReals)
            for i, (_, ex) in enumerate(terms):
                m.c.add(m.um[i] >= ex - m.mt[i] - m.ms[i])
                m.c.add(m.um[i] >= m.mt[i] - ex - m.ms[i])
            m.obj.deactivate()
            m.meter_obj = pe.Objective(expr=m.obj.expr + sum(m.um[i] for i in m.MJ))
        return ctx

    def _set_scan(self, w: _IfaceWindow, ctx, k, prev, static, start=None, end=None) -> bool:
        if not super()._set_scan(w, ctx, k, prev, static, start, end):
            return False
        m = ctx[0]
        pf, _, _, _ = self.att.net.flows(w.vm[k : k + 1], np.rad2deg(w.th[k : k + 1]))
        still = sum(s * pf[0, b] for b, s in w._still)
        m.fixed_flow = float(w.offset[k] + still)
        m.igoal = float(w.igoal[k])
        for i, j in enumerate(w._meters):
            m.mt[i] = float(w.h_true[k, j])
            m.ms[i] = float(w.sig[k, j])
        return True
