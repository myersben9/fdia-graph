"""Am, the multi-snapshot overload FDIA of [1] (TSG 2026, eqs. 12-25): over a window of snapshots the
reported apparent flow of one or two target branches is driven to their ratings, each snapshot the
reading of one AC false state so the operator's residual test stays quiet, while the true flow is below
the rating. The attack tampers the fewest distinct meters (eq. 12).

The attacker knows a local subnetwork around the target (`Attacker.area`, which may span utilities).
Inside it a support S (the buses whose voltages move) frees its attackable loads and generators and
drives the goal flow (`Attacker.false_flow_state`); every bus outside S holds its true voltage, so only
the meters near S change. A candidate search over supports, smallest first, keeps the one that tampers
the fewest meters beyond the noise over the window (eq. 12): spreading the change over more buses can
drop a channel back under the noise, so the cheapest support is not always the smallest. The goal rides
on each branch's own true drift and reaches S_max at the last snapshot (D9, eq. 25); there is no
per-snapshot increment bound (D11). Generators are free within their limits (22)-(23), edge generators
included, and a pretended load changes by at most load_cap times its true load (D16, Yuan TSG 2011,
ours). The paper overloads two lines at once (D17); this tries that first and falls back to one.

Ap, the pattern-aware overload (OURS, not [1]), is Am with one temporal constraint per bus and scan:
an attacker who knows each bus's typical five-minute load step keeps every scan-to-scan change of a
pretended injection inside it, |dD_i(k) - dD_i(k-1)| <= kappa s_i(hour(k)) with dD_i = 0 at the scan
before the window, so the attack ramps in the way the bus's own load moves. s_i(h) is the bus's 95th
percentile five-minute step at that hour over the 2019 pool (`natural_steps`). Only the active step
is bounded (Ben, 2026-10-27): operators forecast and check real power, and the fixed power factor of
the model's loads is a simplification, so a load's reactive change is limited only by the plausibility
cap. The P+Q variant (`reactive_step=True`, also reported in the paper) bounds a load's reactive step
too, in proportion to its true Q/P. A generator bus's column holds its units'
five-minute ramp (`generator_ramps`, NYgrid's maxRampAgc times 5): the change of its reported output
from one scan to the next (the true step plus the pretended one) stays within it, or within the
true step where the pool's dispatch itself moves faster, a nuclear unit not at all; hydro, imports and
units without ramp data keep (22)-(23) only (inf). A generator's reported reactive output moves per scan
by at most the same ramp (N27, the MW ramp taken as the MVAr one, since the data hold no reactive ramp).
The support search and its eq. 12 score are Am's. A support is feasible only when every scan meets the
goal inside every bound. With `bound_rise` False the rise is Am's.

Realism bounds (Ben, 2026-10-27, after the overloads of v7 proved to be mostly fake generator reactive
output and 50 percent load swings): every family's pretended generator reactive change stays within
`Attacker.gen_q_share` (0.2) of its units' Q range (N27), the load cap is 0.1 (N28), and with
`exit_scans` the window ends in an exit (N30): the rise run backwards, the reported flow falling from
the rating to the true flow in equal steps while every pretended injection stays in its step box, so
the attacker withdraws as smoothly as the bounds allow; an attack that still moves a meter beyond its
noise at the last scan does not exist. The exit needs `step_bound` for Am too, whose rise stays
unbounded (`bound_rise=False`).

Both families solve a support's scans in order, each warm-started from the previous scan's false
state carried onto this scan's true state, so the least-norm solution, and with it each pretended
injection, moves continuously through the window instead of being re-derived from the true state.

A target branch is eligible only when it lies inside New York (N16), its flow is metered (the goal is
what the operator sees), it is rated, and its true apparent flow stays below the rating at every
snapshot (else the goal is met with no attack).
"""

from __future__ import annotations

import heapq

import numpy as np

from fdia_graph.errors import ConfigError
from fdia_graph.nyfed.attacks.attacker import Attacker
from fdia_graph.nyfed.attacks.episode import AttackFamily, Episode, Window
from fdia_graph.nyfed.sensing.meters import PF

AM_LINE_TRIES = 8  # target sets an episode tries before it stays benign
MIN_BUDGET = 32  # candidate supports evaluated per target set
NEAR_LIMIT = 0.9  # N36: the share of its rating some candidate line carries when the attacker strikes


class MsFdia(AttackFamily):
    """The overload attack Am, or Ap with a step bound. Holds an `Attacker` and the branch `ratings` [L]
    (MVA, 0 unrated). Settings: `hops` for the area, `n_lines` targets per episode (2 by default, D17),
    `load_cap` the plausibility cap (0.1, D16 with N28), `budget` candidate supports searched, `tries` target sets,
    `target_order` how eligible lines are tried, "loading" (N26, ours) the most loaded line at the
    window's last scan first, the line an attacker who sees the area's flows would pick, or "random".
    Ap (ours): `step_bound [24, N]` each bus's natural five-minute step by hour (MW, `natural_steps`),
    `frame_hour [T]` the clock hour of every frame a window's onset indexes, `kappa` the share of the
    natural step a scan may use (1.0), `reactive_step` whether a load's reactive step is bounded too
    (the P+Q variant; False, the P-only Ap, by default), and `family` the name written on the episode.
    `bound_rise` whether the step bound holds during the rise (default: whenever `step_bound` is given;
    False for Am with an exit) and `exit_scans` the scans at the end of every window spent unwinding
    (N30, 0 for none). `wait_share` (N36, None for none) makes the attacker wait for a loaded grid: a
    window may start only where some candidate line carries that share of its rating at the last rise
    scan (`onset_mask`), since under the realistic bounds an overload exists almost only there."""

    def __init__(
        self,
        attacker: Attacker,
        ratings: np.ndarray,
        *,
        hops: int = 3,
        n_lines: int = 2,
        load_cap: float = 0.1,
        budget: int = MIN_BUDGET,
        tries: int = AM_LINE_TRIES,
        step_bound: np.ndarray | None = None,
        frame_hour: np.ndarray | None = None,
        kappa: float = 1.0,
        reactive_step: bool = False,
        family: str = "Am",
        target_order: str = "loading",
        bound_rise: bool | None = None,
        exit_scans: int = 0,
        wait_share: float | None = None,
    ):
        if exit_scans and step_bound is None:
            raise ConfigError("an exit (exit_scans) unwinds within the step bound, so it needs step_bound")
        if target_order not in ("loading", "random"):
            raise ConfigError(f"target_order must be 'loading' or 'random', not {target_order!r}")
        self.target_order = target_order
        self.att, self.ratings = attacker, np.asarray(ratings, float)
        self.hops, self.n_lines, self.load_cap = hops, n_lines, load_cap
        self.budget, self.tries = budget, tries
        if step_bound is not None and frame_hour is None:
            raise ConfigError("a step bound needs the clock hour of every frame (frame_hour)")
        self.step_bound = None if step_bound is None else np.asarray(step_bound, float)
        self.frame_hour = None if frame_hour is None else np.asarray(frame_hour, int)
        self.kappa, self.reactive_step, self.family = kappa, reactive_step, family
        self.bound_rise = step_bound is not None if bound_rise is None else bound_rise
        self.exit_scans = exit_scans
        self.wait_share = wait_share
        self._flow_branch = _flow_metered(attacker)

    def loaded_frames(self, vm: np.ndarray, va: np.ndarray, share: float, block: int = 4000) -> np.ndarray:
        """[T] whether some candidate target line (inside New York, flow-metered, rated) carries at least
        `share` of its rating in each frame of vm, va [T, N]."""
        att, case = self.att, self.att.net.case
        f, t, owner = case.f, case.t, att.owner
        lines = (owner[f] >= 0) & (owner[t] >= 0) & (f != t) & self._flow_branch & (self.ratings > 0)
        out = np.zeros(vm.shape[0], bool)
        for a in range(0, vm.shape[0], block):
            pf, qf, pt, qt = att.net.flows(vm[a : a + block], va[a : a + block])
            flow = np.maximum(np.hypot(pf, qf), np.hypot(pt, qt))[:, lines]
            with np.errstate(invalid="ignore"):  # unsolved frames (NaN) count as not loaded
                out[a : a + block] = (flow >= share * self.ratings[lines]).any(axis=1)
        return out

    def onset_mask(self, vm: np.ndarray, va: np.ndarray, length: int) -> np.ndarray | None:
        """[T] whether a window of `length` frames may start at each frame (N36): its last rise scan is a
        loaded frame (`loaded_frames` at `wait_share`). None when the family does not wait."""
        if self.wait_share is None:
            return None
        loaded = self.loaded_frames(vm, va, self.wait_share)
        rise = length - self.exit_scans
        mask = np.zeros(len(loaded), bool)
        if len(loaded) >= length:
            mask[: len(loaded) - length + 1] = loaded[rise - 1 : len(loaded) - length + rise]
        return mask

    def design(self, window: Window, rng: np.random.Generator) -> Episode | None:
        onset, vm, va = window
        lines = self._eligible(vm, va)
        if len(lines) == 0:
            self.last_miss = "no_target"
            return None
        rise = vm.shape[0] - self.exit_scans
        for targets in self._target_sets(self._order(lines, vm[:rise], va[:rise], rng)):
            ep = self._attack(onset, vm, va, np.asarray(targets, int))
            if ep is not None:
                return ep
        return None

    def _order(
        self, lines: np.ndarray, vm: np.ndarray, va: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """The eligible lines in the order they are tried (N26): by true apparent flow over rating at the
        last scan, where the goal (D9) must reach the rating, highest first, or at random."""
        if self.target_order == "random":
            return rng.permutation(lines)
        pf, qf, _, _ = self.att.net.flows(vm[-1:], va[-1:])
        return lines[np.argsort(-np.hypot(pf[0, lines], qf[0, lines]) / self.ratings[lines], kind="stable")]

    def _eligible(self, vm: np.ndarray, va: np.ndarray) -> np.ndarray:
        """Target branches worth trying: inside New York (both ends owned), flow-metered, rated, true
        apparent flow below the rating at every snapshot, and an area around an end holding a free
        injection to pretend."""
        att, case = self.att, self.att.net.case
        f, t, owner = case.f, case.t, att.owner
        internal = (owner[f] >= 0) & (owner[t] >= 0) & (f != t)
        pf, qf, pt, qt = att.net.flows(vm, va)
        peak = np.maximum(np.hypot(pf, qf), np.hypot(pt, qt)).max(axis=0)
        ok = internal & self._flow_branch & (self.ratings > 0) & (peak < self.ratings)
        return np.array([b for b in np.flatnonzero(ok) if self._reachable(int(b), vm[0], va[0])], int)

    def _reachable(self, line: int, vm: np.ndarray, va: np.ndarray) -> bool:
        area = self.att.area(int(self.att.net.case.f[line]), self.hops)
        return area is not None and len(self.att.free_injection_buses(vm, va, area)) > 0

    def _target_sets(self, order: np.ndarray) -> list[tuple[int, ...]]:
        """At most `tries` target sets from the eligible branches in `order` (D17): with `n_lines` 1,
        singles; with 2, pairs where the second line's ends lie inside the first's area so one support
        reaches both, then singles as a fallback (the paper's two-line default where feasible, else one,
        the coordinator's instruction)."""
        singles = [(int(b),) for b in order[: self.tries]]
        if self.n_lines == 1:
            return singles
        pairs: list[tuple[int, ...]] = []
        for a in order:
            second = self._partner(int(a), order)
            if second is not None and not any(set(p) == {int(a), second} for p in pairs):
                pairs.append((int(a), second))
            if len(pairs) == self.tries:
                break
        return pairs + singles

    def _partner(self, a: int, order: np.ndarray) -> int | None:
        case = self.att.net.case
        area = self.att.area(int(case.f[a]), self.hops)
        if area is None:
            return None
        inside = set(area.tolist())
        return next((int(b) for b in order if b != a and {int(case.f[b]), int(case.t[b])} <= inside), None)

    def _attack(self, onset: int, vm: np.ndarray, va: np.ndarray, targets: np.ndarray) -> Episode | None:
        att, case = self.att, self.att.net.case
        length = vm.shape[0]
        area = att.area(int(case.f[targets[0]]), self.hops)
        if area is None:
            return None
        # keep only target lines with an end inside the area (the support can control their flow)
        ends_in = np.array([t for t in targets if {int(case.f[t]), int(case.t[t])} & set(area.tolist())], int)
        if len(ends_in) == 0:
            return None
        targets = ends_in
        rise = length - self.exit_scans
        if rise < 1:
            return None
        pf, qf, _, _ = att.net.flows(vm, va)
        true_flow = np.hypot(pf[:, targets], qf[:, targets])  # [L, n_targets]
        rating = self.ratings[targets]
        gap = rating[None, :] - true_flow[rise - 1][None, :]
        # D9 schedule: the share of the gap rises to 1 at the last rise scan (S_max), then (N30) falls back
        # to 0 in equal steps over the exit, the rise run backwards
        share = np.r_[
            np.arange(1, rise + 1) / rise, 1 - np.arange(1, self.exit_scans + 1) / max(self.exit_scans, 1)
        ]
        goal = true_flow + share[:, None] * gap
        goal[rise - 1] = rating
        best = self._search(vm, va, area, targets, goal, self._bounds(onset, vm, va), rise)
        if best is None:
            return None
        a, support = best
        buses = att.free_injection_buses(vm[0], va[0], support)
        changed = np.flatnonzero(att.tampered(vm, va, a).any(axis=0))
        clients = np.unique(att.meas.plan.owner[changed])  # owners of the tampered meters (G5)
        return Episode(self.family, clients, onset, length, buses, int(targets[0]), a)

    def _bounds(self, onset: int, vm: np.ndarray, va: np.ndarray) -> tuple[np.ndarray, ...] | None:
        """Ap's per-scan step bounds (P [L, N] MW, Q [L, N] MVAr) and the shift of each box's centre from
        the previous scan's pretended change [L, N] MW, or None for Am: kappa times each bus's step
        bound at the scan's hour; a load's reactive step unbounded (P-only) or, with `reactive_step`, in
        proportion to its true Q/P at that scan; a generator's reactive output unbounded per step
        within its ramp (N27). A generator bus's bound is on
        its reported output, so its box is shifted by minus its true step (zero into the first scan,
        whose previous true state the window does not hold) and is never narrower than that true step,
        on its active and (N27) its reactive output alike:
        where the pool's own dispatch moves faster than the units' ramp (a commitment change), the
        attack may not add to it but need not hide it."""
        if self.step_bound is None:
            return None
        assert self.frame_hour is not None  # the constructor requires it with a step bound
        gen = self.att.gen_bus
        hours = self.frame_hour[onset : onset + vm.shape[0]]
        bound_p = self.kappa * self.step_bound[hours]
        p, q = self.att.net.injections(vm, va)
        ratio = np.abs(q) / np.where(np.abs(p) > 1e-6, np.abs(p), np.inf)
        with np.errstate(invalid="ignore"):  # an unbounded (inf) column times a zero ratio stays unbounded
            bound_q = np.where(np.isinf(bound_p), np.inf, bound_p * ratio)
        if not self.reactive_step:  # P-only: reactive load limited by the plausibility cap alone
            bound_q[:] = np.inf
        bound_q[:, gen] = bound_p[:, gen]  # N27: a generator's reactive output moves within its ramp
        shift = np.zeros(bound_p.shape, complex)
        true_p, true_q = np.diff(p[:, gen], axis=0), np.diff(q[:, gen], axis=0)
        shift[1:, gen] = -(true_p + 1j * true_q)
        bound_p[1:, gen] = np.maximum(bound_p[1:, gen], np.abs(true_p))
        bound_q[1:, gen] = np.maximum(bound_q[1:, gen], np.abs(true_q))
        return bound_p, bound_q, shift

    def _search(
        self,
        vm: np.ndarray,
        va: np.ndarray,
        area: np.ndarray,
        targets: np.ndarray,
        goal: np.ndarray,
        bounds: tuple[np.ndarray, ...] | None = None,
        rise: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The support that tampers the fewest meters over the window (eq. 12): candidates grow from the
        targets' end buses within the area, smallest first; each is solved at every snapshot and scored
        by the count of meters it moves beyond noise. Returns (attack vector [L, M], the support), or
        None when no candidate carries the goal at every snapshot inside the bounds. The snapshots are
        solved in order, each warm-started from the previous one's false state; with `bounds` each
        snapshot's box is centred on the previous one's injection change (zero before the first), over
        the rise when it is bounded and always over the exit scans after `rise` (N30), whose last scan
        must leave no meter moved beyond its noise."""
        rise = vm.shape[0] if rise is None else rise
        att, case, length = self.att, self.att.net.case, vm.shape[0]
        free = set(att.free_injection_buses(vm[0], va[0], area).tolist())
        if not free:
            return None
        start = frozenset(
            int(b) for b in np.unique(np.r_[case.f[targets], case.t[targets]]) if int(b) in set(area.tolist())
        )
        best_cost, best = None, None
        for support in self._candidates(area, start, free):
            a = np.zeros((length, len(att.meas.plan.kind)))
            ok = True
            prev = np.zeros(vm.shape[1], complex)  # Ap: the pretended injection change of the scan before
            warm = None  # the previous scan's false state, as a deviation from its true state
            for k in range(length):
                start = None if warm is None else (vm[k] + warm[0], va[k] + warm[1])
                boxed = bounds is not None and (k >= rise or self.bound_rise)
                step = (
                    (prev + bounds[2][k], bounds[0][k], bounds[1][k])
                    if boxed and bounds is not None
                    else None
                )
                res = att.false_flow_state(
                    vm[k], va[k], support, targets, goal[k], self.load_cap, step, start
                )
                if res is None:
                    ok = False
                    break
                a[k] = att.attack_vector(vm[k], va[k], res[0], res[1])
                warm = (res[0] - vm[k], res[1] - va[k])
                if bounds is not None:
                    pf, qf = att.net.injections(res[0], res[1])
                    pt, qt = att.net.injections(vm[k], va[k])
                    prev = (pf - pt) + 1j * (qf - qt)
            # N30: unwound by the last scan, no meter still moved beyond its noise
            if not ok or (self.exit_scans and att.tampered(vm[-1:], va[-1:], a[-1:]).any()):
                continue
            cost = int(att.tampered(vm, va, a).any(axis=0).sum())
            if cost >= 1 and (best_cost is None or cost < best_cost):
                best_cost, best = cost, (a, support)
        return best

    def _candidates(self, area: np.ndarray, start: frozenset[int], free: set[int]):
        """Supports inside the area, smallest first: the targets' end buses grown one adjacent area bus
        at a time, each support holding a free injection (a flow goal needs one to pretend). At most
        `budget` of them."""
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
            if s & free:  # a flow goal needs a free injection to pretend
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


def _flow_metered(att: Attacker) -> np.ndarray:
    """[L] True where a branch carries a from-end real-power flow meter (the goal is what the operator
    reads)."""
    plan = att.meas.plan
    metered = np.zeros(len(att.net.case.f), bool)
    metered[plan.branch[(plan.kind == PF) & (plan.end == 0)]] = True
    return metered
