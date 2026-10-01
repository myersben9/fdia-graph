"""`Am`, the multi-snapshot overload attack of [WU26]: the reported flows of its target branches (two
by default, as the paper's case studies overload two lines at once, or one; `OverloadSettings.n_lines`,
the plan's D17) driven to their ratings over the window, with the fewest devices tampered.

[WU26, eqs. 24-25] asks that the apparent flow the tampered measurements carry on a target line grow
snapshot by snapshot until it reaches the line's rating S_max. The rating is by default 1.25 times the
branch's peak true flow over the operating pool (the plan's D15, `use_line_ratings`), or on request the
branch's PGLib-OPF `rate_a` (`fdia_graph.ratings`; the IEEE cases pandapower ships rate every branch
9,900 MVA, which no flow comes near). A branch is a target for a window only when it is rated, its flow is metered
(the goal is what the operator sees) and its true flow stays below the rating at every snapshot of
the window (otherwise the goal is met with no attack). The window's T snapshots are kappa+1 ... kappa+T, kappa the untouched reference
snapshot before it (eq. 25 sums the increments from kappa+1), and the goal at snapshot t (the plan's
D9) is

    S_{l,t} = S_true_{l,t} + (t - kappa)/T (S_max - S_true_{l,kappa+T}),   t = kappa+1 ... kappa+T

on the noiseless reading of the false state: the true flow plus a share of what separates the last
snapshot's true flow from the rating, so the attack adds a steady ramp on top of the load's own
drift instead of cancelling it, and reaches S_max at the window's last snapshot (eq. 25). As in
[WU26], nothing bounds how far the attack moves between snapshots: the paper's noise only decides
which changes its tamper count ignores (`formulas.noise.paper_sigma`, the plan's D8 and D11), and the
attack is stealthy because every snapshot's readings are those of one AC state. Each snapshot's false
state frees the attackable loads and the generators of the support (the latter within their limits,
eqs. 22-23) and holds every other bus's injection
(`FalseStateMixin.solve_flow_local`), and the fewest-tamper search (`MinimizeMixin.min_tamper`)
chooses the support held for the window. The labels of every frame are the free-injection buses of the support: the loads and
generator outputs the attacker pretends.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...formulas.attacks import branch_ratings, generator_output
from ...models.config import OverloadSettings
from ...models.errors import NoLineRatings
from ...models.frames import AmOverloadDesign, AttackVector, FlowGoal, Frame, FrameKnobs
from ...models.grid import NODE
from ...ratings import pglib_branches
from .rref import RrefMixin

# How many target sets (one line, or two, D17) an Am episode tries, drawn from one random order of the
# eligible branches drawn once at onset. A bounded heuristic, not a search over every set: each try is
# a full fewest-tamper search (about 13 s on IEEE-118), so an episode whose first eight sets admit no
# attack stays benign and is counted even if a later set would have.
AM_LINE_TRIES = 8

# [WU26]'s case studies, in MATPOWER bus numbers (`wu26_branch` maps a pair to our branch by its end
# buses): the two lines each scenario overloads at once, and the PMU buses of its metering.
WU26_SCENARIOS: dict[int, tuple[tuple[tuple[int, int], tuple[int, int]], ...]] = {
    14: (((3, 4), (6, 11)), ((1, 2), (4, 5))),  # Section V-A
    118: (((84, 85), (99, 100)),),  # Section V-B, inside the attacker's local network of Fig. 9
}
WU26_PMUS: dict[int, tuple[int, ...]] = {
    14: (1, 4, 6, 13),  # Section V-A, the trusted-PMU sequence 1 -> 4 -> 6 -> 13
    # Fig. 9 ("IEEE 118-bus system topology and PMU distribution", page 12 of the published PDF), the
    # buses drawn in red: 11 bars, each read against its bus label at 1,200 dpi; bus 94's label sits
    # beside bus 95's bar, but the red bar is 94's
    118: (76, 78, 80, 83, 89, 92, 94, 100, 105, 106, 110),
}
# Fig. 9's red dashed region, the attacker's local network on IEEE-118: every bus drawn inside it
WU26_ATTACK_AREA: dict[int, tuple[int, ...]] = {
    118: (
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
    )
    + (101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 118),
}


def _schedule(flows: np.ndarray, rating: float) -> tuple[float, ...]:
    """One branch's goal over a window from its true flows [T, 2] (MW, MVAr): the true flow plus a
    linear share of what separates the last true flow from the rating (the plan's D9). The window's
    snapshots are kappa+1 ... kappa+T after the untouched reference kappa [WU26, eq. 25], so snapshot
    k of the window (k = 1 ... T) carries the share k/T: every snapshot moves the flow and the last
    reaches the rating."""
    true = np.hypot(flows[:, 0], flows[:, 1])
    share = np.arange(1, len(true) + 1) / len(true)
    return tuple(float(x) for x in true + share * (rating - true[-1]))


class OverloadMixin(RrefMixin):
    """Design and step the overload attack of [WU26] on a rated, metered target branch."""

    def use_line_ratings(self, settings: OverloadSettings, X: np.ndarray) -> None:
        """Set the ratings the overload attack drives its lines to (the plan's D15): with "pool", each
        branch's `rating_margin` times its peak true apparent flow over the pool X [T, N, 4], at the
        from end as the goal reads it (a static rating per line, every system); with "pglib", the
        PGLib-OPF ratings (`line_ratings`, NoLineRatings on a case without them)."""
        if settings.rating_source == "pglib":
            self._line_ratings = None
            self.line_ratings()
            return
        peak = np.zeros(self.E)
        for a in range(0, len(X), 4096):  # the pool in slabs: a 72k-state pool is one pass
            peak = np.maximum(peak, np.abs(self.all_flows_from_states(X[a : a + 4096])).max(axis=0))
        self._line_ratings = settings.rating_margin * peak

    def line_ratings(self) -> np.ndarray:
        """[E] each branch's rating in MVA: the ratings `use_line_ratings` set, else PGLib-OPF's `rate_a`
        (matched by end buses), NaN where unrated. Raises `NoLineRatings` on a case the package holds
        no PGLib-OPF ratings for."""
        cached = getattr(self, "_line_ratings", None)
        if cached is not None:
            return cached
        rows = pglib_branches(self.C)
        if rows is None:
            raise NoLineRatings(
                f"IEEE-{self.C} has no line ratings (the overload attack Am needs PGLib-OPF ratings, "
                "held for IEEE-14, 118 and 300); generate without Am, e.g. families=('At',)"
            )
        number = self.base.bus["name"].astype(int).to_numpy()  # MATPOWER bus numbers, pandapower order
        ends = [(int(number[a]), int(number[b])) for a, b in self.ei.T]
        self._line_ratings = branch_ratings(ends, rows)
        return self._line_ratings

    def eligible_lines(self, window: list[np.ndarray], hops: int) -> np.ndarray:
        """The branches an overload attack may target over the snapshots `window`: in service, rated, flow
        metered, true apparent flow below the rating at every snapshot, and an area around its ends
        holding a free injection (an attackable load or a generator). An out-of-service branch (an N-1 outage) keeps its row with a zero
        admittance, so its flow reads zero and no false state can drive it to its rating."""
        rating = self.line_ratings()
        flows = self.clean_flows_from_states(np.stack(window))  # [T, E, 2], unmetered zeroed
        S = np.hypot(flows[..., 0], flows[..., 1]).max(axis=0)  # [E] the window's largest true flow
        metered = np.asarray(self.meters.flow, bool)
        status = self.branch.status
        live = np.ones(len(rating), bool) if status is None else np.asarray(status) > 0
        ok = live & np.isfinite(rating) & metered & (S < np.nan_to_num(rating, nan=-np.inf))
        free = set(self.free_injection_buses().tolist())
        return np.array(
            [int(b) for b in np.flatnonzero(ok) if self._acts_on(int(b), free, hops)], dtype=np.int64
        )

    def _acts_on(self, line: int, free: set[int], hops: int) -> bool:
        """Whether the area within `hops` of the branch's ends holds a free injection to pretend."""
        area = self.local_region(np.unique(self.ei[:, line]), hops)
        return area is not None and bool(free & {int(b) for b in area})

    def overload_goal(self, window: list[np.ndarray], line: int, *more: int) -> FlowGoal:
        """The per-snapshot flow the attack must reach on `line`, the first target, over `window`
        (module docstring), and on each branch of `more` at once, each toward its own rating. The
        generator's episodes pass `OverloadSettings.n_lines` targets: two by default, as the paper's
        case studies overload two lines at once, or one (the plan's D17)."""
        flows = self.clean_flows_from_states(np.stack(window))  # [T, E, 2] MW, MVAr
        rating = self.line_ratings()
        first = _schedule(flows[:, line], float(rating[line]))
        return FlowGoal(
            line, first, more=tuple((int(b), _schedule(flows[:, b], float(rating[b]))) for b in more)
        )

    def am_overload_design(
        self, X: np.ndarray, t: int, length: int, k: FrameKnobs, prev: Optional[AttackVector] = None
    ) -> Optional[AmOverloadDesign]:
        """One `Am` episode starting at t as the overload attack of [WU26]: at most `AM_LINE_TRIES` of the
        eligible branches, in a random order (one draw, only when there is one), each tried until the fewest-tamper search
        finds a support that meets the goal at every snapshot, inside the operating limits, and
        moves at least one device beyond noise. None when no branch has one (the
        span then stays benign and is counted)."""
        frames = range(t, min(t + length, len(X)))
        window = [X[u] for u in frames]
        lines = self.eligible_lines(window, k.hops)
        if len(lines) == 0:
            return None
        for targets in self._target_sets(self.rng.permutation(lines), k.n_lines, k.hops):
            goal = self.overload_goal(window, *targets)
            result = self.min_tamper(window, goal, k, prev)
            if result is not None and result.devices >= 1:
                ratings = tuple(float(self.line_ratings()[b]) for b in targets)
                return AmOverloadDesign(goal, ratings, result.support, result)
        return None

    def _target_sets(self, order: np.ndarray, n_lines: int, hops: int) -> list[tuple[int, ...]]:
        """At most `AM_LINE_TRIES` target sets from the eligible branches in `order` (D17): single
        branches, or pairs one held support can reach, each first branch taken in order with the
        first other branch whose ends lie inside the first one's attack area (the buses within
        `hops` of its ends), each pair once."""
        if n_lines == 1:
            return [(int(b),) for b in order[:AM_LINE_TRIES]]
        pairs: list[tuple[int, ...]] = []
        for a in order:
            second = self._partner(int(a), order, hops)
            if second is not None and not any(set(p) == {int(a), second} for p in pairs):
                pairs.append((int(a), second))
            if len(pairs) == AM_LINE_TRIES:
                break
        return pairs

    def _partner(self, a: int, order: np.ndarray, hops: int) -> Optional[int]:
        """The first branch of `order` other than a whose two ends lie inside a's attack area (the
        buses within `hops` of a's ends), or None."""
        area = self.local_region(np.unique(self.ei[:, a]), hops)
        if area is None:
            return None
        inside = {int(b) for b in area}
        return next((int(b) for b in order if b != a and {int(e) for e in self.ei[:, b]} <= inside), None)

    def wu26_branch(self, a: int, b: int) -> int:
        """Our branch between the MATPOWER buses a and b (either order): the first match."""
        number = self.base.bus["name"].astype(int).to_numpy()
        ends = [{int(number[f]), int(number[t])} for f, t in self.ei.T]
        return next(e for e, pair in enumerate(ends) if pair == {a, b})

    def wu26_buses(self, numbers: tuple[int, ...]) -> np.ndarray:
        """Our bus positions of the MATPOWER bus numbers `numbers`."""
        number = self.base.bus["name"].astype(int).to_numpy()
        return np.array([int(np.flatnonzero(number == n)[0]) for n in numbers], np.int64)

    def overload_step(
        self, design: AmOverloadDesign, Xt: np.ndarray, i: int, k: FrameKnobs
    ) -> tuple[Optional[Frame], np.ndarray]:
        """Frame i of the episode on its state `Xt`: the false state on the held support that reaches
        the goal's flow there, emitted on the true scan, labelled at the pretended loads; and the
        noiseless apparent flow it reaches on each target branch (MVA, `goal.lines` order, NaN when
        the frame could not be built)."""
        Xa, _ = self.goal_state(design.goal, i, Xt, design.support, k)
        if Xa is None:
            return None, np.full(len(design.goal.lines), np.nan)
        # the labels: the free injections of the support, the loads and generator outputs it pretends
        free = np.intersect1d(design.support, self.free_injection_buses())
        dev = self.pretended_change(Xt, Xa, free)
        frame = self.frame_from_state(Xt, Xa, free, dev.astype(float))
        flow = self.clean_flows_from_states(Xa[None])[0, list(design.goal.lines)]
        return frame, np.hypot(flow[:, 0], flow[:, 1])

    def pretended_change(self, Xt: np.ndarray, Xa: np.ndarray, buses: np.ndarray) -> np.ndarray:
        """The relative change the false state pretends at each free-injection bus of `buses`: at a
        generator bus (a bus with a load as well included, whose change the model gives the
        generator) the apparent power change against the true generator output, |dP + j dQ| /
        |P_gen + j Q_gen|; at a load bus the active change against the true load, |dP| / |P_load|."""
        gen = generator_output(Xt, self.load_base, self.gen_base)  # [N, 2] MW, MVAr
        load = self.true_load_by_bus(Xt)
        dP = Xa[:, NODE.p_inj] - Xt[:, NODE.p_inj]
        dQ = Xa[:, NODE.q_inj] - Xt[:, NODE.q_inj]
        is_gen = np.isin(buses, self.generator_buses())
        change = np.where(is_gen, np.hypot(dP[buses], dQ[buses]), np.abs(dP[buses]))
        base = np.where(is_gen, np.hypot(gen[buses, 0], gen[buses, 1]), np.abs(load[buses]))
        return (change / np.maximum(base, 1e-6)).astype(float)

    def true_load_by_bus(self, Xt: np.ndarray) -> np.ndarray:
        """[N] the scan's active load per bus (MW), summed over the load elements at each bus."""
        load = np.zeros(self.C)
        np.add.at(load, self.load_bus, self.true_load(Xt))
        return load
