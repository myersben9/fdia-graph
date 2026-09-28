"""`Am`, the multi-snapshot overload attack of [WU26]: a target branch's reported flow driven to its
rating over the window, with the fewest devices tampered.

[WU26, eqs. 24-25] asks that the apparent flow the tampered measurements carry on a target line grow
snapshot by snapshot until it reaches the line's rating S_max. The rating is the branch's PGLib-OPF
`rate_a` (`fdia_graph.ratings`; the IEEE cases pandapower ships rate every branch 9,900 MVA, which
no flow comes near). A branch is a target for a window only when it is rated, its flow is metered
(the goal is what the operator sees) and its true flow stays below the rating at every snapshot of
the window (otherwise the goal is met with no attack). The goal at snapshot t (the plan's D9) is

    S_{l,t} = S_true_{l,t} + (t - kappa)/T (S_max - S_true_{l,kappa+T})

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

from ...formulas.attacks import branch_ratings
from ...models.errors import NoLineRatings
from ...models.frames import AmOverloadDesign, AttackVector, FlowGoal, Frame, FrameKnobs
from ...models.grid import NODE
from ...ratings import pglib_branches
from .minimize import MinimizeMixin

# How many eligible target branches an Am episode tries, in a random order drawn once at onset. A
# bounded heuristic, not a search over every branch: each try is a full fewest-tamper search (about 13 s
# on IEEE-118), so an episode whose first eight branches admit no attack stays benign and is counted
# even if a later branch would have.
AM_LINE_TRIES = 8


def _schedule(flows: np.ndarray, rating: float) -> tuple[float, ...]:
    """One branch's goal over a window from its true flows [T, 2] (MW, MVAr): the true flow plus a
    linear share of what separates the last true flow from the rating (the plan's D9)."""
    true = np.hypot(flows[:, 0], flows[:, 1])
    T = len(true) - 1
    share = np.arange(len(true)) / T if T > 0 else np.ones(1)
    return tuple(float(x) for x in true + share * (rating - true[-1]))


class OverloadMixin(MinimizeMixin):
    """Design and step the overload attack of [WU26] on a rated, metered target branch."""

    def line_ratings(self) -> np.ndarray:
        """[E] each branch's rating in MVA (PGLib-OPF `rate_a`, matched by end buses), NaN where unrated.
        Raises `NoLineRatings` on a case the package holds no ratings for."""
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
        """The per-snapshot flow the attack must reach on `line` over `window` (module docstring), and
        on each branch of `more` at once, each toward its own rating (the paper's two-line case
        studies; the generator's episodes drive one line)."""
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
        for line in self.rng.permutation(lines)[:AM_LINE_TRIES]:
            goal = self.overload_goal(window, int(line))
            result = self.min_tamper(window, goal, k, prev)
            if result is not None and result.devices >= 1:
                rating = float(self.line_ratings()[int(line)])
                return AmOverloadDesign(goal, rating, result.support, result)
        return None

    def overload_step(
        self, design: AmOverloadDesign, Xt: np.ndarray, i: int, k: FrameKnobs
    ) -> tuple[Optional[Frame], float]:
        """Frame i of the episode on its state `Xt`: the false state on the held support that reaches
        the goal's flow there, emitted on the true scan, labelled at the pretended loads; and the
        noiseless apparent flow it reaches on the target branch (MVA, NaN when the frame could not be
        built)."""
        Xa, _ = self.goal_state(design.goal, i, Xt, design.support, k)
        if Xa is None:
            return None, float("nan")
        # the labels: the free injections of the support, the loads and generator outputs it pretends
        free = np.intersect1d(design.support, self.free_injection_buses())
        dinj = Xa[:, NODE.p_inj] - Xt[:, NODE.p_inj]  # a free bus's injection change is what it pretends
        dev = np.abs(dinj[free]) / np.maximum(np.abs(Xt[free, NODE.p_inj]), 1e-6)
        frame = self.frame_from_state(Xt, Xa, free, dev.astype(float))
        flow = self.clean_flows_from_states(Xa[None])[0, design.goal.line]
        return frame, float(np.hypot(flow[0], flow[1]))

    def true_load_by_bus(self, Xt: np.ndarray) -> np.ndarray:
        """[N] the scan's active load per bus (MW), summed over the load elements at each bus."""
        load = np.zeros(self.C)
        np.add.at(load, self.load_bus, self.true_load(Xt))
        return load
