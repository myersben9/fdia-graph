"""`Am`, the multi-snapshot overload attack of [WU26]: the reported flows of its target branches (one
or two, `OverloadSettings.n_lines`) driven to their ratings S_max (`use_line_ratings`) over the window,
with the fewest devices tampered.

Goal [WU26 eqs. 24-25], on the noiseless reading of the false state, with kappa the untouched reference
snapshot before the window's T snapshots:

    S_{l,t} = S_true_{l,t} + (t - kappa)/T (S_max - S_true_{l,kappa+T}),   t = kappa+1 ... kappa+T

a steady ramp on top of the load's own drift that reaches S_max at the last snapshot. Nothing bounds
the move between snapshots: each snapshot's readings are those of one AC state, which is the stealth.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...constants.channels import EdgeChannel, NodeChannel
from ...errors import NoLineRatings
from ...formulas.attacks import branch_ratings, generator_output
from ...models.config import OverloadSettings
from ...models.frames import AmOverloadDesign, FlowGoal, Frame, FrameKnobs
from ...ratings import pglib_branches
from .minimize import MinimizeMixin

# How many target sets (one line, or two) an Am episode tries, from one random order of the eligible
# branches drawn at onset. A bound, not a search over every set: each try is a full fewest-tamper search
# (about 13 s on IEEE-118). After eight misses the walk moves the episode to another onset
# (`generation.emit.Walk.relocate_later`).
AM_LINE_TRIES = 8


def _schedule(flows: np.ndarray, rating: float) -> tuple[float, ...]:
    """One branch's goal (MVA) over a window from its true flows [T, 2] (MW, MVAr): the true flow plus
    the share k/T (k = 1 ... T) of what separates the last true flow from the rating [WU26 eq. 25], so
    every snapshot moves the flow and the last reaches the rating."""
    true = np.hypot(flows[:, 0], flows[:, 1])
    share = np.arange(1, len(true) + 1) / len(true)
    return tuple(float(x) for x in true + share * (rating - true[-1]))


class OverloadMixin(MinimizeMixin):
    """Design and step the overload attack of [WU26] on a rated, metered target branch."""

    _line_ratings: Optional[np.ndarray] = None  # [E] the static ratings, MVA, once set or read
    _rating_delta: Optional[float] = None  # the step of the per-window "delta" ratings, MVA

    def use_line_ratings(self, settings: OverloadSettings, X: np.ndarray) -> None:
        """Set the ratings the overload attack drives its lines to. "pool": `rating_margin` times each
        branch's peak true from-end apparent flow over the pool X [T, N, 4]. "pglib": the PGLib-OPF
        ratings (`NoLineRatings` on a case without them). "delta": set per window (`window_ratings`)."""
        self._rating_delta = None
        if settings.rating_source == "pglib":
            self._line_ratings = None
            self.line_ratings()
            return
        if settings.rating_source == "delta":  # set per window, from the window's own last flow
            self._rating_delta = settings.rating_delta * self._base_mva
            self._line_ratings = None  # `window_ratings` rates each window from its own flows
            return
        peak = np.zeros(self.E)
        for a in range(0, len(X), 4096):  # the pool in slabs: a 72k-state pool is one pass
            peak = np.maximum(peak, np.abs(self.all_flows_from_states(X[a : a + 4096])).max(axis=0))
        self._line_ratings = settings.rating_margin * peak

    def line_ratings(self) -> np.ndarray:
        """[E] each branch's rating in MVA: the ratings `use_line_ratings` set, else PGLib-OPF's `rate_a`
        (matched by end buses), NaN where unrated. Raises `NoLineRatings` on a case the package holds
        no PGLib-OPF ratings for."""
        if self._line_ratings is not None:
            return self._line_ratings
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
        metered (the goal is what the operator sees), true apparent flow below the rating at every
        snapshot, and a free injection (an attackable load or a generator) in the area around its ends."""
        flows = self.clean_flows_from_states(np.stack(window))  # [T, E, 2], unmetered zeroed
        rating = self.window_ratings(flows)  # finite per window under "delta"
        # [E] the window's largest true flow
        S = np.hypot(flows[..., EdgeChannel.P_FROM], flows[..., EdgeChannel.Q_FROM]).max(axis=0)
        metered = np.asarray(self.meters.flow, bool)
        status = self.branch.status
        live = np.ones(len(rating), bool) if status is None else np.asarray(status) > 0
        ok = live & np.isfinite(rating) & metered & (S < np.nan_to_num(rating, nan=-np.inf))
        free = set(self.free_injection_buses().tolist())
        lines = [int(b) for b in np.flatnonzero(ok) if self._acts_on(int(b), free, hops)]
        return np.array(lines, dtype=np.int64)

    def _acts_on(self, line: int, free: set[int], hops: int) -> bool:
        """Whether the area within `hops` of the branch's ends holds a free injection to pretend."""
        area = self.local_region(np.unique(self.ei[:, line]), hops)
        return area is not None and bool(free & {int(b) for b in area})

    def overload_goal(self, window: list[np.ndarray], line: int, *more: int) -> FlowGoal:
        """The per-snapshot flow the attack must reach on `line` over `window` (module docstring), and on
        each branch of `more` at once, each toward its own rating."""
        flows = self.clean_flows_from_states(np.stack(window))  # [T, E, 2] MW, MVAr
        rating = self.window_ratings(flows)
        first = _schedule(flows[:, line], float(rating[line]))
        others = tuple((int(b), _schedule(flows[:, b], float(rating[b]))) for b in more)
        return FlowGoal(line, first, more=others)

    def window_ratings(self, flows: np.ndarray) -> np.ndarray:
        """[E] the ratings a window's goal drives to, from its true flows [T, E, 2]: the static ratings
        (`line_ratings`), or with rating_source "delta" each branch's true apparent flow at the window's
        last snapshot plus the step (ours, a rating scale matched to [WU26] Fig. 4's attack magnitudes)."""
        if self._rating_delta is None:
            return self.line_ratings()
        return np.hypot(flows[-1, :, 0], flows[-1, :, 1]) + self._rating_delta

    def _target_sets(self, order: np.ndarray, n_lines: int, hops: int) -> list[tuple[int, ...]]:
        """At most `AM_LINE_TRIES` target sets from the eligible branches in `order`: single branches,
        or pairs one held support can reach (each branch with its `_partner`), each pair once."""
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
        inside = set(area.tolist())
        for b in order.tolist():
            if b != a and set(self.ei[:, b].tolist()) <= inside:
                return b
        return None

    def overload_step(
        self, design: AmOverloadDesign, Xt: np.ndarray, i: int, k: FrameKnobs
    ) -> tuple[Optional[Frame], np.ndarray]:
        """Frame i of the episode on its state `Xt`: the false state on the held support that reaches the
        goal's flow, emitted on the true scan; and the noiseless apparent flow it reaches on each target
        branch (MVA, `goal.lines` order). (None, NaN) when the frame cannot be built."""
        Xa = self.goal_state(design.goal, i, Xt, design.support, k)
        if Xa is None:
            return None, np.full(len(design.goal.lines), np.nan)
        # the labels: the free injections of the support, the loads and generator outputs it pretends
        free = np.intersect1d(design.support, self.free_injection_buses())
        dev = self.pretended_change(Xt, Xa, free)
        frame = self.frame_from_state(Xt, Xa, free, dev.astype(float))
        flow = self.clean_flows_from_states(Xa[None])[0, list(design.goal.lines)]
        return frame, np.hypot(flow[:, 0], flow[:, 1])

    def pretended_change(self, Xt: np.ndarray, Xa: np.ndarray, buses: np.ndarray) -> np.ndarray:
        """The relative change the false state pretends at each bus of `buses`: at a generator bus (with
        or without a load) |dP + j dQ| / |P_gen + j Q_gen|; at a load bus |dP| / |P_load|."""
        gen = generator_output(Xt, self.load_base, self.gen_base)  # [N, 2] MW, MVAr
        load = self.true_load_by_bus(Xt)
        dP = Xa[:, NodeChannel.P_INJ] - Xt[:, NodeChannel.P_INJ]
        dQ = Xa[:, NodeChannel.Q_INJ] - Xt[:, NodeChannel.Q_INJ]
        is_gen = np.isin(buses, self.generator_buses())
        change = np.where(is_gen, np.hypot(dP[buses], dQ[buses]), np.abs(dP[buses]))
        base = np.where(is_gen, np.hypot(gen[buses, 0], gen[buses, 1]), np.abs(load[buses]))
        return (change / np.maximum(base, 1e-6)).astype(float)

    def true_load_by_bus(self, Xt: np.ndarray) -> np.ndarray:
        """[N] the scan's active load per bus (MW), summed over the load elements at each bus."""
        load = np.zeros(self.C)
        np.add.at(load, self.load_bus, self.true_load(Xt))
        return load
