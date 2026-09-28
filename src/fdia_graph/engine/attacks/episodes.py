"""What one attack episode is: its targets and the design each of its frames applies, drawn at onset.

The timeline decides when an episode starts and how long it lasts; these designs decide what it
attacks. Each one is tested on every frame it will occupy before it is accepted, since the
operating point drifts along the episode and a design feasible at onset can lose its stealthy state
later; a design that passes cannot fall back to a benign frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ...formulas.attacks import ramp_profile
from ...models.choices import BENIGN_CODE, FAMILIES, FAMILY_CODE, STEALTHY_FAMILIES
from ...models.frames import AmDesign, AttackDesign, FrameKnobs, LoadGoal, RampDesign
from .minimize import MinimizeMixin
from .redistribution import RedistributionMixin
from .stealthy import AQ_FAMILY, AQ_HALVINGS

ONSET_DRAWS = 40  # designs an episode (Aq, At, Am) tries for one with a stealthy state on its frames
AM_DRAWS = ONSET_DRAWS  # the Am redistribution draws, the same budget


def probe_frames(T: int, t: int, length: int) -> range:
    """The frames an episode's design is tested on before it is accepted: every frame it will
    occupy, clipped to the T frames of the pool."""
    return range(t, min(t + length, T))


def pick_targets(rng: np.random.Generator, apos: np.ndarray, fid: int) -> np.ndarray:
    """Attacked load-table positions for an episode: 1 to 6 buses for Aq, up to 4 otherwise."""
    nab = len(apos)
    k = int(rng.integers(1, min(6, nab) + 1)) if fid == AQ_FAMILY else min(4, nab)
    return rng.choice(apos, k, replace=False)


def draw_ramp(rng: np.random.Generator, apos: np.ndarray, ramp_len: int) -> RampDesign:
    """One ramp design over the loads `apos`: a fixed bus set, a direction, the rise and hold
    lengths (four draws)."""
    a = rng.choice(apos, min(5, len(apos)), replace=False)
    direction = 1.0 if rng.random() < 0.5 else -1.0
    rise = max(1, int(rng.uniform(0.2, 0.45) * ramp_len))
    hold = int(rng.uniform(0.0, 0.25) * ramp_len)
    return RampDesign(a, direction, rise, hold)


def ramp_dev(i: int, rise: int, hold: int, rate: float) -> float:
    """The ramp's deviation at step i, never zero: the profile's first frame and its return leg
    floor at one rate, so every frame labelled At carries an attack."""
    return max(rate, ramp_profile(i, rise, hold, rate, rate))


def am_sign(direction: str, rng: Optional[np.random.Generator]) -> float:
    """The sign applied to the engine's redistribution. `lra_delta` orients its delta to LOWER the
    target line's |flow| in the false state, so "mask" (a real overload reads lighter) keeps it (+1)
    and "induce" (a safe line reads as overloaded) flips it (-1); "both" draws one of the two per
    episode (one RNG draw; a given draw maps to the same sign as before the direction fix, though
    files still change where the stealthy load recovery did)."""
    if direction == "both":
        assert rng is not None, "drawing a direction needs the random stream"
        direction = "mask" if rng.random() < 0.5 else "induce"
    return 1.0 if direction == "mask" else -1.0


@dataclass
class _AmShape:
    """The schedule of one Am episode: how much of the held redistribution each frame applies."""

    rate: float  # fraction of the full redistribution added per frame on the rise and the fall
    rise: int
    hold: int

    @classmethod
    def under_floor(cls, rel: float, length: int, am_rate: float, floor: float) -> _AmShape:
        """Rise so that the largest per-bus per-frame load change stays under `am_rate` of the noise
        floor: `rel` is the largest per-bus redistribution fraction, so a step of `rate` of the whole
        moves that bus by rate * rel. The rate is never raised: an episode too short to reach the
        full redistribution ramps as far as it gets and returns, its peak below the full delta."""
        rate = min(1.0, am_rate * floor / max(rel, 1e-9))
        rise = min(math.ceil(1.0 / rate), max(1, length // 2))
        return cls(rate, rise, max(0, length - 2 * rise))

    def at(self, i: int) -> float:
        return min(1.0, ramp_profile(i, self.rise, self.hold, self.rate, self.rate))

    def step(self, i: int) -> float:
        """The fraction applied at frame i, never zero: the first frame and the return leg floor at
        one rate, so every frame labelled Am carries an attack."""
        return max(self.rate, self.at(i))


class EpisodeDesignMixin(RedistributionMixin, MinimizeMixin):
    """Draw the design of an attack episode at its onset, and the design each of its frames applies."""

    def target_counts(self) -> dict[int, int]:
        """The targets the case offers each family code: the stealthy Aq and At need a load off every
        generator bus, Al and Am a line whose subnetwork admits a redistribution, Ad/As/Ar an attackable
        load (checked against the request by `models.inputs.AdmissibleTargets`)."""
        stealthy, lines = len(self.stealthy_pos), len(self._target_lines)
        need = {
            FAMILY_CODE["Aq"]: stealthy,
            FAMILY_CODE["At"]: stealthy,
            FAMILY_CODE["Al"]: lines,
            FAMILY_CODE["Am"]: lines,
        }
        return {f: need.get(f, len(self.attackable_pos)) for f in FAMILIES if f != BENIGN_CODE}

    # ---- At, the slow ramp ------------------------------------------------------------------
    def ramp_step(self, design: RampDesign, i: int, rate: float) -> AttackDesign:
        """The design of the ramp's frame i: its load set scaled by the profile's deviation there, solved
        on the episode's fewest-tamper support when the search chose one."""
        return AttackDesign(
            design.targets, 1 + design.direction * ramp_dev(i, design.rise, design.hold, rate), design.support
        )

    def ramp_design(
        self, X: np.ndarray, t: int, shape: tuple[int, float], k: FrameKnobs
    ) -> Optional[RampDesign]:
        """A ramp starting at t whose every frame has a stealthy state, `shape` = (ramp_len, ramp_rate);
        redrawn up to ONSET_DRAWS times, None when no draw has one (the span then stays benign). With
        `k.min_tamper` the accepted ramp carries the support that tampers the fewest devices over its
        window [WU26, eq. 12], held for every frame; the search spends no random draw."""
        ramp_len, rate = shape
        for _ in range(ONSET_DRAWS):
            design = draw_ramp(self.rng, self.stealthy_pos, ramp_len)  # At is stealthy: no generator bus
            frames = probe_frames(len(X), t, ramp_len)
            if all(self.is_feasible(X[u], self.ramp_step(design, u - t, rate), k) for u in frames):
                return self._fewest_tamper(design, X, frames, t, rate, k) if k.min_tamper else design
        return None

    def _fewest_tamper(
        self, design: RampDesign, X: np.ndarray, frames: range, t: int, rate: float, k: FrameKnobs
    ) -> RampDesign:
        """The accepted ramp with the fewest-tamper support of its window (`frames`) and the search's
        result; unchanged when the search finds none (it always has the region the ramp was accepted on)."""
        goal = LoadGoal(tuple(self.ramp_step(design, u - t, rate) for u in frames))
        result = self.min_tamper([X[u] for u in frames], goal, k)
        if result is None:
            return design
        held = result.support if result.devices >= 0 else None  # -1: no held support met the constraints
        return design._replace(support=held, tamper=result)

    # ---- Aq, Ad, As, Ar: one design held over the episode -----------------------------------
    def single_shot_design(
        self,
        X: np.ndarray,
        t: int,
        fid: int,
        length: int,
        k: FrameKnobs,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> Optional[AttackDesign]:
        """The targets and load multipliers of an episode: the targets, the direction (a load rise or
        a load drop, one draw, like the ramp's) and the per-target scale in the band; an Aq design with
        no stealthy state on any frame of the episode is redrawn, up to ONSET_DRAWS (a case that runs
        below its voltage limits refuses most rises near the low buses, a drop there is the attack
        that fits). None when no draw has one: the span then stays benign and is counted. `rng` is the
        generator's own unless given (the deprecated `timeline._draw_single_shot` passes its caller's)."""
        rng = self.rng if rng is None else rng
        for _ in range(ONSET_DRAWS):
            a = pick_targets(rng, self.stealthy_pos if fid in STEALTHY_FAMILIES else self.attackable_pos, fid)
            direction = 1.0 if fid != AQ_FAMILY or rng.random() < 0.5 else -1.0
            mult = 1 + direction * rng.uniform(0.05, k.intensity, size=len(a))
            design = AttackDesign(a, mult)
            if fid != AQ_FAMILY or all(
                self.is_feasible(X[u], design, k) for u in probe_frames(len(X), t, length)
            ):
                return design
        return None

    # ---- Am, the multi-snapshot redistribution ----------------------------------------------
    def _am_multipliers(self, Xt: np.ndarray, a: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """The load multipliers that add `delta` (MW, per target) to this frame's true load."""
        Lp = self.true_load(Xt)[a]
        return 1.0 + delta / np.where(np.abs(Lp) > 1e-9, Lp, 1e-9)

    def _am_peak_solves(
        self, Xt: np.ndarray, a: np.ndarray, delta: np.ndarray, interior: Optional[np.ndarray], k: FrameKnobs
    ) -> bool:
        """Whether the held redistribution at its peak (`delta`, MW per target) has a stealthy state
        (a local solution inside the operating limits) on the state `Xt`; no random draw is spent."""
        return self.is_feasible(Xt, AttackDesign(a, self._am_multipliers(Xt, a, delta), interior), k)

    def _am_held_delta(
        self,
        X: np.ndarray,
        t: int,
        a: np.ndarray,
        delta: np.ndarray,
        interior: Optional[np.ndarray],
        shape: tuple[int, float],
        k: FrameKnobs,
    ) -> Optional[np.ndarray]:
        """The held redistribution `delta` (MW per target `a`), at its drawn size or the largest
        halving above the noise floor whose peak has a stealthy state on every frame of the plateau;
        None when none has."""
        length, am_rate = shape
        Lp0 = self.true_load(X[t])[a]
        for _ in range(AQ_HALVINGS + 1):
            dev = np.abs(delta) / (np.abs(Lp0) + 1e-6)
            if np.min(dev) < k.floor:
                return None  # a bus inside the noise floor: not an attack by the band's own rule
            sh = _AmShape.under_floor(float(np.max(dev)), length, am_rate, k.floor)
            frames = probe_frames(len(X), t, length)  # every frame's own fraction of the redistribution
            if all(self._am_peak_solves(X[u], a, sh.step(u - t) * delta, interior, k) for u in frames):
                return delta
            delta = delta / 2
        return None

    def am_design(
        self, X: np.ndarray, t: int, shape: tuple[int, float, str], k: FrameKnobs
    ) -> Optional[AmDesign]:
        """One Am episode starting at t: a load redistribution drawn once at onset (the Al
        construction, PTDF-ranked buses, load-conserving), signed by the direction, and the ramp that
        reaches it with a per-bus per-frame step under the noise floor. `shape` = (length, am_rate,
        am_direction). Redrawn up to AM_DRAWS times when the target-line pool gives no redistribution,
        or one without a stealthy state on its plateau at any halving above the floor; None then."""
        length, am_rate, direction = shape
        Lp0 = self.true_load(X[t])
        for _ in range(AM_DRAWS):
            red = self.lra_delta(Lp0, k.intensity, k.lra_k, floor=k.floor, hops=k.hops)
            a = red.buses
            if len(a) == 0:
                continue
            delta = red.delta[a] * am_sign(direction, self.rng)
            delta = self._am_held_delta(X, t, a, delta, red.interior, (length, am_rate), k)
            if delta is not None:
                rel = float(np.max(np.abs(delta) / (np.abs(Lp0[a]) + 1e-6)))
                sh = _AmShape.under_floor(rel, length, am_rate, k.floor)
                return AmDesign(a, delta, red.interior, sh.rate, sh.rise, sh.hold)
        return None

    def am_step(self, design: AmDesign, Xt: np.ndarray, i: int) -> AttackDesign:
        """The design of the Am episode's frame i on its state `Xt`: that frame's fraction of the held
        redistribution, as load multipliers on the targets, over the held interior."""
        fraction = _AmShape(design.rate, design.rise, design.hold).step(i)
        return AttackDesign(
            design.targets, self._am_multipliers(Xt, design.targets, fraction * design.delta), design.interior
        )
