"""At, the slow ramp: a false load that rises a little each frame, every step below the natural
five-minute load swing, so no single snapshot looks anomalous while the total drifts the state away.

The attacker scales a local subnetwork's loads and re-solves it with the boundary voltages held true
(`Attacker.false_state`): the other attackable loads and generators of the area and its edge absorb the
change, every other bus, a zero-injection one among them, keeps its true injection, so every snapshot's
readings are those of one AC state and the operator's global residual test sees noise only [1]. The ramp follows the triangle profile of [1]: rise for a
stretch of frames, hold at the peak, then return. Each per-step increment is capped below the bus's own
natural frame-to-frame load variation over the window, which is what keeps the ramp hidden from a
per-snapshot check; the cumulative peak still clears the meter noise floor, or the window stays benign
(a sub-noise attack has no effect on an accurate estimator). The attack vector is global: every meter
it moves beyond noise is written, and `Episode.clients` records their owners.
"""

from __future__ import annotations

import numpy as np

from fdia_graph.nyfed.attacks.attacker import Attacker
from fdia_graph.nyfed.attacks.episode import AttackFamily, Episode, Window


def _profile(i: int, rise: int, hold: int) -> float:
    """The triangle deviation at step i in units of one per-step increment [1]: i during the rise, the
    peak `rise` while it holds, then down one per step, never below zero."""
    if i < rise:
        return float(i)
    if i < rise + hold:
        return float(rise)
    return max(0.0, float(rise) - (i - rise - hold))


class Ramp(AttackFamily):
    """The slow ramp At. Holds an `Attacker` and the ramp settings: `hops` for the area, `n_buses`
    load buses ramped at once, `rate` the per-step load fraction, `rise_frac`/`hold_frac` the shape as
    shares of the window, `nat_cap` the share of a bus's natural per-step swing a step may use, and
    `draws` centre tries per window."""

    def __init__(
        self,
        attacker: Attacker,
        *,
        hops: int = 2,
        n_buses: int = 4,
        rate: float = 0.03,
        rise_frac: float = 0.4,
        hold_frac: float = 0.2,
        nat_cap: float = 0.9,
        draws: int = 20,
    ):
        self.att = attacker
        self.hops, self.n_buses, self.rate = hops, n_buses, rate
        self.rise_frac, self.hold_frac, self.nat_cap, self.draws = rise_frac, hold_frac, nat_cap, draws

    def design(self, window: Window, rng: np.random.Generator) -> Episode | None:
        onset, vm, va = window
        att, (length, n) = self.att, vm.shape
        rise = max(1, int(self.rise_frac * length))
        hold = int(self.hold_frac * length)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        # the natural five-minute load swing per bus over the window: the step a ramp must stay under
        true_load = np.stack([att.bus_load(vm[k], va[k]) for k in range(length)])  # [L, N] MW
        natural = np.abs(np.diff(true_load, axis=0)).max(axis=0) if length > 1 else np.zeros(n)
        loads = att.load_buses(vm[0], va[0])

        for _ in range(self.draws):
            if not len(loads):
                return None
            area = att.area(int(rng.choice(loads)), self.hops)
            if area is None:
                continue
            ramped = att.load_buses(vm[0], va[0], area)  # local load buses, any utility
            if len(ramped) > self.n_buses:
                ramped = np.sort(rng.choice(ramped, self.n_buses, replace=False))
            if not len(ramped):
                continue
            ep = self._build(onset, vm, va, true_load, natural, area, ramped, direction, rise, hold)
            if ep is not None:
                return ep
        return None

    def _build(
        self,
        onset: int,
        vm: np.ndarray,
        va: np.ndarray,
        true_load: np.ndarray,
        natural: np.ndarray,
        area: np.ndarray,
        ramped: np.ndarray,
        direction: float,
        rise: int,
        hold: int,
    ) -> Episode | None:
        """Solve every frame of the ramp on the held area, cap each step below the natural swing, and
        write the global attack vectors; None when a frame has no local solution or the peak stays
        sub-noise."""
        att, (length, n) = self.att, vm.shape
        p, q = att.net.injections(vm[0], va[0])
        p0, q0 = -p[ramped], -q[ramped]  # true load at the ramped buses (MW, MVAr)
        # per-step load increment per ramped bus (MW, MVAr), capped below its natural per-step swing
        step_p = np.minimum(self.rate * np.abs(p0), self.nat_cap * natural[ramped])
        step_q = np.where(np.abs(p0) > 1e-6, step_p * (q0 / np.where(np.abs(p0) > 1e-6, p0, 1.0)), 0.0)
        if not np.any(step_p > 0):
            return None
        a = np.zeros((length, len(att.meas.plan.kind)))
        for k in range(length):
            dev = direction * _profile(k, rise, hold)
            load_change = np.zeros(n, complex)
            load_change[ramped] = dev * (step_p + 1j * step_q)
            state = att.false_state(vm[k], va[k], area, load_change)
            if state is None:
                return None
            a[k] = att.attack_vector(vm[k], va[k], state[0], state[1])
        changed = np.flatnonzero(att.tampered(vm, va, a).any(axis=0))
        if not len(changed):  # the peak stays sub-noise: no effect on an accurate estimator
            return None
        clients = np.unique(att.meas.plan.owner[changed])  # owners of the tampered meters (G5)
        return Episode("At", clients, onset, length, ramped, -1, a)
