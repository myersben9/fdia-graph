"""What one attack occupies on the timeline, and the interface its two families share."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

# A window handed to a family's `design`: the onset frame of the pool it covers, and the true state
# over that window, `vm [L, N]` p.u. and `va [L, N]` degrees (the frames an episode would occupy, all
# SOLVED). A plain tuple, not a record: the timeline block (D1) slices the pool into one per candidate
# onset. `L` is the window length, `N` the bus count.
Window = tuple[int, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Episode:
    """One attack over a window of `length` frames from `onset`. `family` is "Am", "Ap" or "At". `clients`
    are the owners of the tampered meters, one or several: the attacker's local subnetwork may span
    utilities and it tampers whatever meters its support needs (G5). `buses` are the attacked buses
    (the pretended loads and generators, the localization label). `target_branch` is the overloaded
    branch for Am and Ap (its first, when two are driven), -1 for At. `a [length, M]` is the attack vector
    added to each frame's benign scan, the noiseless reading of the local false state minus that of the
    true state (h(x_false) - h(x_true)) in full, so the operator's one residual test over all meters
    sees noise (global stealth). The tampered meters, those it moves beyond their noise, come from
    `Attacker.tampered`; `clients` are their owners."""

    family: str
    clients: np.ndarray
    onset: int
    length: int
    buses: np.ndarray
    target_branch: int
    a: np.ndarray


class AttackFamily(ABC):
    """A family of attacks (Am, Ap or At). `design` draws one episode for a window of true states, or
    None when it finds no stealthy attack there (the window stays benign). The one base class of the
    attacks package: the families are composed around an `Attacker`, not stacked on mixins."""

    @abstractmethod
    def design(self, window: Window, rng: np.random.Generator) -> Episode | None:
        """One episode over `window = (onset, vm, va)`, or None."""
