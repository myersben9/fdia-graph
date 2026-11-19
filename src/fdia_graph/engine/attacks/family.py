"""The interface the attack families share. The design stage asks a family one thing, the episode of
a window (`EpisodeFamily.designed`), and the walk emits what it gets (`DesignedEpisode`), whichever
family made it. A family of your own (`AttackFamily`) designs one `Episode` (`models.plan`), an
attack vector over its window, or a miss.

Why a design found no attack is a miss reason, kept in `last_miss` and in the design stage's log
(`Designer.log`):
"no_target"    nothing to attack in the window;
"infeasible"   targets existed but no design met every bound;
"solver_error" a solver the family runs stopped abnormally, so "infeasible" may be wrong.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from fdia_graph.constants.channels import NodeChannel
from fdia_graph.constants.families import FAMILIES, FAMILY_CODE
from fdia_graph.models.frames import AttackVector, Frame
from fdia_graph.models.plan import Episode, EpisodeRecord
from fdia_graph.systems.states import States

if TYPE_CHECKING:
    from fdia_graph.engine.core import FdiaGenerator

# A window handed to a New York family's `design`: the onset frame of the pool it covers, and the true
# state over that window, `vm [L, N]` p.u. and `va [L, N]` degrees (the frames an episode would occupy,
# all solved). A plain tuple: the timeline slices the pool into one per candidate onset.
Window = tuple[int, np.ndarray, np.ndarray]


class DesignedEpisode(ABC):
    """One designed episode as the walk emits it (`generation.emit.Walk.run`): `begin` at its onset,
    then `frame` for each of its scans in order, then `record`. `code` is its family's in a file."""

    code: int

    def begin(self, g: FdiaGenerator, X: States, t: int, prev: AttackVector) -> bool:
        """Whether the episode can start at frame t of the pool X, `prev` being the attack vector of
        the frame before. An episode designed before the walk can; a search ramp is designed here."""
        return True

    @abstractmethod
    def frame(self, g: FdiaGenerator, X: np.ndarray, i: int) -> Frame | None:
        """Scan i of the episode on its true state X [N, 4], emitted by the generator `g`; None when
        the scan cannot be built (it is then stored benign)."""

    @abstractmethod
    def record(self, onset: int, length: int, buses: np.ndarray) -> EpisodeRecord:
        """What the file records of the episode written over `length` frames from `onset`; `buses`
        [N] marks the buses its frames labelled."""

    def row(self, onset: int, length: int, buses: np.ndarray) -> Episode:
        """The episode-table row of an episode whose frames carry the attack: the buses they
        labelled, no clients and no stored vector."""
        none = np.zeros(0)
        return Episode(FAMILIES[self.code], none, onset, length, np.flatnonzero(buses), -1, none)


class VectorEpisode(DesignedEpisode):
    """An episode designed as one attack vector over its window (`Episode.a` [length, M], what an
    `AttackFamily` designs): each scan is the true scan plus its row of the vector."""

    def __init__(self, episode: Episode) -> None:
        self.episode, self.code = episode, FAMILY_CODE[episode.family]

    def frame(self, g: FdiaGenerator, X: np.ndarray, i: int) -> Frame | None:
        return g.attack_frames.frame_from_vector(X, self.episode.a[i])

    def record(self, onset: int, length: int, buses: np.ndarray) -> EpisodeRecord:
        return EpisodeRecord(replace(self.episode, onset=onset))


class EpisodeFamily(ABC):
    """An attack family as the design stage asks it: the episode of one window, or none."""

    #: why the last design found no attack, a miss reason (None after a success, and on a family that
    #: gives none)
    last_miss: str | None = None

    @abstractmethod
    def designed(self, onset: int, X: np.ndarray, rng: np.random.Generator) -> DesignedEpisode | None:
        """The episode over the window of true states X [L, N, 4] that starts at pool frame `onset`,
        its random draws from `rng`; None when the family finds no attack there."""


class AttackFamily(EpisodeFamily):
    """A family of attacks (Am or At) that designs an attack vector. `design` draws one episode for a
    window of true states, or None when it finds no stealthy attack there (the timeline then moves
    the episode)."""

    #: solves the family's OPF stopped abnormally on (an OPF family counts them; the others never do)
    solver_errors: int = 0

    def designed(self, onset: int, X: np.ndarray, rng: np.random.Generator) -> DesignedEpisode | None:
        window = (onset, X[:, :, NodeChannel.V], X[:, :, NodeChannel.THETA])
        episode, _ = self.design_explained(window, rng)
        return None if episode is None else VectorEpisode(episode)

    @abstractmethod
    def design(self, window: Window, rng: np.random.Generator) -> Episode | None:
        """One episode over `window = (onset, vm, va)`, or None."""

    def design_explained(self, window: Window, rng: np.random.Generator) -> tuple[Episode | None, str | None]:
        """`design`, and why it found no attack: (the episode, None) or (None, a miss reason), the reason
        also kept in `last_miss`. A solver failure makes the miss "solver_error", since the support it
        failed on might have held an attack; a None the family leaves unexplained is "infeasible"."""
        self.last_miss, errors = None, self.solver_errors
        ep = self.design(window, rng)
        if ep is not None:
            self.last_miss = None
        elif self.solver_errors > errors:
            self.last_miss = "solver_error"
        else:
            self.last_miss = self.last_miss or "infeasible"
        return ep, self.last_miss
