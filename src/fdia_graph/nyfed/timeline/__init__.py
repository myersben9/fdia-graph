"""The dataset: a year of noisy scans with attacks placed at random and labelled.

`TimelineBuilder` splits the year into train, validation and test weeks, places episodes uniformly among the onsets
each family allows, has the families design them on the pool's true states, and writes every frame's readings,
labels and the static tables to `timeline.h5`. `production_builder` (in `fdia_graph.nyfed`) builds it as released.
The build's summary counts what it placed, built and, by reason, missed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph.nyfed._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from .builder import TimelineBuilder
    from .schedule import Slot, place_episodes, split_frames, split_weeks, week_index

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "Slot": ".schedule",
    "place_episodes": ".schedule",
    "split_weeks": ".schedule",
    "split_frames": ".schedule",
    "week_index": ".schedule",
    "TimelineBuilder": ".builder",
}

__all__ = [
    "Slot",
    "TimelineBuilder",
    "place_episodes",
    "split_frames",
    "split_weeks",
    "week_index",
]
__getattr__, __dir__ = attach(__name__, _PUBLIC)
