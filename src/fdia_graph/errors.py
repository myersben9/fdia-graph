"""Every error the package raises on purpose, in one place.

Two kinds. An input a model does not allow is a `ConfigError` (and a dataset view lacking what a
consumer needs a `MissingCapability`), raised by the validation engine in `models.validation` when
the model is built. A condition that only the data can reveal (no benign records in a split, no
room left for an episode, an outage that islands the grid) is one of the named errors below, raised
where the data is read. Every one subclasses ValueError, the type these checks raised before, so a
caller's `except ValueError` keeps working.
"""

from __future__ import annotations

from .models.errors import (
    CountOverflow,
    DataConditionError,
    GridIslanded,
    NoAdmissibleTarget,
    NoAttackedRecords,
    NoBenignRecords,
    NoLineRatings,
    NoOperatingLimits,
    NoRoomForEpisode,
    NotFitted,
    SlackMismatch,
    UnknownColumnOrder,
    VaryingReference,
)
from .models.validation import ConfigError, MissingCapability

__all__ = [
    "ConfigError",
    "MissingCapability",
    "DataConditionError",
    "NoBenignRecords",
    "NoAttackedRecords",
    "NotFitted",
    "NoAdmissibleTarget",
    "NoRoomForEpisode",
    "GridIslanded",
    "SlackMismatch",
    "VaryingReference",
    "CountOverflow",
    "UnknownColumnOrder",
    "NoLineRatings",
    "NoOperatingLimits",
]
