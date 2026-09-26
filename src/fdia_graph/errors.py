"""Every error the package raises on purpose, in one place.

Two kinds. An input a model does not allow is a `ConfigError` (and a dataset view lacking what a
consumer needs a `MissingCapability`), raised by the validation engine in `models.validation` when
the model is built. A condition that only the data can reveal (no benign records in a split, no
room left for an episode, an outage that islands the grid) is one of the named errors below, raised
where the data is read. Every one subclasses ValueError, the type these checks raised before, so a
caller's `except ValueError` keeps working.
"""

from __future__ import annotations

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
]


class DataConditionError(ValueError):
    """A condition the data read does not meet; base of the named errors below."""


class NoBenignRecords(DataConditionError):
    """A fit that calibrates on benign records was given none (a filtered view instead of a split)."""


class NoAttackedRecords(DataConditionError):
    """A threshold tuned on labelled validation records was given no attacked (or no benign) ones."""


class NotFitted(DataConditionError):
    """A step that needs an earlier one was called first."""


class NoAdmissibleTarget(DataConditionError):
    """A requested attack family has nothing to attack on this case."""


class NoRoomForEpisode(DataConditionError):
    """The attacked fraction cannot be placed: no free span is long enough for an episode."""


class GridIslanded(DataConditionError):
    """A line outage splits the grid or leaves a bus isolated."""


class SlackMismatch(DataConditionError):
    """The dataset's slack bus is not the case's."""


class VaryingReference(DataConditionError):
    """The slack angle varies across the training frames, so there is no one reference to fix."""
