"""Every error the package raises on purpose, defined in this one module.

Two kinds. An input a model does not allow is a `ConfigError` (and a dataset view lacking what a
consumer needs a `MissingCapability`), raised by the validation engine in `models.validation` when
the model is built. A condition only the data can reveal (no benign records in a split, no room
left for an episode) is one of the named errors below, raised where the data is read, or by a model
that names it in `Validated.error`. Every one is a ValueError, so a caller's `except ValueError`
catches it. `SolverUnavailable` is the exception: a RuntimeError for an environment that cannot run
the optimization at all.
"""

from __future__ import annotations


class ConfigError(ValueError):
    """An input that its model does not allow."""


class MissingCapability(ConfigError):
    """A dataset view that lacks what a consumer needs (`dataset.base.DatasetBase.require`)."""


class DataConditionError(ValueError):
    """A condition the data read does not meet; base of the named errors below."""


class NotATimeline(DataConditionError):
    """The file is not a timeline (its `kind` attribute says so): a record shard of a data release
    before v0.8.0, or no fdia-graph file at all."""


class NoBenignRecords(DataConditionError):
    """A fit that calibrates on benign records was given none (a filtered view instead of a split)."""


class NoAttackedRecords(DataConditionError):
    """A threshold tuned on labelled validation records was given no attacked (or no benign) ones."""


class NotFitted(DataConditionError):
    """A step that needs an earlier one was called first."""


class NoAdmissibleTarget(DataConditionError):
    """A requested attack family has nothing to attack on this case."""


class NoRoomForEpisode(DataConditionError):
    """The attacked fraction cannot be placed: no free span is long enough for an episode. `family`
    names the episode's family when the placement knows it."""

    def __init__(self, message: str, family: object = None) -> None:
        super().__init__(message)
        self.family = family


class SlackMismatch(DataConditionError):
    """The dataset's slack bus is not the case's."""


class VaryingReference(DataConditionError):
    """The slack angle varies across the training frames, so there is no one reference to fix."""


class CountOverflow(DataConditionError):
    """Pooled record counts past the float range."""


class UnknownColumnOrder(DataConditionError):
    """A state pool whose column order cannot be read from its values."""


class NoLineRatings(DataConditionError):
    """The overload attack (`Am`, [WU26 eqs. 24-25]) needs real line ratings, and the case has none
    (only IEEE-14, 118 and 300 carry the PGLib-OPF ratings)."""


class NoSuchResult(DataConditionError, LookupError):
    """A results query that must find one record found none, or several."""


class SolverUnavailable(RuntimeError):
    """The optimization solver a step needs (Ipopt, through Pyomo) is not installed or does not start.
    Not a property of the data, so not a `DataConditionError`."""
