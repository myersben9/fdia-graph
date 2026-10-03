"""The named errors of conditions only the data reveals, beside `ConfigError` so a model can raise
one (`Validated.error`). `fdia_graph.errors` re-exports every one."""

from __future__ import annotations


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


class NoSuchBranch(DataConditionError):
    """No branch joins the two buses named (a [WU26] target line not in the case)."""


class SnapshotDrawFailed(DataConditionError):
    """Too few of [WU26]'s steady-state draws converged to fill a window (`trust.wu26_snapshots`)."""
