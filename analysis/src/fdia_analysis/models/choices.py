"""The fixed sets the analysis settings choose from, beside `fdia_graph.models.choices` (its `Choice`
is their base): a member is its string value, and a settings model declares a field's set as
``Annotated[str, OneOf(Features)]``.
"""

from __future__ import annotations

from fdia_graph.models.choices import Choice


# ---- estimation and localization -----------------------------------------------------------------
class Calibrate(Choice):
    """Where `fit` takes the meter errors and the reference from: the training truth (the
    estimation benchmark) or equipment data and measurements alone (any detector path)."""

    TRUTH = "truth"
    MEASURED = "measured"


class Reweight(Choice):
    """The robust reweighting an estimator adds on top of its solve."""

    HUBER = "huber"


class Features(Choice):
    """The per-bus vector a learned localizer reads (channels in `localization.learned.FEATURE_SETS`)."""

    MEAS = "meas"
    FULL14 = "full14"
    FULL14_PREV = "full14+prev"
    FULL14_JAC = "full14+jac"
    FULL14_PREV_JAC = "full14+prev+jac"
    JAC = "jac"


class FrOver(Choice):
    """The records a per-bus false-alarm rate counts: every non-attacked cell, or benign records."""

    ALL = "all"
    BENIGN = "benign"


class Buses(Choice):
    """The bus set a per-bus table reports: attacked somewhere in the view, or labelled in training."""

    ACTIVE = "active"
    ATTACKABLE = "attackable"


class Kcl(Choice):
    """Where a federated client's KCL residual comes from: its own buses and branches, or the grid."""

    LOCAL = "local"
    GLOBAL = "global"


class Reduce(Choice):
    """How a per-meter quantity aggregates to a bus: summed (energies) or the largest (changes)."""

    SUM = "sum"
    MAX = "max"
