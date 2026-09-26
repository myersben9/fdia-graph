"""Every argument that takes one value from a fixed set, in one place.

A member is its string value (``Units.PU == "pu"``, ``f"{Units.PU}" == "pu"``), so the public
signatures keep accepting plain strings. A config model (`models.config`) declares a field's set as
``Annotated[str, OneOf(Units)]``, and the engine in `models.validation` converts the value to the
canonical string or refuses it; nothing else checks these values.
"""

from __future__ import annotations

import re
from enum import Enum


class Choice(str, Enum):
    """A string option from a fixed set."""

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Choice:
        raise ValueError(f"{cls.param()} must be one of {cls.values()}, got {value!r}")

    @classmethod
    def param(cls) -> str:
        """The class name in snake case (``AmDirection`` reads as ``am_direction``)."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()

    @classmethod
    def values(cls) -> list[str]:
        """Every allowed value, in declaration order."""
        return [m.value for m in cls]


# ---- the dataset ---------------------------------------------------------------------------------
class Split(Choice):
    """The chronological partitions of a file."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Units(Choice):
    """The unit system of the returned measurements: as stored, or per-unit with angles in radians."""

    PHYSICAL = "physical"
    PU = "pu"


class Order(Choice):
    """The record order of a view: the file's (chronological on a timeline) or a seeded permutation."""

    TIME = "time"
    RANDOM = "random"


class RecordFormat(Choice):
    """What indexing a dataset returns: a dict of tensors, or a PyG `Data`."""

    TORCH = "torch"
    PYG = "pyg"


class Format(Choice):
    """The array flavour `export` returns."""

    NUMPY = "numpy"
    TORCH = "torch"
    TF = "tf"
    PANDAS = "pandas"


class Label(Choice):
    """What a window's label is: per frame, attacked at any frame, or the last frame's."""

    FRAME = "frame"
    ANY = "any"
    LAST = "last"


class Layer(Choice):
    """The measurement layer a window reads: observed, attack-removed or noiseless."""

    NODE_X = "node_x"
    BENIGN = "benign"
    CLEAN = "clean"


class Capability(Choice):
    """What a dataset view can offer a consumer (the table is `dataset.base.CAPABILITIES`)."""

    TIMELINE = "timeline"
    BENIGN_LAYER = "benign_layer"
    CLEAN_LAYER = "clean_layer"
    PHYSICAL_UNITS = "physical_units"
    TIME_ORDER = "time_order"
    CONSECUTIVE = "consecutive"


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


# ---- generation ----------------------------------------------------------------------------------
class AmDirection(Choice):
    """What an Am episode does to its target line: reads lighter (a real overload hidden), reads
    more loaded than it is, or either, drawn per episode."""

    MASK = "mask"
    INDUCE = "induce"
    BOTH = "both"


class Iso(Choice):
    """The system operators a load profile can be fetched from. Matched without regard to case."""

    CAISO = "caiso"
    NYISO = "nyiso"
    ERCOT = "ercot"

    @classmethod
    def _missing_(cls, value: object) -> Choice:
        folded = str(value).lower()
        return cls(folded) if folded in cls.values() else super()._missing_(value)
