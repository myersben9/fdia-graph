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


# ---- the attack families -------------------------------------------------------------------------
# Their codes (the file's data/family), the stealthy ones, and the names older releases used.
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al", 7: "Am"}
FAMILY_CODE = {name: code for code, name in FAMILIES.items()}  # "Aq" -> 1 (aliases: FAMILY_ALIAS)
BENIGN_CODE = FAMILY_CODE["benign"]
# local false states that pass the residual test
STEALTHY_FAMILIES = {FAMILY_CODE[n] for n in ("Aq", "At", "Al", "Am")}
HELDOUT_FAMILIES = ("As", "Ar")  # kept out of train and val in the unseen-attack protocol [BOY22]
ONE_FRAME_FAMILIES = ("Aq", "Al")  # the single-snapshot stealthy families
# new generation makes the multi-snapshot families only; the single-snapshot ones stay loadable from
# released files and are deprecated for generation (retire in 0.22)
GENERATED_FAMILIES = ("At", "Am")
DEPRECATED_FOR_GENERATION = ("Aq", "Ad", "As", "Ar", "Al")
# the families of data release v0.8.3 and the frozen test timeline, in their rotation order
LEGACY_FAMILIES = ("Aq", "Ad", "As", "Ar", "At", "Al", "Am")
FAMILY_ALIAS = {"Ao": 1, "SLS": 1, "ramp": 5, "LRA": 6}  # backward-compatible family-name aliases


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
    SPLIT = "split"
    SWING = "swing"
    TEMPORAL = "temporal"
    PMU_CURRENTS = "pmu_currents"


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
class AmAttack(Choice):
    """What an Am episode is: the overload attack of [WU26] (a target branch's reported flow driven
    to its rating, the fewest devices tampered), or the held load redistribution of data release
    v0.8.3 and earlier, kept to reproduce those files."""

    OVERLOAD = "overload"
    REDISTRIBUTION = "redistribution"


class CostUnit(Choice):
    """What the attack cost of [WU26]'s defense counts (the plan's E3): the tampered measurements, the
    l0 of eqs. (28) and (33) and the unit of Table II's percentages, or the tampered devices, the unit
    of its extra-device counts (Table II, Fig. 12)."""

    CHANNELS = "channels"
    DEVICES = "devices"


class SupportMethod(Choice):
    """How the overload attack picks the buses its false state moves [WU26, eq. 12]: the fewest-tamper
    search over the area's supports (`MinimizeMixin.min_tamper`), or the paper's row reduction of the
    transposed attack-area Jacobian with column exchanges (`RrefMixin.rref_support`, Sec. IV-D1 after
    [YAN17])."""

    SEARCH = "search"
    RREF = "rref"


class RatingSource(Choice):
    """Where the overload attack's line ratings S_max come from (the plan's D15): each branch's peak
    true flow over the operating pool times a margin (every system), or PGLib-OPF's `rate_a`
    (IEEE-14, 118 and 300)."""

    POOL = "pool"
    PGLIB = "pglib"


class CutFamily(Choice):
    """A family of valid cuts the certifier adds to its relaxation (`engine/attacks/relax_cuts.py`):
    bound tightening of each bus's voltage move, the QC relaxation, and bus angles closing every
    cycle (docs/plans/RELAX_CERTIFIER_PLAN.md, section 2.1)."""

    BOUNDS = "bounds"
    QC = "qc"
    CYCLE = "cycle"


class CertifyVerdict(Choice):
    """What a certificate says (`models.frames.Certificate.verdict`): the search's count is globally
    minimal over the area, a gap remains between the bounds, or the bound rests on a result inside
    SCIP's numerical tolerances (a contradiction between relaxation levels, or an infeasibility the
    loosened re-solve does not confirm), so nothing is claimed."""

    CERTIFIED = "certified"
    GAP = "gap"
    UNCERTAIN = "uncertain"


class MeterModel(Choice):
    """What the meters of a generated file measure (the plan's D10). "hybrid": a SCADA voltmeter reads
    the voltage magnitude only, the voltage angle is a PMU channel, and every PMU also reads the
    current phasor of each in-service branch at its bus [WU26, eqs. 17-20]. "v083": the meter plan of
    data release v0.8.3 and earlier, an angle at every voltmeter bus and no branch currents, kept to
    reproduce those files."""

    HYBRID = "hybrid"
    V083 = "v083"


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
