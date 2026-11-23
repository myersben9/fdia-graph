"""Every argument that takes one value from a fixed set, in one place.

A member is its string value (``Units.PU == "pu"``), so the public signatures accept plain strings.
A config model (`models.config`) declares a field's set as ``Annotated[str, OneOf(Units)]``, and
`models.validation` converts the value to the canonical string or refuses it.
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


class SplitBy(Choice):
    """How a timeline's splits are cut: three blocks in time order, or whole weeks drawn into the
    three splits (a pool whose frames carry their time)."""

    WEEK = "week"
    TIME = "time"


class LoadRule(Choice):
    """How a pool's injection at a bus divides into load and generation: by the one factor a scaled
    pool applied to both, or, for a pool whose units were dispatched (an OPF), a bus with a unit is
    all generation and any other bus all load."""

    SCALED = "scaled"
    NET = "net"


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

    BENIGN_LAYER = "benign_layer"
    CLEAN_LAYER = "clean_layer"
    PHYSICAL_UNITS = "physical_units"
    TIME_ORDER = "time_order"
    CONSECUTIVE = "consecutive"
    SPLIT = "split"
    SWING = "swing"
    TEMPORAL = "temporal"
    PMU_CURRENTS = "pmu_currents"


# ---- generation ----------------------------------------------------------------------------------
class AmAttack(Choice):
    """What an Am episode is: the overload attack of [WU26], a target branch's reported flow driven
    to its rating with the fewest devices tampered. In data release v0.8.3 and earlier an Am record is
    a load redistribution, not this overload."""

    OVERLOAD = "overload"


class CostUnit(Choice):
    """What the fewest-tamper search minimizes first (`FrameKnobs.objective`): the tampered measurements,
    the l0 of [WU26]'s eqs. (12), (28) and (33) taken literally, or the tampered devices, a SCADA terminal
    or a PMU counted once however many of its channels move. The other count breaks ties."""

    CHANNELS = "channels"
    DEVICES = "devices"


class SupportMethod(Choice):
    """How the overload attack picks the buses its false state moves [WU26 eq. 12]: the fewest-tamper
    search over the area's supports (`Minimizer.min_tamper`), or the paper's row reduction of the
    transposed attack-area Jacobian with column exchanges (`engine.attacks.rref.RrefSupport`, Sec. IV-D1 after
    [YAN17])."""

    SEARCH = "search"
    RREF = "rref"


class AreaRule(Choice):
    """How the attacker's area is chosen when no area is given [WU26] Sec. III-A: the buses within
    `hops` of the goal, or the area the paper's four principles pick (`AttackArea.rule_area`; rules
    1 and 2 as checks, 3 and 4 as a score of ours)."""

    HOPS = "hops"
    RULES = "rules"


class LimitSlack(Choice):
    """How far past a generator limit [WU26 eqs. 22-23] an implied output may sit before the limit
    counts as passed: by what the bus's injection meter resolves (its accuracy class at the true
    reading, ours: a change under the meter's noise is one no reader of the file can tell), or by
    the numerical slack alone (`LIMIT_TOL_PQ`), which is the paper's hard limit."""

    METER = "meter"
    HARD = "hard"


class RatingSource(Choice):
    """Where the overload attack's line ratings S_max come from: each branch's peak
    true flow over the operating pool times a margin (every system), PGLib-OPF's `rate_a`
    (IEEE-14, 118 and 300), or each target's true flow at the window's end plus a fixed step
    ("delta", ours: a rating scale matched to [WU26] Fig. 4's attack magnitudes)."""

    POOL = "pool"
    PGLIB = "pglib"
    DELTA = "delta"


class SolveMethod(Choice):
    """How one step of the pool is solved, in AC either way: a power flow (the generator outputs are
    given, the voltages are solved) or an optimal power flow (the outputs are chosen at least cost
    within the limits of the units and the voltages)."""

    POWER_FLOW = "power_flow"
    OPF = "opf"


class SolveEngine(Choice):
    """What runs the solve: pandapower (its Newton power flow, its interior-point OPF) or the
    package's own OPF, a Pyomo model solved by Ipopt (`data.ipopt_solver.IpoptSolver`)."""

    PANDAPOWER = "pandapower"
    IPOPT = "ipopt"


class MeterModel(Choice):
    """What the meters of a generated file measure. "hybrid": a SCADA voltmeter reads the voltage
    magnitude only, the voltage angle is a PMU channel, and every PMU also reads the current phasor
    of each in-service branch at its bus [WU26 eqs. 17-20]. The loader also reads data release
    v0.8.3 files, which have an angle at every voltmeter bus and no branch currents."""

    HYBRID = "hybrid"


class Iso(Choice):
    """The system operators a load profile can be fetched from. Matched without regard to case."""

    CAISO = "caiso"
    NYISO = "nyiso"
    ERCOT = "ercot"

    @classmethod
    def _missing_(cls, value: object) -> Choice:
        folded = str(value).lower()
        return cls(folded) if folded in cls.values() else super()._missing_(value)
