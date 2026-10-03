"""The settings every consumer accepts, one model each, checked when built (`models.validation`).

The public signatures do not change: a constructor or function takes its keyword arguments as
before, builds its model here and reads the checked values from it. What a value may be is stated
once, on the field, and nowhere in the consumer.
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Annotated, Optional, Union, cast

import numpy as np

from .choices import (
    GENERATED_FAMILIES,
    AmAttack,
    AreaRule,
    Buses,
    Calibrate,
    CostUnit,
    Features,
    Format,
    FrOver,
    Iso,
    Kcl,
    Label,
    Layer,
    MeterModel,
    Order,
    RatingSource,
    RecordFormat,
    Reweight,
    Split,
    SupportMethod,
    Units,
)
from .validation import (
    AsTuple,
    AtLeast,
    ConfigError,
    Finite,
    InRange,
    Integer,
    NonEmpty,
    OneOf,
    Parses,
    Positive,
    Validated,
)

Fraction = Annotated[float, InRange(0.0, 1.0)]  # (0, 1), a false-alarm target or a split share
Share = Annotated[float, InRange(0.0, 1.0, hi_closed=True)]  # (0, 1]
Count = Annotated[int, Integer(), AtLeast(1)]  # a whole number of rounds, frames, clients
Norm = Annotated[Optional[float], Finite(), Positive()]  # an optional gradient-norm bound
Scale = Annotated[float, Positive(), Finite()]  # a finite positive constant (a Huber c, a rate)
Tolerance = Annotated[float, Finite(), AtLeast(0)]  # a convergence tolerance; 0 runs every pass


# ---- the dataset --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadOptions(Validated):
    """What `fg.load` and `FdiaGraph` are asked for; checked before any download or file read."""

    split: Annotated[Optional[str], OneOf(Split)] = None
    units: Annotated[str, OneOf(Units)] = "physical"
    order: Annotated[str, OneOf(Order)] = "time"
    format: Annotated[str, OneOf(RecordFormat)] = "torch"


def field_names(fields: object) -> tuple[str, ...]:
    """A list or tuple of field names as a tuple. A lone string is refused rather than split into
    letters, and a mapping or a set is refused rather than read for its keys."""
    if not isinstance(fields, (list, tuple)) or not all(isinstance(f, str) for f in fields):
        raise TypeError(fields)
    return tuple(fields)


@dataclass(frozen=True)
class ExportRequest(Validated):
    """What `export` is asked for."""

    format: Annotated[str, OneOf(Format)] = "torch"
    fields: Annotated[
        Optional[Sequence[str]], Parses(field_names, "must be a list or tuple of field names")
    ] = None

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            not (self.format == "pandas" and self.fields),
            "a pandas frame carries every field; pass fields with an array format",
        )


@dataclass(frozen=True)
class WindowSpec(Validated):
    """A sliding-window request over a view of T frames."""

    T: Count
    W: Annotated[int, Integer()]
    stride: Annotated[int, Integer()] = 1
    label: Annotated[str, OneOf(Label)] = "frame"
    layer: Annotated[str, OneOf(Layer)] = "node_x"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            1 <= self.W <= self.T and self.stride >= 1,
            f"need integers 1 <= W <= {self.T} frames and stride >= 1, got W={self.W!r}, stride={self.stride!r}",
        )


# ---- estimation --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SolveConfig(Validated):
    """The chord-Newton solve every estimator shares."""

    npass: Annotated[int, Integer(), AtLeast(1)] = 40  # reweighting passes
    iters: Annotated[int, Integer(), AtLeast(1)] = 8  # chord-Newton steps inside each solve
    # [WU26 eq. 3] pseudo voltage phasors at the far ends of PMU-metered branches (hybrid-meter files)
    pmu_pseudo: bool = False


@dataclass(frozen=True)
class FitOptions(Validated):
    """How an estimator calibrates on the benign training records."""

    n_calib: Annotated[int, Integer(), AtLeast(1)] = 600
    calibrate: Annotated[str, OneOf(Calibrate)] = "truth"


@dataclass(frozen=True)
class HuberConfig(Validated):
    """Iteratively reweighted least squares with Huber weights."""

    c: Scale = 1.5
    tol: Tolerance = 1e-4


@dataclass(frozen=True)
class RemovalConfig(Validated):
    """Largest-normalized-residual removal."""

    threshold: Scale = 4.0
    cond_mult: Annotated[float, AtLeast(1)] = 100.0


@dataclass(frozen=True)
class PriorConfig(Validated):
    """The low-rank benign prior, optionally with Huber reweighting."""

    rank_frac: Share = 0.5
    reweight: Annotated[Optional[str], OneOf(Reweight)] = None
    c: Scale = 1.5
    tol: Tolerance = 1e-4


@dataclass(frozen=True)
class JacobianWeightingConfig(Validated):
    """Huber weights from the unexplained scan-to-scan change, optionally with classical passes."""

    c: Scale = 3.0
    reweight: Annotated[Optional[str], OneOf(Reweight)] = None
    huber_c: Scale = 1.5
    tol: Tolerance = 1e-4


@dataclass(frozen=True)
class GateConfig(Validated):
    """A localizer gating the proposed estimator's weights: a fitted localizer or "oracle"."""

    gate: object = None  # "oracle" or a fitted localizer; the invariant says which
    gate_factor: Share = 1e-3

    @property
    def is_oracle(self) -> bool:
        """The string "oracle" (compared only once it is known to be a string, so an array gate
        never reaches an ambiguous `==`)."""
        return isinstance(self.gate, str) and self.gate == "oracle"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.is_oracle or callable(getattr(self.gate, "localize", None)),  # the localizer interface
            f"pass gate=<fitted localizer> or gate='oracle', got {self.gate!r}",
        )


# ---- localization and trust --------------------------------------------------------------------------
@dataclass(frozen=True)
class LocalizerConfig(Validated):
    """The false-alarm budget every localizer calibrates to."""

    fa_target: Fraction = 0.01


@dataclass(frozen=True)
class LearnedConfig(Validated):
    """The encoder a learned localizer builds, the vector it reads and how it trains."""

    layers: Annotated[int, Integer(), AtLeast(1)] = 4
    hidden: Annotated[int, Integer(), AtLeast(8)] = 128
    features: Annotated[str, OneOf(Features)] = "full14"
    dropout: Annotated[float, InRange(0.0, 1.0, lo_closed=True)] = 0.1
    lr: Scale = 5e-4
    weight_decay: Annotated[float, Finite(), AtLeast(0)] = 0.01
    batch_size: Count = 256
    epochs: Count = 60
    pos_weight: Scale = 1.0
    seed: Annotated[int, Integer()] = 123


@dataclass(frozen=True)
class TrainerConfig(Validated):
    """The training loop's gradient-norm bound."""

    clip: Norm = None


@dataclass(frozen=True)
class PerBusReport(Validated):
    """Which buses and records a per-bus table counts."""

    buses: Annotated[str, OneOf(Buses)] = "active"
    fr_over: Annotated[str, OneOf(FrOver)] = "all"


@dataclass(frozen=True)
class TrustConfig(Validated):
    """A trusted-meter budget and the residual test's false-alarm target."""

    k: Annotated[int, Integer(), AtLeast(1)]
    fa_target: Fraction = 0.01


@dataclass(frozen=True)
class FederatedSettings(Validated):
    """A federated fit: clients, rounds, local passes, the halo, the clip and where KCL comes from."""

    K: Count = 2
    rounds: Count = 60
    local_epochs: Count = 3
    halo: Annotated[int, Integer(), AtLeast(0)] = 0
    grad_clip: Norm = 1.0
    kcl: Annotated[str, OneOf(Kcl)] = "local"
    # K of a partition passed in, which must agree
    partition_clients: Annotated[Optional[int], Integer(), AtLeast(1)] = None
    # the centralized knob, refused: a federated fit counts rounds
    epochs: Annotated[Optional[int], Integer()] = None

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield self.epochs is None, "a federated fit trains rounds x local_epochs; pass those, not epochs"
        yield (
            self.partition_clients in (None, self.K),
            f"the partition has {self.partition_clients} clients but K={self.K}",
        )


# ---- generation --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitSettings(Validated):
    """The chronological train/val/test fractions a timeline is cut into before any episode is placed
    (`generate_timeline(split=...)`): three, each finite in (0, 1), summing to 1."""

    fractions: Annotated[Sequence[float], AsTuple()] = (0.6, 0.2, 0.2)

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield len(self.fractions) == 3, "takes three fractions: train, val and test"
        yield (
            all(math.isfinite(f) and 0.0 < f < 1.0 for f in self.fractions),
            "each fraction must be in (0, 1)",
        )
        yield abs(sum(self.fractions) - 1.0) < 1e-9, "the fractions must sum to 1"


@dataclass(frozen=True)
class TimelineKnobs(Validated):
    """The knobs of one timeline walk that the walk cannot recover from."""

    attacked_frac: Annotated[float, InRange(0.0, 1.0, lo_closed=True, hi_closed=True)] = 0.5
    hops: Count = 2
    ramp_len: Annotated[int, Integer(), AtLeast(1)] = 60
    am_len: Annotated[Optional[int], Integer(), AtLeast(1)] = None  # None: as long as a ramp
    min_tamper: bool = True  # [WU26 eq. 12]: each At episode on the support tampering the fewest devices
    min_budget: Count = 256  # candidate supports the search solves per episode before it settles
    am_attack: Annotated[str, OneOf(AmAttack)] = "overload"  # [WU26]'s overload attack
    # a multiplier on At's stealth bound, whose unit is the rated accuracy [D7]; Am has no bound [D11]
    stealth_scale: Scale = 1.0

    @property
    def am_frames(self) -> int:
        """The Am episode length: `am_len`, or `ramp_len` when none is given."""
        return self.ramp_len if self.am_len is None else self.am_len


@dataclass(frozen=True)
class OverloadSettings(Validated):
    """The overload attack's settings, as `generate_timeline(am_attack=...)` takes them in a dict of
    these fields (the string "overload" is these defaults): the line ratings [D15], S_max
    of each branch `rating_margin` times its peak true apparent flow over the operating pool
    ("pool", every system) or PGLib-OPF's `rate_a` ("pglib", IEEE-14, 118 and 300); the
    load-plausibility cap `load_cap` on every load bus the attack moves [D16]; and `n_lines`, the
    lines one episode overloads at once [D17]."""

    rating_source: Annotated[str, OneOf(RatingSource)] = "pool"
    rating_margin: Annotated[float, Finite(), InRange(1.0, math.inf)] = 1.25  # > 1
    # the load-plausibility cap tau: no load bus the attack moves shows a change beyond tau times its
    # true load [YUA11] ([D16]; a rule of ours, Yuan's 20% to 50%, the upper end by default)
    load_cap: Annotated[float, Finite(), InRange(0.0, 1.0, hi_closed=True)] = 0.5
    # the lines each episode overloads at once [D17]: [WU26]'s case studies always drive two; 1 or 2,
    # since every added line multiplies the target sets an episode tries and the paper uses no more
    n_lines: Annotated[int, Integer(), InRange(1, 2, lo_closed=True, hi_closed=True)] = 2
    # how an episode's support is chosen: the fewest-tamper search, or [WU26]'s row reduction ("rref")
    support_method: Annotated[str, OneOf(SupportMethod)] = "search"
    # with rating_source "delta": each target's S_max is its true flow at the window's end plus this
    # many per unit of the system base (ours, matched to [WU26] Fig. 4's 0.10 to 0.22 pu attacks;
    # experiment `minlp.rating_delta`)
    rating_delta: Annotated[float, Finite(), InRange(0.0, math.inf)] = 0.10

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.support_method != SupportMethod.WU_L1.value,
            ("the wu_l1 attack is the [WU26] reproduction's (trust.wu26), not the generator's"),
        )

    @staticmethod
    def of(am_attack: Union[str, dict]) -> tuple[str, Optional[OverloadSettings]]:
        """(the Am attack's kind, its settings): a dict of these fields means the overload attack with
        them; a string names the kind (checked by `TimelineKnobs`), None settings for an unknown one."""
        if isinstance(am_attack, dict):
            return AmAttack.OVERLOAD.value, OverloadSettings(**am_attack)
        return am_attack, OverloadSettings() if am_attack == AmAttack.OVERLOAD.value else None


@dataclass(frozen=True)
class TrustSchedule(Validated):
    """[WU26]'s dynamic trusted-PMU configuration (eqs. 26-32) as a constraint on the attack window: the
    PMU at bus `buses[i]` (the generator's bus index) is trusted from snapshot `slots[i]` of the window
    on. Trust accumulates (eqs. 30-31: h^S_t = h^S_t-1 + dh^S_t), so at snapshot t every PMU trusted at
    a slot <= t is pinned and a PMU trusted later can still be tampered with. A trusted PMU pins its
    own |V| and angle rows (eqs. 27, 32: two rows per trusted PMU, "the rank of hS becomes 2n x 2"),
    so the attack's state deviation is zero there (eq. 29); its branch currents stay in the nonsecure
    set h^S' (eq. 27). One support is held for the window, a trusted bus dropping out of it from its
    slot (E13's support per slot never changed an answer in any run and was removed)."""

    buses: Annotated[Sequence[int], AsTuple()]
    slots: Annotated[Sequence[int], AsTuple()]

    def invariants(self) -> Iterable[tuple[bool, str]]:
        buses, slots = np.asarray(self.buses), np.asarray(self.slots)
        yield len(buses) == len(slots), "needs one slot per trusted bus"
        yield _indices(buses), "buses must be non-negative bus indices"
        yield len(set(self.buses)) == len(self.buses), "trusts each PMU once"
        yield _indices(slots), "slots must be non-negative snapshot indices"

    def pinned(self, t: int) -> frozenset[int]:
        """The buses whose PMU is trusted at snapshot t (every slot at or before t)."""
        return frozenset(int(b) for b, s in zip(self.buses, self.slots) if s <= t)


@dataclass(frozen=True)
class MeterSettings(Validated):
    """The meter plan a timeline walks, as `generate_timeline(redundancy=...)` takes it: the coverage
    fractions of the voltage meters, the PMUs and the flow meters, and what the meters measure (the
    plan's D10, D12): "hybrid" (angles at the PMUs only, PMU branch currents)."""

    vbus_frac: float = 0.6
    pmu_frac: float = 0.2
    flow_frac: float = 0.9
    meter_model: Annotated[str, OneOf(MeterModel)] = "hybrid"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.meter_model != MeterModel.WU26.value,
            ("the wu26 meter plan is the [WU26] reproduction's (trust.wu26), not a generator plan"),
        )

    @property
    def coverage(self) -> dict[str, float]:
        """The coverage fractions as the generator takes them."""
        return {"vbus_frac": self.vbus_frac, "pmu_frac": self.pmu_frac, "flow_frac": self.flow_frac}


@dataclass(frozen=True)
class RampSettings(Validated):
    """The slow ramp At: its per-frame growth `rate`, the episode `length` in frames and `stealth_scale`,
    a multiplier on its stealth bound whose unit is the meters' rated accuracy ([D7]; the
    overload Am has no such bound, D11)."""

    rate: Scale = 0.002
    length: Annotated[int, Integer(), AtLeast(1)] = 60
    stealth_scale: Scale = 1.0


@dataclass(frozen=True)
class SearchSettings(Validated):
    """The attacker's subnetwork and the fewest-tamper search [WU26 eq. 12]: `hops`, the buses within
    this many branches of the attacked loads (At) or target lines (Am); `min_tamper`, hold each At
    episode on the support tampering the fewest devices (off: the whole subnetwork); `min_budget`,
    the candidate supports solved per episode before the search settles; `area_rule`, how the
    attacker's area is chosen (`AreaRule`: the buses within `hops`, or [WU26] Sec. III-A's principles)."""

    hops: Count = 2
    min_tamper: bool = True
    min_budget: Count = 256
    area_rule: Annotated[str, OneOf(AreaRule)] = "hops"


# the flat keywords `generate_timeline` takes, by the settings field each one sets (section, field)
_FLAT: dict[str, tuple[str, str]] = {
    "ramp_rate": ("ramp", "rate"),
    "ramp_len": ("ramp", "length"),
    "stealth_scale": ("ramp", "stealth_scale"),
    "hops": ("search", "hops"),
    "min_tamper": ("search", "min_tamper"),
    "min_budget": ("search", "min_budget"),
}
_TOP = ("attacked_frac", "families", "am_len", "max_load_mw", "workers")


@dataclass(frozen=True)
class TimelineSettings(Validated):
    """Everything one timeline walk is built from, nested by subject (the plan's S2 item 4):
    `attacked_frac` of each split's frames under an episode, the `families` generated (At and Am),
    the ramp (`RampSettings`), the overload Am's length `am_len` (None: the ramp's) and settings
    (`OverloadSettings`), the meter plan (`MeterSettings`), the train/val/test cut (`SplitSettings`),
    the search (`SearchSettings`), the load cap `max_load_mw` (None: off) and the processes the Am
    designs run on (`workers`, the result does not depend on it). `generate_timeline` also takes every
    field as a flat keyword (`TimelineSettings.of`)."""

    attacked_frac: Annotated[float, InRange(0.0, 1.0, lo_closed=True, hi_closed=True)] = 0.5
    families: Annotated[Sequence[str], AsTuple()] = GENERATED_FAMILIES
    ramp: RampSettings = field(default_factory=RampSettings)
    am_len: Annotated[Optional[int], Integer(), AtLeast(1)] = None
    overload: OverloadSettings = field(default_factory=OverloadSettings)
    meters: MeterSettings = field(default_factory=MeterSettings)
    split: SplitSettings = field(default_factory=SplitSettings)
    search: SearchSettings = field(default_factory=SearchSettings)
    max_load_mw: Annotated[Optional[float], Positive()] = 2000.0
    workers: Count = 1
    am_attack: Annotated[str, OneOf(AmAttack)] = "overload"  # [WU26]'s overload attack, the one kind

    @property
    def am_frames(self) -> int:
        """The Am episode length: `am_len`, or the ramp's length when none is given."""
        return self.ramp.length if self.am_len is None else self.am_len

    def as_record(self) -> dict[str, object]:
        """The settings as nested plain values (JSON-ready), what a file and the results store record;
        `workers` is left out, since the timeline does not depend on it."""
        record = asdict(self)
        del record["workers"]
        return cast("dict[str, object]", _plain(record))

    @classmethod
    def of(cls, settings: Optional[TimelineSettings] = None, **knobs: object) -> TimelineSettings:
        """The settings `generate_timeline` walks: `settings` (default the defaults) with each flat
        keyword applied over it, every name `generate_timeline` has always taken: attacked_frac,
        families, ramp_rate, ramp_len, am_len, hops, redundancy (a `MeterSettings`), split (fractions or
        a `SplitSettings`), max_load_mw, min_tamper, min_budget, am_attack ("overload" or an
        `OverloadSettings`), stealth_scale, workers. A dict for `am_attack` or `redundancy` still works
        and warns: dicts go in 0.22."""
        base = settings or cls()
        unknown = sorted(set(knobs) - set(_FLAT) - set(_TOP) - {"redundancy", "split", "am_attack"})
        if unknown:
            raise ConfigError(f"generate_timeline got unknown settings: {', '.join(unknown)}")
        sections = {name: {} for name in ("ramp", "search")}
        for key, (section, name) in _FLAT.items():
            if key in knobs:
                sections[section][name] = knobs[key]
        top = {k: knobs[k] for k in _TOP if k in knobs}
        top.update(_section_knobs(knobs))
        return replace(
            base,
            ramp=replace(base.ramp, **sections["ramp"]),
            search=replace(base.search, **sections["search"]),
            **top,
        )


def _section_knobs(knobs: dict[str, object]) -> dict[str, object]:
    """The settings the flat `redundancy`, `split` and `am_attack` keywords stand for."""
    out: dict[str, object] = {}
    if "redundancy" in knobs:
        out["meters"] = _model(MeterSettings, knobs["redundancy"], "redundancy")
    if "split" in knobs:
        split = knobs["split"]
        out["split"] = (
            split if isinstance(split, SplitSettings) else SplitSettings(cast(Sequence[float], split))
        )
    if "am_attack" in knobs:
        am = knobs["am_attack"]
        if isinstance(am, str):
            out["am_attack"] = am
        else:
            out["overload"] = _model(OverloadSettings, am, "am_attack")
    return out


def _model(model: type, value: object, name: str) -> object:
    """A settings model from its instance, None (the defaults) or a dict (deprecated, warned)."""
    if value is None:
        return model()
    if isinstance(value, dict):
        warnings.warn(
            f"generate_timeline({name}=dict) is deprecated and goes in 0.22: pass a {model.__name__}",
            DeprecationWarning,
            stacklevel=5,  # the caller of generate_timeline or generate
        )
        return model(**value)
    if not isinstance(value, model):
        raise ConfigError(f"{name} must be a {model.__name__}")
    return value


def _plain(value: object) -> object:
    """Nested settings as JSON-ready values (tuples as lists)."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class GeneratorOptions(Validated):
    """The generator's cap on what counts as a single load (MW, None disables it) and the meter model
    its plan follows."""

    max_load_mw: Annotated[Optional[float], Positive()] = 2000.0
    meter_model: Annotated[str, OneOf(MeterModel)] = "hybrid"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.meter_model != MeterModel.WU26.value,
            ("the wu26 meter plan is the [WU26] reproduction's (trust.wu26), not a generator plan"),
        )


@dataclass(frozen=True)
class ShardRun(Validated):
    """How many pool timesteps a shard generation walks; None walks them all."""

    frames: Annotated[Optional[int], Integer(), AtLeast(1)] = None


# How each operator's downloadable CSV export names its (timestamp, load) columns.
EXPORT_COLUMNS = {"caiso": ("interval_start_local", "load"), "nyiso": ("Time Stamp", "Load")}


@dataclass(frozen=True)
class IsoExport(Validated):
    """A directory of one operator's load export; the operator must have a known export format."""

    iso: Annotated[str, OneOf(Iso)]
    directory: str = "."

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            isinstance(self.directory, (str, os.PathLike)),
            f"directory must be a path, got {self.directory!r}",
        )
        yield (
            self.iso in EXPORT_COLUMNS,
            f"no CSV export format is known for {self.iso}; one of {sorted(EXPORT_COLUMNS)}",
        )


@dataclass(frozen=True)
class ProfileFetch(Validated):
    """A load-profile download: the operator and the optional resampling cadence (minutes)."""

    iso: Annotated[str, OneOf(Iso)]
    resample_min: Annotated[Optional[int], Integer(), AtLeast(1)] = None


@dataclass(frozen=True)
class WuDefenseConfig(Validated):
    """[WU26]'s trusted-PMU MDP (Sec. IV-D2, Fig. 1; `trust.WuDefenseEnv`): the candidate PMU buses (the
    generator's bus indices), the snapshot of the window at which each configuration step trusts one of
    them ("one PMU per step"; IEEE-14's steps at snapshots 2, 4, 6 and 8, IEEE-118's one per snapshot),
    and the unit the attack cost counts [E3]."""

    pmus: Annotated[Sequence[int], AsTuple(), NonEmpty()]
    slots: Annotated[Sequence[int], AsTuple(), NonEmpty()]
    unit: Annotated[str, OneOf(CostUnit)] = "channels"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        pmus, slots = np.asarray(self.pmus), np.asarray(self.slots)
        yield _indices(pmus), "pmus must be non-negative bus indices"
        yield len(set(self.pmus)) == len(self.pmus), "lists each PMU once"
        yield _indices(slots), "slots must be non-negative snapshot indices"
        yield (
            bool((np.diff(slots) >= 0).all()),
            "slots must not decrease: each step trusts at or after the last",
        )
        yield len(slots) <= len(pmus), "cannot have more steps than PMUs to trust"


@dataclass(frozen=True)
class Wu26Attack(Validated):
    """[WU26]'s attack as `trust.wu26` reproduces it: the meter plan "wu26" and the solver "wu_l1" (eq. 12
    read literally, the l1 of the window's summed attack), with the settings the paper leaves open, all
    ours: the overload `rho` (each target's flow ramps to rho times its true value by the window's end;
    no rating is stated), the weight `tau` of the per-snapshot l1 that bounds the intermediate snapshots
    (0.5 on IEEE-14, 0.05 on IEEE-118, calibrated to Fig. 4's magnitude and Table V's set), the
    per-snapshot trust region of `dv` pu and `da` radians around the true state, and the voltage band
    [`vmin`, `vmax`] of eq. (21)."""

    rho: Annotated[float, Finite(), InRange(1.0, math.inf)] = 1.5
    tau: Annotated[float, Finite(), InRange(0.0, math.inf, lo_closed=True)] = 0.05
    dv: Annotated[float, Finite(), Positive()] = 0.04
    da: Annotated[float, Finite(), Positive()] = 0.12
    vmin: Annotated[float, Finite(), Positive()] = 0.94
    vmax: Annotated[float, Finite(), Positive()] = 1.06
    # IPOPT's iteration cap per window (ours): the reproduction's numbers were measured at 400, where a
    # 40-bus IEEE-118 window often stops at the cap with every goal met
    max_iter: Annotated[int, Integer(), AtLeast(1)] = 400
    meter_model: Annotated[str, OneOf(MeterModel)] = "wu26"
    support_method: Annotated[str, OneOf(SupportMethod)] = "wu_l1"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield self.vmin < self.vmax, f"vmin {self.vmin} must be below vmax {self.vmax}"
        yield self.meter_model == MeterModel.WU26.value, "the reproduction reads the wu26 meter plan"
        yield self.support_method == SupportMethod.WU_L1.value, "the reproduction solves with wu_l1"


@dataclass(frozen=True)
class WuDqnConfig(Validated):
    """[WU26]'s DQN (Sec. IV-D2, Algorithm 1, Sec. V and the Appendix; `trust.TrustedPMUsDQN`): the
    stated hyperparameters by default, the discount gamma 0.9, the learning rate 0.005 (Adam), the
    replay buffer of 2,500 transitions, minibatches of 25, the target network updated every 20
    iterations, the exploration rate e^(-0.002 ep) and about 250 episodes (Fig. 13). The network's
    width is ours (the paper gives none): two hidden layers of `hidden` units."""

    episodes: Count = 250
    gamma: Share = 0.9
    lr: Scale = 0.005
    buffer: Count = 2500
    batch: Count = 25
    target_every: Count = 20
    epsilon_decay: Scale = 0.002
    hidden: Count = 128
    seed: Annotated[int, Integer(), AtLeast(0)] = 123

    def invariants(self) -> Iterable[tuple[bool, str]]:
        # a minibatch larger than the buffer is never drawn, so training would make no update
        yield self.batch <= self.buffer, f"batch {self.batch} cannot exceed the replay buffer {self.buffer}"


def _indices(a: np.ndarray) -> bool:
    """Whether `a` holds only non-negative integers (an empty sequence does)."""
    return a.size == 0 or (np.issubdtype(a.dtype, np.integer) and bool((a >= 0).all()))
