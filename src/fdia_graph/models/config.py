"""The settings every consumer accepts, one model each, checked when built (`models.validation`).

The public signatures do not change: a constructor or function takes its keyword arguments as
before, builds its model here and reads the checked values from it. What a value may be is stated
once, on the field, and nowhere in the consumer.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Optional

from .choices import (
    AmDirection,
    Buses,
    Calibrate,
    Features,
    Format,
    FrOver,
    Iso,
    Kcl,
    Label,
    Layer,
    Order,
    RecordFormat,
    Reweight,
    Split,
    Units,
)
from .validation import AtLeast, Finite, InRange, Integer, OneOf, Parses, Positive, Validated

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


def field_names(fields: Any) -> tuple[str, ...]:
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

    gate: Any = None
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
class TimelineKnobs(Validated):
    """The knobs of one timeline walk that the walk cannot recover from."""

    attacked_frac: Annotated[float, InRange(0.0, 1.0, lo_closed=True, hi_closed=True)] = 0.5
    am_rate: Scale = 0.9
    hops: Count = 2
    am_direction: Annotated[str, OneOf(AmDirection)] = "both"
    ramp_len: Annotated[int, Integer(), AtLeast(1)] = 60
    am_len: Annotated[Optional[int], Integer(), AtLeast(1)] = None  # None: as long as a ramp
    corrupt_len: Annotated[Optional[int], Integer(), AtLeast(1)] = 1

    @property
    def am_frames(self) -> int:
        """The Am episode length: `am_len`, or `ramp_len` when none is given."""
        return self.ramp_len if self.am_len is None else self.am_len


@dataclass(frozen=True)
class GeneratorOptions(Validated):
    """The generator's cap on what counts as a single load (MW); None disables it."""

    max_load_mw: Annotated[Optional[float], Positive()] = 2000.0


@dataclass(frozen=True)
class ShardRun(Validated):
    """How many pool timesteps a shard generation walks; None walks them all."""

    frames: Annotated[Optional[int], Integer(), AtLeast(1)] = None


@dataclass(frozen=True)
class SplitFractions(Validated):
    """A train / validation share of a stream, the rest the test, and the measurement layer read (the
    deprecated torch_data helpers); checked before the stream loads."""

    train_frac: Fraction = 0.6
    val_frac: Annotated[float, InRange(0.0, 1.0, lo_closed=True)] = 0.2
    max_test: Annotated[Optional[int], Integer(), AtLeast(0)] = None
    layer: Annotated[str, OneOf(Layer)] = "node_x"

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.train_frac + self.val_frac < 1.0,
            f"need train_frac + val_frac < 1, got {self.train_frac} + {self.val_frac}",
        )


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
