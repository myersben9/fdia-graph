"""The settings every consumer accepts, one model each, checked when built (`models.validation`).

The public signatures do not change: a constructor or function takes its keyword arguments as
before, builds its model here and reads the checked values from it. What a value may be is stated
once, on the field, and nowhere in the consumer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Any, Optional

from .choices import (
    AmDirection,
    Buses,
    Calibrate,
    Features,
    Format,
    FrOver,
    Kcl,
    Label,
    Layer,
    Order,
    RecordFormat,
    Reweight,
    Split,
    Units,
)
from .validation import AtLeast, Finite, InRange, Integer, OneOf, Positive, Validated

Fraction = Annotated[float, InRange(0.0, 1.0)]  # (0, 1), a false-alarm target or a split share
Share = Annotated[float, InRange(0.0, 1.0, hi_closed=True)]  # (0, 1]
Count = Annotated[int, Integer(), AtLeast(1)]  # a whole number of rounds, frames, clients
Norm = Annotated[Optional[float], Finite(), Positive()]  # an optional gradient-norm bound


# ---- the dataset --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadOptions(Validated):
    """What `fg.load` and `FdiaGraph` are asked for; checked before any download or file read."""

    split: Annotated[Optional[str], OneOf(Split)] = None
    units: Annotated[str, OneOf(Units)] = "physical"
    order: Annotated[str, OneOf(Order)] = "time"
    format: Annotated[str, OneOf(RecordFormat)] = "torch"


@dataclass(frozen=True)
class ExportRequest(Validated):
    """What `export` is asked for."""

    format: Annotated[str, OneOf(Format)] = "torch"
    fields: Optional[tuple[str, ...]] = None

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            not (self.format == "pandas" and self.fields),
            "a pandas frame carries every field; pass fields with an array format",
        )


@dataclass(frozen=True)
class WindowSpec(Validated):
    """A sliding-window request over a view of T frames."""

    T: int
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

    c: Annotated[float, Positive()] = 1.5
    tol: float = 1e-4


@dataclass(frozen=True)
class RemovalConfig(Validated):
    """Largest-normalized-residual removal."""

    threshold: Annotated[float, Positive()] = 4.0
    cond_mult: Annotated[float, AtLeast(1)] = 100.0


@dataclass(frozen=True)
class PriorConfig(Validated):
    """The low-rank benign prior, optionally with Huber reweighting."""

    rank_frac: Share = 0.5
    reweight: Annotated[Optional[str], OneOf(Reweight)] = None
    c: Annotated[float, Positive()] = 1.5
    tol: float = 1e-4


@dataclass(frozen=True)
class JacobianWeightingConfig(Validated):
    """Huber weights from the unexplained scan-to-scan change, optionally with classical passes."""

    c: Annotated[float, Positive()] = 3.0
    reweight: Annotated[Optional[str], OneOf(Reweight)] = None
    huber_c: Annotated[float, Positive()] = 1.5
    tol: float = 1e-4


@dataclass(frozen=True)
class GateConfig(Validated):
    """A localizer gating the proposed estimator's weights: a fitted localizer or "oracle"."""

    gate: Any = None
    gate_factor: Share = 1e-3

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.gate == "oracle" or hasattr(self.gate, "localize"),  # the localizer interface
            f"pass gate=<fitted localizer> or gate='oracle', got {self.gate!r}",
        )


# ---- localization and trust --------------------------------------------------------------------------
@dataclass(frozen=True)
class LocalizerConfig(Validated):
    """The false-alarm budget every localizer calibrates to."""

    fa_target: Fraction = 0.01


@dataclass(frozen=True)
class LearnedConfig(Validated):
    """The encoder a learned localizer builds and the vector it reads."""

    layers: Annotated[int, Integer(), AtLeast(1)] = 4
    hidden: Annotated[int, Integer(), AtLeast(8)] = 128
    features: Annotated[str, OneOf(Features)] = "full14"


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
    partition_clients: Optional[int] = None  # K of a partition passed in, which must agree
    epochs: Optional[int] = None  # the centralized knob, refused: a federated fit counts rounds

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
    am_rate: Annotated[float, Positive()] = 0.9
    hops: Count = 2
    am_direction: Annotated[str, OneOf(AmDirection)] = "both"
    ramp_len: Annotated[int, Integer(), AtLeast(1)] = 60
    am_len: Annotated[int, Integer(), AtLeast(1)] = 60
    corrupt_len: Annotated[Optional[int], Integer(), AtLeast(1)] = 1


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
    """A train / validation share of a stream, the rest the test (the deprecated torch_data helpers)."""

    train_frac: Fraction = 0.6
    val_frac: Annotated[float, InRange(0.0, 1.0, lo_closed=True)] = 0.2
    max_test: Annotated[Optional[int], Integer(), AtLeast(0)] = None

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.train_frac + self.val_frac < 1.0,
            f"need train_frac + val_frac < 1, got {self.train_frac} + {self.val_frac}",
        )
