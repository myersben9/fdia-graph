"""The settings the estimators, the localizers, the trust selectors and the federated trainers accept,
one model each, checked when built by the validation engine of `fdia_graph.models.validation`.

A constructor takes its keyword arguments, builds its model here and reads the checked values from
it. What a value may be is stated once, on the field, and nowhere in the consumer.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated, Optional

from fdia_graph.models.config import Count, Scale
from fdia_graph.models.validation import (
    AtLeast,
    Boolean,
    Finite,
    InRange,
    Integer,
    OneOf,
    Positive,
    Validated,
)

from .choices import Buses, Calibrate, Features, FrOver, Kcl, Reweight

Fraction = Annotated[float, InRange(0.0, 1.0)]  # (0, 1), a false-alarm target
Share = Annotated[float, InRange(0.0, 1.0, hi_closed=True)]  # (0, 1]
Norm = Annotated[Optional[float], Finite(), Positive()]  # an optional gradient-norm bound
Tolerance = Annotated[float, Finite(), AtLeast(0)]  # a convergence tolerance; 0 runs every pass


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
    attackable_only: Annotated[bool, Boolean()] = True  # train and score on the attackable buses only
    device: Optional[str] = None  # None: cuda when available, else cpu
    kernel: Annotated[int, Integer(), AtLeast(1)] = 3  # BusCNN's convolution width along the bus axis


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
class DqnConfig(Validated):
    """The DQN trusted-meter selector's training: episodes, discount, Adam step, MLP width and seed."""

    episodes: Count = 200
    gamma: Share = 0.95
    lr: Scale = 1e-3
    hidden: Count = 128
    seed: Annotated[int, Integer(), AtLeast(0)] = 123


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

    def invariants(self) -> Iterable[tuple[bool, str]]:
        yield (
            self.partition_clients in (None, self.K),
            f"the partition has {self.partition_clients} clients but K={self.K}",
        )
