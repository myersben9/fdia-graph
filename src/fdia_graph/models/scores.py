"""The result tables: an estimator's error pair per record class, a localizer's metric rows per
class, and the Jacobian feature outputs. Each is a Bundle, indexable by family name as before."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import Bundle


@dataclass(frozen=True, eq=False)
class ErrorPair(Bundle):
    """Mean absolute error of one record class: angles in degrees, voltage magnitudes per unit."""

    angle_mae_deg: float  # mean |theta_hat - theta| over buses and records, degrees
    voltage_mae_pu: float  # mean ||V|_hat - |V|| over buses and records, per unit


@dataclass(frozen=True, eq=False)
class EstimatorScores(Bundle):
    """`SEBase.score`: the error pair of every record class present and their geometric mean
    (`geo`, the estimation paper's table cell). Indexable by family name as before, `geo` last."""

    _tail = ("geo",)
    geo: ErrorPair  # geometric mean over the classes present
    benign: Optional[ErrorPair] = None  # attack-free records
    Aq: Optional[ErrorPair] = None  # stealthy re-solve attack
    Ad: Optional[ErrorPair] = None  # additive bias
    As: Optional[ErrorPair] = None  # scaling
    Ar: Optional[ErrorPair] = None  # replay
    At: Optional[ErrorPair] = None  # slow ramp
    Al: Optional[ErrorPair] = None  # load redistribution
    Am: Optional[ErrorPair] = None  # multi-snapshot (timeline files)


@dataclass(frozen=True, eq=False)
class OverallMetrics(Bundle):
    """Pooled over every record, benign included: the papers' per-bus macro F1, detection rate
    and false-positive rate over the active buses (attacked somewhere in the records scored), and the
    micro node F1."""

    macro_f1: float  # mean per-bus F1 over the active buses
    macro_dr: float  # mean per-bus detection rate over the active buses
    macro_fr: float  # mean per-bus false-positive rate on benign records
    node_f1: float  # micro F1 over every bus call


@dataclass(frozen=True, eq=False)
class BenignMetrics(Bundle):
    """On benign records: the record-level false-alarm rate and the mean per-bus alarm rate."""

    false_alarm_rate: float  # benign records with any bus flagged
    bus_alarm_rate: float  # mean per-bus flag rate on benign records (calibrated to fa_target)


@dataclass(frozen=True, eq=False)
class FamilyMetrics(Bundle):
    """On one attacked family: strict localization accuracy, micro node precision/recall/F1, per-bus
    macro F1 over the buses the family attacks, per-sample F1, and the record-level detection rate."""

    strict_acc: float  # records whose flagged set equals the attacked set
    node_precision: float  # micro precision over bus calls
    node_recall: float  # micro recall over bus calls
    node_f1: float  # micro F1 over bus calls
    macro_f1: float  # mean per-bus F1 over the buses the family attacks
    sample_f1: float  # mean per-record F1
    detection_rate: float  # records with any bus flagged


@dataclass(frozen=True, eq=False)
class LocalizerScores(Bundle):
    """`LocalizerBase.score`: `all` (pooled), `benign`, and one entry per attacked family present.
    Indexable by family name as before."""

    all: OverallMetrics  # pooled over every record
    benign: Optional[BenignMetrics] = None  # attack-free records
    Aq: Optional[FamilyMetrics] = None  # stealthy re-solve attack
    Ad: Optional[FamilyMetrics] = None  # additive bias
    As: Optional[FamilyMetrics] = None  # scaling
    Ar: Optional[FamilyMetrics] = None  # replay
    At: Optional[FamilyMetrics] = None  # slow ramp
    Al: Optional[FamilyMetrics] = None  # load redistribution
    Am: Optional[FamilyMetrics] = None  # multi-snapshot (timeline files)


@dataclass(frozen=True, eq=False)
class PerBusMetrics(Bundle):
    """One block of per-bus localization metrics at the localizer's thresholds, the federated
    paper's node-wise table: arrays aligned to `bus_index`, and their means over those buses."""

    bus_index: np.ndarray  # [B] the buses reported
    threshold: np.ndarray  # [B] each bus's decision threshold
    f1: np.ndarray  # [B] per-bus F1
    dr: np.ndarray  # [B] per-bus detection rate
    fr: np.ndarray  # [B] per-bus false-alarm rate over the negatives counted (see fr_over)
    auprc: np.ndarray  # [B] per-bus average precision of the score, NaN where never attacked
    n_pos: np.ndarray  # [B] attacked records per bus
    macro_f1: float  # mean of f1
    macro_dr: float  # mean of dr
    macro_fr: float  # mean of fr
    macro_auprc: float  # mean of auprc over the buses where it exists (NaN if none)


@dataclass(frozen=True, eq=False)
class PerBusScores(Bundle):
    """`LocalizerBase.score_perbus`: `all` over every record, and per attacked family the block
    over that family's records plus the benign ones (the paper's per-type convention)."""

    all: PerBusMetrics  # every record
    Aq: Optional[PerBusMetrics] = None  # stealthy re-solve attack (+ benign)
    Ad: Optional[PerBusMetrics] = None  # additive bias (+ benign)
    As: Optional[PerBusMetrics] = None  # scaling (+ benign)
    Ar: Optional[PerBusMetrics] = None  # replay (+ benign)
    At: Optional[PerBusMetrics] = None  # slow ramp (+ benign)
    Al: Optional[PerBusMetrics] = None  # load redistribution (+ benign)
    Am: Optional[PerBusMetrics] = None  # multi-snapshot (+ benign)


@dataclass(frozen=True, eq=False)
class GridScores(Bundle):
    """`LearnedLocalizer.score_grid`: record-level detection, a record flagged when its highest
    attackable-bus probability exceeds `tau` (tuned on validation for grid F1)."""

    tau: float  # the grid threshold
    false_alarm: float  # benign records flagged
    detection_rate: float  # attacked records flagged (every family)
    by_family: dict  # detection rate per attacked family present


@dataclass(frozen=True, eq=False)
class TrustScores(Bundle):
    """`TrustedMeters.score`: what securing the selected meters does. `cost` is the attack cost
    after each secured meter (the meters the cheapest stealthy attack still has to touch, inf once
    none is left); `detected_before` / `detected_after` the fraction of attacked records of each
    family present whose largest normalized residual crosses the benign alarm level, with the
    attacker free to write every meter and with the secured meters reading their un-attacked value;
    `false_alarm` the benign record fraction over the alarm level (the calibration target)."""

    order: list[int]  # the secured meters, in the order they were secured (masked measurement index)
    cost: list[float]  # the attack cost after each
    detected_before: dict[str, float]  # per family present: detection rate with every meter writable
    detected_after: dict[str, float]  # per family present: detection rate with the secured meters pinned
    false_alarm: float  # benign records over the alarm level


@dataclass(frozen=True, eq=False)
class JacobianOutputs(Bundle):
    """`JacobianFeatures.transform`: the per-bus block [n, N, 8], the global features [n, 4]
    (under the dict key "global"), the implied state change [n, SD] and the unexplained residual
    [n, m]."""

    _keys = {"global_": "global"}
    bus: np.ndarray  # [n, N, 8] per-bus features
    global_: np.ndarray  # [n, 4] per-record features, dict key "global"
    dx_hat: np.ndarray  # [n, 2N-1] implied state change (H^T W H)^-1 H^T W dz
    r_perp: np.ndarray  # [n, m] residual the Jacobian cannot explain, (I - P) dz
