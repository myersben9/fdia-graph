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
    and false-positive rate over the attackable buses, and the micro node F1."""

    macro_f1: float  # mean per-bus F1 over the attackable buses
    macro_dr: float  # mean per-bus detection rate over the attackable buses
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
class JacobianOutputs(Bundle):
    """`JacobianFeatures.transform`: the per-bus block [n, N, 8], the global features [n, 4]
    (under the dict key "global"), the implied state change [n, SD] and the unexplained residual
    [n, m]."""

    _keys = {"global_": "global"}
    bus: np.ndarray  # [n, N, 8] per-bus features
    global_: np.ndarray  # [n, 4] per-record features, dict key "global"
    dx_hat: np.ndarray  # [n, 2N-1] implied state change (H^T W H)^-1 H^T W dz
    r_perp: np.ndarray  # [n, m] residual the Jacobian cannot explain, (I - P) dz
