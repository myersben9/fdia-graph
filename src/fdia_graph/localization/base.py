"""Per-bus FDIA localization on fdia-graph shards — the shared machinery behind every method class.

LocalizerBase owns what all localizers have in common: pulling the per-record arrays from a shard,
calibrating per-bus alarm thresholds on BENIGN training records at a chosen false-alarm budget, and
the scoring protocol (per-family node precision/recall/F1, strict localization accuracy, per-sample
macro-F1, and record-level detection rate — with the benign false-alarm rate always reported next to
it, because a detector judged by detection rate alone can simply flag everything). Subclasses change
only the per-bus score, mirroring fdia_graph.se where subclasses change only the estimator.

Numpy-only by default. ResidualLocalizer composes an estimator from fdia_graph.se and therefore
needs the [se] extra; the threshold methods run anywhere the loader runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..formulas.metrics import average_precision, perbus_counts, perbus_f1_from_counts, perbus_rates
from ..models.scores import (  # noqa: F401  re-exported: defined here before the models package
    BenignMetrics,
    FamilyMetrics,
    LocalizerScores,
    OverallMetrics,
    PerBusMetrics,
    PerBusScores,
)

if TYPE_CHECKING:
    from ..dataset import FdiaGraph

# Per-record fields that only exist on newer shards, and the FdiaGraph flag that says so — checked
# up front so a missing field is a clear message instead of an h5py KeyError mid-read.
_FIELD_FLAG = {"swing": "has_swing", "temporal_delta": "has_temporal", "clean": "has_clean"}


class LocalizerBase:
    """Threshold localization at a fixed per-bus false-alarm budget.

    Usage:
        loc  = SwingThreshold().fit(fg.load("ieee118", split="train"))
        flag = loc.localize(test_ds)      # [n, N] bool, which buses are called attacked
        rep  = loc.score(test_ds)         # per-family localization metrics + benign false alarms

    fit() computes the method's per-bus score on the benign training records and sets each bus's
    threshold at the (1 - fa_target) benign quantile, so every method is calibrated to the same
    false-alarm budget and differences between methods are differences in the score, not the tuning.
    """

    def __init__(self, fa_target: float = 0.01) -> None:
        if not 0.0 < fa_target < 1.0:
            raise ValueError(f"fa_target must be in (0, 1), got {fa_target}")
        self.fa_target = fa_target  # per-bus benign alarm rate the threshold is calibrated to

    # ---- subclass hooks ---------------------------------------------------------------------
    def _fields(self) -> list[str]:
        """Per-record arrays the score needs (beyond y/family, which fit/score always pull)."""
        raise NotImplementedError

    def _fit_stats(self, d: dict[str, np.ndarray], ben: np.ndarray, ds: FdiaGraph) -> None:
        """Learn anything the score needs from the benign training records (default: nothing)."""

    def _score(self, d: dict[str, np.ndarray], ds: FdiaGraph) -> np.ndarray:
        """Per-bus attack score [n, N]; higher means more suspicious. The one thing methods change.
        d holds the pulled arrays; ds is the dataset they came from, for methods that need more
        than its arrays (a composed estimator's per-record weights, the unit system)."""
        raise NotImplementedError

    # ---- data -------------------------------------------------------------------------------
    def _pull(self, ds: FdiaGraph, extra: Sequence[str] = ()) -> dict[str, np.ndarray]:
        want = list(dict.fromkeys(list(self._fields()) + list(extra)))  # ordered de-dup
        for k in want:
            flag = _FIELD_FLAG.get(k)
            if flag is not None and not getattr(ds, flag):
                raise ValueError(f"dataset has no '{k}' field; this method needs a newer shard")
        return ds.export(want)

    # ---- fitting ----------------------------------------------------------------------------
    def fit(self, ds: FdiaGraph) -> LocalizerBase:
        d = self._pull(ds, extra=["family"])
        ben = np.where(d["family"] == 0)[0]
        if not len(ben):
            raise ValueError("fit needs benign records; pass the train split unfiltered")
        self._fit_stats(d, ben, ds)
        s = self._score(d, ds)[ben]
        # Per-bus threshold at the (1 - fa_target) benign quantile: each bus alarms on ~fa_target
        # of benign scans by construction, so the operating point is set before any attack is seen.
        self.thr = np.quantile(s, 1.0 - self.fa_target, axis=0)
        return self

    # ---- public API -------------------------------------------------------------------------
    def scores(self, ds: FdiaGraph) -> np.ndarray:
        """Continuous per-bus attack scores [n, N] in record order."""
        return self._score(self._pull(ds), ds)

    def localize(self, ds: FdiaGraph) -> np.ndarray:
        """Boolean per-bus attack calls [n, N]: score above the bus's calibrated threshold."""
        return self.scores(ds) > self.thr[None, :]

    def score(self, ds: FdiaGraph, scores: Optional[np.ndarray] = None) -> LocalizerScores:
        """Per-family localization metrics against the per-bus labels. Pass `scores` (a previous
        `scores(ds)`, record order of `ds`) to skip recomputing them, e.g. from a cache.

        For each attacked family: strict localization accuracy (predicted attacked set equals the
        true set exactly), micro node precision/recall/F1 over bus calls, per-bus macro-F1 over the
        buses that family attacks, per-sample macro-F1, and the record-level detection rate (any
        bus flagged). For benign: the record-level false-alarm rate and the mean per-bus alarm rate
        (which fit calibrated to fa_target). The "all" entry is always present and pools every
        record, benign included; its macro_f1 (per-bus F1 averaged over attackable buses) is the
        papers' headline number, and reads 0.0 when the dataset holds no attacked bus at all.
        """
        from ..dataset import FAMILIES

        d = self._pull(ds, extra=["family", "y"]) if scores is None else ds.export(["family", "y"])
        s = self._score(d, ds) if scores is None else np.asarray(scores, np.float64)
        if s.shape != (len(ds), ds.N):
            raise ValueError(f"scores must be [{len(ds)}, {ds.N}], got {s.shape}")
        pred = s > self.thr[None, :]
        y = d["y"].astype(bool)
        out: dict[str, Any] = {"all": _overall_metrics(pred, y, d["family"] == 0)}
        for fid, name in FAMILIES.items():
            m = d["family"] == fid
            if m.any():
                out[name] = _benign_metrics(pred[m]) if fid == 0 else _family_metrics(pred[m], y[m])
        return LocalizerScores(**out)

    def score_perbus(
        self,
        ds: FdiaGraph,
        scores: Optional[np.ndarray] = None,
        buses: str = "active",
        fr_over: str = "all",
        auprc: bool = True,
    ) -> PerBusScores:
        """The federated paper's node-wise table at this localizer's thresholds: per-bus F1, DR,
        FR and AUPRC, over every record (`all`) and per attacked family over that family's records
        plus the benign ones.

        buses   : "active" (buses attacked somewhere in ds) or "attackable" (buses labelled in the
                  training records, the paper's set; learned localizers only)
        fr_over : "all" (the paper's Table IV FR: false alarms over every non-attacked cell, attacked
                  records included) or "benign" (benign records only); with buses="active" and
                  fr_over="benign" the `all` block's means equal `score()["all"]`'s macro F1, DR and FR
        """
        from ..dataset import FAMILIES

        if fr_over not in ("all", "benign"):
            raise ValueError(f"fr_over must be 'all' or 'benign', got {fr_over!r}")
        d = self._pull(ds, extra=["family", "y"]) if scores is None else ds.export(["family", "y"])
        s = self._score(d, ds) if scores is None else np.asarray(scores, np.float64)
        if s.shape != (len(ds), ds.N):
            raise ValueError(f"scores must be [{len(ds)}, {ds.N}], got {s.shape}")
        y, fam = d["y"].astype(bool), d["family"]
        cols = self._report_buses(y, buses)
        out: dict[str, Any] = {
            "all": self._perbus_rows(s, y, cols, fam, np.ones(len(fam), bool), fr_over, auprc)
        }
        for fid, name in FAMILIES.items():
            rows = (fam == fid) | (fam == 0)
            if fid and (fam == fid).any():
                out[name] = self._perbus_rows(s[rows], y[rows], cols, fam[rows], rows[rows], fr_over, auprc)
        return PerBusScores(**out)

    def _report_buses(self, y: np.ndarray, buses: str) -> np.ndarray:
        """The bus set a per-bus table reports."""
        if buses == "active":
            return np.flatnonzero(y.any(axis=0))
        if buses == "attackable" and hasattr(self, "_attackable"):
            return np.flatnonzero(getattr(self, "_attackable"))
        raise ValueError(f"buses must be 'active' or, for a learned localizer, 'attackable'; got {buses!r}")

    def _perbus_rows(
        self,
        s: np.ndarray,
        y: np.ndarray,
        cols: np.ndarray,
        fam: np.ndarray,
        rows: np.ndarray,
        fr_over: str,
        auprc: bool,
    ) -> PerBusMetrics:
        """One block: the negatives for FR are every record, or the benign ones."""
        negatives = rows if fr_over == "all" else fam == 0
        return perbus_block(s, y, np.asarray(self.thr, np.float64), cols, negatives, auprc)


def perbus_block(
    s: np.ndarray, y: np.ndarray, thr: np.ndarray, buses: np.ndarray, negatives: np.ndarray, auprc: bool
) -> PerBusMetrics:
    """The per-bus metrics of the chosen buses over the chosen records (`formulas.metrics`); the
    false-alarm rate counts only the rows in `negatives` (all records, or the benign ones)."""
    pred = s[:, buses] > thr[None, buses]
    t = y[:, buses]
    f1, dr, _ = perbus_rates(pred, t)
    _, _, fr = perbus_rates(pred[negatives], t[negatives])
    ap = np.full(len(buses), np.nan)
    if auprc:
        for j in np.flatnonzero(t.any(axis=0)):
            ap[j] = average_precision(s[:, buses[j]], t[:, j])
    return PerBusMetrics(
        bus_index=buses,
        threshold=thr[buses],
        f1=f1,
        dr=dr,
        fr=fr,
        auprc=ap,
        n_pos=t.sum(axis=0),
        macro_f1=float(f1.mean()) if len(buses) else 0.0,
        macro_dr=float(dr.mean()) if len(buses) else 0.0,
        macro_fr=float(fr.mean()) if len(buses) else 0.0,
        macro_auprc=float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan"),
    )


def _overall_metrics(pred: np.ndarray, y: np.ndarray, ben: np.ndarray) -> OverallMetrics:
    """Pooled over every record, benign included: the papers' per-bus macro scores over the
    attackable set (F1 and recall accumulate over every record, the false-positive rate over
    benign records only) and the micro node F1. macro_f1 reads 0.0 when no bus is ever attacked."""
    act = y.any(axis=0)
    tp = (pred & y).sum(axis=0).astype(np.float64)
    fn = (~pred & y).sum(axis=0).astype(np.float64)
    return OverallMetrics(
        macro_f1=float(_perbus_f1(pred, y)[act].mean()) if act.any() else 0.0,
        macro_dr=float((tp / np.maximum(tp + fn, 1e-9))[act].mean()) if act.any() else 0.0,
        macro_fr=float(pred[ben][:, act].mean()) if act.any() and ben.any() else 0.0,
        node_f1=_micro_f1(pred, y),
    )


def _benign_metrics(p: np.ndarray) -> BenignMetrics:
    """On benign records: the record-level false-alarm rate and the mean per-bus alarm rate (which
    fit calibrated to fa_target)."""
    return BenignMetrics(false_alarm_rate=float(p.any(axis=1).mean()), bus_alarm_rate=float(p.mean()))


def _family_metrics(p: np.ndarray, t: np.ndarray) -> FamilyMetrics:
    """On one attacked family: strict localization accuracy (predicted set equals the true set),
    micro node precision/recall/F1 over bus calls, per-bus macro-F1 over the buses the family
    attacks, per-sample macro-F1, and the record-level detection rate (any bus flagged)."""
    tp = float((p & t).sum())
    prec = tp / max(float(p.sum()), 1e-12)
    rec = tp / max(float(t.sum()), 1e-12)
    inter = (p & t).sum(axis=1).astype(np.float64)
    denom = np.maximum(p.sum(axis=1) + t.sum(axis=1), 1e-12)
    act = t.any(axis=0)
    return FamilyMetrics(
        strict_acc=float((p == t).all(axis=1).mean()),
        node_precision=prec,
        node_recall=rec,
        node_f1=2 * prec * rec / max(prec + rec, 1e-12),
        macro_f1=float(_perbus_f1(p, t)[act].mean()),
        sample_f1=float((2 * inter / denom).mean()),
        detection_rate=float(p.any(axis=1).mean()),
    )


def _perbus_f1(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """F1 per bus [N] over the record axis (`formulas.metrics`). Its mean over attackable buses is
    the papers' localization macro-F1."""
    return perbus_f1_from_counts(*perbus_counts(pred, truth))


def _micro_f1(pred: np.ndarray, truth: np.ndarray) -> float:
    """One F1 over every (record, bus) call."""
    tp = float((pred & truth).sum())
    prec = tp / max(float(pred.sum()), 1e-12)
    rec = tp / max(float(truth.sum()), 1e-12)
    return 2 * prec * rec / max(prec + rec, 1e-12)
