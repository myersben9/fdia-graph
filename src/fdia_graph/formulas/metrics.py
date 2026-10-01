"""Per-bus localization metrics from confusion counts, so they can be summed across clients or
thresholds before any ratio is taken [KEC25]."""

from __future__ import annotations

import numpy as np

from ..models.inputs import LabelGrids, RankedLabels, TauSearch


def perbus_counts(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True positives, false positives and false negatives per bus over the record axis [KEC25].

    pred, truth : [n, N] bool
    returns     : (tp [N], fp [N], fn [N]) as float64
    """
    g = LabelGrids(pred, truth)
    pred, truth = g.pred, g.truth
    tp = (pred & truth).sum(axis=0).astype(np.float64)
    fp = (pred & ~truth).sum(axis=0).astype(np.float64)
    fn = (~pred & truth).sum(axis=0).astype(np.float64)
    return tp, fp, fn


def perbus_counts_at(
    score: np.ndarray, truth: np.ndarray, taus: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`perbus_counts` of `score > tau` at every candidate threshold at once.

    score : [n, N] float;  truth : [n, N] bool;  taus : [K]
    returns : (tp, fp, fn), each [K, N] float64
    """
    g = LabelGrids(np.zeros(np.shape(score), bool), truth)
    pred = np.asarray(score)[None] > np.asarray(taus, np.float64)[:, None, None]  # [K, n, N]
    t = g.truth[None]
    tp = (pred & t).sum(axis=1).astype(np.float64)
    fp = (pred & ~t).sum(axis=1).astype(np.float64)
    fn = (~pred & t).sum(axis=1).astype(np.float64)
    return tp, fp, fn


def perbus_f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """F1 per bus from its counts [KEC25]:  F1 = 2 TP / (2 TP + FP + FN), with 1e-9 in the
    denominator so a bus never attacked nor flagged scores 0.

    tp, fp, fn : [..., N]
    returns    : [..., N]
    """
    return 2 * tp / (2 * tp + fp + fn + 1e-9)


def tau_from_counts(
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, active: np.ndarray, taus: np.ndarray
) -> float:
    """The papers' global threshold: the tau whose mean per-bus F1 over the active buses is
    largest [FED26], the first one on ties.

    tp, fp, fn : [n_taus, N] counts at each candidate tau
    active     : [N] bool, the buses the mean runs over
    taus       : [n_taus]
    returns    : the chosen tau
    """
    search = TauSearch((tp, fp, fn), active, taus)
    taus, active = search.taus, search.active
    f1 = perbus_f1_from_counts(tp, fp, fn)[:, active].mean(axis=1)
    return float(taus[int(np.argmax(f1))])


def perbus_rates(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F1, detection rate and false-alarm rate of every bus over the record axis [KEC25]:

        F1 = 2 TP / (2 TP + FP + FN),   DR = TP / (TP + FN),   FR = FP / (FP + TN)

    with DR and FR taken as 0 where their denominator is 0 (a bus never attacked, or always).

    pred, truth : [n, N] bool
    returns     : (f1 [N], dr [N], fr [N])
    """
    tp, fp, fn = perbus_counts(pred, truth)
    tn = (~np.asarray(pred, bool) & ~np.asarray(truth, bool)).sum(axis=0).astype(np.float64)
    return perbus_f1_from_counts(tp, fp, fn), tp / np.maximum(tp + fn, 1.0), fp / np.maximum(fp + tn, 1.0)


def micro_prf(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Micro precision, recall and F1 over every (record, bus) call, each 0 when its denominator is
    0 (scikit-learn's precision_recall_fscore_support with average="micro", zero_division=0).

    pred, truth : [n, N] bool
    """
    g = LabelGrids(pred, truth)
    tp = float((g.pred & g.truth).sum())
    prec = tp / max(float(g.pred.sum()), 1e-12)
    rec = tp / max(float(g.truth.sum()), 1e-12)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-12)


def sample_f1(pred: np.ndarray, truth: np.ndarray) -> float:
    """Per-sample F1 averaged over the records, 0 for a record with nothing predicted nor true
    (scikit-learn's f1_score with average="samples", zero_division=0).

    pred, truth : [n, N] bool
    """
    g = LabelGrids(pred, truth)
    inter = (g.pred & g.truth).sum(axis=1).astype(np.float64)
    denom = np.maximum(g.pred.sum(axis=1) + g.truth.sum(axis=1), 1e-12)
    return float((2 * inter / denom).mean())


def strict_accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    """Strict localization accuracy: the share of records whose predicted set equals the true set
    exactly (scikit-learn's accuracy_score on multilabel rows).

    pred, truth : [n, N] bool
    """
    g = LabelGrids(pred, truth)
    return float((g.pred == g.truth).all(axis=1).mean())


def average_precision(score: np.ndarray, truth: np.ndarray) -> float:
    """Area under the precision-recall curve as a step sum over the distinct score thresholds,
    highest first [DG06]:  AP = sum_n (R_n - R_{n-1}) P_n  (scikit-learn's average_precision_score).

    score : [n] float
    truth : [n] bool, at least one positive
    """
    ranked = RankedLabels(score, truth)
    score, truth = ranked.score, ranked.truth
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], truth[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # the last record at each distinct threshold
    tps = np.cumsum(t)[last].astype(np.float64)
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
