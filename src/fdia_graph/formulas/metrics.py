"""Per-bus localization metrics from confusion counts, so they can be summed across clients or
thresholds before any ratio is taken [KEC25]."""

from __future__ import annotations

import numpy as np


def perbus_counts(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True positives, false positives and false negatives per bus over the record axis [KEC25].

    pred, truth : [n, N] bool
    returns     : (tp [N], fp [N], fn [N]) as float64
    """
    pred, truth = np.asarray(pred, bool), np.asarray(truth, bool)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError(f"need two [n, N] boolean arrays of one shape, got {pred.shape} and {truth.shape}")
    tp = (pred & truth).sum(axis=0).astype(np.float64)
    fp = (pred & ~truth).sum(axis=0).astype(np.float64)
    fn = (~pred & truth).sum(axis=0).astype(np.float64)
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
    taus, active = np.asarray(taus), np.asarray(active, bool)
    if taus.ndim != 1 or active.ndim != 1:
        raise ValueError(
            f"taus and active must be one-dimensional, got shapes {taus.shape} and {active.shape}"
        )
    shape = (len(taus), len(active))
    if not len(taus) or not active.any() or any(np.shape(c) != shape for c in (tp, fp, fn)):
        raise ValueError(f"need [n_taus, N] counts of shape {shape}, a non-empty tau grid and an active bus")
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


def average_precision(score: np.ndarray, truth: np.ndarray) -> float:
    """Area under the precision-recall curve as a step sum over the distinct score thresholds,
    highest first [DG06]:  AP = sum_n (R_n - R_{n-1}) P_n  (scikit-learn's average_precision_score).

    score : [n] float
    truth : [n] bool, at least one positive
    """
    score, truth = np.asarray(score, np.float64), np.asarray(truth, bool)
    if score.ndim != 1 or score.shape != truth.shape or not truth.any():
        raise ValueError("average_precision needs matching 1-D scores and labels with a positive")
    order = np.argsort(-score, kind="mergesort")
    s, t = score[order], truth[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # the last record at each distinct threshold
    tps = np.cumsum(t)[last].astype(np.float64)
    precision = tps / (last + 1)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
