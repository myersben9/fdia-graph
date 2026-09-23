"""Per-bus localization metrics from confusion counts, so they can be summed across clients or
thresholds before any ratio is taken [KEC25]."""

from __future__ import annotations

import numpy as np


def perbus_counts(pred: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True positives, false positives and false negatives per bus over the record axis [KEC25].

    pred, truth : [n, N] bool
    returns     : (tp [N], fp [N], fn [N]) as float64
    """
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
    f1 = perbus_f1_from_counts(tp, fp, fn)[:, active].mean(axis=1)
    return float(taus[int(np.argmax(f1))])
