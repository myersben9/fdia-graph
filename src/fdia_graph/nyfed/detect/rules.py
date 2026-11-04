"""When a detector alarms: the rule that turns a statistic per scan into a first alarm.

A rule is anything with `first_alarm(stat) -> int`, the first scan it alarms on or -1 (`AlarmRule`), so a different
rule (a CUSUM, a moving average) plugs in where `PersistenceRule` is used. The persistence rule alarms when `k` of the
last `n` statistics pass a threshold fitted on benign days, which a single noisy scan cannot trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = ["AlarmRule", "PersistenceRule"]


@runtime_checkable
class AlarmRule(Protocol):
    """Anything that names the first scan a statistic per scan alarms on (-1 for none)."""

    def first_alarm(self, stat: np.ndarray) -> int: ...


@dataclass(frozen=True)
class PersistenceRule:
    """Alarm when `k` of the last `n` statistics exceed `theta` (N79). A NaN statistic (a scan that could not be
    evaluated) counts as below the threshold."""

    theta: float
    k: int = 2
    n: int = 3

    def first_alarm(self, stat: np.ndarray) -> int:
        """The first scan at which `k` of the last `n` values of `stat [T]` exceed `theta`, or -1."""
        over = np.nan_to_num(np.asarray(stat, float), nan=0.0) > self.theta
        count = np.convolve(over.astype(int), np.ones(self.n, int))[: len(over)]
        hit = np.flatnonzero(count >= self.k)
        return int(hit[0]) if len(hit) else -1

    @classmethod
    def fit(cls, benign: np.ndarray, q: float = 99.9, k: int = 2, n: int = 3) -> PersistenceRule:
        """The rule whose threshold is the `q`th percentile of the benign statistics `benign` (any shape, NaN
        ignored): the calibration of N79 on whole benign January to March days."""
        return cls(float(np.nanpercentile(np.asarray(benign, float), q)), k, n)
