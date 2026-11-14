"""When a detector alarms: the rule that turns a statistic per scan into a first alarm.

A rule is anything with `first_alarm(stat) -> int`, the first scan it alarms on or -1 (`AlarmRule`), so a different
rule (a CUSUM, a moving average) plugs in where `PersistenceRule` is used. The persistence rule alarms when `k` of the
last `n` statistics pass a threshold fitted on benign days, which a single noisy scan cannot trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from fdia_graph.errors import NoBenignRecords

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
        """The first scan at which `k` of the last `n` values of `stat [T]` exceed `theta`, or -1 (also for no
        scans). A NaN never exceeds, whatever the sign of `theta`."""
        stat = np.asarray(stat, float)
        if stat.size == 0:
            return -1
        over = stat > self.theta  # a comparison with NaN is False
        count = np.convolve(over.astype(int), np.ones(self.n, int))[: len(over)]
        hit = np.flatnonzero(count >= self.k)
        return int(hit[0]) if len(hit) else -1

    @classmethod
    def fit(cls, benign: np.ndarray, q: float = 99.9, k: int = 2, n: int = 3) -> PersistenceRule:
        """The rule whose threshold is the `q`th percentile of the benign statistics `benign` (any shape, NaN
        ignored): the calibration of N79 on whole benign January to March days.

        Raises:
            NoBenignRecords: `benign` holds no evaluated statistic, so there is nothing to calibrate on (a rule
                fitted on nothing would never alarm).
        """
        benign = np.asarray(benign, float)
        if not np.isfinite(benign).any():
            raise NoBenignRecords("PersistenceRule.fit needs at least one evaluated benign statistic")
        return cls(float(np.nanpercentile(benign, q)), k, n)
