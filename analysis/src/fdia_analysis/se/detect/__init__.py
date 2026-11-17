"""Detection: when a detector alarms, as a building block.

A detector gives one statistic per scan (a residual sum, a gap between two meters, any score). `PersistenceRule`
turns that series into a first alarm: k of the last n statistics past a threshold fitted on benign days
(`PersistenceRule.fit`). Any object with `first_alarm(stat)` is an `AlarmRule` and plugs in instead. A scan that
cannot be evaluated is NaN, which the persistence rule counts as below its threshold.
"""

from __future__ import annotations

from .rules import AlarmRule, PersistenceRule

__all__ = [
    "AlarmRule",
    "PersistenceRule",
]
