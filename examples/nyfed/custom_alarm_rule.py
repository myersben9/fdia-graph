#!/usr/bin/env python
"""A detector rule of your own: a CUSUM on the residual statistic, next to the package's persistence rule.

Any object with `first_alarm(stat) -> int` is an `AlarmRule`, so it plugs in wherever the package's
`PersistenceRule` is used (the detection tools under tools/nyfed/ take one). This example runs on a synthetic
residual series, so it needs no data.

Run: python examples/nyfed/custom_alarm_rule.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdia_graph.se.detect.rules import AlarmRule, PersistenceRule


@dataclass(frozen=True)
class Cusum:
    """Alarm when the running sum of the residual's excess over `target` passes `h` (a one-sided CUSUM)."""

    target: float
    h: float

    def first_alarm(self, stat: np.ndarray) -> int:
        s = 0.0
        for t, x in enumerate(np.asarray(stat, float)):
            if np.isnan(x):  # a scan that could not be evaluated adds nothing
                continue
            s = max(0.0, s + x - self.target)
            if s > self.h:
                return t
        return -1


def main() -> None:
    rng = np.random.default_rng(0)
    benign = 400 + 8 * rng.standard_normal(
        2000
    )  # the residual sum of benign scans, about its degrees of freedom
    attacked = 400 + 8 * rng.standard_normal(30) + np.linspace(0, 40, 30)  # an attack that drifts it upward

    persistence = PersistenceRule.fit(benign, q=99.9)  # the package's rule, calibrated on benign scans
    cusum = Cusum(target=float(np.median(benign)) + 8, h=60)
    for rule in (persistence, cusum):
        assert isinstance(rule, AlarmRule)  # both satisfy the protocol
        print(
            f"{type(rule).__name__:16s} benign alarm at {rule.first_alarm(benign[:300]):4d}, "
            f"attack alarm at scan {rule.first_alarm(attacked)}"
        )


if __name__ == "__main__":
    main()
