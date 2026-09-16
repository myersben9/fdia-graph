"""Attack shapes defined by the dataset [DAT26] that are pure functions of their parameters."""

from __future__ import annotations


def ramp_profile(i: int, rise: int, hold: int, rate_up: float, rate_down: float) -> float:
    """Deviation of the slow ramp At at step i of a sequence [DAT26]: rise at rate_up for `rise`
    steps to the peak, hold there for `hold` steps, then return at rate_down and never below zero.
    The direction (surge or dip) is applied by the caller as 1 +/- dev.

        dev(i) = rate_up * i                                  for i < rise
               = rate_up * rise                               for rise <= i < rise + hold
               = max(0, rate_up * rise - rate_down * (i - rise - hold))   after
    """
    peak = rate_up * rise
    if i < rise:
        return rate_up * i
    if i < rise + hold:
        return peak
    return max(0.0, peak - rate_down * (i - rise - hold))
