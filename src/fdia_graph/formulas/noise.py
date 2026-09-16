"""The meter error model: accuracy-class standard deviations split into a constant per-meter
bias and a per-scan jitter [ASP14], our split."""

from __future__ import annotations

from typing import Dict, Tuple


def bias_jitter_split(
    sd: Dict[str, float], jitter_frac: float = 0.25
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Split each accuracy-class std into a per-scan jitter and a per-meter bias [ASP14], our split.

        jitter = jitter_frac * SD,   bias = sqrt(1 - jitter_frac^2) * SD,   so bias^2 + jitter^2 = SD^2

    The accuracy-class error is mostly systematic (a constant calibration offset drawn once per
    meter) with a small per-scan random part; treating all of SD as per-scan noise would
    over-jitter one-minute traces and drown the temporal channel. The root-sum-square keeps the
    class total.

    sd      : std per channel kind, e.g. {"pf": 0.017, "v": 0.0012, ...}
    returns : (jitter std per kind, bias std per kind)
    """
    jitter = {k: v * jitter_frac for k, v in sd.items()}
    bias = {k: v * (1.0 - jitter_frac * jitter_frac) ** 0.5 for k, v in sd.items()}
    return jitter, bias
