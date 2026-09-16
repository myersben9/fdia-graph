"""The mathematics of the package as named, pure functions: numpy in, numpy out, no files, no
dataset objects, no torch. Each function carries its equation and a source key from
docs/reference/REFERENCES.md; docs/reference/FORMULAS.md is the catalogue.

Provisional until 1.0: names may still be adjusted once, with a deprecation alias.
"""

from .attacks import ramp_profile
from .network import (
    Admittances,
    BranchModel,
    branch_admittances,
    branch_flows,
    bus_injections,
    complex_voltages,
    series_admittance,
)
from .noise import bias_jitter_split
from .temporal import recent_change_scale, swing_zscore, temporal_delta

__all__ = [
    "Admittances",
    "BranchModel",
    "bias_jitter_split",
    "branch_admittances",
    "branch_flows",
    "bus_injections",
    "complex_voltages",
    "ramp_profile",
    "recent_change_scale",
    "series_admittance",
    "swing_zscore",
    "temporal_delta",
]
