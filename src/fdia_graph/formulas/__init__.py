"""The mathematics of the generation toolkit as named, pure functions: numpy in, numpy out, no files,
no dataset objects, no torch. Each function carries its equation and a source key from
docs/reference/REFERENCES.md; docs/reference/FORMULAS.md is the catalogue. The formulas of
estimation, localization, federated training and trust selection are the analysis toolkit's
(`fdia_analysis.formulas`).

Provisional until 1.0: a name may still be adjusted, as a changelog entry with no alias kept.
"""

from .attacks import ramp_profile, tampered_channels, tampered_devices
from .network import (
    branch_admittances,
    branch_currents,
    branch_flows,
    bus_injections,
    complex_voltages,
    edge_adjacency,
    flow_sensitivities,
    hop_distances,
    local_ac_solve,
    series_admittance,
    subnetwork,
)
from .noise import (
    accuracy_sigma,
    bias_jitter_split,
    biased_current,
    current_magnitude,
    current_sigma,
    jitter_sigma,
    paper_current_sigma,
)
from .temporal import recent_change_scale, swing_zscore, temporal_delta, wrap_degrees
from .trust import rref

__all__ = [
    "flow_sensitivities",
    "accuracy_sigma",
    "bias_jitter_split",
    "branch_admittances",
    "branch_currents",
    "branch_flows",
    "bus_injections",
    "complex_voltages",
    "edge_adjacency",
    "hop_distances",
    "biased_current",
    "current_magnitude",
    "current_sigma",
    "jitter_sigma",
    "paper_current_sigma",
    "local_ac_solve",
    "ramp_profile",
    "recent_change_scale",
    "rref",
    "series_admittance",
    "subnetwork",
    "swing_zscore",
    "tampered_channels",
    "tampered_devices",
    "temporal_delta",
    "wrap_degrees",
]
