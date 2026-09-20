"""The mathematics of the package as named, pure functions: numpy in, numpy out, no files, no
dataset objects, no torch. Each function carries its equation and a source key from
docs/reference/REFERENCES.md; docs/reference/FORMULAS.md is the catalogue.

Provisional until 1.0: names may still be adjusted once, with a deprecation alias.
"""

from .attacks import ramp_profile
from .estimation import (
    critical_measurements,
    floored_covariance,
    gate_weights,
    huber_weights,
    normal_matrix,
    normalized_residual,
    residual_covariance_diag,
    weighted_objective,
    whitened_svd_basis,
    wls_step,
    wls_step_batched,
)
from .linalg import batched_normal_matrices, condition_number, guarded_inverse
from .network import (
    Admittances,
    BranchModel,
    ac_jacobian,
    ac_measurement,
    branch_admittances,
    branch_flows,
    bus_injections,
    complex_voltages,
    local_ac_solve,
    series_admittance,
    subnetwork,
)
from .noise import bias_jitter_split
from .projection import (
    bus_incidence,
    direction_coefficients,
    explained_unexplained,
    leverage,
    meters_to_buses,
    weak_directions,
    weak_move,
    weighted_pseudoinverse,
)
from .temporal import recent_change_scale, swing_zscore, temporal_delta
from .trust import attack_cost, attack_subspace, greedy_trusted_meters, rref, sparse_basis

__all__ = [
    "Admittances",
    "BranchModel",
    "ac_jacobian",
    "ac_measurement",
    "attack_cost",
    "attack_subspace",
    "batched_normal_matrices",
    "bias_jitter_split",
    "branch_admittances",
    "branch_flows",
    "bus_incidence",
    "bus_injections",
    "complex_voltages",
    "condition_number",
    "critical_measurements",
    "direction_coefficients",
    "explained_unexplained",
    "floored_covariance",
    "gate_weights",
    "greedy_trusted_meters",
    "guarded_inverse",
    "huber_weights",
    "leverage",
    "local_ac_solve",
    "meters_to_buses",
    "normal_matrix",
    "normalized_residual",
    "ramp_profile",
    "recent_change_scale",
    "residual_covariance_diag",
    "rref",
    "series_admittance",
    "sparse_basis",
    "subnetwork",
    "swing_zscore",
    "temporal_delta",
    "weak_directions",
    "weak_move",
    "weighted_objective",
    "weighted_pseudoinverse",
    "whitened_svd_basis",
    "wls_step",
    "wls_step_batched",
]
