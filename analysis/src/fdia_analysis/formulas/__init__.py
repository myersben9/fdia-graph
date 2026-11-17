"""The mathematics of the analysis toolkit as named, pure functions: numpy in, numpy out, no files,
no dataset objects, no torch. Each function carries its equation and a source key from
docs/reference/REFERENCES.md; docs/reference/FORMULAS.md is the catalogue. The network model, the
meter noise and the attack shapes are the generation toolkit's (`fdia_graph.formulas`).

Provisional until 1.0: a name may still be adjusted, as a changelog entry with no alias kept.
"""

from .estimation import (
    critical_measurements,
    floored_covariance,
    gate_weights,
    huber_weights,
    normal_matrix,
    normalized_residual,
    pmu_pseudo_links,
    pmu_pseudo_phasors,
    pmu_pseudo_voltages,
    residual_covariance_diag,
    weighted_objective,
    whitened_svd_basis,
    wls_step,
    wls_step_batched,
)
from .federated import (
    attackable_affinity,
    block_diagonal_basis,
    channel_moments,
    cut_edge_count,
    fedavg,
    halo_nodes,
    hop_distance,
    interior_boundary,
    pool_moments,
)
from .linalg import batched_normal_matrices, condition_number, guarded_inverse
from .measurement import ac_jacobian, ac_measurement
from .metrics import (
    average_precision,
    micro_prf,
    perbus_counts,
    perbus_f1_from_counts,
    perbus_rates,
    sample_f1,
    strict_accuracy,
    tau_from_counts,
)
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
from .trust import attack_cost, attack_subspace, greedy_trusted_meters, sparse_basis

__all__ = [
    "ac_jacobian",
    "ac_measurement",
    "attack_cost",
    "attack_subspace",
    "attackable_affinity",
    "average_precision",
    "batched_normal_matrices",
    "block_diagonal_basis",
    "bus_incidence",
    "channel_moments",
    "condition_number",
    "critical_measurements",
    "cut_edge_count",
    "direction_coefficients",
    "explained_unexplained",
    "fedavg",
    "floored_covariance",
    "gate_weights",
    "greedy_trusted_meters",
    "guarded_inverse",
    "halo_nodes",
    "hop_distance",
    "huber_weights",
    "interior_boundary",
    "leverage",
    "meters_to_buses",
    "normal_matrix",
    "normalized_residual",
    "pmu_pseudo_links",
    "pmu_pseudo_phasors",
    "pmu_pseudo_voltages",
    "micro_prf",
    "perbus_counts",
    "perbus_f1_from_counts",
    "perbus_rates",
    "pool_moments",
    "residual_covariance_diag",
    "sample_f1",
    "sparse_basis",
    "strict_accuracy",
    "tau_from_counts",
    "weak_directions",
    "weak_move",
    "weighted_objective",
    "weighted_pseudoinverse",
    "whitened_svd_basis",
    "wls_step",
    "wls_step_batched",
]
