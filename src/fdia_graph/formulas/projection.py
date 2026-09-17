"""Projections through the measurement Jacobian: the explained and unexplained parts of a
measurement change, leverage, the weak directions, and the meter-to-bus aggregation of the
Jacobian-informed features [JAC26], [HAN75].

Shapes: m measurements, k state coordinates, n records. `Hw = W^{1/2} H` is the whitened
Jacobian and `Ai = (HᵀWH)⁻¹` the inverse normal matrix, both from a fitted estimator. Every
function keeps the exact expression `se.jacobian.JacobianFeatures` used before it existed.
"""

from __future__ import annotations

import numpy as np


def bus_incidence(n_bus: int, n_branch: int, edge_index: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Masked-measurement indices touching each bus: its own V, P, Q, θ channels plus the flows of
    every incident branch (a flow meter reacts to both endpoints) [JAC26].

    n_bus, n_branch : N, E
    edge_index      : [2, E] from and to bus of every branch
    mask            : [4N + 2E] bool, which measurement slots exist, in the order V, P, Q, θ, Pf, Qf
    returns         : per bus, the indices into the masked measurement vector
    """
    N, E = n_bus, n_branch
    inc = np.zeros((N, 4 * N + 2 * E), bool)
    for c in range(4):
        inc[np.arange(N), c * N + np.arange(N)] = True
    for c in range(2):
        cols = 4 * N + c * E + np.arange(E)
        inc[edge_index[0], cols] = True
        inc[edge_index[1], cols] = True
    incm = inc[:, mask]
    return [np.where(incm[b])[0] for b in range(N)]


def weighted_pseudoinverse(H: np.ndarray, w: np.ndarray, Ai: np.ndarray) -> np.ndarray:
    """H_W⁺ = (HᵀWH)⁻¹ HᵀW, the map from a measurement change to the implied state change (the
    WLS step as a matrix) [SCH70, part II].

    H       : [m, k]
    w       : [m]
    Ai      : [k, k]
    returns : [k, m]
    """
    return Ai @ (H.T * w[None, :])


def leverage(Hw: np.ndarray, Ai: np.ndarray) -> np.ndarray:
    """Diagonal of the hat matrix P = Hw (HᵀWH)⁻¹ Hwᵀ, the leverage of every measurement [HAN75].

    Hw      : [m, k] whitened Jacobian W^{1/2} H
    Ai      : [k, k]
    returns : [m] clipped to [0, 1]
    """
    P = Hw @ Ai @ Hw.T
    return np.clip(np.diag(P), 0.0, 1.0)


def weak_directions(Hw: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SVD of the whitened Jacobian and its k weakest right-singular directions [JAC26].

    Hw      : [m, d]
    k       : number of weak directions
    returns : (U [m, d], S [d], V_weak [d, k], weak_rows [k]) with weak_rows the indices of the
              k smallest singular values, in increasing order of index
    """
    U, S, Vt = np.linalg.svd(Hw, full_matrices=False)
    return U, S, Vt[-k:].T, np.arange(len(S) - k, len(S))


def explained_unexplained(
    dz: np.ndarray, H: np.ndarray, pinv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a measurement change into what a state change explains and what it cannot [JAC26].

        dx̂ = H_W⁺ Δz,   r∥ = H dx̂,   r⊥ = Δz − r∥

    dz      : [n, m] measurement change
    H       : [m, k]
    pinv    : [k, m] the weighted pseudo-inverse
    returns : (dx̂ [n, k], r∥ [n, m], r⊥ [n, m])
    """
    dx = dz @ pinv.T
    r_par = dx @ H.T
    return dx, r_par, dz - r_par


def direction_coefficients(dz_w: np.ndarray, U: np.ndarray) -> np.ndarray:
    """α = Uᵀ W^{1/2} Δz, the change expressed in the left-singular directions of Hw [JAC26].

    dz_w    : [n, m] whitened measurement change W^{1/2} Δz
    U       : [m, d]
    returns : [n, d]
    """
    return dz_w @ U


def weak_move(dx: np.ndarray, V_weak: np.ndarray) -> np.ndarray:
    """The implied state change restricted to the weak subspace, V_weak V_weakᵀ dx̂ [JAC26].

    dx      : [n, d]
    V_weak  : [d, k]
    returns : [n, d]
    """
    return (dx @ V_weak) @ V_weak.T


def meters_to_buses(values: np.ndarray, incidence: list[np.ndarray], reduce: str) -> np.ndarray:
    """Aggregate a per-meter quantity to each bus over the meters incident to it [JAC26].

    values    : [n, m]
    incidence : per bus, the meter indices (see bus_incidence)
    reduce    : "sum" (energies) or "max" (changes)
    returns   : [n, N], zero for a bus with no incident meter
    """
    if reduce not in ("sum", "max"):
        raise ValueError(f"reduce must be 'sum' or 'max', got {reduce!r}")
    out = np.zeros((values.shape[0], len(incidence)))
    for b, ix in enumerate(incidence):
        if len(ix):
            out[:, b] = values[:, ix].sum(axis=1) if reduce == "sum" else values[:, ix].max(axis=1)
    return out
