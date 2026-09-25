"""Weighted least squares state estimation and its robust variants, one function per equation
[SCH70], [HAN75], [HUB64], [EST26].

The estimator classes in `fdia_graph.se` own the measurement function h(x), the chord Jacobian,
the iteration loops and their divergence guard; every algebraic step inside those loops is one
of the functions here. Each keeps the exact floating-point expression the estimator used before
it existed, so cached results reproduce bit for bit.

Shapes: m measurements, k state (or basis) coordinates, n records. `w` is a weight vector
(1/σ²) per measurement, `H` the m×k Jacobian, `Ai` the inverse normal matrix (HᵀWH)⁻¹.
"""

from __future__ import annotations

import numpy as np


def normal_matrix(H: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The gain (normal) matrix of weighted least squares, G = Hᵀ W H [SCH70, part II].

    H       : [m, k] Jacobian (or Jacobian times a basis)
    w       : [m] measurement weights 1/σ²
    returns : [k, k]
    """
    return H.T @ (w[:, None] * H)


def wls_step(residual: np.ndarray, w: np.ndarray, B: np.ndarray, Ai: np.ndarray) -> np.ndarray:
    """One Gauss-Newton step of weighted least squares with a shared Jacobian [SCH70, part II].

        Δc = (Bᵀ W B)⁻¹ Bᵀ W (z − h(x))

    residual : [n, m] z − h(x) per record
    w        : [m] shared measurement weights
    B        : [m, k] the (chord) Jacobian, or Jacobian times a basis
    Ai       : [k, k] inverse normal matrix (Bᵀ W B)⁻¹
    returns  : [n, k] the step for every record
    """
    return (residual * w) @ B @ Ai.T


def wls_step_batched(residual: np.ndarray, w: np.ndarray, B: np.ndarray, Ai: np.ndarray) -> np.ndarray:
    """The same step with per-record weights, so each record has its own normal matrix.

    residual : [n, m] z − h(x) per record
    w        : [n, m] per-record measurement weights
    B        : [m, k]
    Ai       : [n, k, k] per-record inverse normal matrices
    returns  : [n, k]
    """
    return np.einsum("bij,bj->bi", Ai, (residual * w) @ B)


def weighted_objective(residual: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The weighted least-squares objective J(x) = Σ_i w_i (z_i − h_i(x))² per record [SCH70].

    residual : [n, m]
    w        : [n, m] or [m]
    returns  : [n]
    """
    return (w * residual**2).sum(axis=1)


def residual_covariance_diag(H: np.ndarray, w: np.ndarray, Ai: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal of the residual covariance Ω = R − H G⁻¹ Hᵀ [HAN75], with R = diag(1/w).

    H       : [m, k] Jacobian
    w       : [m] measurement weights 1/σ²
    Ai      : [k, k] inverse normal matrix G⁻¹
    returns : (Ω_ii [m], R_ii [m])
    """
    R = 1.0 / w
    om = R - np.einsum("ij,jk,ik->i", H, Ai, H)
    return om, R


def critical_measurements(om: np.ndarray, R: np.ndarray, rel: float = 1e-6) -> np.ndarray:
    """A measurement is critical when its residual variance is structurally zero, Ω_ii < rel R_ii
    [HAN75]; its residual carries no bad-data information and it must never be removed.

    returns : [m] bool
    """
    return om < rel * R


def floored_covariance(om: np.ndarray, R: np.ndarray, rel: float = 1e-12) -> np.ndarray:
    """Ω_ii floored at rel R_ii so a critical measurement's normalized residual stays finite."""
    return np.maximum(om, rel * R)


def normalized_residual(residual: np.ndarray, om: np.ndarray) -> np.ndarray:
    """The normalized residual r_N,i = |z_i − h_i(x̂)| / √Ω_ii [HAN75].

    residual : [n, m] z − h(x̂)
    om       : [m] residual covariance diagonal (floored)
    returns  : [n, m] non-negative
    """
    return np.abs(residual) / np.sqrt(om)[None, :]


def huber_weights(r_n: np.ndarray, c: float, eps: float = 1e-9) -> np.ndarray:
    """The Huber weight a_i = min(1, c / |r_N,i|) [HUB64]: full weight inside the band, decaying
    outside it, so a gross error is bounded in influence.

    r_n     : [n, m] normalized residuals (non-negative)
    c       : the band half-width in normalized units
    eps     : floor on |r_N,i| against division by zero
    returns : [n, m] in (0, 1]
    """
    return np.minimum(1.0, c / np.maximum(r_n, eps))


def whitened_svd_basis(X: np.ndarray, rank_frac: float) -> tuple[int, np.ndarray]:
    """The learned operating-point prior [EST26]: whiten the benign states per coordinate, take
    the SVD, keep the leading rank_frac fraction of directions, un-whiten and re-orthonormalize.

    X         : [n, d] benign training states
    rank_frac : fraction of d to keep, in (0, 1]
    returns   : (K, V_K [d, K]) with V_Kᵀ V_K = I; the estimate is restricted to x = mean + V_K c

    The whitened basis is otherwise catastrophically ill conditioned near full rank, which is
    why the QR re-orthonormalization is part of the formula.
    """
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-9)  # angle and voltage differ ~10x in scale
    _, _, Vt = np.linalg.svd((X - mean) / std, full_matrices=False)
    K = max(1, int(round(rank_frac * X.shape[1])))
    VK = np.linalg.qr(Vt[:K].T * std[:, None])[0]  # un-whiten, re-orthonormalize
    return K, VK


def gate_weights(w: np.ndarray, flags: np.ndarray, incidence: list[np.ndarray], factor: float) -> np.ndarray:
    """Localization-gated weights [EST26]: every meter incident to a flagged bus is down-weighted
    by `factor` before the solve, so the prior supplies the state there.

    w         : [m] shared measurement weights
    flags     : [n, N] bool, the flagged buses per record
    incidence : per bus, the masked-measurement indices touching it (see projection.bus_incidence)
    factor    : in (0, 1]
    returns   : [n, m] per-record weights
    """
    out = np.repeat(w[None, :], flags.shape[0], axis=0)
    for b, ix in enumerate(incidence):
        if len(ix):
            hit = flags[:, b]
            out[np.ix_(hit, ix)] *= factor
    return out


def accuracy_class_sigma(
    mean_abs: np.ndarray, cls: np.ndarray, relative: np.ndarray, floor: float
) -> np.ndarray:
    """Each meter's error std from its accuracy class [ASP14]: a relative class scales the meter's
    mean absolute reading, an absolute class is the std itself,

        sigma_i = c_i |z_i| + floor   (relative: power injections and flows)
        sigma_i = c_i                 (absolute: voltage magnitude and angle)

    Equipment data and measurements only; no state enters.

    mean_abs : [m] mean |z_i| over the calibration scans
    cls      : [m] accuracy class per meter
    relative : [m] bool, True where the class is relative
    floor    : absolute floor of a relative meter's std (its reading's units)
    returns  : [m]
    """
    mean_abs, cls = np.asarray(mean_abs, np.float64), np.asarray(cls, np.float64)
    return np.where(np.asarray(relative, bool), cls * mean_abs + floor, cls)
