"""Numerical linear algebra the estimators lean on: a guarded inverse of a normal matrix, its
condition number without an eigen-decomposition, and per-record normal matrices in sub-batches.

These carry no equation from a source; they are the numerical guards the estimation kernel
needs, kept here so `formulas.estimation` reads as mathematics. Every function keeps the exact
expression the estimator used before it existed (docs/plans/READABILITY_PLAN.md, section 0).
"""

from __future__ import annotations

import numpy as np


def _scipy_linalg():
    try:
        import scipy.linalg

        return scipy.linalg
    except ImportError as e:
        raise ImportError("state estimation needs scipy: pip install 'fdia-graph[se]'") from e


def guarded_inverse(A: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive-definite normal matrix, pseudo-inverse when singular.

    A       : [k, k] symmetric (symmetrized here), a normal matrix HᵀWH
    returns : [k, k] its inverse, or the pseudo-inverse with rcond = 100 eps when A is not
              positive definite or its condition number exceeds 1 / (100 eps)

    Positive-definiteness is decided by a Cholesky factorization and near-singularity by LAPACK's
    condition estimate of the triangular factor (cond(A) ~ cond(L)²), both O(k²) after the
    factorization; an eigen-decomposition here cost 240 ms per call at IEEE-300 size.
    """
    lapack = _scipy_linalg().lapack

    A = np.asarray(A)
    eps = np.finfo(A.dtype if np.issubdtype(A.dtype, np.floating) else np.float64).eps
    S = 0.5 * (A + A.T)
    try:
        L = np.linalg.cholesky(S)
        rcond, _ = lapack.dtrcon(L, norm="1", uplo="L", diag="N")
        if rcond**2 > 100 * eps:  # cond(A) ~ cond(L)^2; the same 1e-14 relative floor as before
            return np.linalg.inv(S)
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(S, rcond=100 * eps)


def condition_number(A: np.ndarray, its: int = 40) -> float:
    """Spectral condition number of a symmetric positive-definite matrix, inf when it is not PD.

    A       : [k, k] symmetric (symmetrized here)
    its     : power-iteration steps for each extreme eigenvalue
    returns : λ_max / λ_min

    Cholesky for the PD test, then power iteration for the largest eigenvalue and inverse
    iteration through the Cholesky factor for the smallest; tens of milliseconds where a full
    eigen-decomposition takes hundreds, and equal to it to three decimals on the real normal
    matrices (checked on IEEE 14/118 with and without removed meters).
    """
    sl = _scipy_linalg()
    cho_factor, cho_solve = sl.cho_factor, sl.cho_solve

    S = 0.5 * (A + A.T)
    try:
        cf = cho_factor(S, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return float("inf")
    n = S.shape[0]
    v = np.ones(n) / np.sqrt(n)
    for _ in range(its):
        v = S @ v
        v /= np.linalg.norm(v)
    lmax = float(v @ (S @ v))
    u = np.ones(n) / np.sqrt(n)
    for _ in range(its):
        u = cho_solve(cf, u, check_finite=False)
        u /= np.linalg.norm(u)
    lmin = float(u @ (S @ u))
    return lmax / max(lmin, 1e-300)


def batched_normal_matrices(w: np.ndarray, B: np.ndarray, sub: int = 50) -> np.ndarray:
    """Per-record weighted normal matrices Bᵀ diag(w_i) B, built in sub-batches.

    w       : [n, m] per-record measurement weights
    B       : [m, k] the (chord) Jacobian, or the Jacobian times a basis
    sub     : records per sub-batch, so the [sub, k, m] intermediate stays small
    returns : [n, k, k]

    One einsum over the whole chunk materialized an 8 GB intermediate at IEEE-300 size and took
    260 s per 200 records; this takes 1.4 s.
    """
    if sub < 1:
        raise ValueError(f"sub must be a positive sub-batch size, got {sub}")
    n, k = w.shape[0], B.shape[1]
    out = np.empty((n, k, k), dtype=np.result_type(w, B))
    BT = B.T[None]  # [1, k, m]
    for a in range(0, n, sub):
        out[a : a + sub] = (BT * w[a : a + sub, None, :]) @ B
    return out
