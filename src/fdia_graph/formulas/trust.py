"""Trusted-meter selection against stealthy attacks [WU26]: the attack subspace a secured set
leaves open, the cost of the cheapest attack in it, and the greedy selection that closes it.

A stealthy false-data injection on a linearized state estimator is a = H c for some state change
c: it moves every measurement the way a real state change would, so the residual does not see it
[AE04, ch. 5]. Meters the attacker cannot write (secured, trusted) pin their rows: a_S = 0, so
H_S c = 0 and c lies in the null space of H_S. The attacker's freedom is the subspace
{H c : H_S c = 0}; its cost is the fewest meters an attack in that subspace has to touch. Securing
rows shrinks the null space, and once H_S has full column rank no stealthy attack exists.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np

_TOL = 1e-9


def rref(A: np.ndarray, tol: float = _TOL) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form of A and its pivot columns (Gauss-Jordan with partial pivoting).

    A       : [r, n]
    returns : ([r, n] the reduced rows, zero rows last; the pivot column of every nonzero row)
    """
    R = np.array(A, dtype=np.float64, copy=True)
    rows, cols = R.shape
    pivots: list[int] = []
    lead = 0
    for col in range(cols):
        if lead >= rows:
            break
        p = lead + int(np.argmax(np.abs(R[lead:, col])))
        if abs(R[p, col]) <= tol:
            continue
        R[[lead, p]] = R[[p, lead]]
        R[lead] /= R[lead, col]
        others = np.arange(rows) != lead
        R[others] -= np.outer(R[others, col], R[lead])
        pivots.append(col)
        lead += 1
    R[np.abs(R) <= tol] = 0.0
    return R, pivots


def attack_subspace(H: np.ndarray, secured: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """A basis of the attack vectors the secured meters leave open [WU26].

        {H c : H_S c = 0} = H N,   N a basis of null(H_S)

    H       : [m, n] measurement Jacobian
    secured : [k] row indices the attacker cannot write (may be empty)
    returns : [m, d] columns spanning the open attack subspace; d = 0 when no stealthy attack exists
    """
    H = np.asarray(H, np.float64)
    m, n = H.shape
    secured = np.asarray(secured, int)
    if len(secured) == 0:
        return H @ np.eye(n)
    _, s, vt = np.linalg.svd(H[secured], full_matrices=True)
    rank = int((s > tol * max(1.0, s[0] if len(s) else 1.0)).sum())
    N = vt[rank:].T  # [n, n - rank] null space of H_S
    return H @ N


def sparse_basis(H: np.ndarray, secured: np.ndarray, tol: float = _TOL) -> np.ndarray:
    """The reduced row echelon basis of the open attack subspace: [d, m] rows, each an attack, in
    the sparsest form row reduction gives; empty when the secured rows leave no stealthy attack."""
    B = attack_subspace(H, secured, tol)
    if B.shape[1] == 0:
        return np.zeros((0, H.shape[0]))
    R, pivots = rref(B.T, tol)
    return R[: len(pivots)]


def attack_cost(H: np.ndarray, secured: np.ndarray, tol: float = _TOL) -> tuple[float, Optional[np.ndarray]]:
    """The cost of the cheapest stealthy attack the secured meters leave open, and that attack.

    The cost is the number of meters the attack touches, min ‖a‖₀ over the open subspace. The exact
    minimum is the sparsest vector of a subspace (NP-hard); what is returned is the sparsest row of
    the reduced row echelon basis of that subspace, the bound the row-reduction heuristic of [WU26]
    works with, which is exact whenever the sparsest attack is one of the basis rows.

    H       : [m, n] measurement Jacobian
    secured : row indices the attacker cannot write
    returns : (cost, a [m]) or (inf, None) when the secured rows leave no stealthy attack
    """
    R = sparse_basis(H, secured, tol)
    if not len(R):
        return float("inf"), None
    nnz = (np.abs(R) > tol).sum(axis=1)
    i = int(np.argmin(nnz))
    return float(nnz[i]), R[i]


def greedy_trusted_meters(H: np.ndarray, k: int, tol: float = _TOL) -> tuple[list[int], list[float]]:
    """Secure meters one at a time, each the meter that raises the attack cost most (the
    row-reduction selection of [WU26]): at every step take the cheapest open attack and secure the meter
    on its support whose protection leaves the attacker the costliest cheapest attack; among
    meters that tie, the one the most attacks of the open basis pass through.

    H       : [m, n] measurement Jacobian
    k       : how many meters to secure at most (stops early once no stealthy attack is left)
    returns : (the meters in the order they were secured, the attack cost after each)
    """
    order: list[int] = []
    costs: list[float] = []
    for _ in range(k):
        R = sparse_basis(H, np.array(order, int), tol)
        if not len(R):
            break
        nnz = (np.abs(R) > tol).sum(axis=1)
        a = R[int(np.argmin(nnz))]
        through = (np.abs(R) > tol).sum(axis=0)  # how many basis attacks pass through each meter
        support = [int(i) for i in np.flatnonzero(np.abs(a) > tol) if i not in order]
        best, best_key = support[0], (-1.0, -1)
        for i in support:
            key = (attack_cost(H, np.array(order + [i], int), tol)[0], int(through[i]))
            if key > best_key:
                best, best_key = i, key
        order.append(best)
        costs.append(best_key[0])
    return order, costs


def sparsest_rows(
    M: np.ndarray,
    tol: float = _TOL,
    rounds: int = 50,
    eligible: Optional[Callable[[np.ndarray], bool]] = None,
) -> np.ndarray:
    """The rows of M's row space in the sparsest reduced echelon form row reduction with column
    exchanges reaches ([WU26] Sec. IV-D1 steps 2-5, after [YAN17]): reduce M to RREF, take the row with the
    fewest nonzeros, move its nonzero columns to the end (the column order is the tracking matrix),
    reduce again, and repeat until the fewest nonzeros stops falling; the rows come back in M's own
    column order (the inverse of the tracking matrix), sparsest first. With M the transpose of an
    attack-area Jacobian, each row is an attack vector h c, the first the sparsest found. The first
    exchange moves the sparsest row of M itself to the end (step 1's transpose, whose rows are the
    single-variable attacks; a choice of ours, the paper starts from the first reduction). `eligible`
    restricts the row the exchanges chase to rows it accepts (an attack that reaches a goal); the
    eligible rows then come first, sparsest first.

    M        : [n, m] (rows: a basis of the space; columns: the measurements)
    eligible : which rows count, or None for every row
    returns  : [r, m] the rows of the last improving reduction, r the rank of M
    """
    M = np.asarray(M, np.float64)
    ok = eligible if eligible is not None else (lambda row: True)
    # step 1 is the transpose itself, whose rows are attacks too (one state variable each): the chase
    # starts with the sparsest of them that `ok` accepts moved to the end, the first exchange
    perm = np.arange(M.shape[1])
    seed = _chased_row(M, tol, ok)
    if seed is not None:
        moved = np.abs(M[seed[1]]) > tol
        perm = np.concatenate([perm[~moved], perm[moved]])
    fewest, rows = np.inf, np.zeros((0, M.shape[1]))
    for _ in range(rounds):
        Rp, pivots = rref(M[:, perm], tol)
        Rp = Rp[: len(pivots)]
        R = np.zeros_like(Rp)
        R[:, perm] = Rp  # back to M's column order
        found = _chased_row(R, tol, ok)
        if found is None or found[0] >= fewest:
            break
        fewest, rows = found[0], R
        moved = np.abs(Rp[found[1]]) > tol
        perm = np.concatenate([perm[~moved], perm[moved]])  # the chased row's columns to the end
    nnz = (np.abs(rows) > tol).sum(axis=1)
    first = np.array([not ok(r) for r in rows], dtype=bool)
    return rows[np.lexsort((nnz, first))]


def _chased_row(R: np.ndarray, tol: float, ok: Callable[[np.ndarray], bool]) -> Optional[tuple[int, int]]:
    """(The nonzero count, the index) of the sparsest row of R that `ok` accepts, or None."""
    nnz = (np.abs(R) > tol).sum(axis=1)
    for i in np.argsort(nnz, kind="stable"):
        if ok(R[i]):
            return int(nnz[i]), int(i)
    return None


__all__ = ["rref", "attack_subspace", "sparse_basis", "attack_cost", "greedy_trusted_meters", "sparsest_rows"]
