"""Row reduction for stealthy attacks under trusted meters [WU26]: the reduced row echelon form of
a matrix, and the sparsest rows of a row space, which the attack generator searches for an attack
vector the trusted meters leave open.

A stealthy false-data injection on a linearized state estimator is a = H c for some state change
c: it moves every measurement the way a real state change would, so the residual does not see it
[AE04, ch. 5]. Meters the attacker cannot write (secured, trusted) pin their rows: a_S = 0, so
H_S c = 0 and c lies in the null space of H_S. The selection of the meters to trust, and the cost
of the attacks a selection leaves open, are the analysis toolkit's (`fdia_analysis.formulas.trust`),
built on `rref` here and on `constants.numerics.ZERO_TOL`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional

import numpy as np

from fdia_graph.constants.numerics import ZERO_TOL


def rref(A: np.ndarray, tol: float = ZERO_TOL) -> tuple[np.ndarray, list[int]]:
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


def sparsest_rows(
    M: np.ndarray,
    tol: float = ZERO_TOL,
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
