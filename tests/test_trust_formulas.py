"""The row reduction the attack search and the trusted-meter kernel share (the kernel itself, the open
attack subspace, its cost and the greedy selection, is checked in analysis/tests/test_trust_formulas.py)."""

import numpy as np

from fdia_graph.formulas.trust import rref


def test_rref_reduces_and_reports_pivots():
    A = np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    R, pivots = rref(A)
    assert pivots == [0, 2]
    assert np.allclose(R, [[1, 2, 0], [0, 0, 1], [0, 0, 0]])
    assert np.linalg.matrix_rank(A) == len(pivots)
