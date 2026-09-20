"""The trusted-meter kernel: row reduction, the open attack subspace, its cost, the greedy selection."""

import numpy as np
import pytest

from fdia_graph.formulas.trust import attack_cost, attack_subspace, greedy_trusted_meters, rref


def test_rref_reduces_and_reports_pivots():
    A = np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 4.0], [0.0, 0.0, 0.0]])
    R, pivots = rref(A)
    assert pivots == [0, 2]
    assert np.allclose(R, [[1, 2, 0], [0, 0, 1], [0, 0, 0]])
    assert np.linalg.matrix_rank(A) == len(pivots)


def test_secured_rows_close_the_attack_subspace():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(8, 3))  # 8 meters, 3 states
    assert attack_subspace(H, np.array([], int)).shape == (8, 3)  # nothing secured: every H c
    assert attack_subspace(H, np.array([0])).shape == (8, 2)  # one secured row pins one direction
    cost, a = attack_cost(H, np.array([0, 1, 2]))  # three independent secured rows: full rank
    assert cost == float("inf") and a is None
    cost, a = attack_cost(H, np.array([], int))
    assert a is not None and cost == (np.abs(a) > 1e-9).sum() and cost <= 8
    cost, a = attack_cost(H, np.array([0, 1]))
    assert a is not None
    c = np.linalg.lstsq(H, a, rcond=None)[0]  # the attack is a state change seen through H ...
    assert np.allclose(H @ c, a, atol=1e-8)
    assert np.allclose(H[[0, 1]] @ c, 0, atol=1e-8)  # ... that leaves the secured rows untouched


def test_greedy_selection_raises_the_cost_and_stops_when_closed():
    rng = np.random.default_rng(1)
    H = rng.normal(size=(10, 4))
    order, costs = greedy_trusted_meters(H, k=10)
    assert len(order) == len(set(order)) and len(order) <= 4  # four independent rows close it
    assert costs[-1] == float("inf")
    assert all(b >= a for a, b in zip(costs, costs[1:]))
    # a sparse Jacobian: securing the one meter every attack touches is found first
    H = np.zeros((5, 2))
    H[0] = [1, 1]
    H[1] = [1, 0]
    H[2] = [0, 1]
    H[3] = [2, 0]
    H[4] = [0, 3]
    order, costs = greedy_trusted_meters(H, k=5)
    assert costs[-1] == float("inf") and len(order) == 2


def test_attack_cost_matches_the_definition_on_a_small_case():
    """With H = I every meter is its own state: the cheapest attack touches one meter, and securing
    all but one leaves exactly one open."""
    H = np.eye(4)
    cost, a = attack_cost(H, np.array([], int))
    assert cost == 1
    cost, a = attack_cost(H, np.array([0, 1, 2]))
    assert cost == 1 and a is not None and np.flatnonzero(np.abs(a) > 1e-9).tolist() == [3]
    with pytest.raises(ValueError):
        rref(np.zeros((2,)))  # not a matrix
