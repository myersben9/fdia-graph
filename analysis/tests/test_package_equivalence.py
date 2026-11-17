"""The analysis toolkit's part of tests/test_package_equivalence.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
Hand-written loops and formulas replaced by library calls or array operations: each test pins
the replacement to the code it replaced (identical, or to tolerance where the arithmetic order
changes), so the simplification cannot drift."""

import numpy as np
import pytest

from fdia_analysis.formulas.metrics import (
    micro_prf,
    perbus_counts,
    perbus_counts_at,
    sample_f1,
    strict_accuracy,
)
from fdia_analysis.formulas.projection import leverage


def _labels(seed: int = 0, n: int = 60, N: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = rng.random((n, N)) < 0.2
    truth[: n // 4] = False  # benign records: nothing true, so the zero-division cases are exercised
    score = np.clip(truth * 0.6 + rng.random((n, N)) * 0.5, 0, 1)
    return score, score > 0.5, truth


def test_localization_metrics_match_scikit_learn():
    """micro_prf, sample_f1 and strict_accuracy equal scikit-learn's multilabel scores with
    zero_division=0, the papers' conventions for strict localization accuracy and node F1."""
    metrics = pytest.importorskip("sklearn.metrics")
    _, pred, truth = _labels()
    prec, rec, f1 = micro_prf(pred, truth)
    p, r, f, _ = metrics.precision_recall_fscore_support(truth, pred, average="micro", zero_division=0)
    assert (prec, rec, f1) == pytest.approx((p, r, f), abs=1e-9)
    assert sample_f1(pred, truth) == pytest.approx(
        metrics.f1_score(truth, pred, average="samples", zero_division=0), abs=1e-9
    )
    assert strict_accuracy(pred, truth) == pytest.approx(metrics.accuracy_score(truth, pred))


def test_counts_at_every_threshold_equal_one_threshold_at_a_time():
    """perbus_counts_at (every tau in one array operation) gives exactly the per-tau loop's counts."""
    score, _, truth = _labels(1)
    taus = np.linspace(0.05, 0.95, 19)
    looped = [np.stack(c) for c in zip(*(perbus_counts(score > tau, truth) for tau in taus))]
    for got, want in zip(perbus_counts_at(score, truth, taus), looped):
        np.testing.assert_array_equal(got, want)


def test_leverage_is_the_hat_matrix_diagonal():
    rng = np.random.default_rng(2)
    Hw = rng.standard_normal((40, 7))
    Ai = np.linalg.inv(Hw.T @ Hw)
    np.testing.assert_allclose(leverage(Hw, Ai), np.clip(np.diag(Hw @ Ai @ Hw.T), 0, 1), atol=1e-12)


def test_replay_keeps_the_newest_in_arrival_order():
    from fdia_analysis.trust.meter_dqn import _Replay

    buf = _Replay(3)
    for i in range(5):
        buf.append((i,))
    assert len(buf) == 3
    assert buf.sample(np.array([0, 1, 2])) == [(2,), (3,), (4,)]  # index 0 is the oldest kept


def _bfs_subnetwork(edge_index, seeds, hops, n_bus):
    """The hand-written breadth-first search `formulas.network.subnetwork` used before csgraph."""
    adj = [set() for _ in range(n_bus)]
    for a, b in zip(edge_index[0], edge_index[1]):
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))
    interior = {int(b) for b in seeds}
    frontier = set(interior)
    for _ in range(hops):
        frontier = {j for i in frontier for j in adj[i]} - interior
        interior |= frontier
    boundary = {j for i in interior for j in adj[i]} - interior
    return sorted(interior), sorted(boundary)


def test_subnetwork_and_hop_distance_match_breadth_first_search():
    """`subnetwork` and `hop_distance` (scipy's shortest path) give the breadth-first search's
    interiors, boundaries and hop counts, unreachable buses included."""
    pytest.importorskip("scipy")
    from fdia_analysis.formulas.federated import hop_distance
    from fdia_graph.formulas.network import subnetwork

    rng = np.random.default_rng(4)
    n = 30
    edges = np.array([(i, i + 1) for i in range(24)] + [(3, 17), (5, 20), (26, 27)]).T  # 28, 29 isolated
    A = np.zeros((n, n))
    A[edges[0], edges[1]] = A[edges[1], edges[0]] = 1
    for _ in range(100):
        seeds = rng.choice(n, int(rng.integers(1, 4)), replace=False)
        hops = int(rng.integers(0, 5))
        interior, boundary = subnetwork(edges, seeds, hops, n)
        assert (interior.tolist(), boundary.tolist()) == _bfs_subnetwork(edges, seeds, hops, n)
        want = np.full(n, -1)
        for h in range(n):
            reach = _bfs_subnetwork(edges, seeds, h, n)[0]
            want[[b for b in reach if want[b] < 0]] = h
        np.testing.assert_array_equal(hop_distance(A, seeds), want)
