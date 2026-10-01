"""Hand-written loops and formulas replaced by library calls or array operations: each test pins
the replacement to the code it replaced (identical, or to tolerance where the arithmetic order
changes), so the simplification cannot drift."""

from types import SimpleNamespace

import numpy as np
import pytest

from fdia_graph.formulas.metrics import micro_prf, perbus_counts, perbus_counts_at, sample_f1, strict_accuracy
from fdia_graph.formulas.projection import leverage


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


def test_jitter_draws_in_the_per_channel_order():
    """`_jitter_in_order` takes one vector draw that equals the per-channel draws it replaced: row
    by row, and within a row in the given column order, only where the mask is set."""
    from fdia_graph.engine.measurement import _NODE_DRAW_ORDER, MeasurementMixin

    rng = np.random.default_rng(3)
    mask = (rng.random((12, 4)) < 0.6).astype(np.uint8)
    mask[:, 2] = mask[:, 1]  # P and Q are metered together
    sig = rng.random((12, 4)) + 0.1
    gen = SimpleNamespace(_jitter=np.random.default_rng(7))
    got = MeasurementMixin._jitter_in_order(gen, mask, sig, _NODE_DRAW_ORDER)  # type: ignore[arg-type]
    ref = np.random.default_rng(7)
    want = np.zeros((12, 4))
    for b in range(12):
        for c in _NODE_DRAW_ORDER:
            if mask[b, c]:
                want[b, c] = ref.normal(0, sig[b, c])
    np.testing.assert_array_equal(got, want)


def test_replay_keeps_the_newest_in_arrival_order():
    from fdia_graph.trust.dqn import _Replay

    buf = _Replay(3)
    for i in range(5):
        buf.append((i,))
    assert len(buf) == 3
    assert buf.sample(np.array([0, 1, 2])) == [(2,), (3,), (4,)]  # index 0 is the oldest kept


def test_shunt_draws_leave_the_injections_like_the_loop():
    """Two shunts on one bus both come off it, in order, as the per-shunt loop did."""
    pd = pytest.importorskip("pandas")
    from fdia_graph.profiles import _remove_shunt_injections

    base = SimpleNamespace(
        shunt=pd.DataFrame({"bus": [3, 1, 3, 9]}),
        res_shunt=pd.DataFrame({"p_mw": [0.5, 1.25, 0.125, 7.0], "q_mvar": [-2.0, 0.75, 0.3, 1.0]}),
    )
    pos = {1: 0, 3: 2}  # bus 9 is outside the nodelist and is skipped
    z = np.arange(12, dtype=float).reshape(4, 3)
    want = z.copy()
    for b, ps, qs in zip([3, 1, 3, 9], [0.5, 1.25, 0.125, 7.0], [-2.0, 0.75, 0.3, 1.0]):
        if b in pos:
            want[pos[b], 1] -= ps
            want[pos[b], 2] -= qs
    _remove_shunt_injections(z, base, pos)  # type: ignore[arg-type]
    np.testing.assert_array_equal(z, want)


def _bfs_subnetwork(edge_index, seeds, hops, n_bus):
    """The hand-written breadth-first search `formulas.subnetwork` used before csgraph."""
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
    from fdia_graph.formulas import hop_distance, subnetwork

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


def test_edge_adjacency_is_symmetric_zero_one():
    pytest.importorskip("scipy")
    from fdia_graph.formulas import edge_adjacency

    A = edge_adjacency(np.array([[0, 0, 1, 2], [1, 1, 2, 0]]), 4).toarray()  # a parallel pair on 0-1
    np.testing.assert_array_equal(A, A.T)
    np.testing.assert_array_equal(A, [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]])


def test_temporal_layers_equal_the_per_frame_kernels(tmp_path):
    """`write_temporal_layers` over small blocks equals `temporal_delta` and `swing_zscore` frame by
    frame, the first frame (its own previous) and the frames across a block boundary included."""
    h5py = pytest.importorskip("h5py")
    from fdia_graph import schema
    from fdia_graph.formulas.temporal import SWING_WINDOW, recent_change_scale, swing_zscore, temporal_delta
    from fdia_graph.generation.write import write_temporal_layers

    rng = np.random.default_rng(5)
    T, N = 23, 6
    nx = (rng.standard_normal((T, N, 4)) * 40).astype(np.float32)
    with h5py.File(tmp_path / "t.h5", "w") as f:
        f[schema.NODE_X] = nx
        f.create_dataset(schema.TEMPORAL_DELTA, (T, N, 2), np.float32)
        f.create_dataset(schema.SWING, (T, N, 2), np.float32)
        write_temporal_layers(f, block=5)  # boundaries at frames 5, 10, 15, 20
        got_d, got_s = f[schema.TEMPORAL_DELTA][()], f[schema.SWING][()]
    scale = recent_change_scale(nx.astype(np.float64), SWING_WINDOW, N)  # the writer runs it in float64
    every = np.ones(N, bool)
    for t in range(T):
        before = nx[t] if t == 0 else nx[t - 1]
        np.testing.assert_array_equal(got_d[t], temporal_delta(nx[t], before, every))
        np.testing.assert_array_equal(got_s[t], swing_zscore(nx[t], before, scale[t], every))
