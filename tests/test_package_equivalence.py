"""Hand-written loops and formulas replaced by library calls or array operations: each test pins
the replacement to the code it replaced (identical, or to tolerance where the arithmetic order
changes), so the simplification cannot drift."""

from types import SimpleNamespace

import numpy as np
import pytest


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


def test_edge_adjacency_is_symmetric_zero_one():
    pytest.importorskip("scipy")
    from fdia_graph.formulas.network import edge_adjacency

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
