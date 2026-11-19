"""Hand-written loops and formulas replaced by library calls or array operations: each test pins
the replacement to the code it replaced (identical, or to tolerance where the arithmetic order
changes), so the simplification cannot drift."""

from types import SimpleNamespace

import numpy as np
import pytest


def test_jitter_draws_in_the_per_channel_order():
    """A scan's jitter is one vector draw that equals the per-channel draws it replaced: an IEEE
    plan's rows run bus by bus (|V|, angle, P, Q, where metered), then the flows and then the PMU
    currents, branch by branch, and draw i scales meter row i."""
    pytest.importorskip("pandapower")
    from fdia_graph.constants.channels import TO_END_OFFSET, CurrentChannel
    from fdia_graph.constants.meters import BUS_KINDS, FLOW_KINDS, METER_SLOT, PHASOR_KINDS
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.systems.network import Network
    from tests.helpers import FixedDraws

    g = FdiaGenerator(Network.from_ieee(14), seed=7)
    plan, masks = g.plan, g.masks
    volt = masks.node[:, METER_SLOT[BUS_KINDS[0]][1]] > 0
    assert volt.any() and not volt.all() and len(g.pmu_buses) < volt.sum()  # a mixed plan
    want = []  # (kind, bus, branch, end) of every meter, in draw order
    for b in range(g.C):
        want += [(int(k), b, -1, -1) for k in BUS_KINDS if masks.node[b, METER_SLOT[k][1]]]
    for e in range(g.E):
        at = int(g.ei[0, e])
        want += [(int(k), at, e, 0) for k in FLOW_KINDS if masks.edge[e, METER_SLOT[k][1]]]
    for e in range(g.E):
        for c in CurrentChannel:
            end = c // TO_END_OFFSET
            if masks.current[e, c]:
                want.append((int(PHASOR_KINDS[c % TO_END_OFFSET]), int(g.ei[end, e]), e, end))
    rows = zip(plan.kind.tolist(), plan.bus.tolist(), plan.branch.tolist(), plan.end.tolist())
    assert list(rows) == want

    z = g.emitter.exact(np.stack([g.case.vm, g.case.pd, g.case.qd, g.case.va], axis=1))
    sig = g.noise.jitter(z)
    got = g.noise.sample(z, np.random.default_rng(7))
    ref = np.random.default_rng(7)
    want_scan = g.noise.sample(z, FixedDraws(0.0))  # type: ignore[arg-type]
    for i in range(len(z)):
        want_scan[i] += ref.normal(0, 1) * sig[i]
    assert (sig > 0).all()
    np.testing.assert_array_equal(got, want_scan)


def test_shunt_draws_leave_the_injections_like_the_loop():
    """Two shunts on one bus both come off it, in order, as the per-shunt loop did."""
    pd = pytest.importorskip("pandas")
    from fdia_graph.data.ieee_pool import _remove_shunt_injections

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
    from fdia_graph.formulas.temporal import recent_change_scale, swing_zscore, temporal_delta
    from fdia_graph.generation.write import TimelineWriter
    from fdia_graph.schema import SWING_WINDOW_SCANS

    rng = np.random.default_rng(5)
    T, N = 23, 6
    nx = (rng.standard_normal((T, N, 4)) * 40).astype(np.float32)
    with h5py.File(tmp_path / "t.h5", "w") as f:
        f[schema.NODE_X] = nx
        f.create_dataset(schema.TEMPORAL_DELTA, (T, N, 2), np.float32)
        f.create_dataset(schema.SWING, (T, N, 2), np.float32)
        TimelineWriter.write_temporal_layers(f, block=5)  # boundaries at frames 5, 10, 15, 20
        got_d, got_s = f[schema.TEMPORAL_DELTA][()], f[schema.SWING][()]
    scale = recent_change_scale(nx.astype(np.float64), SWING_WINDOW_SCANS, N)  # the writer runs it in float64
    every = np.ones(N, bool)
    for t in range(T):
        before = nx[t] if t == 0 else nx[t - 1]
        np.testing.assert_array_equal(got_d[t], temporal_delta(nx[t], before, every))
        np.testing.assert_array_equal(got_s[t], swing_zscore(nx[t], before, scale[t], every))
