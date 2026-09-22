"""The trusted-meter selectors on the tiny timeline: the greedy selection raises the attack cost,
the secured meters expose the stealthy families to the residual test, and the DQN learns a
selection that closes the attack subspace too."""

import numpy as np
import pytest

pytest.importorskip("pandapower")

import fdia_graph as fg  # noqa: E402


@pytest.fixture(scope="module")
def selector(timeline):
    from fdia_graph.trust import TrustedMeters

    return TrustedMeters(k=12).fit(fg.load(timeline, split="train"))


def test_greedy_selection_raises_the_attack_cost(selector):
    tm = selector
    assert 0 < len(tm.order) <= 12 and len(set(tm.order)) == len(tm.order)
    assert all(0 <= i < tm.m for i in tm.order)
    assert all(b >= a for a, b in zip(tm.cost, tm.cost[1:]))
    assert tm.attack_cost([]) <= tm.cost[0] and tm.attack_cost() == tm.cost[len(tm.select()) - 1]
    assert len(tm.select(3)) == 3


def test_secured_meters_expose_the_stealthy_families(selector, timeline):
    from fdia_graph.models import TrustScores

    rep = selector.score(fg.load(timeline, split="test", order="time"))
    # the level is set on the fixture's 600 train frames and its 200 test frames sit 400 minutes
    # later on the one-minute pool, so the benign rate there is coarse; the full files hold 1%
    assert isinstance(rep, TrustScores) and rep.false_alarm < 0.5
    assert selector.level > 0
    assert set(rep.detected_before) == set(rep.detected_after)
    # the re-solve families are what the trust exposes; Am is not residual-stealthy to begin with
    # (its untouched meters contradict the moved state), so pinning meters can move it either way
    for fam in ("Aq", "Al"):
        if fam in rep.detected_after:
            assert rep.detected_after[fam] >= rep.detected_before[fam] - 0.05, fam
    assert rep["order"] == rep.order and len(rep.cost) == len(rep.order)
    with pytest.raises(ValueError, match="benign layer"):
        from conftest import SHARD_V072

        from fdia_graph.dataset import FdiaGraph

        selector.score(FdiaGraph(SHARD_V072))


def test_dqn_learns_a_selection(timeline):
    pytest.importorskip("torch")
    from fdia_graph.trust import TrustedMetersDQN

    rl = TrustedMetersDQN(k=8, episodes=12, seed=0).fit(fg.load(timeline, split="train"))
    assert 0 < len(rl.order) <= 8 and len(set(rl.order)) == len(rl.order)
    assert len(rl.history) == 12 and len(rl.cost) == len(rl.order)
    assert rl.attack_cost() >= rl.attack_cost([])
    with pytest.raises(ValueError, match="k must be"):
        TrustedMetersDQN(k=0)


def test_secured_copy_pins_the_selection_and_keeps_the_temporal_kernels(selector, timeline, tmp_path):
    """The copy reads the benign layer at the secured meters and nothing else changes; with no
    meter secured the recomputed temporal features match the stored ones (to the float32 precision
    of the stored layers), which pins the writer's kernels and the clean layer as the swing pool."""
    import h5py

    from fdia_graph import schema
    from fdia_graph.formulas.projection import meter_positions

    ds = fg.load(timeline, order="time")
    same = selector.secured_copy(ds, str(tmp_path / "none.h5"), k=0)
    with h5py.File(ds.path) as a, h5py.File(same) as b:
        for key in (schema.NODE_X, schema.EDGE_X, schema.NODE_TAMPER):
            assert np.array_equal(a[key][...], b[key][...]), key
        for key in (schema.TEMPORAL_DELTA, schema.SWING):  # float32 layers in, the writer's float64 out
            assert np.allclose(a[key][...], b[key][...], rtol=1e-4, atol=1e-3), key
    out = selector.secured_copy(ds, str(tmp_path / "secured.h5"), name="tiny_ieee14_secured")
    nodes, edges = meter_positions(selector.est.N, selector.est.E, selector.est.mask, selector.select())
    assert len(nodes) + len(edges) == len(selector.select())
    with h5py.File(ds.path) as a, h5py.File(out) as b:
        nx, bnx, nx2 = a[schema.NODE_X][...], a[schema.NODE_BENIGN][...], b[schema.NODE_X][...]
        pinned = np.zeros(nx.shape[1:], bool)
        for bus, col in nodes:
            pinned[bus, col] = True
            assert np.array_equal(nx2[:, bus, col], bnx[:, bus, col])
            assert not b[schema.NODE_TAMPER][:, bus, col].any()
        assert np.array_equal(nx2[:, ~pinned], nx[:, ~pinned])
        for branch, col in edges:
            assert np.array_equal(b[schema.EDGE_X][:, branch, col], a[schema.EDGE_BENIGN][:, branch, col])
            assert not b[schema.EDGE_TAMPER][:, branch, col].any()
    sec = fg.load("tiny_ieee14_secured", split="test", order="time")
    assert len(sec) == len(fg.load(timeline, split="test")) and sec.has_benign


def test_gated_prior_keeps_secured_meters_at_full_weight(selector, timeline):
    """With `secured`, the gate leaves those meters at their fitted weight on every record and
    still down-weights the rest of a flagged bus."""
    from fdia_graph.se import GatedPrior

    train = fg.load(timeline, split="train")
    test = fg.load(timeline, split="test")
    secured = selector.select(4)
    plain = GatedPrior(gate="oracle", rank_frac=0.3).fit(train)
    aware = GatedPrior(gate="oracle", rank_frac=0.3, secured=secured).fit(train)
    w0, w1 = plain.gated_weights(test), aware.gated_weights(test)
    assert np.array_equal(w1[:, secured], np.broadcast_to(aware.Wk[secured], (len(test), len(secured))))
    assert (w0 < np.broadcast_to(plain.Wk, w0.shape)).any()  # the gate does fire somewhere
    others = np.setdiff1d(np.arange(w0.shape[1]), secured)
    assert np.array_equal(w0[:, others], w1[:, others])
