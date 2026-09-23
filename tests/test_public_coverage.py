"""Direct tests for public pieces the rest of the suite reached only through a whole pipeline: the
replay policy, the linear-algebra and network kernels on cases small enough to check by hand, the
episode reader, the release lookup, the trust selector base, the removal guard and the learned
localizer's threshold."""

import h5py
import numpy as np
import pytest

import fdia_graph as fg
from fdia_graph import registry

# ---- the replay families' benign buffer ------------------------------------------------------------


def test_replay_frame_follows_its_policy():
    from fdia_graph.engine.records import REPLAY_MIN_LAG, replay_frame

    rng = np.random.default_rng(0)
    assert replay_frame([], None, rng) is None and replay_frame([], 5, rng) is None
    buf = [np.full(2, i) for i in range(50)]
    assert replay_frame(buf, 3, rng)[0] == 47  # a fixed lag: exactly that far back
    assert replay_frame(buf, 500, rng)[0] == 0  # clamped to the oldest the buffer holds
    lags = {50 - int(replay_frame(buf, None, rng)[0]) for _ in range(200)}
    assert min(lags) >= REPLAY_MIN_LAG  # a random lag reaches at least the minimum back
    assert replay_frame(buf[:REPLAY_MIN_LAG], None, rng)[0] == 0  # too shallow: the oldest


def test_remember_benign_is_a_bounded_fifo():
    from types import SimpleNamespace

    from fdia_graph.engine.records import BENIGN_BUFFER, remember_benign

    g = SimpleNamespace(benign_buf=[])
    for i in range(BENIGN_BUFFER + 5):
        remember_benign(g, np.array([i]))
    assert len(g.benign_buf) == BENIGN_BUFFER and g.benign_buf[0][0] == 5


# ---- kernels, checked by hand ------------------------------------------------------------------------


def test_batched_normal_matrices_equal_the_explicit_products():
    from fdia_graph.formulas.linalg import batched_normal_matrices

    rng = np.random.default_rng(1)
    B, w = rng.normal(size=(7, 3)), rng.uniform(0.1, 2.0, size=(5, 7))
    want = np.stack([B.T @ np.diag(wi) @ B for wi in w])
    assert np.allclose(batched_normal_matrices(w, B, sub=2), want)
    with pytest.raises(ValueError, match="sub"):
        batched_normal_matrices(w, B, sub=0)


def test_local_ac_solve_recovers_a_two_bus_voltage():
    from fdia_graph.formulas.network import local_ac_solve

    y = 1.0 / (0.01 + 0.1j)
    Y = np.array([[y, -y], [-y, y]])
    V_true = np.array([1.0 + 0j, 0.97 * np.exp(-0.05j)])  # bus 0 held, bus 1 the interior
    S1 = V_true[1] * np.conj(Y[1] @ V_true)
    V = local_ac_solve(Y, np.array([1.0 + 0j, 1.0 + 0j]), np.array([1]), np.array([S1]))
    assert V is not None and abs(V[1] - V_true[1]) < 1e-8 and V[0] == 1.0


def test_sparse_basis_opens_and_closes_with_the_secured_rows():
    from fdia_graph.formulas.trust import sparse_basis

    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    open_ = sparse_basis(H, np.array([], int))
    assert open_.shape == (2, 3)  # two independent stealthy attacks a = H c
    assert np.allclose(open_ @ np.array([1.0, 1.0, -1.0]), 0.0)  # each is orthogonal to the null of H^T
    assert sparse_basis(H, np.array([0, 1])).shape == (0, 3)  # both states pinned: no attack left


# ---- sequences and episodes ----------------------------------------------------------------------------


def test_per_bus_sequences_regroup_by_bus():
    from fdia_graph.dataset.sequence import per_bus_sequences

    Xw = np.arange(2 * 3 * 4 * 1, dtype=float).reshape(2, 3, 4, 1)  # [n, W, N, C]
    yf = (Xw[..., 0] > 10).astype(np.uint8)  # [n, W, N]
    X, y = per_bus_sequences(Xw, yf, "frame")
    assert X.shape == (8, 3, 1) and y.shape == (8, 3)
    assert np.array_equal(X[5, :, 0], Xw[1, :, 1, 0])  # record 1, bus 1
    assert np.array_equal(y[5], yf[1, :, 1])  # its per-frame labels follow it
    assert np.array_equal(y, yf.transpose(0, 2, 1).reshape(8, 3))
    ya = yf.any(axis=1)  # [n, N] window labels
    X, y = per_bus_sequences(Xw, ya, "any")
    assert y.shape == (8,) and np.array_equal(y, ya.reshape(8)) and y[5] == ya[1, 1]


def test_read_episodes_reads_the_group(tmp_path):
    from fdia_graph import schema
    from fdia_graph.dataset.sequence import read_episodes

    path = tmp_path / "e.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group(schema.Group.EPISODES)
        g[schema.EPISODE_ONSET] = [3, 10]
        g[schema.EPISODE_LENGTH] = [2, 1]
        g[schema.EPISODE_FAMILY] = [1, 4]
        g[schema.EPISODE_BUS_PTR] = [0, 2, 3]
        g[schema.EPISODE_BUS_IDX] = [5, 7, 2]
    with h5py.File(path, "r") as f:
        e = read_episodes(f, True)
        assert read_episodes(f, False) is None
    assert e.onset.tolist() == [3, 10] and e.family.tolist() == [1, 4]
    assert [b.tolist() for b in e.buses] == [[5, 7], [2]]


# ---- the release lookup ---------------------------------------------------------------------------------


def test_latest_release_online_and_offline(monkeypatch):
    import requests

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {"tag_name": "data-v0.9.1"},
                {"tag_name": "v0.19.0"},
                {"tag_name": "data-v0.9.2", "draft": True},
            ]

    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    assert registry.latest_release() == "v0.9.1"  # newest published data tag; drafts and packages ignored

    def offline(*a, **k):
        raise requests.ConnectionError("no network")

    monkeypatch.setattr(requests, "get", offline)
    assert registry.latest_release() == registry._RELEASE


# ---- trust, removal, learned localizer ---------------------------------------------------------------------


def test_trust_selector_base(splits):
    pytest.importorskip("pandapower")
    from fdia_graph.trust import TrustedMeters
    from fdia_graph.trust.base import TrustSelector

    with pytest.raises(NotImplementedError):
        TrustSelector(k=2).fit(splits["train"])
    tm = TrustedMeters(k=4).fit(splits["train"])
    assert len(tm.select()) == 4 and tm.select(2).tolist() == tm.order[:2]
    assert tm.attack_cost([]) <= tm.attack_cost()  # securing meters never makes the attack cheaper


def test_removal_keeps_every_meter_the_guard_refuses(splits):
    pytest.importorskip("pandapower")
    from fdia_graph.se import ResidualRemoval

    est = ResidualRemoval(threshold=0.5).fit(splits["train"])  # nearly every residual is "bad"
    d = splits["test"].export(["node_x", "edge_x", "clean"])
    z = est._z_of(d["node_x"], d["edge_x"])[:20]
    thsl = est._truth_of(d["clean"])["thsl"][:20]
    asked = []

    def refuse(w):  # every removal would break observability
        asked.append(1)
        return False

    est._observable = refuse
    full = est._w_solve(z, np.broadcast_to(est.Wk, z.shape), thsl)
    assert np.allclose(est._solve(z, thsl), full)  # nothing removed, and the loop still ends
    assert len(asked) >= 20  # the guard was consulted, at least once per record


def test_tune_threshold_picks_one_global_tau(splits, timeline):
    pytest.importorskip("torch")
    from fdia_graph.localization import BusMLP

    loc = BusMLP(epochs=2, device="cpu").fit(splits["train"], val=splits["val"])
    assert loc.tau is not None and np.round(loc.tau / 0.05) * 0.05 == pytest.approx(loc.tau)
    assert np.all(loc.thr[loc._attackable] == loc.tau) and np.all(np.isinf(loc.thr[~loc._attackable]))
    with pytest.raises(ValueError, match="attacked records"):
        loc.tune_threshold(fg.load(timeline, split="val", families=["benign"]))
