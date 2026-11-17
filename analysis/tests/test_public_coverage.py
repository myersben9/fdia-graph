"""The analysis toolkit's part of tests/test_public_coverage.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
Direct tests for public pieces the rest of the suite reached only through a whole pipeline: the
replay policy, the linear-algebra and network kernels on cases small enough to check by hand, the
episode reader, the release lookup, the trust selector base, the removal guard and the learned
localizer's threshold."""

import numpy as np
import pytest

import fdia_graph as fg
from fdia_analysis.models.config import LearnedConfig, RemovalConfig, TrustConfig

# ---- the replay families' benign buffer ------------------------------------------------------------


# ---- kernels, checked by hand ------------------------------------------------------------------------


def test_batched_normal_matrices_equal_the_explicit_products():
    from fdia_analysis.formulas.linalg import batched_normal_matrices

    rng = np.random.default_rng(1)
    B, w = rng.normal(size=(7, 3)), rng.uniform(0.1, 2.0, size=(5, 7))
    want = np.stack([B.T @ np.diag(wi) @ B for wi in w])
    assert np.allclose(batched_normal_matrices(w, B, sub=2), want)
    with pytest.raises(ValueError, match="sub"):
        batched_normal_matrices(w, B, sub=0)


def test_sparse_basis_opens_and_closes_with_the_secured_rows():
    from fdia_analysis.formulas.trust import sparse_basis

    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    open_ = sparse_basis(H, np.array([], int))
    assert open_.shape == (2, 3)  # two independent stealthy attacks a = H c
    assert np.allclose(open_ @ np.array([1.0, 1.0, -1.0]), 0.0)  # each is orthogonal to the null of H^T
    assert sparse_basis(H, np.array([0, 1])).shape == (0, 3)  # both states pinned: no attack left


# ---- trust, removal, learned localizer ---------------------------------------------------------------------


def test_trust_selector_base(splits):
    pytest.importorskip("pandapower")
    from fdia_analysis.trust.base import TrustedMeters, TrustSelector

    with pytest.raises(NotImplementedError):
        TrustSelector(TrustConfig(k=2)).fit(splits["train"])
    tm = TrustedMeters(TrustConfig(k=4)).fit(splits["train"])
    assert len(tm.select()) == 4 and tm.select(2).tolist() == tm.order[:2]
    assert tm.attack_cost([]) <= tm.attack_cost()  # securing meters never makes the attack cheaper


def test_removal_keeps_every_meter_the_guard_refuses(splits):
    pytest.importorskip("pandapower")
    from fdia_analysis.se.methods import ResidualRemoval

    est = ResidualRemoval(removal=RemovalConfig(threshold=0.5)).fit(
        splits["train"]
    )  # nearly every residual is "bad"
    d = splits["test"].export(["node_x", "edge_x"])
    z = est._z_of(d["node_x"], d["edge_x"])[:20]
    asked = []

    def refuse(w):  # every removal would break observability
        asked.append(1)
        return False

    est._observable = refuse
    full = est._w_solve(z, np.broadcast_to(est.Wk, z.shape))
    assert np.allclose(est._solve(z), full)  # nothing removed, and the loop still ends
    assert len(asked) >= 20  # the guard was consulted, at least once per record


def test_tune_threshold_picks_one_global_tau(splits, timeline):
    pytest.importorskip("torch")
    from fdia_analysis.localization.learned import BusMLP

    loc = BusMLP(learned=LearnedConfig(epochs=2, device="cpu")).fit(splits["train"], val=splits["val"])
    assert loc.tau is not None and np.round(loc.tau / 0.05) * 0.05 == pytest.approx(loc.tau)
    assert np.all(loc.thr[loc._attackable] == loc.tau) and np.all(np.isinf(loc.thr[~loc._attackable]))
    with pytest.raises(ValueError, match="attacked records"):
        loc.tune_threshold(fg.load(timeline, split="val", families=["benign"]))
