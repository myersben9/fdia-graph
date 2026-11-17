"""The analysis toolkit's part of tests/test_formulas.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
The formulas kernel: each function on a case small enough to check by hand, and the estimator's
torch measurement function pinned to the numpy kernel so there is one AC model in the package."""

import numpy as np
import pytest

from fdia_analysis.models import HuberConfig, SolveConfig
from fdia_graph.formulas import (
    branch_flows,
    bus_injections,
    complex_voltages,
)


def test_estimator_measurement_function_is_the_kernel(splits):
    """SEBase._h is `ac_measurement`, and the torch twin `_h_t` (kept for callers that differentiate
    through it) equals it on real states."""
    torch = pytest.importorskip("torch")
    from fdia_analysis.se import WLS

    est = WLS().fit(splits["train"])
    d = splits["test"].export(["clean"])
    tr = est._truth_of(d["clean"][:8])
    h_np = est._h(tr["x"], tr["thsl"])  # [n, m] masked measurements in pu and rad
    with torch.no_grad():
        h_torch = est._h_t(torch.tensor(tr["x"]), torch.tensor(tr["thsl"])).numpy()[:, est.mask]
    assert np.allclose(h_torch, h_np, rtol=1e-12, atol=1e-12)
    ns, N = len(est.keep), est.N
    th = np.zeros((8, N))
    th[:, est.keep] = tr["x"][:, :ns]
    th[:, est.slack] = tr["thsl"]
    Vc_pp = complex_voltages(tr["x"][:, ns:], np.rad2deg(th))
    Vc = np.zeros((8, est._nppc), complex)
    Vc[:, est._lut] = Vc_pp
    Ybus = est._Ybus.numpy()
    Yft = est._Yft.numpy()
    Sb = bus_injections(Vc, Ybus)  # generation positive; the shard's injections are load positive
    Sf = branch_flows(Vc, Yft, est._fb)
    lut = est._lut
    full = np.concatenate([tr["x"][:, ns:], -Sb.real[:, lut], -Sb.imag[:, lut], th, Sf.real, Sf.imag], axis=1)
    assert np.allclose(h_np, full[:, est.mask], rtol=1e-10, atol=1e-10)


def test_closed_form_jacobian_matches_autograd(splits):
    """`ac_jacobian` at the chord point equals the automatic-differentiation Jacobian of the torch
    twin, which is how the estimator computed H before the closed form (max difference 1e-14 on
    the tiny shard; the tolerance leaves room for other BLAS builds)."""
    torch = pytest.importorskip("torch")
    from torch.func import jacrev, vmap

    from fdia_analysis.se import WLS

    est = WLS().fit(splits["train"])
    x0 = torch.tensor(est.xmean, dtype=torch.float64)[None]
    thsl = est._truth_of(splits["train"].export(["clean"])["clean"][:1])["thsl"]
    t0 = torch.tensor([float(thsl[0])], dtype=torch.float64)

    def h1(xi, ti):
        return est._h_t(xi[None], ti[None])[0]

    H_auto = vmap(jacrev(h1))(x0, t0)[0].numpy()[est.mask]
    assert H_auto.shape == est.H.shape
    assert np.abs(H_auto - est.H).max() < 1e-9 * max(1.0, np.abs(est.H).max())


def test_estimator_fits_and_scores_without_torch(splits, monkeypatch):
    """The estimator needs neither torch nor autograd: block the import and fit, estimate, score."""
    import builtins
    import sys

    real_import = builtins.__import__

    def no_torch(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch blocked for this test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    for m in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, m)
    from fdia_analysis.se import AdaptiveWeighting

    est = AdaptiveWeighting(huber=HuberConfig(c=1.5), solve=SolveConfig(npass=2)).fit(splits["train"])
    scores = est.score(splits["test"])
    assert np.isfinite(scores.geo.angle_mae_deg) and not hasattr(est, "_Ybus")


# ---- estimation, linalg and projection kernels ---------------------------------------------------


def _toy_system():
    """Two states, four measurements with a hand-checkable Jacobian and weights."""
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    w = np.array([1.0, 1.0, 4.0, 0.25])
    return H, w


def test_normal_matrix_and_wls_step_by_hand():
    from fdia_analysis.formulas import guarded_inverse, normal_matrix, wls_step, wls_step_batched

    H, w = _toy_system()
    G = normal_matrix(H, w)
    assert np.allclose(G, [[5.25, 3.75], [3.75, 5.25]])  # sum_i w_i h_i h_i^T: 1 + 4 + 0.25 on the diagonal
    Ai = guarded_inverse(G)
    assert np.allclose(Ai @ G, np.eye(2))
    r = np.array([[1.0, 2.0, 3.0, -1.0]])
    step = wls_step(r, w, H, Ai)
    assert np.allclose(step, np.linalg.solve(G, H.T @ (w * r[0])))  # the normal equations
    stacked = wls_step_batched(
        np.repeat(r, 3, axis=0), np.repeat(w[None], 3, axis=0), H, np.repeat(Ai[None], 3, axis=0)
    )
    assert np.allclose(stacked, np.repeat(step, 3, axis=0))


def test_residual_covariance_and_normalized_residual():
    from fdia_analysis.formulas import (
        critical_measurements,
        floored_covariance,
        guarded_inverse,
        normal_matrix,
        normalized_residual,
        residual_covariance_diag,
        weighted_objective,
    )

    H, w = _toy_system()
    Ai = guarded_inverse(normal_matrix(H, w))
    om, R = residual_covariance_diag(H, w, Ai)
    assert np.allclose(R, 1 / w)
    S = np.diag(1 / w) - H @ Ai @ H.T  # the full residual covariance
    assert np.allclose(om, np.diag(S)) and np.all(om > 0)  # four meters, two states: nothing critical
    assert not critical_measurements(om, R).any()
    om2 = om.copy()
    om2[0] = 0.0
    assert critical_measurements(om2, R)[0] and floored_covariance(om2, R)[0] == 1e-12 * R[0]
    r = np.array([[0.5, -1.0, 2.0, 0.0]])
    assert np.allclose(normalized_residual(r, om), np.abs(r) / np.sqrt(om))
    assert weighted_objective(r, w)[0] == pytest.approx(0.25 + 1.0 + 16.0)


def test_huber_weights_by_hand():
    from fdia_analysis.formulas import huber_weights

    a = huber_weights(np.array([[0.5, 1.5, 3.0, 0.0]]), c=1.5)
    assert a.tolist() == [
        [1.0, 1.0, 0.5, 1.0]
    ]  # inside the band 1; outside c / r; a zero residual is floored


def test_whitened_svd_basis_is_orthonormal():
    from fdia_analysis.formulas import whitened_svd_basis

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)) * np.array([3.0, 0.3, 0.03, 0.003, 0.0003, 0.00003])
    K, VK = whitened_svd_basis(X, 0.5)
    assert K == 3 and VK.shape == (6, 3)
    assert np.allclose(VK.T @ VK, np.eye(3), atol=1e-12)


def test_gate_weights_and_bus_incidence():
    from fdia_analysis.formulas import bus_incidence, gate_weights

    N, E = 3, 2
    ei = np.array([[0, 1], [1, 2]])  # branches 0-1 and 1-2
    mask = np.ones(4 * N + 2 * E, bool)
    inc = bus_incidence(N, E, ei, mask)
    assert inc[0].tolist() == [0, 3, 6, 9, 12, 14]  # V, P, Q, theta of bus 0 and both flows of branch 0
    assert inc[1].tolist() == [1, 4, 7, 10, 12, 13, 14, 15]  # bus 1 touches both branches
    w = np.ones(16)
    flags = np.array([[True, False, False], [False, False, True]])
    g = gate_weights(w, flags, inc, 0.1)
    assert g.shape == (2, 16) and g[0, inc[0]].tolist() == [0.1] * 6 and g[0, 1] == 1.0
    assert g[1, inc[2]].tolist() == [0.1] * len(inc[2]) and g[1, 0] == 1.0


def test_projection_split_is_exact_and_orthogonal():
    from fdia_analysis.formulas import (
        direction_coefficients,
        explained_unexplained,
        guarded_inverse,
        leverage,
        meters_to_buses,
        normal_matrix,
        weak_directions,
        weak_move,
        weighted_pseudoinverse,
    )

    H, w = _toy_system()
    Ai = guarded_inverse(normal_matrix(H, w))
    pinv = weighted_pseudoinverse(H, w, Ai)
    assert np.allclose(pinv @ H, np.eye(2))  # a left inverse of H
    dz = np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 1.0, -1.0]])
    dx, r_par, r_perp = explained_unexplained(dz, H, pinv)
    assert np.allclose(r_par + r_perp, dz)
    assert np.allclose((r_perp * w) @ H, 0.0, atol=1e-12)  # the unexplained part is W-orthogonal to range(H)
    assert np.allclose(r_perp[1], 0.0)  # dz = H [0, 1] exactly: nothing unexplained
    Hw = np.sqrt(w)[:, None] * H
    lev = leverage(Hw, Ai)
    assert lev.sum() == pytest.approx(2.0) and np.all(
        (lev >= 0) & (lev <= 1)
    )  # trace of the hat matrix = rank
    U, S, Vweak, rows = weak_directions(Hw, 1)
    assert rows.tolist() == [1] and Vweak.shape == (2, 1) and S[0] >= S[1]
    assert np.allclose(np.linalg.norm(weak_move(dx, Vweak), axis=1), np.abs(dx @ Vweak[:, 0]))
    assert direction_coefficients(dz * np.sqrt(w), U).shape == (2, 2)
    inc = [np.array([0, 2]), np.array([], int), np.array([1, 3])]
    assert meters_to_buses(dz, inc, "sum").tolist() == [[4.0, 0.0, 6.0], [1.0, 0.0, 0.0]]
    assert meters_to_buses(dz, inc, "max").tolist() == [[3.0, 0.0, 4.0], [1.0, 0.0, 1.0]]


def test_guarded_inverse_and_condition_number():
    from fdia_analysis.formulas import condition_number, guarded_inverse

    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    assert np.allclose(guarded_inverse(A), np.linalg.inv(A))
    assert condition_number(A) == pytest.approx(np.linalg.cond(A), rel=1e-6)
    S = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular: pseudo-inverse, infinite condition
    assert np.allclose(guarded_inverse(S), np.linalg.pinv(S))
    assert condition_number(S) == float("inf")


def test_triangular_rcond_falls_back_without_dtrcon():
    """SciPy on Python 3.9 has no lapack.dtrcon: Hager's estimate stands in, bounded by the exact value."""
    import scipy.linalg as sl

    from fdia_analysis.formulas.linalg import _triangular_rcond

    A = np.random.default_rng(3).normal(size=(30, 30))
    L = np.linalg.cholesky(A @ A.T + np.eye(30))
    exact = 1.0 / (np.linalg.norm(L, 1) * np.linalg.norm(np.linalg.inv(L), 1))
    fallback = _triangular_rcond(L, object())  # an estimate of the inverse's norm from below
    assert exact <= fallback <= 3 * exact
    if hasattr(sl.lapack, "dtrcon"):  # LAPACK's estimate bounds it from above, within a small factor
        assert exact <= _triangular_rcond(L, sl.lapack) <= 3 * exact


def test_pool_moments_equals_the_union_and_leaves_one_part_unchanged():
    from fdia_analysis.formulas.federated import channel_moments, pool_moments

    rng = np.random.default_rng(0)
    parts = [rng.normal(size=(n, 5, 3)) for n in (7, 2, 11)]
    n, mu, var = pool_moments([channel_moments(X) for X in parts])
    allX = np.concatenate([X.reshape(-1, 3) for X in parts])
    assert (
        n == len(allX)
        and np.allclose(mu, allX.mean(0), atol=1e-12)
        and np.allclose(var, allX.var(0), atol=1e-12)
    )
    one = channel_moments(parts[0])
    n1, mu1, var1 = pool_moments([one])
    assert n1 == one[0] and mu1 is one[1] and var1 is one[2]  # bit for bit: no arithmetic on one part


def test_tau_from_counts_matches_the_direct_f1_rule():
    from fdia_analysis.formulas.metrics import perbus_counts, tau_from_counts

    rng = np.random.default_rng(1)
    p, t = rng.random((200, 6)), rng.random((200, 6)) < 0.3
    taus = np.linspace(0.05, 0.95, 19)
    active = t.any(axis=0)
    direct = []
    for tau in taus:
        pr = p > tau
        tp, fp, fn = ((pr & t).sum(0), (pr & ~t).sum(0), (~pr & t).sum(0))
        direct.append((2 * tp / (2 * tp + fp + fn + 1e-9))[active].mean())
    tp, fp, fn = (np.stack(c) for c in zip(*(perbus_counts(p > tau, t) for tau in taus)))
    assert tau_from_counts(tp, fp, fn, active, taus) == float(taus[int(np.argmax(direct))])


def test_count_and_moment_formulas_refuse_mismatched_shapes():
    from fdia_analysis.formulas.federated import pool_moments
    from fdia_analysis.formulas.metrics import perbus_counts, tau_from_counts

    with pytest.raises(ValueError, match="moments of shape"):
        pool_moments([(4.0, np.zeros(3), np.ones(3)), (2.0, np.zeros(2), np.ones(2))])
    for bad in (0.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="finite positive count"):
            pool_moments([(bad, np.zeros(3), np.ones(3))])
    n, mu, var = pool_moments([(1e300, np.zeros(2), np.ones(2)), (1e300, np.full(2, 2.0), np.ones(2))])
    assert n == 2e300 and np.allclose(mu, 1.0) and np.allclose(var, 2.0)  # huge counts pool without overflow
    with pytest.raises(ValueError, match="overflows"):
        pool_moments([(1e308, np.zeros(3), np.ones(3)), (1e308, np.zeros(3), np.ones(3))])
    with pytest.raises(ValueError, match="one shape"):
        perbus_counts(np.zeros((3, 2), bool), np.zeros((3, 4), bool))
    z = np.zeros((2, 3))
    with pytest.raises(ValueError, match="n_taus, N"):
        tau_from_counts(z, z, z, np.ones(4, bool), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="one-dimensional"):
        tau_from_counts(z, z, z, np.ones((1, 3), bool), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="one-dimensional"):
        tau_from_counts(z, z, z, np.ones(3, bool), np.array([[0.1, 0.2]]))
    with pytest.raises(ValueError, match="active bus"):
        tau_from_counts(z, z, z, np.zeros(3, bool), np.array([0.1, 0.2]))


def test_average_precision_matches_scikit_learn_with_ties():
    sk = pytest.importorskip("sklearn.metrics")
    from fdia_analysis.formulas.metrics import average_precision

    rng = np.random.default_rng(3)
    for _ in range(5):
        score = np.round(rng.random(200), 1)  # many ties
        truth = rng.random(200) < 0.3
        assert average_precision(score, truth) == pytest.approx(
            sk.average_precision_score(truth, score), abs=1e-12
        )
    with pytest.raises(ValueError, match="positive"):
        average_precision(np.ones(3), np.zeros(3, bool))


def test_perbus_rates_by_hand():
    from fdia_analysis.formulas.metrics import perbus_rates

    pred = np.array([[1, 0], [1, 0], [0, 0], [1, 1]], bool)
    truth = np.array([[1, 0], [0, 0], [1, 0], [1, 0]], bool)
    f1, dr, fr = perbus_rates(pred, truth)
    # bus 0: TP 2, FP 1, FN 1, TN 0; bus 1: TP 0, FP 1, FN 0, TN 3
    assert np.allclose(f1, [4 / 6, 0]) and np.allclose(dr, [2 / 3, 0]) and np.allclose(fr, [1.0, 0.25])
