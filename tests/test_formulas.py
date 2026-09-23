"""The formulas kernel: each function on a case small enough to check by hand, and the estimator's
torch measurement function pinned to the numpy kernel so there is one AC model in the package."""

import numpy as np
import pytest

from fdia_graph.formulas import (
    BranchModel,
    bias_jitter_split,
    branch_admittances,
    branch_flows,
    bus_injections,
    complex_voltages,
    ramp_profile,
    recent_change_scale,
    series_admittance,
    swing_zscore,
    temporal_delta,
)


def test_series_admittance_by_hand():
    ys = series_admittance(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.5, 0.0]))
    assert ys[0] == pytest.approx(0.5 - 0.5j)  # 1 / (1 + j) = (1 - j) / 2
    assert ys[1] == pytest.approx(-2j)  # 1 / (0.5 j)
    assert ys[2] == 0  # no impedance: no series path


def test_two_bus_line_flow_and_injection():
    """One line of reactance 0.1 pu between two buses, 1 pu volts, 0.1 rad apart: the from-end
    active flow is sin(0.1)/0.1 pu and the two injections are the sending and receiving ends."""
    r, x = np.array([0.0]), np.array([0.1])
    zeros = np.zeros(1)
    Y, Yf, Yt = branch_admittances(
        BranchModel(r, x, zeros, zeros, np.ones(1), zeros), np.array([[0], [1]]), 2
    )
    V = complex_voltages(np.array([1.0, 1.0]), np.rad2deg(np.array([0.1, 0.0])))
    Sf = branch_flows(V, Yf, np.array([0]))
    S = bus_injections(V, Y)
    assert Sf[0].real == pytest.approx(np.sin(0.1) / 0.1)
    assert S[0].real == pytest.approx(np.sin(0.1) / 0.1)  # sending end injects what the line carries
    assert S[1].real == pytest.approx(-np.sin(0.1) / 0.1)  # receiving end absorbs it (lossless)
    assert (Yf + Yt).sum() == pytest.approx(0)  # from-end and to-end currents cancel on a lossless line


def test_branch_flows_stack_matches_single():
    rng = np.random.default_rng(0)
    N, E = 5, 7
    ei = np.array([rng.integers(0, N, E), rng.integers(0, N, E)])
    ei[1] = (ei[0] + 1 + rng.integers(0, N - 1, E)) % N  # no self loops
    branch = BranchModel(
        rng.uniform(0.01, 0.1, E),
        rng.uniform(0.05, 0.5, E),
        np.zeros(E),
        np.zeros(E),
        np.ones(E),
        np.zeros(E),
    )
    Y, Yf, Yt = branch_admittances(branch, ei, N)
    V = complex_voltages(rng.uniform(0.95, 1.05, (3, N)), rng.uniform(-5, 5, (3, N)))
    stack = branch_flows(V, Yf, ei[0], 100.0)
    for t in range(3):
        assert np.allclose(stack[t], branch_flows(V[t], Yf, ei[0], 100.0), rtol=1e-12, atol=1e-12)


def test_loader_admittances_match_kernel(timeline):
    import fdia_graph as fg

    ds = fg.load(timeline)
    p = ds._phys
    branch = BranchModel(
        p["edge_r"], p["edge_x"], p["edge_b"], p["edge_g"], p["edge_tap"], p["edge_shift"], p["edge_status"]
    )
    Y, Yf, Yt = branch_admittances(
        branch, ds.edge_index_np, ds.N, p["bus_shunt_g"], p["bus_shunt_b"], ds.baseMVA
    )
    assert np.array_equal(Y, ds.ybus_np) and np.array_equal(Yf, ds.yf_np) and np.array_equal(Yt, ds.yt_np)


def test_estimator_measurement_function_is_the_kernel(splits):
    """SEBase._h is `ac_measurement`, and the torch twin `_h_t` (kept for callers that differentiate
    through it) equals it on real states."""
    torch = pytest.importorskip("torch")
    from fdia_graph.se import WLS

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

    from fdia_graph.se import WLS

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
    from fdia_graph.se import AdaptiveWeighting

    est = AdaptiveWeighting(c=1.5, npass=2).fit(splits["train"])
    scores = est.score(splits["test"])
    assert np.isfinite(scores.geo.angle_mae_deg) and not hasattr(est, "_Ybus")


def test_bias_jitter_split_keeps_the_class_total():
    jitter, bias = bias_jitter_split({"v": 0.0012, "pf": 0.017}, jitter_frac=0.25)
    assert jitter["v"] == pytest.approx(0.0003) and bias["pf"] == pytest.approx(0.017 * (1 - 0.0625) ** 0.5)
    for k in ("v", "pf"):
        assert bias[k] ** 2 + jitter[k] ** 2 == pytest.approx({"v": 0.0012, "pf": 0.017}[k] ** 2)


def test_ramp_profile_shape():
    """Rise for 4 steps at 0.5, hold 2, return at 0.25, never below zero."""
    devs = [ramp_profile(i, 4, 2, 0.5, 0.25) for i in range(12)]
    assert devs[:4] == [0.0, 0.5, 1.0, 1.5]
    assert devs[4:6] == [2.0, 2.0]
    assert devs[6:] == pytest.approx([2.0 - 0.25 * k for k in range(6)])
    assert ramp_profile(100, 4, 2, 0.5, 0.25) == 0.0


def test_temporal_features_by_hand():
    prev = np.array([[1.0, 10.0, 5.0, 0.0], [1.0, 20.0, 8.0, 0.0]])
    nx = np.array([[1.0, 13.0, 4.0, 0.0], [1.0, 21.0, 8.0, 0.0]])
    metered = np.array([True, False])
    td = temporal_delta(nx, prev, metered)
    assert td.tolist() == [[3.0, -1.0], [0.0, 0.0]]  # unmetered bus stays zero
    sw = swing_zscore(nx, prev, np.array([[1.5, 0.5], [1.0, 1.0]]), metered)
    assert sw.tolist() == [[2.0, -2.0], [0.0, 0.0]]


def test_recent_change_scale_matches_windowed_std():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3, 4))
    scale = recent_change_scale(X, 10, 3)
    D = np.abs(np.diff(X[:, :, 1:3], axis=0))
    t = 25  # window covers D[15 .. 23], ten changes strictly before t
    expected = D[15:24].std(axis=0) + 1e-3
    assert np.allclose(scale[t], expected, rtol=1e-5)
    assert np.all(scale[:4] == np.float32(1e-3))  # too few changes: the floor alone


# ---- estimation, linalg and projection kernels ---------------------------------------------------


def _toy_system():
    """Two states, four measurements with a hand-checkable Jacobian and weights."""
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    w = np.array([1.0, 1.0, 4.0, 0.25])
    return H, w


def test_normal_matrix_and_wls_step_by_hand():
    from fdia_graph.formulas import guarded_inverse, normal_matrix, wls_step, wls_step_batched

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
    from fdia_graph.formulas import (
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
    from fdia_graph.formulas import huber_weights

    a = huber_weights(np.array([[0.5, 1.5, 3.0, 0.0]]), c=1.5)
    assert a.tolist() == [
        [1.0, 1.0, 0.5, 1.0]
    ]  # inside the band 1; outside c / r; a zero residual is floored


def test_whitened_svd_basis_is_orthonormal():
    from fdia_graph.formulas import whitened_svd_basis

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)) * np.array([3.0, 0.3, 0.03, 0.003, 0.0003, 0.00003])
    K, VK = whitened_svd_basis(X, 0.5)
    assert K == 3 and VK.shape == (6, 3)
    assert np.allclose(VK.T @ VK, np.eye(3), atol=1e-12)


def test_gate_weights_and_bus_incidence():
    from fdia_graph.formulas import bus_incidence, gate_weights

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
    from fdia_graph.formulas import (
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
    from fdia_graph.formulas import condition_number, guarded_inverse

    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    assert np.allclose(guarded_inverse(A), np.linalg.inv(A))
    assert condition_number(A) == pytest.approx(np.linalg.cond(A), rel=1e-6)
    S = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular: pseudo-inverse, infinite condition
    assert np.allclose(guarded_inverse(S), np.linalg.pinv(S))
    assert condition_number(S) == float("inf")


def test_bus_load_undoes_the_pools_common_scale():
    from fdia_graph.formulas.attacks import bus_load

    # bus 0: load 30, gen 50, both at scale 1.2 -> injection 1.2 * (30 - 50) = -24, load 36
    # bus 1: load only, 10 at scale 0.8 -> injection 8, load 8; bus 2: gen only, 40 at 1.1 -> load 0
    load_base = np.array([[30.0, 5.0], [10.0, 2.0], [0.0, 0.0]])
    gen_base = np.array([[50.0, 0.0], [0.0, 0.0], [40.0, 0.0]])
    X = np.zeros((3, 4))
    X[:, 1] = [-24.0, 8.0, -44.0]
    assert np.allclose(bus_load(X, load_base, gen_base), [36.0, 8.0, 0.0])
    # the base-generation shortcut the engine used is wrong where a generator sits
    assert not np.isclose(X[0, 1] + gen_base[0, 0], 36.0)


def test_element_loads_split_a_bus_by_base_shares():
    from fdia_graph.formulas.attacks import element_loads

    # bus 2 holds two loads, 10 and 30 MW base, and scans at 60 MW: they read 15 and 45, not 60 each
    bus_p = np.array([0.0, 8.0, 60.0])
    load_bus = np.array([1, 2, 2])
    p0 = np.array([10.0, 10.0, 30.0])
    assert np.allclose(element_loads(bus_p, load_bus, p0), [8.0, 15.0, 45.0])
    assert element_loads(np.array([5.0]), np.array([0]), np.array([0.0]))[0] == 0.0  # no base P: no share
