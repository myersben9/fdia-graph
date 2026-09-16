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
    series_admittance,
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


def test_loader_admittances_match_kernel(shard):
    import fdia_graph as fg

    ds = fg.load(shard)
    p = ds._phys
    branch = BranchModel(
        p["edge_r"], p["edge_x"], p["edge_b"], p["edge_g"], p["edge_tap"], p["edge_shift"], p["edge_status"]
    )
    Y, Yf, Yt = branch_admittances(
        branch, ds.edge_index_np, ds.N, p["bus_shunt_g"], p["bus_shunt_b"], ds.baseMVA
    )
    assert np.array_equal(Y, ds.ybus_np) and np.array_equal(Yf, ds.yf_np) and np.array_equal(Yt, ds.yt_np)


def test_estimator_measurement_function_is_the_kernel(splits):
    """SEBase._h_t (torch, kept for the autograd Jacobian) equals the numpy kernel on real states."""
    pytest.importorskip("torch")
    from fdia_graph.se import WLS

    est = WLS().fit(splits["train"])
    d = splits["test"].to_numpy(["clean"])
    tr = est._truth_of(d["clean"][:8])
    h_torch = est._h(tr["x"], tr["thsl"])  # [n, m] masked measurements in pu and rad
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
    assert np.allclose(h_torch, full[:, est.mask], rtol=1e-10, atol=1e-10)


def test_bias_jitter_split_keeps_the_class_total():
    jitter, bias = bias_jitter_split({"v": 0.0012, "pf": 0.017}, jitter_frac=0.25)
    assert jitter["v"] == pytest.approx(0.0003) and bias["pf"] == pytest.approx(0.017 * (1 - 0.0625) ** 0.5)
    for k in ("v", "pf"):
        assert bias[k] ** 2 + jitter[k] ** 2 == pytest.approx({"v": 0.0012, "pf": 0.017}[k] ** 2)
