"""The generation toolkit's formulas: each function on a case small enough to check by hand. The
analysis toolkit's formulas, and the estimator's torch measurement function pinned to the numpy kernel
here, are checked in analysis/tests/test_formulas.py."""

import numpy as np
import pytest

from fdia_graph.formulas.attacks import ramp_profile
from fdia_graph.formulas.network import (
    branch_admittances,
    branch_flows,
    bus_injections,
    complex_voltages,
    series_admittance,
)
from fdia_graph.formulas.noise import bias_jitter_split
from fdia_graph.formulas.temporal import recent_change_scale, swing_zscore, temporal_delta, wrap_degrees
from fdia_graph.models.grid import BranchModel


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


def test_flow_sensitivities_match_a_resolved_power_flow():
    """[MP19, dSbus_dV, dSbr_dV] One more unit of active power at a bus, taken at the slack, moves every
    branch's active flow by the bus's column of `flow_sensitivities`: checked against the AC power flow solved
    again at the moved injection (every bus but the slack holding its injections), on IEEE-14."""
    pytest.importorskip("pandapower")
    from scipy.optimize import fsolve

    from fdia_graph.constants.channels import NodeChannel
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.formulas.network import _dense, flow_sensitivities
    from fdia_graph.generation import _load_states

    g = FdiaGenerator(14, seed=1)
    X = _load_states(14, None)[0]
    lut = g._ppc_row[np.arange(g.C)]
    V0 = np.ones(np.shape(g._Ybus)[0], np.complex128)
    V0[lut] = complex_voltages(X[:, NodeChannel.V], X[:, NodeChannel.THETA])
    Y, F = _dense(g._Ybus), _dense(g._Yf)
    slack = int(lut[g.slack_bus])
    keep = np.flatnonzero(np.arange(len(V0)) != slack)
    sens = flow_sensitivities(V0, Y, F, g._from_bus_ppc, slack)
    assert sens.shape == (g.E, len(V0)) and not sens[:, slack].any()

    def state(x: np.ndarray) -> np.ndarray:
        th, vm = np.angle(V0).copy(), np.abs(V0).copy()
        th[keep], vm[keep] = x[: len(keep)], x[len(keep) :]
        return vm * np.exp(1j * th)

    S0, f0 = bus_injections(V0, Y), branch_flows(V0, F, g._from_bus_ppc).real
    start = np.r_[np.angle(V0)[keep], np.abs(V0)[keep]]
    eps = 1e-4
    for b in keep[::3]:
        target = S0.copy()
        target[b] += eps

        def mismatch(x: np.ndarray, target: np.ndarray = target) -> np.ndarray:
            S = bus_injections(state(x), Y) - target
            return np.r_[S.real[keep], S.imag[keep]]

        moved = branch_flows(state(fsolve(mismatch, start, xtol=1e-13)), F, g._from_bus_ppc).real
        assert np.allclose((moved - f0) / eps, sens[:, b], atol=1e-4)


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


def test_wrap_degrees_by_hand():
    """A difference across the +-180 wrap comes back small, 180 stays 180 and -180 becomes 180 (the interval
    is (-180, 180]); the dtype of the readings is kept."""
    d = np.array([0.0, 10.0, -10.0, 350.0, -350.0, 180.0, -180.0, 540.0])
    assert wrap_degrees(d).tolist() == [0.0, 10.0, -10.0, -10.0, 10.0, 180.0, 180.0, 180.0]
    assert wrap_degrees(d.astype(np.float32)).dtype == np.float32


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


def test_accuracy_class_sigma_is_relative_or_absolute():
    from fdia_graph.formulas.noise import accuracy_class_sigma

    s = accuracy_class_sigma(np.array([2.0, 2.0]), np.array([0.01, 0.003]), np.array([True, False]), 1e-5)
    assert np.allclose(s, [0.02 + 1e-5, 0.003])
