"""The analysis toolkit's part of tests/test_pmu_channels.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
The hybrid meter model: angles at the PMU buses only, the PMU branch-current
phasors [WU26 eqs. 19-20] emitted, stored, loaded and attacked, and the eq. (3) pseudo-measurements
an estimator can derive from them. The v0.8.3 meter model stays the default and writes none of it."""

import numpy as np
import pytest

from fdia_analysis.formulas.estimation import pmu_pseudo_links, pmu_pseudo_voltages
from fdia_analysis.models.config import SolveConfig
from fdia_graph.constants.channels import NodeChannel
from fdia_graph.formulas.network import branch_currents

pytest.importorskip("pandapower")


def _case14():
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.pypower.makeYbus import makeYbus

    net = pn.case14()
    pp.runpp(net)
    ppc = net._ppc
    _, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    V = (
        ppc["internal"]["V"]
        if "internal" in ppc
        else ppc["bus"][:, 7] * np.exp(1j * np.deg2rad(ppc["bus"][:, 8]))
    )
    f, t = ppc["branch"][:, 0].real.astype(int), ppc["branch"][:, 1].real.astype(int)
    Yf, Yt = np.asarray(Yf.todense()), np.asarray(Yt.todense())
    pmu = np.zeros(len(V), bool)
    pmu[[1, 5, 8]] = True  # a transformer end (bus 5, IEEE-14's 4-6 tap) among them
    cm = np.zeros((len(f), 4))
    cm[pmu[f], 0:2] = 1
    cm[pmu[t], 2:4] = 1
    return V, Yf, Yt, f, t, pmu, cm


def test_eq3_is_exact_without_noise():
    # [WU26 eq. 3]
    V, Yf, Yt, f, t, pmu, cm = _case14()
    links = pmu_pseudo_links(pmu, cm, f, t, Yf, Yt)
    pv = pmu_pseudo_voltages(
        links, len(V), np.abs(V), np.angle(V), branch_currents(V, Yf, Yt) * cm, 0.0, 0.0, 0.0
    )
    r = pv.reached
    assert r.sum() >= 5 and not r[pmu].any()
    np.testing.assert_allclose(pv.v[r], np.abs(V)[r], atol=1e-12)
    np.testing.assert_allclose(pv.theta[r], np.angle(V)[r], atol=1e-12)
    assert np.isnan(pv.v[~r]).all()


def test_eq3_noise_matches_its_propagated_variance():
    V, Yf, Yt, f, t, pmu, cm = _case14()
    links = pmu_pseudo_links(pmu, cm, f, t, Yf, Yt)
    cur = branch_currents(V, Yf, Yt) * cm
    sv, st, si = 2e-3, 3e-3, 4e-3
    rng = np.random.default_rng(0)
    n, N = 20000, len(V)
    vv = np.abs(V) + rng.normal(0, sv, (n, N))
    tt = np.angle(V) + rng.normal(0, st, (n, N))
    cc = cur + rng.normal(0, si, (n, *cur.shape)) * cm
    mc = pmu_pseudo_voltages(links, N, vv, tt, cc, sv, st, si)
    an = pmu_pseudo_voltages(links, N, np.abs(V), np.angle(V), cur, sv, st, si)
    r = an.reached
    # 20000 draws: the sample variance is within about 3% of the true one (2 sigma of its own spread)
    np.testing.assert_allclose(mc.v[:, r].var(0), an.var_v[r], rtol=0.05)
    np.testing.assert_allclose(mc.theta[:, r].var(0), an.var_theta[r], rtol=0.05)


def test_pmu_pseudo_fills_only_unmetered_slots_and_needs_currents(pmu_files):
    from fdia_analysis.se.methods import WLS
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.errors import MissingCapability

    old, new = pmu_files
    with pytest.raises(MissingCapability, match="PMU currents"):
        WLS(solve=SolveConfig(pmu_pseudo=True)).fit(FdiaGraph(old, split="train"))
    plain = WLS().fit(FdiaGraph(new, split="train"))
    est = WLS(solve=SolveConfig(pmu_pseudo=True)).fit(FdiaGraph(new, split="train"))
    assert est.m == plain.m + len(est._pseudo_v) + len(est._pseudo_th) and len(est._pseudo_th) > 0
    x = est.estimate(FdiaGraph(new, split="test"))
    assert np.isfinite(x).all()


def test_jacobian_weighting_takes_the_pseudo_measurements(pmu_files):
    from fdia_analysis.se.methods import JacobianWeighting
    from fdia_graph.dataset import FdiaGraph

    _, new = pmu_files
    est = JacobianWeighting(solve=SolveConfig(pmu_pseudo=True)).fit(FdiaGraph(new, split="train"))
    assert len(est._pseudo_th) > 0
    assert np.isfinite(est.estimate(FdiaGraph(new, split="test"))).all()


@pytest.mark.parametrize("method", ["WLS", "JacobianWeighting"])
def test_pmu_pseudo_with_the_measured_calibration(pmu_files, method):
    """The measured calibration weights each pseudo slot by eq. (3)'s propagated noise: `_pseudo_sig`
    is set from the calibration scans before the accuracy classes, and `_class_sigma` places it at
    the pseudo slots; fit and estimate run, WLS and JacobianWeighting (which reads prev_pmu_i)."""
    from fdia_analysis.se import methods
    from fdia_graph.dataset import FdiaGraph

    _, new = pmu_files
    est = getattr(methods, method)(solve=SolveConfig(pmu_pseudo=True)).fit(
        FdiaGraph(new, split="train"), calibrate="measured"
    )
    n_pseudo = len(est._pseudo_v) + len(est._pseudo_th)
    assert est._pseudo_sig.shape == (n_pseudo,) and np.isfinite(est._pseudo_sig).all()
    assert (est._pseudo_sig > 0).all()
    np.testing.assert_array_equal(est.sig[est._pseudo_pos], np.maximum(est._pseudo_sig, 1e-9))
    x = est.estimate(FdiaGraph(new, split="test"))
    assert np.isfinite(x).all()


def test_estimate_builds_the_pseudo_measurements_chunk_by_chunk(pmu_files):
    """Review fix: estimate builds the measurement vectors, pseudo slots included, one chunk at a
    time and reads the pseudo values alone. The result is the whole-split path's, bit for bit at the
    same chunk, and the same to solver tolerance at any chunk size."""
    from fdia_analysis.se.methods import WLS
    from fdia_graph.dataset import FdiaGraph

    _, new = pmu_files
    train, test = FdiaGraph(new, split="train"), FdiaGraph(new, split="test")
    est = WLS(solve=SolveConfig(pmu_pseudo=True)).fit(train)
    d = test.export(["node_x", "edge_x", "pmu_i"])
    whole = est._estimate_arrays(est._z_of(d["node_x"], d["edge_x"], d["pmu_i"]), None, 1000)
    np.testing.assert_array_equal(est.estimate(test, chunk=1000), whole)
    np.testing.assert_allclose(est.estimate(test, chunk=7), whole, rtol=0, atol=1e-10)
    # the values the solve reads are the propagating formula's, without its covariances
    from fdia_analysis.formulas.estimation import pmu_pseudo_phasors

    pv = est._pseudo_of(d["node_x"], d["pmu_i"])
    V_f = pmu_pseudo_phasors(
        est._links,
        est.N,
        d["node_x"][:, :, NodeChannel.V],
        np.deg2rad(d["node_x"][:, :, NodeChannel.THETA]),
        d["pmu_i"],
    )
    np.testing.assert_array_equal(np.abs(V_f), pv.v)
    np.testing.assert_array_equal(np.angle(V_f), pv.theta)


def test_a_file_in_another_layout_is_refused_when_opened(pmu_files, tmp_path):
    """A file that declares currents as magnitude and angle, or flows at both branch ends, is refused
    by a named error when it is opened, before a column of it is read as something else."""
    import shutil

    import h5py

    from fdia_graph import schema
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.errors import NotATimeline

    _, rectangular = pmu_files
    FdiaGraph(rectangular)  # the one layout opens
    for attr, value, says in (
        (schema.Attr.CURRENT_FORM, "polar", "real and imaginary part"),
        (schema.Attr.EDGE_FEAT, "P_from,Q_from,P_to,Q_to", "flows at the from end"),
    ):
        other = tmp_path / f"{attr}.h5"
        shutil.copy(rectangular, other)
        with h5py.File(other, "a") as f:
            f.attrs[attr] = value
        with pytest.raises(NotATimeline, match=says):
            FdiaGraph(other)


def test_an_estimator_refuses_a_file_of_no_ieee_case(pmu_files, tmp_path):
    """The estimators build their network from the IEEE case of the file's `system`; a file of another grid
    (a New York file, 140 buses) is refused by a named error, not a KeyError."""
    import shutil

    import h5py

    from fdia_analysis.se.methods import WLS
    from fdia_graph import schema
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.errors import MissingCapability

    other = tmp_path / "other.h5"
    shutil.copy(pmu_files[1], other)
    with h5py.File(other, "a") as f:
        f.attrs[schema.Attr.SYSTEM] = 140
    with pytest.raises(MissingCapability, match="an IEEE case"):
        WLS().fit(FdiaGraph(other, split="train"))
