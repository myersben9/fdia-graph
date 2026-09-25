"""Estimator behaviour the frozen scores cannot see: the per-unit guard on every public path, one
estimator whatever calls it, the weighted solve keeping its best step, classical one-at-a-time
residual removal, and the spread-out meter-sigma calibration."""

import numpy as np
import pytest

import fdia_graph as fg

pytest.importorskip("pandapower")


@pytest.fixture(scope="module")
def wls(splits):
    from fdia_graph.se import WLS

    return WLS().fit(splits["train"])


@pytest.fixture(scope="module")
def test_arrays(splits, wls):
    d = splits["test"].export(["node_x", "edge_x", "clean", "family"])
    z = wls._z_of(d["node_x"], d["edge_x"])
    thsl = wls._truth_of(d["clean"])["thsl"]
    return z, thsl, d["family"]


def test_every_public_path_refuses_a_per_unit_view(timeline, splits, wls):
    from fdia_graph.localization import ResidualLocalizer
    from fdia_graph.se import JacobianFeatures, JacobianWeighting
    from fdia_graph.trust import TrustedMeters

    pu = fg.load(timeline, split="test", units="pu")
    with pytest.raises(ValueError, match="units='physical'"):
        wls.estimate(pu)
    with pytest.raises(ValueError, match="units='physical'"):
        wls.score(pu)
    with pytest.raises(ValueError, match="units='physical'"):
        ResidualLocalizer(estimator=wls).fit(splits["train"]).scores(pu)
    with pytest.raises(ValueError, match="units='physical'"):
        JacobianFeatures().fit(fg.load(timeline, split="train", units="pu"))
    with pytest.raises(ValueError, match="units='physical'"):
        JacobianWeighting().fit(splits["train"]).estimate(pu)
    with pytest.raises(ValueError, match="units='physical'"):
        TrustedMeters(k=2).fit(splits["train"]).score(pu)


def test_residual_localizer_keeps_a_fitted_estimator(splits, wls):
    from fdia_graph.localization import ResidualLocalizer

    H = wls.H
    ResidualLocalizer(estimator=wls).fit(splits["val"])
    assert wls.H is H  # not refitted on the localizer's split


def test_a_composed_gated_estimator_is_the_gated_estimator(splits):
    from fdia_graph.localization import ResidualLocalizer
    from fdia_graph.se import GatedPrior

    est = GatedPrior(gate="oracle", rank_frac=0.5, reweight="huber").fit(splits["train"])
    loc = ResidualLocalizer(estimator=est).fit(splits["train"])
    test = splits["test"]
    d = test.export(["node_x", "edge_x", "clean"])
    z = est._z_of(d["node_x"], d["edge_x"])
    thsl = est._truth_of(d["clean"])["thsl"]
    xhat = est.estimate(test)
    expected = np.stack([est._nres(xhat, z, thsl)[:, ix].max(axis=1) for ix in loc._inc], axis=1)
    assert np.allclose(loc.scores(test), expected)


def test_weighted_solve_considers_its_last_step(wls, test_arrays):
    from fdia_graph.formulas.estimation import weighted_objective

    # With uniform weights both solves take the same steps; the guarded one keeps the lowest-objective
    # iterate (the chord iteration wobbles at convergence), which must include the plain solve's last.
    z, thsl, _ = test_arrays
    w = np.broadcast_to(wls.Wk, z.shape)
    J_w = weighted_objective(z - wls._h(wls._w_solve(z, w, thsl), thsl), w)
    J_p = weighted_objective(z - wls._h(wls._solve_plain(z, thsl), thsl), w)
    assert (J_w <= J_p + 1e-9).all()


def test_removal_takes_out_one_gross_error_and_keeps_its_neighbours(splits, test_arrays):
    from fdia_graph.se import ResidualRemoval

    est = ResidualRemoval(threshold=4.0).fit(splits["train"])
    z, thsl, fam = test_arrays
    z, thsl = z[fam == 0].copy(), thsl[fam == 0]

    def removable(m):  # not critical, and the observability guard lets it go
        w = est.Wk.copy()
        w[m] = 0.0
        return not est.critical[m] and est._observable(w)

    j = next(m for m in range(len(est.Wk)) if removable(m))
    z[:, j] += 50.0 * est.sig[j]  # one gross error per record
    keep = np.ones_like(z)
    keep[:, j] = 0.0
    only_j = est._w_solve(z, est.Wk * keep, thsl)  # the answer with exactly that meter removed
    # the frames where nothing else crosses the threshold once it is gone: one removal is the answer
    rest = np.where(keep > 0, np.abs(est._nres(only_j, z, thsl)), 0.0).max(axis=1) < est.threshold
    assert rest.sum() >= 5
    z, thsl, only_j = z[rest][:5], thsl[rest][:5], only_j[rest][:5]
    assert np.abs(est._solve(z, thsl) - only_j).max() < 1e-9


def test_meter_sigma_is_calibrated_across_the_benign_set(splits):
    from fdia_graph.se import WLS

    train = splits["train"]
    d = train.export(["node_x", "edge_x", "clean", "family"])
    ben = np.flatnonzero(d["family"] == 0)
    est = WLS().fit(train, n_calib=5)
    c = ben[np.linspace(0, len(ben) - 1, 5).round().astype(int)]  # first to last, evenly
    assert c[0] == ben[0] and c[-1] == ben[-1]
    tr = est._truth_of(d["clean"][c])
    r = est._z_of(d["node_x"][c], d["edge_x"][c]) - est._h(tr["x"], tr["thsl"])
    assert np.allclose(est.sig, np.maximum(np.sqrt((r**2).mean(axis=0)), 1e-9))


def test_fit_refuses_a_dataset_whose_slack_disagrees(timeline):
    from fdia_graph.se import WLS

    ds = fg.load(timeline, split="train")
    ds.slack = 3
    with pytest.raises(ValueError, match="slack"):
        WLS().fit(ds)


def test_the_angle_reference_is_one_constant_from_the_fit(timeline):
    """fit takes the reference from the training split and refuses a slack angle that varies."""

    from fdia_graph.se import WLS

    est = WLS().fit(fg.load(timeline, split="train"))
    clean = fg.load(timeline, split="test").export(["clean"])["clean"]
    assert np.allclose(np.deg2rad(clean[:, est.slack, 3]), est.theta_ref)
    with pytest.raises(ValueError, match="varies"):
        est._fit_reference(np.array([0.0, 0.1]))


def test_measured_calibration_reads_no_clean_layer(timeline, tmp_path):
    """calibrate="measured" fits from equipment data and measurements: rewriting the clean layer of the
    training file leaves the fitted model unchanged, and the sigma is the accuracy class."""
    import shutil

    import h5py

    from fdia_graph import schema
    from fdia_graph.engine.core import ACCURACY_CLASS
    from fdia_graph.se import WLS

    a = WLS().fit(fg.load(timeline, split="train"), calibrate="measured")
    copy = str(tmp_path / "scrambled.h5")
    shutil.copyfile(fg.load(timeline).path, copy)
    with h5py.File(copy, "r+") as f:
        f[schema.NODE_CLEAN][...] = f[schema.NODE_CLEAN][:] * 1.5 + 0.1
    from fdia_graph import FdiaGraph

    b = WLS().fit(FdiaGraph(copy, split="train"), calibrate="measured")
    assert np.array_equal(a.sig, b.sig) and np.array_equal(a.xmean, b.xmean) and a.theta_ref == b.theta_ref
    v = a.mask[: a.N].sum()  # the metered |V| slots come first
    assert np.allclose(a.sig[:v], ACCURACY_CLASS["v"])
    with pytest.raises(ValueError, match="calibrate"):
        WLS().fit(fg.load(timeline, split="train"), calibrate="oracle")


def test_measured_calibration_fits_without_a_clean_layer(timeline, tmp_path):
    """A file with no clean layer fits with calibrate="measured", the subspace prior included (its
    calibration passes solve in the full state before the subspace exists), and still refuses the
    truth calibration."""
    import shutil

    import h5py

    from fdia_graph import FdiaGraph, schema
    from fdia_graph.se import WLS, SubspacePrior

    copy = str(tmp_path / "no_clean.h5")
    shutil.copyfile(fg.load(timeline).path, copy)
    with h5py.File(copy, "r+") as f:
        del f[schema.NODE_CLEAN], f[schema.EDGE_CLEAN]
    ds = FdiaGraph(copy, split="train")
    assert not ds.has_clean
    for est in (WLS(), SubspacePrior(reweight="huber")):
        est.fit(ds, calibrate="measured")
        assert np.isfinite(est.xmean).all() and not est._full_state
    assert SubspacePrior().fit(ds, calibrate="measured")._basis().shape[1] < 2 * ds.N - 1
    with pytest.raises(ValueError, match="clean layer"):
        WLS().fit(ds)
