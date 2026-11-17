"""The analysis toolkit's part of tests/test_sdk.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
Behavioral tests for the SDK surface: the contracts the docs promise, checked on a tiny timeline."""

import numpy as np
import pytest

import fdia_graph as fg
from fdia_analysis.models.config import (
    GateConfig,
    JacobianWeightingConfig,
    LearnedConfig,
    LocalizerConfig,
    PriorConfig,
    TrainerConfig,
)

# ---- features ----------------------------------------------------------------------------------


def test_kcl_residual_matches_the_papers_builder(splits):
    from fdia_analysis.localization.learned import full14, kcl_residual

    d = splits["test"].export(["node_x", "node_m", "edge_x", "temporal_delta", "swing"])
    nx, ex, ei = d["node_x"].astype(float), d["edge_x"].astype(float), d["edge_index"]
    n, N = nx.shape[:2]
    inP, inQ = np.zeros((n, N)), np.zeros((n, N))
    np.add.at(inP, (slice(None), ei[1]), ex[:, :, 0])
    np.add.at(inP, (slice(None), ei[0]), -ex[:, :, 0])
    np.add.at(inQ, (slice(None), ei[1]), ex[:, :, 1])
    np.add.at(inQ, (slice(None), ei[0]), -ex[:, :, 1])
    ref = np.stack([inP - nx[:, :, 1], inQ - nx[:, :, 2]], -1)
    assert np.allclose(kcl_residual(nx, ex, ei), ref)
    assert full14(d).shape == (n, N, 14)


# ---- localization ------------------------------------------------------------------------------


def test_threshold_localizer_protocol(timeline, splits):
    from fdia_analysis.localization.methods import SwingThreshold

    loc = SwingThreshold(localizer=LocalizerConfig(fa_target=0.05)).fit(splits["train"])
    flags = loc.localize(splits["test"])
    assert flags.shape == (len(splits["test"]), splits["test"].N) and flags.dtype == bool
    rep = loc.score(splits["test"])
    assert {"all", "benign"} <= set(rep)
    for k in ("macro_f1", "macro_dr", "macro_fr", "node_f1"):
        assert 0.0 <= rep["all"][k] <= 1.0
    fam = [k for k in rep if k not in ("all", "benign")][0]
    assert {"strict_acc", "node_f1", "macro_f1", "sample_f1", "detection_rate"} <= set(rep[fam])
    # The pooled entry survives a benign-only evaluation set.
    ben_only = fg.load(timeline, split="test", families=[0])
    assert loc.score(ben_only)["all"]["macro_f1"] == 0.0
    # Scoring from precomputed scores (the docs cache path) equals scoring from scratch.
    assert loc.score(splits["test"], scores=loc.scores(splits["test"])) == rep
    with pytest.raises(ValueError):
        loc.score(splits["test"], scores=np.zeros((3, splits["test"].N)))


def test_learned_localizer_smoke(splits):
    pytest.importorskip("torch")
    from fdia_analysis.localization.learned import BusCNN, BusMLP

    for cls in (BusMLP, BusCNN):
        loc = cls(learned=LearnedConfig(epochs=2, device="cpu")).fit(splits["train"], val=splits["val"])
        assert loc.tau is not None and 0.0 < loc.tau < 1.0
        s = loc.scores(splits["test"])
        assert s.shape == (len(splits["test"]), splits["test"].N)
        assert np.all((s >= 0.0) & (s <= 1.0))
        assert "all" in loc.score(splits["test"])


# ---- state estimation --------------------------------------------------------------------------


def test_wls_estimates_the_classical_state(splits):
    pytest.importorskip("torch")
    from fdia_analysis.se.methods import WLS

    est = WLS().fit(splits["train"])
    x = est.estimate(splits["test"])
    N = splits["test"].N
    assert x.shape == (len(splits["test"]), 2 * N - 1)
    assert est.slack == splits["test"].slack
    rep = est.score(splits["test"])
    assert "geo" in rep and rep["geo"]["angle_mae_deg"] < 1.0
    assert rep["benign"]["voltage_mae_pu"] < 0.01
    # Scoring from precomputed estimates (the docs cache path) equals scoring from scratch.
    rep2 = est.score(splits["test"], xhat=x)
    assert all(np.isclose(rep2[f][k], rep[f][k], rtol=1e-6) for f in rep for k in rep[f])
    with pytest.raises(ValueError):
        est.score(splits["test"], xhat=x[:3])


# ---- Jacobian-informed features ----------------------------------------------------------------


def test_jacobian_features_split_stealthy_from_corruption():
    """Unexplained energy fires on in-place corruption (Ad) and not on the stealthy re-solve (Aq);
    the explained energy fires on an Aq episode's first frame, where the previous frame's estimate
    is still benign (later frames of the episode are measured against an estimate the false state
    already moved). The digest's central claims, with only observed data in the feature, checked on
    the v0.8.3 release that carries both families (the generator no longer makes them)."""
    pytest.importorskip("torch")
    from fdia_analysis.se.jacobian import JacobianFeatures
    from fdia_graph.dataset import FdiaGraph
    from tests.helpers import TIMELINE_V083

    jf = JacobianFeatures().fit(FdiaGraph(TIMELINE_V083, split="train"))
    d = FdiaGraph(TIMELINE_V083).export(
        ["node_x", "edge_x", "prev_node_x", "prev_edge_x", "prev_timestep", "family", "y"]
    )
    F = jf.transform(d)
    n, N = d["node_x"].shape[:2]
    assert F["bus"].shape == (n, N, 8) and F["global"].shape == (n, 4)
    assert np.isfinite(F["bus"]).all() and np.isfinite(F["global"]).all()
    assert jf.kappa > 1.0 and len(jf.singular_values) == jf.est.SD
    fam = d["family"]
    ben, aq, ad = fam == 0, fam == 1, fam == 2
    onset = aq & (np.r_[0, fam[:-1]] != 1)  # an Aq frame whose previous frame was not Aq
    q_perp, q_par = F["global"][:, 0], F["global"][:, 1]
    assert onset.any()
    assert np.median(q_perp[ad]) > 2 * np.median(q_perp[ben])  # corruption leaves an unexplained part
    assert np.median(q_perp[aq]) < 2 * np.median(q_perp[ben])  # a stealthy re-solve does not
    assert np.median(q_par[onset]) > 2 * np.median(q_par[ben])  # but its onset moves the explained part


def test_learned_localizer_feature_sets(splits):
    pytest.importorskip("torch")
    from fdia_analysis.localization.learned import FEATURE_SETS, BusMLP

    for feats, width in FEATURE_SETS.items():
        loc = BusMLP(learned=LearnedConfig(epochs=1, device="cpu", features=feats)).fit(splits["train"])
        assert loc.n_feat == width and loc.mu.shape == (width,)
        assert loc.scores(splits["test"]).shape == (len(splits["test"]), splits["test"].N)
    with pytest.raises(ValueError):
        BusMLP(learned=LearnedConfig(features="nope"))


def test_jacobian_weighting_estimator():
    """On the v0.8.3 release, which carries the in-place bias Ad: the weights never raise a meter,
    and the estimate is never worse than WLS on the bias."""
    pytest.importorskip("torch")
    from fdia_analysis.se.methods import WLS, JacobianWeighting
    from fdia_graph.dataset import FdiaGraph
    from tests.helpers import TIMELINE_V083

    splits = {s: FdiaGraph(TIMELINE_V083, split=s) for s in ("train", "test")}
    est = JacobianWeighting(weighting=JacobianWeightingConfig(c=3.0)).fit(splits["train"])
    w = est.weights(splits["test"])
    assert w.shape == (len(splits["test"]), est.m) and np.all(w <= est.Wk[None, :] + 1e-12)
    x = est.estimate(splits["test"])
    assert x.shape == (len(splits["test"]), 2 * splits["test"].N - 1)
    rep = est.score(splits["test"], xhat=x)
    base = WLS().fit(splits["train"]).score(splits["test"])
    assert rep["geo"]["angle_mae_deg"] < 1.0
    assert rep["Ad"]["angle_mae_deg"] <= base["Ad"]["angle_mae_deg"] * 1.05  # never worse on bias


def test_gated_prior_uses_the_gate(splits):
    pytest.importorskip("torch")
    from fdia_analysis.se.methods import GatedPrior

    est = GatedPrior(
        GateConfig(gate="oracle"), prior=PriorConfig(rank_frac=0.5, reweight="huber", c=1.5)
    ).fit(splits["train"])
    w = est.gated_weights(splits["test"])
    y = splits["test"].export(["y"])["y"].astype(bool)
    assert w.shape == (len(splits["test"]), est.m)
    assert np.all(w[~y.any(axis=1)] == est.Wk[None, :])  # benign records keep the full weights
    assert (w[y.any(axis=1)] < est.Wk[None, :]).any(axis=1).all()  # attacked ones lose some
    x = est.estimate(splits["test"])
    assert x.shape == (len(splits["test"]), 2 * splits["test"].N - 1)
    with pytest.raises(ValueError):
        GatedPrior(GateConfig(gate=None))


def test_local_trainer_runs_persist_and_honour_owned_and_clip():
    """Two one-epoch runs equal one two-epoch run (optimizer and batch order persist); with `owned`
    the labels of the other buses cannot matter; a clip changes training only when it binds."""
    torch = pytest.importorskip("torch")
    from fdia_analysis.localization.learned import LocalTrainer, _mlp_net
    from fdia_analysis.models.training import OptimConfig

    was = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        _trainer_contract(torch, LocalTrainer, OptimConfig, _mlp_net)
    finally:
        torch.use_deterministic_algorithms(was)  # a process-wide switch: leave it as found


def _trainer_contract(torch, LocalTrainer, OptimConfig, _mlp_net):
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(64, 5, 14)).astype(np.float32)
    Y = (rng.random((64, 5)) < 0.3).astype(np.float32)
    cfg = OptimConfig(1e-2, 0.01, 16, 1.0)

    def train(Y, runs, owned=None, clip=None):
        torch.manual_seed(7)
        net = _mlp_net(14, 16, 2, 0.0)
        tr = LocalTrainer(trainer=TrainerConfig(clip=clip), net=net, cfg=cfg, dev="cpu", seed=3)
        Xt, Yt = tr.stage(Xs, Y)
        for e in runs:
            tr.run(Xt, Yt, e, owned=owned)
        return [v.clone() for v in net.state_dict().values()]

    def same(a, b):
        return all(torch.equal(x, y) for x, y in zip(a, b))

    assert same(train(Y, [2]), train(Y, [1, 1]))
    Y_other = Y.copy()
    Y_other[:, 3:] = 1 - Y_other[:, 3:]  # flip every label outside the first three buses
    assert same(train(Y, [2], owned=3), train(Y_other, [2], owned=3))
    assert not same(train(Y, [2]), train(Y_other, [2]))
    assert same(train(Y, [2], clip=1e9), train(Y, [2]))  # a clip that never binds changes nothing
    assert not same(train(Y, [2], clip=1e-4), train(Y, [2]))


def test_predict_runs_in_eval_mode_after_training():
    pytest.importorskip("torch")
    from fdia_analysis.localization.learned import _mlp_net, predict

    net = _mlp_net(14, 16, 2, 0.5)
    net.train()  # as a training run leaves it
    Xs = np.random.default_rng(0).normal(size=(8, 5, 14)).astype(np.float32)
    assert np.array_equal(predict(net, Xs, "cpu"), predict(net, Xs, "cpu"))  # dropout off: repeatable
    assert not net.training


def test_score_perbus_agrees_with_score_and_reports_the_paper_columns(splits):
    from fdia_analysis.localization.methods import SwingThreshold

    loc = SwingThreshold().fit(splits["train"])
    te = splits["test"]
    s = loc.scores(te)
    ref = loc.score(te, scores=s)["all"]
    pb = loc.score_perbus(te, scores=s, fr_over="benign")["all"]
    assert pb["macro_f1"] == pytest.approx(ref["macro_f1"]) and pb["macro_dr"] == pytest.approx(
        ref["macro_dr"]
    )
    assert pb["macro_fr"] == pytest.approx(ref["macro_fr"])
    everyone = loc.score_perbus(te, scores=s, fr_over="all")
    y = te.export(["y"])["y"].astype(bool)
    cols = everyone["all"]["bus_index"]
    pred = s[:, cols] > loc.thr[None, cols]
    fp, tn = (pred & ~y[:, cols]).sum(0), (~pred & ~y[:, cols]).sum(0)  # every non-attacked cell
    assert np.allclose(everyone["all"]["fr"], fp / np.maximum(fp + tn, 1))
    with pytest.raises(ValueError, match="scores must be"):
        loc.score_perbus(te, scores=s[:, :-1])
    fams = [k for k in ("Aq", "Ad", "As", "Ar", "At", "Al", "Am") if k in everyone]
    assert fams and all(len(everyone[k]["f1"]) == len(everyone["all"]["bus_index"]) for k in fams)
    with pytest.raises(ValueError, match="attackable"):
        loc.score_perbus(te, scores=s, buses="attackable")


def test_learned_localizer_grid_detection_and_attackable_table(splits, timeline):
    pytest.importorskip("torch")
    from fdia_analysis.localization.learned import BusMLP

    loc = BusMLP(learned=LearnedConfig(epochs=2, device="cpu")).fit(splits["train"], val=splits["val"])
    with pytest.raises(ValueError, match="tune_grid_threshold"):
        loc.score_grid(splits["test"])
    import fdia_graph as fg

    with pytest.raises(ValueError, match="both attacked and benign"):
        loc.tune_grid_threshold(fg.load(timeline, split="val", families=["benign"]))
    g = loc.tune_grid_threshold(splits["val"]).score_grid(splits["test"])
    assert 0.05 <= g["tau"] <= 0.95 and 0 <= g["false_alarm"] <= 1 and 0 <= g["detection_rate"] <= 1
    assert g["by_family"] and all(0 <= v <= 1 for v in g["by_family"].values())
    pb = loc.score_perbus(splits["test"], buses="attackable")["all"]
    assert pb["bus_index"].tolist() == np.flatnonzero(loc._attackable).tolist()
    loc.fit(splits["train"])  # a refit forgets the previous fit's grid threshold
    with pytest.raises(ValueError, match="tune_grid_threshold"):
        loc.score_grid(splits["test"])


def test_jacobian_features_never_read_the_true_state(timeline, splits):
    """The change is taken against the previous frame's estimate, referenced to the case's slack
    angle (a network parameter): the transform reads measurements only, and the case's reference is
    the angle every true state of the timeline carries at the slack."""
    pytest.importorskip("pandapower")
    from fdia_analysis.se.jacobian import JacobianFeatures

    jf = JacobianFeatures().fit(splits["train"])
    d = splits["test"].export(["node_x", "edge_x", "prev_node_x", "prev_edge_x", "prev_timestep"])
    before = jf.transform(d)["bus"]  # d holds the measurements and their frame index, no truth
    assert np.isfinite(before).all() and not hasattr(jf, "_pool")
    clean = splits["test"].export(["clean"])["clean"]
    assert np.allclose(np.deg2rad(clean[:, jf.est.slack, 3]), jf.est.theta_ref)
    empty = {k: v[:0] for k, v in d.items()}  # an empty view transforms to empty features
    assert jf.transform(empty)["bus"].shape == (0, before.shape[1], 8)


def test_previous_swing_feature_set(timeline, splits):
    """prev_swing is the swing of file row - 1, and "full14+prev" appends it after the papers' 14
    channels; the frame after a one-frame attack reads the attack's swing there, sign reversed."""
    pytest.importorskip("torch")
    from fdia_analysis.localization.learned import BusCNN, full14

    full = fg.load(timeline).export(["swing", "family"])
    ds = splits["test"]
    d = ds.export(["prev_swing"])
    assert np.array_equal(d["prev_swing"], full["swing"][np.maximum(ds.idx - 1, 0)])
    loc = BusCNN(learned=LearnedConfig(epochs=1, device="cpu", features="full14+prev")).fit(splits["train"])
    x = loc._pull(ds)
    F = loc._features(x)
    assert (
        F.shape[-1] == 16
        and np.array_equal(F[..., :14], full14(x))
        and np.array_equal(F[..., 14:], x["prev_swing"])
    )
