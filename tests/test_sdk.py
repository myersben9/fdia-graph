"""Behavioral tests for the SDK surface: the contracts the docs promise, checked on a tiny timeline."""

import numpy as np
import pytest

import fdia_graph as fg
from fdia_graph.registry import system_id

# ---- names and ids ----------------------------------------------------------------------------


def test_system_id_accepts_every_spelling():
    assert system_id("ieee118") == 118
    assert system_id("IEEE14") == 14
    assert system_id("300") == 300
    assert system_id(57) == 57
    with pytest.raises(ValueError):
        system_id("case14")


def test_family_codes_are_the_documented_ones():
    assert fg.FAMILIES == {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al", 7: "Am"}
    assert fg.STEALTHY_FAMILIES == {1, 5, 6, 7}


# ---- timeline layout ------------------------------------------------------------------------------


def test_node_x_is_voltage_first_and_clean_matches(splits):
    d = splits["test"].export(["node_x", "node_m", "clean", "family"])
    nx, nm, cl = d["node_x"], d["node_m"], d["clean"]
    metered_v = nm[:, :, 0] > 0
    # Column 0 is |V| in per-unit near 1.0; column 3 is an angle in degrees.
    assert np.all(np.abs(nx[:, :, 0][metered_v] - 1.0) < 0.3)
    assert np.all(np.abs(cl[:, :, 3]) < 180.0)
    # On benign records the metered voltage reading is the clean value plus a small meter error.
    ben = d["family"] == 0
    err = np.abs(nx[ben][:, :, 0] - cl[ben][:, :, 0])[metered_v[ben]]
    assert err.max() < 0.02
    # Unmetered channels are zero-filled in node_x but the clean layer covers every bus.
    assert np.all(nx[:, :, 0][~metered_v] == 0.0)
    assert np.all(cl[:, :, 0] > 0.5)


def test_slack_is_the_pinned_bus(timeline):
    ds = fg.load(timeline)
    assert ds.slack == 0  # case14's ext_grid sits on bus 0
    th = ds._clean_np[:, :, 3]
    assert th[:, ds.slack].std() < 1e-6  # pinned angle: no variation beyond float noise
    assert (th.std(axis=0) < 1e-6).sum() == 1  # and it is the only such bus


def test_labels_y_and_family_agree(splits):
    d = splits["test"].export(["y", "family"])
    y, fam = d["y"].astype(bool), d["family"]
    assert not y[fam == 0].any()  # benign records flag no bus
    assert y[fam > 0].any(axis=1).all()  # every attacked record flags at least one bus
    assert set(np.unique(fam)) <= set(fg.FAMILIES)


def test_family_filter_and_pu_units(timeline):
    sub = fg.load(timeline, families=["Aq", "At"])
    assert set(np.unique(sub.export(["family"])["family"])) <= {1, 5}
    phys = fg.load(timeline, split="test").export(["node_x"])["node_x"]
    pu = fg.load(timeline, split="test", units="pu").export(["node_x"])["node_x"]
    base = fg.load(timeline).baseMVA
    assert np.allclose(pu[:, :, 1], phys[:, :, 1] / base)
    assert np.allclose(pu[:, :, 3], np.deg2rad(phys[:, :, 3]))
    assert np.allclose(pu[:, :, 0], phys[:, :, 0])  # |V| is per-unit either way


def test_ybus_matches_engine_and_clean_injections(timeline):
    """ds.ybus equals the engine's makeYbus matrix and reproduces the clean injections at the
    shunt-free buses, so bus order and the branch model are both right."""
    pytest.importorskip("pandapower")
    from fdia_graph.engine import FdiaGenerator

    ds = fg.load(timeline)
    Y = ds.ybus_np
    g = FdiaGenerator(14, seed=1)
    with pytest.warns(DeprecationWarning, match="nl is deprecated"):
        assert g.nl == g.n_lines  # the 0.17 name still answers
    lut = np.asarray(g._ppc_row)  # timeline bus i -> ppc row lut[i]
    ref = g._Ybus.toarray()[np.ix_(lut, lut)]
    assert Y.shape == (ds.N, ds.N) and np.allclose(Y, ref, atol=1e-9)
    # S = V conj(Y V) is the net injection (generation positive); the clean layer stores P/Q with
    # pandapower's consumption-positive sign and the shunt draw removed, so compare -S where there
    # is no shunt.
    cl = ds._clean_np[:64].astype(float)  # [T, N, 4] = [|V|, P, Q, theta]
    V = cl[:, :, 0] * np.exp(1j * np.deg2rad(cl[:, :, 3]))
    S = V * np.conj(V @ Y.T) * ds.baseMVA
    shunt = (ds._phys["bus_shunt_b"] != 0) | (ds._phys["bus_shunt_g"] != 0)
    assert np.allclose(-S.real[:, ~shunt], cl[:, ~shunt, 1], atol=1e-2)
    assert np.allclose(-S.imag[:, ~shunt], cl[:, ~shunt, 2], atol=1e-2)
    assert tuple(ds.ybus.shape) == (ds.N, ds.N)
    # Branch admittance matrices: equal to the engine's, and Yf reproduces the clean from-end flows
    # (edge_clean is zero on unmetered branches, so compare where a flow meter exists).
    assert np.allclose(ds.yf_np, g._Yf.toarray()[:, lut], atol=1e-9)
    assert np.allclose(ds.yt_np, g._Yt.toarray()[:, lut], atol=1e-9)
    f = ds.edge_index_np[0]
    Sf = V[:, f] * np.conj(V @ ds.yf_np.T) * ds.baseMVA  # [T, E]
    ec = ds._eclean_np[:64].astype(float)
    metered = ds.export(["edge_m"])["edge_m"][0, :, 0] > 0
    assert np.allclose(Sf.real[:, metered], ec[:, metered, 0], atol=1e-2)
    assert np.allclose(Sf.imag[:, metered], ec[:, metered, 1], atol=1e-2)
    assert tuple(ds.yf.shape) == (ds.E, ds.N) and tuple(ds.yt.shape) == (ds.E, ds.N)
    # edge_clean_full: the same flows on EVERY branch, equal to edge_clean where metered and to the
    # Yf construction everywhere (so unmetered branches are no longer zero).
    assert ds.has_clean_full
    full = ds._clean_flows_full()[:64].astype(float)
    assert np.allclose(full[:, :, 0], Sf.real, atol=1e-2) and np.allclose(full[:, :, 1], Sf.imag, atol=1e-2)
    assert np.allclose(full[:, metered], ec[:, metered], atol=1e-2)
    if (~metered).any():
        assert np.abs(full[:, ~metered]).max() > 0
    rec = ds[0]
    t = rec["timestep"]
    assert tuple(rec["edge_clean_full"].shape) == (ds.E, 2)
    assert np.allclose(rec["edge_clean_full"].numpy(), ds._clean_flows_full()[t])
    npy = ds.export(["edge_clean_full", "timestep"])
    assert np.allclose(npy["edge_clean_full"][0], ds._clean_flows_full()[npy["timestep"][0]])
    pu = fg.load(timeline, units="pu")
    assert np.allclose(pu[0]["edge_clean_full"].numpy(), rec["edge_clean_full"].numpy() / ds.baseMVA)


def test_state_pool_order_is_detected_and_unified(timeline):
    """The engine works in one column order, [|V|, P, Q, theta]; older P-first pools convert on load."""
    from fdia_graph.generation import as_v_first

    clean = fg.load(timeline)._clean_np.astype(float)  # [Tpool, N, 4] already voltage-first
    assert np.array_equal(as_v_first(clean), clean)
    p_first = clean[:, :, [1, 2, 0, 3]]  # the pre-0.12 layout
    assert np.array_equal(as_v_first(p_first), clean)
    with pytest.raises(ValueError):
        as_v_first(np.zeros((3, 5, 4)))  # neither column looks like a voltage


# ---- formats agree -----------------------------------------------------------------------------


def test_pyg_data_matches_dict_record(timeline):
    pytest.importorskip("torch_geometric")
    import torch

    dict_ds = fg.load(timeline, split="test")
    pyg_ds = fg.load(timeline, split="test", format="pyg")
    for i in (0, 1, len(dict_ds) - 1):
        item, data = dict_ds[i], pyg_ds[i]
        assert torch.equal(data.x, item["node_x"])
        assert torch.equal(data.edge_attr, item["edge_x"])
        assert torch.equal(data.edge_x, item["edge_x"])
        assert torch.equal(data.node_mask, item["node_m"])
        assert torch.equal(data.edge_mask, item["edge_m"])
        assert torch.equal(data.y, item["y"])
        assert torch.equal(data.edge_index, item["edge_index"])
        assert data.family == item["family"] and data.slack == dict_ds.slack
        for k in ("stealthy", "seq_id", "timestep"):
            assert getattr(data, k) == item[k]
        for k in ("temporal_delta", "swing", "clean", "edge_clean", "edge_clean_full"):
            assert torch.equal(getattr(data, k), item[k])
        assert torch.equal(data.edge_phys, item["edge_attr"])  # static [E,8] physics, renamed
        # Every per-record key reaches PyG under its own name or the documented rename.
        rename = {"node_x": "x", "node_m": "node_mask", "edge_m": "edge_mask", "edge_attr": "edge_phys"}
        for k in item:
            assert hasattr(data, rename.get(k, k)), k
    batch = next(iter(pyg_ds.loader(batch_size=3, shuffle=False)))
    assert tuple(batch.edge_x.shape) == (3 * dict_ds.E, 2)
    assert tuple(batch.edge_phys.shape) == (3 * dict_ds.E, 8)
    assert batch.slack.tolist() == [dict_ds.slack] * 3


def test_pyg_stream_matches_dataset_pyg_contract(timeline):
    """The stream PyG helper and fg.load(format='pyg') expose the same attribute names."""
    pytest.importorskip("torch_geometric")
    import torch

    from fdia_graph.generation import _load_states

    X = _load_states(14, None)[:60]  # a short pool slice keeps the stream build to seconds
    with pytest.warns(DeprecationWarning, match="generate_stream is deprecated"):
        s = fg.generate_stream(14, states=X, seed=1)
    with pytest.warns(DeprecationWarning, match="pyg_stream is deprecated"):
        tr, te = fg.pyg_stream(stream=s, train_frac=0.5)
    d = tr[0]
    assert tuple(d.x.shape) == (14, 4) and tuple(d.edge_attr.shape) == (20, 2)
    assert torch.equal(d.edge_attr, d.edge_x)
    assert tuple(d.edge_phys.shape) == (20, 8)
    assert tuple(d.node_mask.shape) == (14, 4) and tuple(d.edge_mask.shape) == (20, 2)
    assert torch.equal(d.edge_x, torch.as_tensor(s["edge_x"][0], dtype=torch.float32))
    with pytest.warns(DeprecationWarning):
        (trc, tec) = fg.pyg_stream(stream=s, train_frac=0.5, layer="clean")
    assert torch.equal(trc[0].edge_x, torch.as_tensor(s["edge_clean"][0], dtype=torch.float32))


def test_branch_physics_names(timeline):
    import torch

    ds = fg.load(timeline)
    assert torch.equal(ds.branch_x, ds.edge_attr[:, 1])
    assert torch.equal(ds.branch_gs, ds.edge_attr[:, 4])
    with pytest.warns(DeprecationWarning, match="branch flows"):
        old = ds.edge_x  # the dataset-level reactance under its clashing old name
    assert torch.equal(old, ds.branch_x)


def test_dict_loader_batches_every_documented_key(splits):
    batch = next(iter(splits["test"].loader(batch_size=4, shuffle=False)))
    for k in (
        "node_x",
        "node_m",
        "edge_x",
        "edge_m",
        "edge_index",
        "y",
        "family",
        "swing",
        "temporal_delta",
        "clean",
        "edge_clean",
        "edge_clean_full",
        "edge_attr",
    ):
        assert k in batch, k
    assert tuple(batch["edge_clean_full"].shape) == (4, splits["test"].E, 2)
    N, E = splits["test"].N, splits["test"].E
    assert tuple(batch["node_x"].shape) == (4, N, 4)
    assert tuple(batch["edge_x"].shape) == (4, E, 2)
    assert tuple(batch["edge_attr"].shape) == (E, 8)


# ---- features ----------------------------------------------------------------------------------


def test_kcl_residual_matches_the_papers_builder(splits):
    from fdia_graph.localization.learned import full14, kcl_residual

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
    from fdia_graph.localization import SwingThreshold

    loc = SwingThreshold(fa_target=0.05).fit(splits["train"])
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
    from fdia_graph.localization import BusCNN, BusMLP

    for cls in (BusMLP, BusCNN):
        loc = cls(epochs=2, device="cpu").fit(splits["train"], val=splits["val"])
        assert loc.tau is not None and 0.0 < loc.tau < 1.0
        s = loc.scores(splits["test"])
        assert s.shape == (len(splits["test"]), splits["test"].N)
        assert np.all((s >= 0.0) & (s <= 1.0))
        assert "all" in loc.score(splits["test"])


# ---- state estimation --------------------------------------------------------------------------


def test_wls_estimates_the_classical_state(splits):
    pytest.importorskip("torch")
    from fdia_graph.se import WLS

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


def test_jacobian_features_split_stealthy_from_corruption(timeline, splits):
    """Unexplained energy fires on in-place corruption (Ad) and not on the stealthy re-solve (Aq);
    the explained energy fires on an Aq episode's first frame, where the previous frame's estimate
    is still benign (later frames of the episode are measured against an estimate the false state
    already moved). The digest's central claims, with only observed data in the feature.
    Transformed over the whole timeline so every family has frames (the test split of the tiny
    timeline can miss a long-episode family)."""
    pytest.importorskip("torch")
    from fdia_graph.se.jacobian import JacobianFeatures

    jf = JacobianFeatures().fit(splits["train"])
    d = fg.load(timeline).export(
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
    from fdia_graph.localization import BusMLP
    from fdia_graph.localization.learned import FEATURE_SETS

    for feats, width in FEATURE_SETS.items():
        loc = BusMLP(epochs=1, device="cpu", features=feats).fit(splits["train"])
        assert loc.n_feat == width and loc.mu.shape == (width,)
        assert loc.scores(splits["test"]).shape == (len(splits["test"]), splits["test"].N)
    with pytest.raises(ValueError):
        BusMLP(features="nope")


def test_jacobian_weighting_estimator(splits):
    pytest.importorskip("torch")
    from fdia_graph.se import WLS, JacobianWeighting

    est = JacobianWeighting(c=3.0).fit(splits["train"])
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
    from fdia_graph.se import GatedPrior

    est = GatedPrior(gate="oracle", rank_frac=0.5, reweight="huber", c=1.5).fit(splits["train"])
    w = est.gated_weights(splits["test"])
    y = splits["test"].export(["y"])["y"].astype(bool)
    assert w.shape == (len(splits["test"]), est.m)
    assert np.all(w[~y.any(axis=1)] == est.Wk[None, :])  # benign records keep the full weights
    assert (w[y.any(axis=1)] < est.Wk[None, :]).any(axis=1).all()  # attacked ones lose some
    x = est.estimate(splits["test"])
    assert x.shape == (len(splits["test"]), 2 * splits["test"].N - 1)
    with pytest.raises(ValueError):
        GatedPrior(gate=None)


def test_local_trainer_runs_persist_and_honour_owned_and_clip():
    """Two one-epoch runs equal one two-epoch run (optimizer and batch order persist); with `owned`
    the labels of the other buses cannot matter; a clip changes training only when it binds."""
    torch = pytest.importorskip("torch")
    from fdia_graph.localization.learned import LocalTrainer, OptimConfig, _mlp_net

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
        tr = LocalTrainer(net, cfg, "cpu", seed=3, clip=clip)
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
    from fdia_graph.localization.learned import _mlp_net, predict

    net = _mlp_net(14, 16, 2, 0.5)
    net.train()  # as a training run leaves it
    Xs = np.random.default_rng(0).normal(size=(8, 5, 14)).astype(np.float32)
    assert np.array_equal(predict(net, Xs, "cpu"), predict(net, Xs, "cpu"))  # dropout off: repeatable
    assert not net.training


def test_score_perbus_agrees_with_score_and_reports_the_paper_columns(splits):
    from fdia_graph.localization import SwingThreshold

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
    from fdia_graph.localization import BusMLP

    loc = BusMLP(epochs=2, device="cpu").fit(splits["train"], val=splits["val"])
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


def test_previous_frame_fields_are_the_row_emitted_before(timeline):
    """prev_* hold the readings of file row - 1 for every kept record, whatever split or family that
    row belongs to (an operator sees every frame); a record shard offers none."""
    from conftest import SHARD_V072

    from fdia_graph import FdiaGraph

    full = fg.load(timeline).export(["node_x", "edge_x", "timestep"])
    ds = fg.load(timeline, split="test", families=[0, 1, 2])
    d = ds.export(["prev_node_x", "prev_edge_x", "prev_timestep"])
    prev = np.maximum(ds.idx - 1, 0)
    assert not np.isin(prev, ds.idx).all()  # some previous frames sit outside the filtered view
    assert np.array_equal(d["prev_node_x"], full["node_x"][prev])
    assert np.array_equal(d["prev_edge_x"], full["edge_x"][prev])
    assert np.array_equal(d["prev_timestep"], full["timestep"][prev])
    with pytest.raises(ValueError, match="unknown field"):
        FdiaGraph(SHARD_V072).export(["prev_node_x"])


def test_jacobian_features_never_read_the_true_state(timeline, splits):
    """The change is taken against the previous frame's estimate, referenced to the case's slack
    angle (a network parameter): the transform reads measurements only, and the case's reference is
    the angle every true state of the timeline carries at the slack."""
    pytest.importorskip("pandapower")
    from fdia_graph.se.jacobian import JacobianFeatures

    jf = JacobianFeatures().fit(splits["train"])
    d = splits["test"].export(["node_x", "edge_x", "prev_node_x", "prev_edge_x", "prev_timestep"])
    before = jf.transform(d)["bus"]  # d holds the measurements and their frame index, no truth
    assert np.isfinite(before).all() and not hasattr(jf, "_pool")
    clean = splits["test"].export(["clean"])["clean"]
    assert np.allclose(np.deg2rad(clean[:, jf.est.slack, 3]), jf.est.theta_ref)
    empty = {k: v[:0] for k, v in d.items()}  # an empty view transforms to empty features
    assert jf.transform(empty)["bus"].shape == (0, before.shape[1], 8)
    with pytest.raises(ValueError, match="timeline"):
        from conftest import SHARD_V072

        from fdia_graph import FdiaGraph

        JacobianFeatures().fit(FdiaGraph(SHARD_V072))


def test_previous_swing_feature_set(timeline, splits):
    """prev_swing is the swing of file row - 1, and "full14+prev" appends it after the papers' 14
    channels; the frame after a one-frame attack reads the attack's swing there, sign reversed."""
    pytest.importorskip("torch")
    from fdia_graph.localization import BusCNN
    from fdia_graph.localization.learned import full14

    full = fg.load(timeline).export(["swing", "family"])
    ds = splits["test"]
    d = ds.export(["prev_swing"])
    assert np.array_equal(d["prev_swing"], full["swing"][np.maximum(ds.idx - 1, 0)])
    loc = BusCNN(epochs=1, device="cpu", features="full14+prev").fit(splits["train"])
    x = loc._pull(ds)
    F = loc._features(x)
    assert (
        F.shape[-1] == 16
        and np.array_equal(F[..., :14], full14(x))
        and np.array_equal(F[..., 14:], x["prev_swing"])
    )
