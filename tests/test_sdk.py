"""Behavioral tests for the SDK surface: the contracts the docs promise, checked on a tiny shard."""

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
    assert fg.FAMILIES == {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
    assert fg.STEALTHY_FAMILIES == {1, 5, 6}


# ---- shard layout ------------------------------------------------------------------------------


def test_node_x_is_voltage_first_and_clean_matches(splits):
    d = splits["test"].to_numpy(["node_x", "node_m", "clean", "family"])
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


def test_slack_is_the_pinned_bus(shard):
    ds = fg.load(shard)
    assert ds.slack == 0  # case14's ext_grid sits on bus 0
    th = ds._clean_np[:, :, 3]
    assert th[:, ds.slack].std() < 1e-6  # pinned angle: no variation beyond float noise
    assert (th.std(axis=0) < 1e-6).sum() == 1  # and it is the only such bus


def test_labels_y_and_family_agree(splits):
    d = splits["test"].to_numpy(["y", "family"])
    y, fam = d["y"].astype(bool), d["family"]
    assert not y[fam == 0].any()  # benign records flag no bus
    assert y[fam > 0].any(axis=1).all()  # every attacked record flags at least one bus
    assert set(np.unique(fam)) <= set(fg.FAMILIES)


def test_family_filter_and_pu_units(shard):
    sub = fg.load(shard, families=["Aq", "At"])
    assert set(np.unique(sub.to_numpy(["family"])["family"])) <= {1, 5}
    phys = fg.load(shard, split="test").to_numpy(["node_x"])["node_x"]
    pu = fg.load(shard, split="test", units="pu").to_numpy(["node_x"])["node_x"]
    base = fg.load(shard).baseMVA
    assert np.allclose(pu[:, :, 1], phys[:, :, 1] / base)
    assert np.allclose(pu[:, :, 3], np.deg2rad(phys[:, :, 3]))
    assert np.allclose(pu[:, :, 0], phys[:, :, 0])  # |V| is per-unit either way


def test_ybus_matches_engine_and_clean_injections(shard):
    """ds.ybus equals the engine's makeYbus matrix and reproduces the clean injections at the
    shunt-free buses, so bus order and the branch model are both right."""
    pytest.importorskip("pandapower")
    from fdia_graph.engine import FdiaGenerator

    ds = fg.load(shard)
    Y = ds.ybus_np
    g = FdiaGenerator(14, seed=1)
    lut = np.asarray(g._lut)  # shard bus i -> ppc row lut[i]
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


def test_state_pool_order_is_detected_and_unified(shard):
    """The engine works in one column order, [|V|, P, Q, theta]; older P-first pools convert on load."""
    from fdia_graph.generation import as_v_first

    clean = fg.load(shard)._clean_np.astype(float)  # [Tpool, N, 4] already voltage-first
    assert np.array_equal(as_v_first(clean), clean)
    p_first = clean[:, :, [1, 2, 0, 3]]  # the pre-0.12 layout
    assert np.array_equal(as_v_first(p_first), clean)
    with pytest.raises(ValueError):
        as_v_first(np.zeros((3, 5, 4)))  # neither column looks like a voltage


# ---- formats agree -----------------------------------------------------------------------------


def test_pyg_data_matches_dict_record(shard):
    pytest.importorskip("torch_geometric")
    import torch

    dict_ds = fg.load(shard, split="test")
    pyg_ds = fg.load(shard, split="test", format="pyg")
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
        for k in ("temporal_delta", "swing", "clean", "edge_clean"):
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


def test_pyg_stream_matches_dataset_pyg_contract(shard):
    """The stream PyG helper and fg.load(format='pyg') expose the same attribute names."""
    pytest.importorskip("torch_geometric")
    import torch

    from fdia_graph.generation import _load_states

    X = _load_states(14, None)[:60]  # a short pool slice keeps the stream build to seconds
    s = fg.generate_stream(14, states=X, seed=1)
    tr, te = fg.pyg_stream(stream=s, train_frac=0.5)
    d = tr[0]
    assert tuple(d.x.shape) == (14, 4) and tuple(d.edge_attr.shape) == (20, 2)
    assert torch.equal(d.edge_attr, d.edge_x)
    assert tuple(d.edge_phys.shape) == (20, 8)
    assert tuple(d.node_mask.shape) == (14, 4) and tuple(d.edge_mask.shape) == (20, 2)
    assert torch.equal(d.edge_x, torch.as_tensor(s["edge_x"][0], dtype=torch.float32))
    (trc, tec) = fg.pyg_stream(stream=s, train_frac=0.5, layer="clean")
    assert torch.equal(trc[0].edge_x, torch.as_tensor(s["edge_clean"][0], dtype=torch.float32))


def test_branch_physics_names(shard):
    import torch

    ds = fg.load(shard)
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
        "edge_attr",
    ):
        assert k in batch, k
    N, E = splits["test"].N, splits["test"].E
    assert tuple(batch["node_x"].shape) == (4, N, 4)
    assert tuple(batch["edge_x"].shape) == (4, E, 2)
    assert tuple(batch["edge_attr"].shape) == (E, 8)


# ---- features ----------------------------------------------------------------------------------


def test_kcl_residual_matches_the_papers_builder(splits):
    from fdia_graph.localization.learned import full14, kcl_residual

    d = splits["test"].to_numpy(["node_x", "node_m", "edge_x", "temporal_delta", "swing"])
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


def test_threshold_localizer_protocol(shard, splits):
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
    ben_only = fg.load(shard, split="test", families=[0])
    assert loc.score(ben_only)["all"]["macro_f1"] == 0.0


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
