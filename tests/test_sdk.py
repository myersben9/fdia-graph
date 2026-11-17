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


def test_branch_physics_names(timeline):
    import torch

    ds = fg.load(timeline)
    assert torch.equal(ds.branch_x, ds.edge_attr[:, 1])
    assert torch.equal(ds.branch_gs, ds.edge_attr[:, 4])
    assert not hasattr(ds, "edge_x")  # on a record edge_x is the branch flows; the dataset has branch_x


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


def test_previous_frame_fields_are_the_row_emitted_before(timeline):
    """prev_* hold the readings of file row - 1 for every kept record, whatever split or family that
    row belongs to (an operator sees every frame)."""
    full = fg.load(timeline).export(["node_x", "edge_x", "timestep"])
    ds = fg.load(timeline, split="test", families=[0, 1, 2])
    d = ds.export(["prev_node_x", "prev_edge_x", "prev_timestep"])
    prev = np.maximum(ds.idx - 1, 0)
    assert not np.isin(prev, ds.idx).all()  # some previous frames sit outside the filtered view
    assert np.array_equal(d["prev_node_x"], full["node_x"][prev])
    assert np.array_equal(d["prev_edge_x"], full["edge_x"][prev])
    assert np.array_equal(d["prev_timestep"], full["timestep"][prev])
