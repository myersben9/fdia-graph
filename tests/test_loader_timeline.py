"""`fg.load` on a timeline file (docs/plans/ONE_DATASET_PLAN.md step 2): the record table and the
timeline from one file, `order`, the benign layer on records, batches and exports, `ds.windows`,
`ds.episodes`, and the torch helpers on a dataset. Uses the `timeline` fixture of conftest."""

import numpy as np
import pytest

pytest.importorskip("pandapower")
pytest.importorskip("torch")

import fdia_graph as fg  # noqa: E402


def test_generate_rejects_a_bad_frame_cap(tmp_path):
    for bad in (0, -5, 2.5, True):
        with pytest.raises(ValueError, match="frames"):
            fg.generate("ieee14", "never", frames=bad, out=str(tmp_path / "x.h5"))


def test_generate_registers_a_timeline_that_load_reads(timeline):
    assert fg.list_datasets()[timeline] == "local"
    ds = fg.load(timeline)
    assert ds.is_timeline and ds.has_benign and ds.has_clean and ds.has_clean_full
    assert len(ds) == 1000 and ds.system == 14 and ds.N == 14 and ds.E == 20
    assert np.array_equal(ds.idx, np.arange(1000))
    summ = ds.summary()
    assert summ.n == 1000 and set(summ.families) == {fg.FAMILIES[k] for k in range(8)}


def test_record_batch_and_export_carry_the_benign_layer(timeline):
    import torch

    ds = fg.load(timeline)
    rec = ds[0]
    assert tuple(rec["benign"].shape) == (14, 4) and tuple(rec["edge_benign"].shape) == (20, 2)
    assert list(rec)[-2:] == ["benign", "edge_benign"]
    batch = ds.collate([ds[i] for i in range(3)])
    assert tuple(batch["benign"].shape) == (3, 14, 4) and tuple(batch["edge_benign"].shape) == (3, 20, 2)
    a = ds.export()
    assert a["benign"].shape == (1000, 14, 4) and a["edge_benign"].shape == (1000, 20, 2)
    assert "benign" in ds._default_fields() and torch.equal(rec["benign"], torch.as_tensor(a["benign"][0]))
    # observed == benign on benign frames; the attack is the difference elsewhere
    ben = a["family"] == 0
    assert np.array_equal(a["node_x"][ben], a["benign"][ben])
    assert (a["node_x"][~ben] != a["benign"][~ben]).any()
    pu = fg.load(timeline, units="pu")
    assert torch.allclose(pu[0]["benign"][:, 1], rec["benign"][:, 1] / ds.baseMVA)
    assert torch.allclose(pu[0]["edge_benign"], rec["edge_benign"] / ds.baseMVA)
    pre = fg.load(timeline, split="test", preload=True)
    assert torch.equal(pre[0]["benign"], fg.load(timeline, split="test")[0]["benign"])


def test_random_order_is_the_same_frames_permuted_by_the_seed(timeline):
    import torch

    time = fg.load(timeline)
    r0 = fg.load(timeline, order="random")
    r0b = fg.load(timeline, order="random", seed=0)
    r1 = fg.load(timeline, order="random", seed=1)
    assert len(r0) == len(time) and np.array_equal(r0.idx, time.idx)  # the same rows, a view permutation
    t0 = [int(r0[i]["timestep"]) for i in range(20)]
    assert t0 == [int(r0b[i]["timestep"]) for i in range(20)] and t0 != list(range(20))
    assert t0 != [int(r1[i]["timestep"]) for i in range(20)]
    assert sorted(r0.export(["timestep"])["timestep"].tolist()) == list(range(1000))
    for i in (0, 7, 999):
        assert torch.equal(r0[i]["node_x"], time[int(r0[i]["timestep"])]["node_x"])
    a = r0.export(["node_x", "timestep", "clean", "benign"])
    assert a["timestep"][:20].tolist() == t0
    for i in (0, 5):
        assert torch.equal(r0[i]["node_x"], torch.as_tensor(a["node_x"][i]))
        assert torch.equal(r0[i]["clean"], torch.as_tensor(a["clean"][i]))
    pre = fg.load(timeline, order="random", preload=True)
    assert torch.equal(pre[3]["benign"], r0[3]["benign"])
    with pytest.raises(ValueError, match="order must be"):
        fg.load(timeline, order="shuffled")


def test_windows_on_the_time_ordered_view(timeline):
    ds = fg.load(timeline, split="test")
    T = len(ds)
    Xw, yw = ds.windows(8, stride=4)
    n = (T - 8) // 4 + 1
    assert Xw.shape == (n, 8, 14, 4) and yw.shape == (n, 14)
    a = ds.export(["node_x", "y", "benign"])
    assert np.array_equal(Xw[1], a["node_x"][4:12]) and np.array_equal(yw[1], a["y"][4:12].max(0))
    Xb, yf = ds.windows(8, stride=4, label="frame", layer="benign")
    assert np.array_equal(Xb[0], a["benign"][:8]) and yf.shape == (n, 8, 14)
    pu = fg.load(timeline, split="test", units="pu")
    assert np.allclose(pu.windows(8, stride=4)[0][0, :, :, 1], Xw[0, :, :, 1] / ds.baseMVA)
    with pytest.raises(ValueError, match="order='time'"):
        fg.load(timeline, order="random").windows(8)
    with pytest.raises(ValueError, match="consecutive frames"):
        fg.load(timeline, families=["Aq"]).windows(8)
    with pytest.raises(ValueError, match="layer must be"):
        ds.windows(8, layer="swing")
    with pytest.raises(ValueError, match="need integers"):
        ds.windows(T + 1)


def test_the_v072_record_shard_still_loads():
    """The pre-0.18 layout (gap rows, the clean pool once per pool timestep) reads as before."""
    from conftest import SHARD_V072

    from fdia_graph.dataset import FdiaGraph

    ds = FdiaGraph(SHARD_V072)
    assert not ds.is_timeline and not ds.has_benign and ds.has_clean and len(ds) > 0
    assert len(FdiaGraph(SHARD_V072, include_gaps=True)) >= len(ds)
    rec = ds[0]
    assert "benign" not in rec and "benign" not in ds.export() and tuple(rec["clean"].shape) == (14, 4)
    assert set(np.unique(ds.export(["family"])["family"]).tolist()) <= set(range(7))
    train, test = FdiaGraph(SHARD_V072, split="train"), FdiaGraph(SHARD_V072, split="test")
    assert len(train) > len(test) > 0
    with pytest.raises(ValueError, match="timeline file"):
        ds.windows(4)
    with pytest.raises(AttributeError, match="no episodes"):
        ds.episodes
    rnd = FdiaGraph(SHARD_V072, order="random")
    assert sorted(int(rnd[i]["timestep"]) for i in range(len(rnd))) == sorted(
        int(ds[i]["timestep"]) for i in range(len(ds))
    )


def test_episodes_table_follows_the_view(timeline):
    from fdia_graph.models import EpisodeTable

    ds = fg.load(timeline)
    ep = ds.episodes
    assert isinstance(ep, EpisodeTable) and len(ep) > 10
    assert len(ep.buses) == len(ep) and all(b.ndim == 1 for b in ep.buses)
    a = ds.export(["family", "seq_id", "y"])
    for k in range(len(ep)):
        rows = slice(ep.onset[k], ep.onset[k] + ep.length[k])
        assert set(np.unique(a["family"][rows]).tolist()) <= {0, int(ep.family[k])}
        assert set(np.where(a["y"][rows].any(0))[0].tolist()) == set(ep.buses[k].tolist())
    test = fg.load(timeline, split="test")
    sub = test.episodes
    assert 0 < len(sub) < len(ep) and np.isin(sub.onset, test.idx).all()
    assert len(fg.load(timeline, families=["At"]).episodes) == int((ep.family == 5).sum())


def test_edge_attr_np_is_torch_free_and_equal_to_edge_attr(timeline):
    import torch

    ds = fg.load(timeline)
    assert np.array_equal(ds.edge_attr_np, ds.edge_attr.numpy().astype(np.float32))
    ds._phys["edge_gs"] = None  # a file that predates the stored series admittance derives it
    assert np.allclose(ds.edge_attr_np, ds.edge_attr.numpy(), rtol=1e-6)
    assert isinstance(torch.as_tensor(ds.edge_attr_np), torch.Tensor)


def test_stream_of_refuses_a_shuffled_or_filtered_view(timeline):
    from fdia_graph.streams import stream_of

    with pytest.raises(ValueError, match="order='time'"):
        stream_of(fg.load(timeline, order="random"))
    with pytest.raises(ValueError, match="consecutive frames"):
        stream_of(fg.load(timeline, families=["Aq"]))
    s = stream_of(fg.load(timeline, split="test"))
    assert s.node_x.shape[0] == len(fg.load(timeline, split="test"))


def test_torch_helpers_take_a_dataset(timeline):
    import torch

    ds = fg.load(timeline, split="test")
    with pytest.warns(DeprecationWarning, match="torch_windows is deprecated"):
        (Xtr, ytr), (Xte, yte) = fg.torch_windows(dataset=ds, W=8, stride=4, train_frac=0.5)
    T = len(ds)
    assert Xtr.shape[1:] == (8, 4) and Xtr.shape[0] % 14 == 0 and ytr.shape == (Xtr.shape[0],)
    assert Xtr.shape[0] // 14 + Xte.shape[0] // 14 <= (T - 8) // 4 + 1
    a = ds.export(["node_x"])
    assert torch.equal(Xtr[0], torch.as_tensor(a["node_x"][:8, 0], dtype=torch.float32))
    torch_geometric = pytest.importorskip("torch_geometric")
    with pytest.warns(DeprecationWarning, match="pyg_stream is deprecated"):
        tr, te = fg.pyg_stream(dataset=ds, train_frac=0.5, layer="benign")
    assert len(tr) + len(te) == T and isinstance(tr[0], torch_geometric.data.Data)
    assert tuple(tr[0].edge_phys.shape) == (20, 8) and tuple(tr[0].node_mask.shape) == (14, 4)
    assert torch.equal(tr[0].x, ds[0]["benign"])
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="order='time'"):
        fg.torch_windows(dataset=fg.load(timeline, order="random"), W=8)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="dataset="):
        fg.torch_windows(W=8)
    # the replacement: per-bus sequences from the view itself, the file's split as the split
    Xb, yb = ds.windows(8, stride=4, label="last", per_bus=True)
    assert Xb.shape[1:] == (8, 4) and Xb.shape[0] == ((T - 8) // 4 + 1) * 14 and yb.shape == (Xb.shape[0],)
    assert np.array_equal(Xb[0], ds.export(["node_x"])["node_x"][:8, 0])
    Xf, yf = ds.windows(8, stride=4, label="frame", per_bus=True)
    assert yf.shape == (Xf.shape[0], 8)


def test_the_stream_entry_points_warn(timeline):
    from fdia_graph.generation import _load_states

    X = _load_states(14, None)[:30]
    with pytest.warns(DeprecationWarning, match="generate_stream is deprecated"):
        s = fg.generate_stream(14, states=X, seed=1)
    with pytest.warns(DeprecationWarning, match="windows.*deprecated"):
        fg.windows(s, W=4)


def test_estimator_and_localizer_score_a_timeline_with_am(timeline):
    from fdia_graph.localization import SwingThreshold
    from fdia_graph.se import WLS

    train, test = fg.load(timeline, split="train"), fg.load(timeline, split="test")
    assert 7 in set(test.export(["family"])["family"].tolist())
    se = WLS().fit(train).score(test)
    assert se.Am is not None and se["Am"].angle_mae_deg > 0 and list(se)[-1] == "geo"
    loc = SwingThreshold(fa_target=0.01).fit(train).score(test)
    assert loc.Am is not None and 0 <= loc["Am"].detection_rate <= 1


def test_windows_view_equals_the_copy(timeline):
    ds = fg.load(timeline, order="time")
    Xc, yc = ds.windows(10, stride=4, label="last")
    Xv, yv = ds.windows(10, stride=4, label="last", copy=False)
    assert np.array_equal(Xc, Xv) and np.array_equal(yc, yv)
    assert Xc.flags.writeable and not Xv.flags.writeable
