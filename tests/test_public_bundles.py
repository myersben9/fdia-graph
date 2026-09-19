"""The public dual-access bundles (data models step 3): each is the dict it used to be, with the
same keys in the same order, and its attribute view agrees with the dict view."""

import json
import os

import numpy as np
import pytest

from fdia_graph.models import Bundle

FROZEN = os.path.join(os.path.dirname(__file__), "frozen")


def _agree(b: Bundle) -> None:
    assert isinstance(b, dict) and isinstance(b, Bundle)
    for name in b._names():
        v = getattr(b, name)
        if v is None:
            assert b.key_of(name) not in b
        else:
            assert b[b.key_of(name)] is v


def test_record_batch_and_arrays(splits):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from fdia_graph.dataset import ArraysBundle, BatchBundle, FdiaGraph, RecordBundle, Summary

    ds = splits["test"]
    rec = ds[0]
    assert isinstance(rec, RecordBundle)
    _agree(rec)
    assert list(rec)[:6] == ["edge_index", "node_x", "node_m", "edge_x", "edge_m", "y"]
    assert rec.family == rec["family"] and torch.equal(rec.node_x, rec["node_x"])

    batch = FdiaGraph.collate([ds[i] for i in range(3)])
    assert isinstance(batch, BatchBundle)
    _agree(batch)
    assert tuple(batch.node_x.shape) == (3, ds.N, 4) and batch.family.tolist() == [
        ds[i].family for i in range(3)
    ]
    loader_batch = next(iter(DataLoader(ds, batch_size=3, collate_fn=FdiaGraph.collate)))
    assert set(loader_batch) == set(batch)
    default_batch = next(iter(DataLoader(ds, batch_size=3)))  # PyTorch's own collate sees a dict
    assert torch.equal(default_batch["node_x"], batch.node_x)

    arrays = ds.to_numpy()
    assert isinstance(arrays, ArraysBundle)
    _agree(arrays)
    assert list(arrays)[:3] == ["edge_index", "edge_reactance", "node_x"]
    assert arrays.node_x.shape == (len(ds), ds.N, 4) and "swing" in arrays
    sub = ds.to_numpy(fields=["node_x", "y"])
    assert list(sub) == ["edge_index", "edge_reactance", "node_x", "y"] and sub.swing is None
    rev = ds.to_numpy(fields=["y", "node_x"])  # the caller's order is the dict order, as before
    assert list(rev) == ["edge_index", "edge_reactance", "y", "node_x"]
    assert list(ds.to_torch(fields=["y", "node_x"])) == list(rev)
    tens = ds.to_torch(fields=["node_x"])
    assert isinstance(tens, ArraysBundle) and torch.equal(tens.node_x, torch.as_tensor(sub.node_x))

    summ = ds.summary()
    assert isinstance(summ, Summary) and summ.n == len(ds) == summ["n"]
    json.dumps(summ)  # scalar bundles serialize as the dicts they are


def test_estimator_and_localizer_scores(shard):
    import fdia_graph as fg
    from fdia_graph.localization import SwingThreshold
    from fdia_graph.localization.base import LocalizerScores, OverallMetrics
    from fdia_graph.se import WLS
    from fdia_graph.se.base import ErrorPair, EstimatorScores

    train, test = fg.load(shard, split="train"), fg.load(shard, split="test")
    se = WLS().fit(train).score(test)
    assert isinstance(se, EstimatorScores) and isinstance(se.geo, ErrorPair)
    _agree(se)
    assert se["geo"]["angle_mae_deg"] == se.geo.angle_mae_deg
    assert list(se)[-1] == "geo" and list(se)[0] == "benign"
    ref = json.load(open(os.path.join(FROZEN, "tiny_ieee14_se.json")))["wls"]
    assert set(se) == set(ref) and all(set(se[k]) == set(ref[k]) for k in se)  # the reference is key-sorted

    loc = SwingThreshold(fa_target=0.01).fit(train).score(test)
    assert isinstance(loc, LocalizerScores) and isinstance(loc.all, OverallMetrics)
    _agree(loc)
    assert loc["all"]["macro_f1"] == loc.all.macro_f1
    ref = json.load(open(os.path.join(FROZEN, "tiny_ieee14_localization.json")))["swing"]
    assert set(loc) == set(ref) and all(set(loc[k]) == set(ref[k]) for k in loc)
    json.dumps(loc)


def test_jacobian_outputs(shard):
    import fdia_graph as fg
    from fdia_graph.se.jacobian import JacobianFeatures, JacobianOutputs

    train, test = fg.load(shard, split="train"), fg.load(shard, split="test")
    out = JacobianFeatures().fit(train).transform(test.to_numpy())
    assert isinstance(out, JacobianOutputs)
    _agree(out)
    assert list(out) == ["bus", "global", "dx_hat", "r_perp"]
    assert out["global"] is out.global_ and out.bus.shape[:2] == (len(test), test.N)


def test_stream_bundle():
    from fdia_graph.streams import Stream

    z = np.load(os.path.join(FROZEN, "ieee14_stream.npz"))  # the frozen file flattens the episode list
    raw = {k: z[k] for k in z.files if not k.startswith("episode_")}
    raw["episodes"] = [
        {"onset": int(o), "length": int(n), "family": int(f), "buses": [int(b) for b in bs.split(",") if b]}
        for o, n, f, bs in zip(
            z["episode_onset"], z["episode_length"], z["episode_family"], z["episode_buses"]
        )
    ]
    s = Stream(**raw)
    _agree(s)
    assert list(s) == list(raw) and s.system is None and "system" not in s  # the file carries arrays only
    from fdia_graph.streams import stream_summary

    summ = stream_summary(raw)  # what load_stream adds on top of the file
    assert summ["system"] == raw["node_x"].shape[1] == 14
    assert summ["attacked_frac"] == float((raw["y"].sum(axis=1) > 0).mean()) and 0 < summ["attacked_frac"] < 1
    full = Stream(**raw, **summ)
    assert full.system == 14 and full["attacked_frac"] == summ["attacked_frac"]
    assert s.episodes[0]["onset"] >= 0 and len(s.episodes) > 1
    assert s.node_x.shape[0] == s.y.shape[0] == len(s.timestep)


def test_load_stream_fills_the_summary_fields(tmp_path, monkeypatch):
    """`fg.load_stream` itself, with the download replaced by a file built from the frozen stream's
    arrays: `system` and `attacked_frac` come back filled (they were None before 0.18)."""
    import fdia_graph as fg
    import fdia_graph.download as download

    z = np.load(os.path.join(FROZEN, "ieee14_stream.npz"))
    arrays = {k: z[k] for k in z.files if not k.startswith("episode_")}
    episodes = [
        {"onset": int(o), "length": int(n), "family": int(f), "buses": [int(b) for b in bs.split(",") if b]}
        for o, n, f, bs in zip(
            z["episode_onset"], z["episode_length"], z["episode_family"], z["episode_buses"]
        )
    ]
    path = tmp_path / "stream_ieee14.npz"
    np.savez_compressed(path, **arrays, episodes=np.array(episodes, dtype=object))
    monkeypatch.setattr(download, "ensure_local", lambda spec: str(path))

    with pytest.warns(DeprecationWarning, match="load_stream is deprecated"):
        s = fg.load_stream("ieee14")
    assert s.system == 14 and s["system"] == 14
    assert s.attacked_frac == float((arrays["y"].sum(axis=1) > 0).mean()) and 0 < s.attacked_frac < 1
    assert s.node_x.shape == arrays["node_x"].shape and len(s.episodes) == len(episodes)
