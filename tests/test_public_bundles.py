"""The public dual-access bundles (data models step 3): each is the dict it used to be, with the
same keys in the same order, and its attribute view agrees with the dict view."""

import json

import pytest

from tests.helpers import agree


def test_record_batch_and_arrays(splits):
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.models.data import ArraysBundle, BatchBundle, RecordBundle, Summary

    ds = splits["test"]
    rec = ds[0]
    assert isinstance(rec, RecordBundle)
    agree(rec)
    assert list(rec)[:6] == ["edge_index", "node_x", "node_m", "edge_x", "edge_m", "y"]
    assert rec.family == rec["family"] and torch.equal(rec.node_x, rec["node_x"])

    batch = FdiaGraph.collate([ds[i] for i in range(3)])
    assert isinstance(batch, BatchBundle)
    agree(batch)
    assert tuple(batch.node_x.shape) == (3, ds.N, 4) and batch.family.tolist() == [
        ds[i].family for i in range(3)
    ]
    loader_batch = next(iter(DataLoader(ds, batch_size=3, collate_fn=FdiaGraph.collate)))
    assert set(loader_batch) == set(batch)
    default_batch = next(iter(DataLoader(ds, batch_size=3)))  # PyTorch's own collate sees a dict
    assert torch.equal(default_batch["node_x"], batch.node_x)

    arrays = ds.export()
    assert isinstance(arrays, ArraysBundle)
    agree(arrays)
    assert list(arrays)[:2] == ["edge_index", "node_x"]
    assert arrays.node_x.shape == (len(ds), ds.N, 4) and "swing" in arrays
    sub = ds.export(fields=["node_x", "y"])
    assert list(sub) == ["edge_index", "node_x", "y"] and sub.swing is None
    rev = ds.export(fields=["y", "node_x"])  # the caller's order is the dict order, as before
    assert list(rev) == ["edge_index", "y", "node_x"]
    assert list(ds.export(fields=["y", "node_x"], format="torch")) == list(rev)
    tens = ds.export(fields=["node_x"], format="torch")
    assert isinstance(tens, ArraysBundle) and torch.equal(tens.node_x, torch.as_tensor(sub.node_x))
    df = ds.export(format="pandas", flatten_features=False)
    assert len(df) == len(ds) and list(df.columns)[:2] == ["family", "family_id"]
    with pytest.raises(ValueError, match="format"):
        ds.export(format="polars")
    with pytest.raises(ValueError, match="every field"):
        ds.export(fields=["nope"], format="pandas")
    for gone in ("to_numpy", "to_torch", "to_tf", "to_pandas"):  # the exporters of 0.17, retired
        assert not hasattr(ds, gone)

    summ = ds.summary()
    assert isinstance(summ, Summary) and summ.n == len(ds) == summ["n"]
    json.dumps(summ)  # scalar bundles serialize as the dicts they are
