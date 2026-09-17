"""The public edges: wrong arguments fail early with a message naming what is allowed, and the
filters mean what the docs say. The frozen suite proves the happy path; this file is the rest."""

import os

import numpy as np
import pytest

import fdia_graph as fg
from fdia_graph.dataset import FAMILIES, family_ids

# ---- registry and loading -------------------------------------------------------------------------


def test_unknown_dataset_and_system_spellings():
    with pytest.raises(KeyError, match="unknown dataset"):
        fg.resolve("no_such_dataset")
    from fdia_graph.registry import system_id

    with pytest.raises(ValueError, match="system must be"):
        system_id("bogus")


def test_family_ids_accept_names_codes_and_aliases_and_reject_unknowns():
    assert family_ids(["benign", "Aq", "Al"]) == [0, 1, 6]
    assert family_ids([1, 5]) == [1, 5]
    assert family_ids(["Ao", "ramp", "LRA"]) == [1, 5, 6]  # the legacy names
    with pytest.raises(ValueError, match="unknown family"):
        family_ids(["Bogus"])
    with pytest.raises(ValueError, match="unknown family"):
        family_ids([42])
    with pytest.raises(ValueError, match="unknown family"):
        family_ids([1.9])  # a float is not a code, even one that truncates to a valid one
    with pytest.raises(ValueError, match="unknown family"):
        family_ids([True])
    assert family_ids([np.int64(2)]) == [2]


def test_bad_split_or_family_fails_before_any_download(monkeypatch):
    """`fg.load("ieee118", split="bogus")` must not fetch 317 MB first: the argument checks run
    before the registry is consulted."""
    import fdia_graph.registry as registry

    def boom(*a, **k):
        raise AssertionError("resolve was called before the argument check")

    monkeypatch.setattr(registry, "resolve", boom)
    monkeypatch.setattr(fg, "resolve", boom)
    with pytest.raises(ValueError, match="split must be one of"):
        fg.load("ieee118", split="bogus")
    with pytest.raises(ValueError, match="unknown family"):
        fg.load("ieee118", families=["Bogus"])
    with pytest.raises(ValueError, match="units must be"):
        fg.load("ieee118", units="feet")


def test_load_rejects_bad_units_split_and_family(shard):
    with pytest.raises(ValueError, match="units must be"):
        fg.load(shard, units="feet")
    with pytest.raises(ValueError, match="split must be one of"):
        fg.load(shard, split="bogus")
    with pytest.raises(ValueError, match="unknown family"):
        fg.load(shard, families=["Bogus"])


def test_family_filter_aliases_and_heldout_protocol(shard):
    by_alias = fg.load(shard, families=["Ao"])
    by_code = fg.load(shard, families=[1])
    assert len(by_alias) == len(by_code) > 0 and np.array_equal(by_alias.idx, by_code.idx)
    for split in ("train", "val"):
        fams = set(fg.load(shard, split=split, heldout=True).to_numpy(["family"])["family"].tolist())
        assert not fams & {3, 4}, f"As/Ar must be held out of {split}"
    test_fams = set(fg.load(shard, split="test", heldout=True).to_numpy(["family"])["family"].tolist())
    assert test_fams & {3, 4}, "the test split keeps As/Ar"
    assert len(fg.load(shard, include_gaps=True)) >= len(fg.load(shard))


# ---- exports ----------------------------------------------------------------------------------------


def test_to_numpy_rejects_unknown_fields_and_empty_means_default(splits):
    ds = splits["test"]
    with pytest.raises(ValueError, match="unknown field"):
        ds.to_numpy(["node_x", "nope"])
    assert list(ds.to_numpy([])) == list(ds.to_numpy())
    assert list(ds.to_numpy(["y"])) == ["edge_index", "edge_reactance", "y"]


def test_default_fields_follow_the_layers_present(splits):
    """`edge_clean` is offered only when the file carries it, so a requested field is always in
    the returned bundle (the check and the gather use the same list)."""
    ds = splits["test"]
    fields = ds._default_fields()
    assert ("clean" in fields) == (ds._clean_np is not None)
    assert ("edge_clean" in fields) == (ds._eclean_np is not None)
    assert set(ds.to_numpy(fields)) == set(fields) | {"edge_index", "edge_reactance"}


def test_summary_counts_add_up(splits):
    ds = splits["train"]
    s = ds.summary()
    assert s.n == len(ds) == sum(s.families.values())
    assert set(s.families) <= set(FAMILIES.values()) and s.N == ds.N and s.E == ds.E


def test_collate_and_loader_shapes(splits):
    torch = pytest.importorskip("torch")
    from fdia_graph.dataset import FdiaGraph

    ds = splits["test"]
    one = FdiaGraph.collate([ds[0]])
    assert tuple(one.node_x.shape) == (1, ds.N, 4) and one.family.shape == (1,)
    batch = next(iter(ds.loader(batch_size=3, shuffle=False)))
    assert tuple(batch["node_x"].shape) == (3, ds.N, 4)
    assert torch.equal(batch["edge_index"], ds.edge_index)


# ---- streams ------------------------------------------------------------------------------------------


def _tiny_stream(T: int = 12, N: int = 3):
    rng = np.random.default_rng(0)
    return {"node_x": rng.normal(size=(T, N, 4)), "y": (rng.random((T, N)) > 0.7).astype(np.uint8)}


def test_windows_labels_and_bounds():
    s = _tiny_stream()
    Xw, yw = fg.windows(s, W=4, stride=2, label="frame")
    assert Xw.shape == (5, 4, 3, 4) and yw.shape == (5, 4, 3)
    _, y_any = fg.windows(s, W=4, stride=2, label="any")
    _, y_last = fg.windows(s, W=4, stride=2, label="last")
    assert y_any.shape == y_last.shape == (5, 3)
    assert np.array_equal(y_any, yw.max(axis=1)) and np.array_equal(y_last, yw[:, -1])
    with pytest.raises(ValueError, match="label must be"):
        fg.windows(s, W=4, label="bogus")
    with pytest.raises(ValueError, match="need integers 1 <= W"):
        fg.windows(s, W=13)
    with pytest.raises(ValueError, match="stride"):
        fg.windows(s, W=4, stride=0)
    for bad in ({"W": 4.5}, {"W": 4, "stride": 1.5}, {"W": True}):
        with pytest.raises(ValueError, match="need integers"):
            fg.windows(s, **{"W": 4, **bad})


def test_load_stream_rejects_an_unknown_system():
    with pytest.raises(ValueError, match="system must be"):
        fg.load_stream("bogus")


# ---- estimators and localizers ---------------------------------------------------------------------------


def test_estimator_constructor_checks():
    from fdia_graph.se import (
        WLS,
        AdaptiveWeighting,
        GatedPrior,
        JacobianWeighting,
        ResidualRemoval,
        SubspacePrior,
    )

    with pytest.raises(ValueError, match="npass and iters"):
        WLS(npass=0)
    with pytest.raises(ValueError, match="c must be"):
        AdaptiveWeighting(c=0.0)
    with pytest.raises(ValueError, match="threshold"):
        ResidualRemoval(threshold=0.0)
    with pytest.raises(ValueError, match="rank_frac"):
        SubspacePrior(rank_frac=1.5)
    with pytest.raises(ValueError, match="reweight"):
        SubspacePrior(reweight="bogus")
    with pytest.raises(ValueError, match="reweight"):
        JacobianWeighting(reweight="bogus")
    with pytest.raises(ValueError, match="gate"):
        GatedPrior()
    with pytest.raises(ValueError, match="gate_factor"):
        GatedPrior(gate="oracle", gate_factor=2.0)


def test_estimator_fit_and_score_input_checks(shard, splits):
    from fdia_graph.se import WLS

    with pytest.raises(ValueError, match="units='physical'"):
        WLS().fit(fg.load(shard, split="train", units="pu"))
    with pytest.raises(ValueError, match="benign records"):
        WLS().fit(fg.load(shard, split="train", families=["Aq"]))
    est = WLS().fit(splits["train"])
    with pytest.raises(ValueError, match="xhat must be"):
        est.score(splits["test"], xhat=np.zeros((3, est.SD)))


def test_localizer_input_checks(shard, splits):
    from fdia_graph.localization import SwingThreshold

    with pytest.raises(ValueError, match="fa_target"):
        SwingThreshold(fa_target=1.5)
    with pytest.raises(ValueError, match="benign records"):
        SwingThreshold().fit(fg.load(shard, split="train", families=["Aq"]))
    loc = SwingThreshold().fit(splits["train"])
    with pytest.raises(ValueError, match="scores must be"):
        loc.score(splits["test"], scores=np.zeros((2, 2)))


# ---- opt-in: the real IEEE-118 shard --------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("FDIA_SLOW"), reason="set FDIA_SLOW=1: downloads the 317 MB IEEE-118 shard"
)
def test_ieee118_estimator_sanity():
    """WLS on the published IEEE-118 shard: the benign angle error is a fraction of a degree and
    every class scores finite. A loose bound, so a data or solver regression shows without pinning
    the paper's numbers here (docs/se/README.md holds those)."""
    from fdia_graph.se import WLS

    train, test = fg.load("ieee118", split="train"), fg.load("ieee118", split="test")
    s = WLS().fit(train).score(test)
    assert s.benign is not None and 0 < s.benign.angle_mae_deg < 0.5
    assert all(np.isfinite(v.angle_mae_deg) and np.isfinite(v.voltage_mae_pu) for v in s.values())
