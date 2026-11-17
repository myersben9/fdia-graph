"""The public edges: wrong arguments fail early with a message naming what is allowed, and the
filters mean what the docs say. The other test files prove the happy path; this file is the rest."""

import numpy as np
import pytest

import fdia_graph as fg
from fdia_graph.dataset.base import family_ids
from fdia_graph.models.choices import FAMILIES

# ---- registry and loading -------------------------------------------------------------------------


def test_unknown_dataset_and_system_spellings():
    with pytest.raises(KeyError, match="unknown dataset"):
        fg.resolve("no_such_dataset")
    from fdia_graph.registry import system_id

    with pytest.raises(ValueError, match="system must be"):
        system_id("bogus")


def test_family_ids_accept_names_and_codes_and_reject_unknowns():
    assert family_ids(["benign", "Aq", "Al"]) == [0, 1, 6]
    assert family_ids([1, 5]) == [1, 5]
    for old in ("Ao", "SLS", "ramp", "LRA"):  # the names of the first releases are gone
        with pytest.raises(ValueError, match="unknown family"):
            family_ids([old])
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


def test_load_rejects_bad_units_split_and_family(timeline):
    with pytest.raises(ValueError, match="units must be"):
        fg.load(timeline, units="feet")
    with pytest.raises(ValueError, match="split must be one of"):
        fg.load(timeline, split="bogus")
    with pytest.raises(ValueError, match="unknown family"):
        fg.load(timeline, families=["Bogus"])


def test_family_filter_and_heldout_protocol(timeline):
    """On an old release that carries them, the older families select by name or code, and the unseen-attack
    protocol (As and Ar held out of train and val) selects as documented."""
    from fdia_graph.dataset import FdiaGraph
    from tests.helpers import TIMELINE_V083

    by_name = FdiaGraph(TIMELINE_V083, families=["Aq"])
    by_code = FdiaGraph(TIMELINE_V083, families=[1])
    assert len(by_name) == len(by_code) > 0 and np.array_equal(by_name.idx, by_code.idx)
    for split in ("train", "val"):
        fams = set(FdiaGraph(TIMELINE_V083, split=split, heldout=True).export(["family"])["family"].tolist())
        assert not fams & {3, 4}, f"As/Ar must be held out of {split}"
    test_fams = set(
        FdiaGraph(TIMELINE_V083, split="test", heldout=True).export(["family"])["family"].tolist()
    )
    assert test_fams & {3, 4}, "the test split keeps As/Ar"
    assert len(fg.load(timeline, families=["At"])) == len(fg.load(timeline, families=[5])) > 0


# ---- exports ----------------------------------------------------------------------------------------


def test_export_rejects_unknown_fields_and_empty_means_default(splits):
    ds = splits["test"]
    with pytest.raises(ValueError, match="unknown field"):
        ds.export(["node_x", "nope"])
    assert list(ds.export([])) == list(ds.export())
    assert list(ds.export(["y"])) == ["edge_index", "y"]


def test_default_fields_follow_the_layers_present(splits):
    """`edge_clean` is offered only when the file carries it, so a requested field is always in
    the returned bundle (the check and the gather use the same list)."""
    ds = splits["test"]
    fields = ds._default_fields()
    assert ("clean" in fields) == (ds._clean_np is not None)
    assert ("edge_clean" in fields) == (ds._eclean_np is not None)
    assert set(ds.export(fields)) == set(fields) | {"edge_index"}


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


# ---- windows ------------------------------------------------------------------------------------------


def test_windows_labels_and_bounds():
    """The label helper of `ds.windows` and the checks of its arguments."""
    from fdia_graph.dataset.sequence import window_labels
    from fdia_graph.models.config import WindowSpec

    y = (np.random.default_rng(0).random((12, 3)) > 0.7).astype(np.uint8)
    starts = range(0, 12 - 4 + 1, 2)
    yw = window_labels(y, starts, 4, "frame")
    assert yw.shape == (5, 4, 3)
    y_any = window_labels(y, starts, 4, "any")
    y_last = window_labels(y, starts, 4, "last")
    assert y_any.shape == y_last.shape == (5, 3)
    assert np.array_equal(y_any, yw.max(axis=1)) and np.array_equal(y_last, yw[:, -1])
    with pytest.raises(ValueError, match="label must be"):
        WindowSpec(12, 4, 1, "bogus")
    with pytest.raises(ValueError, match="need integers 1 <= W"):
        WindowSpec(12, 13, 1, "any")
    with pytest.raises(ValueError, match="stride"):
        WindowSpec(12, 4, 0, "any")
    for W, stride in ((4.5, 1), (4, 1.5), (True, 1)):
        with pytest.raises(ValueError, match="WindowSpec.(W|stride) must be an integer"):
            WindowSpec(12, W, stride, "any")
