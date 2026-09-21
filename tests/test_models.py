"""Bundle, the dual-access base of the typed records: every dict use the package's callers rely
on must keep working, and the attribute view must agree with it."""

import json
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from fdia_graph.models import Bundle


@dataclass(frozen=True, eq=False)
class _Rec(Bundle):
    node_x: np.ndarray
    y: np.ndarray
    family: int
    swing: Optional[np.ndarray] = None


@dataclass(frozen=True, eq=False)
class _Aliased(Bundle):
    _keys = {"global_": "global"}
    bus: int
    global_: int


def _rec(with_swing: bool) -> _Rec:
    return _Rec(np.ones((3, 4)), np.zeros(3), 2, np.ones((3, 2)) if with_swing else None)


def test_dict_view_equals_attribute_view():
    r = _rec(True)
    assert isinstance(r, dict)
    assert r["family"] == r.family == 2
    assert np.array_equal(r["node_x"], r.node_x)
    assert list(r) == ["node_x", "y", "family", "swing"]
    assert r.to_dict() == {"node_x": r.node_x, "y": r.y, "family": 2, "swing": r.swing}
    assert dict(**r)["family"] == 2  # ** unpacking


def test_optional_fields_are_absent_when_none():
    r = _rec(False)
    assert "swing" not in r and "node_x" in r
    assert list(r) == ["node_x", "y", "family"] and len(r) == 3
    with pytest.raises(KeyError):
        r["swing"]


def test_read_only_json_and_pickle():
    r = _rec(False)
    with pytest.raises(Exception):
        r.family = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        r["family"] = 3
    for mutate in (
        lambda: r.update(family=3),
        lambda: r.pop("family"),
        r.clear,
        r.popitem,
        lambda: r.setdefault("z", 1),
    ):
        with pytest.raises(TypeError):
            mutate()
    with pytest.raises(TypeError):
        r |= {"family": 3}  # type: ignore[misc]
    assert r.family == 2 and r["family"] == 2 and "z" not in r
    assert json.loads(json.dumps(_Aliased(1, 2)))["global"] == 2  # json.dump works on the dict side
    back = pickle.loads(pickle.dumps(r))
    assert back.family == 2 and np.array_equal(back["node_x"], r.node_x)


@dataclass(frozen=True, eq=False)
class _Tailed(Bundle):
    _tail = ("geo",)
    geo: int
    benign: Optional[int] = None


def test_pickle_restores_fields_and_key_order():
    t = pickle.loads(pickle.dumps(_Tailed(geo=1, benign=2)))
    assert t.geo == 1 and t.benign == 2 and list(t) == ["benign", "geo"]  # geo is required yet listed last
    r = pickle.loads(pickle.dumps(_Rec.ordered({"family": 2, "y": np.zeros(3), "node_x": np.ones((3, 4))})))
    assert list(r) == ["family", "y", "node_x"] and r.family == 2


# The dict-key order and the required fields of every shard-shaped bundle, as they were before the
# field groups (docs/plans/FIELD_GROUPS_PLAN.md): the groups must not change what a user sees.
_SCAN = ["node_x", "node_m", "edge_x", "edge_m"]
_IDS = ["family", "stealthy", "seq_id", "timestep"]
_BENIGN = ["benign", "edge_benign"]
_TEMPORAL = ["temporal_delta", "swing"]
_CLEAN = ["clean", "edge_clean", "edge_clean_full"]
EXPECTED = {
    # benign / edge_benign (0.18, timeline files) come after the layers a shard record carries
    "RecordBundle": (
        ["edge_index", *_SCAN, "y", *_IDS, "edge_attr", *_TEMPORAL, *_CLEAN, *_BENIGN],
        ["edge_index", *_SCAN, "y", *_IDS],
    ),
    "BatchBundle": (
        [*_SCAN, "y", *_TEMPORAL, *_CLEAN, *_BENIGN, "edge_index", "edge_attr", *_IDS],
        [*_SCAN, "y"],
    ),
    # edge_attr is new on ArraysBundle (it comes with GraphFields), last and never filled by the exports
    "ArraysBundle": (
        ["edge_index", "edge_reactance", *_SCAN, "y", *_TEMPORAL, *_CLEAN, *_BENIGN, *_IDS, "edge_attr"],
        [],
    ),
    "Stream": (
        [
            "node_x",
            "benign",
            "clean",
            "edge_x",
            "edge_benign",
            "edge_clean",
            "edge_index",
            "edge_attr",
            "node_m",
            "edge_m",
            "y",
            "family",
            "temporal_delta",
            "swing",
            "timestep",
            "episodes",
            "system",
            "attacked_frac",
            "stealthy",
            "seq_id",
            "edge_clean_full",
        ],
        [
            "node_x",
            "benign",
            "clean",
            "edge_x",
            "edge_benign",
            "edge_clean",
            "edge_index",
            "edge_attr",
            "node_m",
            "edge_m",
            "y",
            "family",
            "temporal_delta",
            "swing",
            "timestep",
            "episodes",
        ],
    ),
}


def test_field_groups_keep_every_key_order_and_required_set():
    import dataclasses

    import fdia_graph.models as m

    for name, (order, required) in EXPECTED.items():
        cls = getattr(m, name)
        assert list(cls._names()) == order, name
        assert sorted(dataclasses.fields(cls), key=lambda f: f.name) == sorted(
            dataclasses.fields(cls), key=lambda f: f.name
        )
        assert set(f.name for f in dataclasses.fields(cls)) == set(order), name
        assert list(cls._required) == required, name


def test_group_built_bundles_are_keyword_only():
    """Positional construction would bind to the inherited field order, not the documented one, so
    it is refused outright rather than allowed to mis-bind silently."""
    from fdia_graph.models import RecordBundle, TrueState

    with pytest.raises(TypeError, match="by keyword"):
        RecordBundle(np.zeros((2, 1)), np.ones((3, 4)))
    assert TrueState(np.zeros((2, 5)), np.zeros(2)).thsl.shape == (
        2,
    )  # explicit bundles still take positionals


def test_required_fields_raise_at_construction():
    from fdia_graph.models import RecordBundle

    kw = dict(
        edge_index=np.zeros((2, 1)),
        node_x=np.ones((3, 4)),
        node_m=np.ones((3, 4)),
        edge_x=np.ones((1, 2)),
        edge_m=np.ones((1, 2)),
        y=np.zeros(3),
        family=0,
        stealthy=0,
        seq_id=-1,
        timestep=0,
    )
    rec = RecordBundle(**kw)
    assert list(rec) == EXPECTED["RecordBundle"][0][:10] and rec.swing is None and "swing" not in rec
    with pytest.raises(TypeError, match="missing required"):
        RecordBundle(**{k: v for k, v in kw.items() if k != "node_x"})


def test_ordered_keeps_the_mapping_key_order():
    r = _Rec.ordered({"family": 2, "y": np.zeros(3), "node_x": np.ones((3, 4))})
    assert list(r) == ["family", "y", "node_x"] and r.node_x.shape == (3, 4) and "swing" not in r


def test_aliased_key():
    a = _Aliased(bus=1, global_=2)
    assert a["global"] == a.global_ == 2 and list(a) == ["bus", "global"]


def test_default_collate_treats_a_bundle_as_a_dict():
    """PyTorch's default collate batches a Mapping key by key and falls back to a plain dict when the
    mapping type cannot be built from a dict, so a DataLoader over bundle records batches exactly
    as it batched dict records."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    recs = [_Rec(torch.ones(3, 4) * i, torch.zeros(3), i, None) for i in range(4)]
    batch = next(iter(DataLoader(recs, batch_size=4)))
    assert set(batch) == {"node_x", "y", "family"}
    assert tuple(batch["node_x"].shape) == (4, 3, 4) and batch["family"].tolist() == [0, 1, 2, 3]


def test_column_views_name_the_documented_order():
    """`NodeColumns` is the one definition of the node_x order (|V|, P_inj, Q_inj, theta) and
    `EdgeColumns` of the edge_x order (P_from, Q_from); the views share memory with the array."""
    from fdia_graph.models import BRANCH, EDGE, NODE, BranchColumns, EdgeColumns, NodeColumns, RecordBundle

    assert NODE == (0, 1, 2, 3) and NODE._fields == ("v", "p_inj", "q_inj", "theta")
    assert EDGE == (0, 1) and EDGE._fields == ("p_from", "q_from")
    assert BRANCH._fields == ("r", "x", "b", "g", "gs", "bs", "tap", "shift")
    phys = np.arange(16.0).reshape(2, 8)
    branch = BranchColumns.of(phys)
    assert np.array_equal(branch.tap, phys[:, 6]) and np.shares_memory(branch.shift, phys)
    a = np.arange(24.0).reshape(2, 3, 4)
    cols = NodeColumns.of(a)
    assert np.array_equal(cols.theta, a[:, :, 3]) and np.shares_memory(cols.v, a)
    rec = RecordBundle(
        edge_index=np.zeros((2, 1), int),
        node_x=a[0],
        node_m=np.ones((3, 4)),
        edge_x=np.ones((1, 2)),
        edge_m=np.ones((1, 2)),
        y=np.zeros(3),
        family=0,
        stealthy=0,
        seq_id=-1,
        timestep=0,
    )
    assert np.array_equal(rec.node().p_inj, a[0, :, 1]) and rec.edge().q_from.shape == (1,)
    assert EdgeColumns.of(np.ones((5, 2))).p_from.shape == (5,)


def test_loader_columns_agree_with_the_definition(splits):
    """The clean layer read through the views: voltages near 1 pu, the slack angle exactly zero."""
    from fdia_graph.models import NodeColumns

    ds = splits["test"]
    cols = NodeColumns.of(ds.export(["clean"]).clean)
    assert 0.8 < cols.v.mean() < 1.2
    assert np.allclose(
        cols.theta[:, ds.slack], 0.0, atol=1e-6
    )  # the layer is float32; same tolerance as the loader
