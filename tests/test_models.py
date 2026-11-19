"""Bundle, the dual-access base of the typed records: every dict use the package's callers rely
on must keep working, and the attribute view must agree with it."""

import json
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

from fdia_graph.models.base import Bundle


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
# bundles shared their fields: the shared base must not change what a user sees.
_SCAN = ["node_x", "node_m", "edge_x", "edge_m"]
_IDS = ["family", "stealthy", "seq_id", "timestep"]
_BENIGN = ["benign", "edge_benign"]
_PREV = ["prev_node_x", "prev_edge_x", "prev_timestep", "prev_swing"]
_TEMPORAL = ["temporal_delta", "swing"]
_CLEAN = ["clean", "edge_clean", "edge_clean_full"]
_PMU_I = [
    "pmu_i",
    "pmu_i_m",
    "pmu_i_benign",
]  # hybrid-meter timelines, last so older keys keep their order
EXPECTED = {
    # benign / edge_benign (0.18, timeline files) come after the layers a shard record carries
    "RecordBundle": (
        ["edge_index", *_SCAN, "y", *_IDS, "edge_attr", *_TEMPORAL, *_CLEAN, *_BENIGN, *_PMU_I],
        ["edge_index", *_SCAN, "y", *_IDS],
    ),
    "BatchBundle": (
        [*_SCAN, "y", *_TEMPORAL, *_CLEAN, *_BENIGN, "edge_index", "edge_attr", *_IDS, *_PMU_I],
        [*_SCAN, "y"],
    ),
    # edge_attr is new on ArraysBundle (it comes with RecordFields), last and never filled by the exports
    "ArraysBundle": (
        [
            "edge_index",
            *_SCAN,
            "y",
            *_TEMPORAL,
            *_CLEAN,
            *_BENIGN,
            *_IDS,
            *_PREV,
            "edge_attr",
            *_PMU_I,
            "prev_pmu_i",
        ],
        [],
    ),
}


def test_field_groups_keep_every_key_order_and_required_set():
    import dataclasses

    from fdia_graph.models import data

    for name, (order, required) in EXPECTED.items():
        cls = getattr(data, name)
        assert list(cls._names()) == order, name
        assert sorted(dataclasses.fields(cls), key=lambda f: f.name) == sorted(
            dataclasses.fields(cls), key=lambda f: f.name
        )
        assert set(f.name for f in dataclasses.fields(cls)) == set(order), name
        assert list(cls._required) == required, name


def test_required_fields_raise_at_construction():
    from fdia_graph.models.data import RecordBundle

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


def test_channels_name_the_documented_order():
    """`NodeChannel` is the one definition of the node_x order (|V|, P_inj, Q_inj, theta) and
    `EdgeChannel` of the flow order (P_from, Q_from); the legends a file carries name
    the same columns, and a member indexes the last axis of a numpy array or a tensor with no copy."""
    import torch

    from fdia_graph import schema
    from fdia_graph.constants.channels import (
        EDGE_FEAT_DEFAULT,
        TO_END_OFFSET,
        CurrentChannel,
        EdgeChannel,
        NodeChannel,
    )

    assert [c.name for c in NodeChannel] == ["V", "P_INJ", "Q_INJ", "THETA"]
    assert [c.name for c in EdgeChannel] == ["P_FROM", "Q_FROM"]
    assert schema.NODE_FEAT_LEGEND.upper().split(",") == [c.name for c in NodeChannel]
    assert [name.upper() for name in schema.FLOW_NAMES] == [c.name for c in EdgeChannel]
    assert EDGE_FEAT_DEFAULT == ",".join(schema.FLOW_NAMES)
    for channels in (NodeChannel, CurrentChannel):
        assert [int(c) for c in channels] == [0, 1, 2, 3]
    # the to end of a current sits TO_END_OFFSET columns after its from end
    assert CurrentChannel.RE_TO - CurrentChannel.RE_FROM == TO_END_OFFSET
    assert CurrentChannel.IM_TO - CurrentChannel.IM_FROM == TO_END_OFFSET
    a = np.arange(24.0).reshape(2, 3, 4)
    theta = a[..., NodeChannel.THETA]
    assert np.array_equal(theta, a[:, :, 3]) and np.shares_memory(theta, a)
    t = torch.arange(24.0).reshape(2, 3, 4)
    assert torch.equal(t[..., NodeChannel.P_INJ], t[:, :, 1])


def test_loader_columns_agree_with_the_definition(splits):
    """The clean layer read through the channels: voltages near 1 pu, the slack angle exactly zero."""
    from fdia_graph.constants.channels import NodeChannel

    ds = splits["test"]
    clean = ds.export(["clean"]).clean
    assert 0.8 < clean[..., NodeChannel.V].mean() < 1.2
    # the layer is float32; same tolerance as the loader
    assert np.allclose(clean[..., NodeChannel.THETA][:, ds.slack], 0.0, atol=1e-6)
