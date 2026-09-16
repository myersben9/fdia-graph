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
    assert json.loads(json.dumps(_Aliased(1, 2)))["global"] == 2  # json.dump works on the dict side
    back = pickle.loads(pickle.dumps(r))
    assert back.family == 2 and np.array_equal(back["node_x"], r.node_x)


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
