"""The file protocol lives in one module: the fixture file carries exactly the names it spells,
and no other module spells a dataset path."""

import os
import sys

import h5py
import numpy as np

import fdia_graph as fg
from fdia_graph import schema

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import readability  # noqa: E402


def test_the_fixture_carries_exactly_the_schema(timeline):
    with h5py.File(fg.load(timeline).path, "r") as f:
        names = []
        f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
        groups = {n for n in f}
        attrs = dict(f.attrs)
    per_frame = {v for v in schema.FIELD_PATH.values()} | {
        schema.SPLIT,
        schema.NODE_TAMPER,
        schema.EDGE_TAMPER,
        schema.MAG_PTR,
        schema.MAG_BUS,
        schema.MAG,
    }
    assert per_frame <= set(names)
    assert groups == {
        schema.Group.DATA,
        schema.Group.BENIGN,
        schema.Group.CLEAN,
        schema.Group.GRAPH,
        schema.Group.EPISODES,
        schema.Group.ATTACK,
    }
    episodes = {
        schema.path(schema.Group.EPISODES, k)
        for k in (
            schema.EPISODE_ONSET,
            schema.EPISODE_LENGTH,
            schema.EPISODE_FAMILY,
            schema.EPISODE_BUS_PTR,
            schema.EPISODE_BUS_IDX,
        )
    }
    assert episodes <= set(names)
    known = (
        per_frame
        | episodes
        | {schema.EDGE_INDEX, schema.EDGE_REACTANCE}
        | {schema.path(schema.Group.GRAPH, k) for k in schema.STATIC_PHYSICS}
    )
    assert set(names) <= known, sorted(set(names) - known)  # every dataset in the file has a name here
    assert attrs[schema.Attr.KIND] == schema.KIND_TIMELINE
    for key in (
        schema.Attr.N,
        schema.Attr.E,
        schema.Attr.T,
        schema.Attr.SYSTEM,
        schema.Attr.BASEMVA,
        schema.Attr.SEED,
        schema.Attr.FAMILIES,
        schema.Attr.ATTACKED_FRAC,
        schema.Attr.N_EPISODES,
        schema.Attr.FALLBACK_BENIGN,
    ):
        assert key in attrs, key
    assert np.array_equal(sorted(schema.SPLIT_CODE.values()), [0, 1, 2])


def test_no_module_but_schema_spells_a_dataset_path(tmp_path):
    assert readability.protocol_literals_all() == []
    assert set(readability._GROUPS) == {v for k, v in vars(schema.Group).items() if not k.startswith("_")}
    p = tmp_path / "m.py"
    p.write_text(
        '"""data/x in a docstring is fine."""\nx = f["data/node_x"]\ny = f"graph/{k}"\nz = "node_x"\n'
    )
    found = readability.protocol_literals(str(p))
    assert [(line, lit) for _, line, lit in found] == [(2, "data/node_x"), (3, "graph/")]
