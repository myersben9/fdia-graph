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
    # the fixture is a hybrid-meter file [D10]: it carries the PMU current layers
    per_frame = (
        set(schema.FIELD_PATH.values())
        | set(schema.CURRENT_LAYERS)
        | {
            schema.SPLIT,
            schema.NODE_TAMPER,
            schema.EDGE_TAMPER,
            schema.MAG_PTR,
            schema.MAG_BUS,
            schema.MAG,
        }
    )
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
    # the fewest-tamper search's and the overload attack's per-episode rows (min_*, am_*)
    searched = {
        schema.path(schema.Group.EPISODES, v)
        for k, v in vars(schema).items()
        if k.startswith(("EPISODE_MIN_", "EPISODE_AM_")) and isinstance(v, str)
    }
    known = (
        per_frame
        | episodes
        | searched
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
    assert set(attrs) <= schema.ATTR_KEYS, sorted(set(attrs) - schema.ATTR_KEYS)  # every key is named
    with h5py.File(fg.load(timeline).path, "r") as f:
        assert set(f[schema.Group.GRAPH].attrs) <= schema.ATTR_KEYS
        assert set(f[schema.Group.ATTACK].attrs) <= schema.ATTR_KEYS
    assert np.array_equal(sorted(schema.SPLIT_CODE.values()), [0, 1, 2])


SNIPPET = "\n".join(
    [
        '"""data/x in a docstring is fine."""',
        'x = f["data/node_x"]',
        'y = f"graph/{k}"',
        'z = "node_x"',
        'g = f.create_group("attack")',
        'd = f["data"]',
        'ok = a["clean"]',
        'has = "episodes" in f',
        'g2 = f.create_group(name="graph")',
        "",
    ]
)


def test_no_module_but_schema_spells_a_dataset_path(tmp_path):
    assert readability.protocol_literals_all() == []
    assert set(readability._GROUPS) == {v for k, v in vars(schema.Group).items() if not k.startswith("_")}
    p = tmp_path / "m.py"
    p.write_text(SNIPPET)
    found = readability.protocol_literals(str(p))
    assert [(line, lit) for _, line, lit in found] == [
        (2, "data/node_x"),
        (3, "graph/"),
        (5, "attack"),
        (6, "data"),
        (8, "episodes"),
        (9, "graph"),
    ]  # a["clean"] is a record field, not a group, and is not flagged
    schema_copy = tmp_path / "schema.py"
    schema_copy.write_text('x = f["data/node_x"]\n')
    assert readability.protocol_literals(str(schema_copy))  # only the package's schema.py is exempt


def test_no_input_is_checked_by_hand_outside_the_models():
    """Every input check is declared on a model and run by the one engine (CONTRIBUTING.md);
    a condition only the data reveals raises a named error from fdia_graph.errors."""
    assert readability.hand_checks_all() == []


def test_every_any_is_listed_with_its_reason():
    """An annotation says Any only where the value can be anything (readability.ANY_ALLOWED); every
    other annotation carries its real type, and a listed place that no longer says Any is dropped."""
    assert readability.any_unlisted() == []
    assert readability.any_stale() == []
    assert all(reason.strip() for reason in readability.ANY_ALLOWED.values())


def test_the_any_rule_finds_every_kind_of_annotation(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(
        "from typing import Annotated, Any, Optional\n"
        "Alias = Optional[Any]\n"
        "def f(a: Any, *args: Any, b: dict[str, Any], **kw: Any) -> Any: ...\n"
        "class C:\n"
        "    x: Annotated[Any, 1]\n"
        "    y: int\n"
        "    def g(self, z: list[Any]) -> None:\n"
        "        w: Any = 1\n"
        "        self.v: Any = 2\n"
        "Bare = Any\n"
        "Union604 = int | Any\n"
    )
    prefix = readability._rel(str(p)).replace(os.sep, "/") + ":"
    found = [k[len(prefix) :] for k, _ in readability.any_annotations(str(p))]
    assert found == [
        "Alias",
        "f:a",
        "f:b",
        "f:*args",
        "f:**kw",
        "f:return",
        "C.x",
        "C.g:z",
        "C.g.w",
        "C.g.self.v",
        "Bare",
        "Union604",
    ]


def test_the_equation_map_refuses_a_malformed_wu26_tag(tmp_path):
    """Any citation of the paper in brackets that is neither the bare reference nor a full tag is
    reported (the citations below are assembled, so this file carries none of them)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "equation_map",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "equation_map.py"),
    )
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    src = tmp_path / "x.py"
    bodies = ("WU26", "WU26 eq. 12", "WU26 eq 12", "WU26 eq. (12)", "WU26, eq. 3")
    cites = " ".join("[" + body + "]" for body in bodies)
    src.write_text(f'def f():\n    """{cites}"""\n', encoding="utf-8")
    c = em.Citations()
    c.scan(str(src))
    bad = [p for p in c.problems if "is not a tag" in p]
    assert len(bad) == 3 and c.code["12"]  # the valid tag still counts
    assert em.numbers("21-23, 27") == ["21", "22", "23", "27"]
