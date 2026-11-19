"""The New York grid is a test case of the original generator like any IEEE case: coverage meters and
the search-based designs on the packaged case file and the checked-in days' pool (no solver, no
NYgrid data), through `SearchSource.for_pool`. `tests/test_any_case.py` has the IEEE side."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from fdia_graph import schema
from fdia_graph.constants.channels import NodeChannel
from fdia_graph.constants.families import AM_CODE, AT_CODE
from fdia_graph.constants.grid import LOAD_FLOOR_MW
from fdia_graph.schema import Attr

from .inputs import DAYS_POOL


@pytest.fixture(scope="module")
def grid():
    """(the New York network, the days' pool)"""
    pytest.importorskip("matpowercaseframes")  # the packaged case file's reader (the [ny] extra)
    from fdia_graph.data.pool import read_pool
    from fdia_graph.systems.case import nygrid_case
    from fdia_graph.systems.network import Network

    return Network(nygrid_case()), read_pool(DAYS_POOL)


def _settings(**knobs):
    """The original designs with the rating of each window set from its own flow ("delta") and a 10%
    load cap."""
    from fdia_graph.models.config import OverloadSettings, TimelineSettings

    am = OverloadSettings(rating_source="delta", load_cap=0.1)
    return TimelineSettings.of(ramp_len=12, attacked_frac=0.3, min_budget=32, am_attack=am, **knobs)


def test_the_loads_of_a_dispatched_pool_are_read_at_the_grids_own_buses(grid):
    from fdia_graph.generation.search import SearchSource

    net, pool = grid
    source = SearchSource.for_pool(net, pool, _settings())
    g, case = source.g, net.case
    has_unit = np.isin(np.arange(len(case.bus_id)), case.unit_bus)
    targets = g.load_bus[g.stealthy_pos]
    # a ramp's targets: New York buses with no unit whose mean load in the pool is above the floor
    assert len(targets) and case.own_bus[targets].all() and not has_unit[targets].any()
    assert (source.X[:, targets, NodeChannel.P_INJ].mean(axis=0) > LOAD_FLOOR_MW).all()
    # a unit the pool dispatches is not idle, whatever the case's base state says: its bus is no junction
    assert not set(g.zero_inj) & set(case.unit_bus.tolist())
    # the units an overload may move are New York's
    assert len(g.generator_buses()) and case.own_bus[g.generator_buses()].all()
    # a bus with a unit is all generation, any other bus all load
    X = source.X[0]
    assert (g.split.load(X)[has_unit] == 0).all()
    assert np.array_equal(g.split.load(X)[~has_unit], X[~has_unit, NodeChannel.P_INJ])


@pytest.mark.parametrize("split_by", ["time", "week"])
def test_the_original_generator_takes_the_new_york_case(grid, tmp_path, split_by):
    from fdia_graph.constants.newyork import SolveKind
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.generation.search import SearchSource

    net, pool = grid
    solved = pool.solve_kind == SolveKind.SOLVED
    out = SearchSource.for_pool(net, pool, _settings(split_by=split_by), seed=5).generate(tmp_path / "ny.h5")
    with h5py.File(out, "r") as f:
        attrs = dict(f.attrs)
        family = f[schema.path(schema.Group.EPISODES, schema.EPISODE_FAMILY)][:]
        labelled = f[schema.path(schema.Group.EPISODES, schema.EPISODE_BUS_IDX)][:]
        seq, split, time = f[schema.SEQ_ID][:], f[schema.SPLIT][:], f[schema.TIME][:]
        seen, benign = f[schema.NODE_X][:], f[schema.BENIGN_PATH["benign"]][:]
        clean = f[schema.FIELD_PATH["clean"]][:]
    # the file: one frame per solved pool frame, on the 140-bus case, with each frame's time
    assert attrs[Attr.T] == int(solved.sum()) == len(seq) and len(net.case.bus_id) == seen.shape[1]
    assert (time == pool.time[solved].astype("datetime64[s]").astype(np.int64)).all()
    assert json.loads(attrs[Attr.SETTINGS])["loads"] == "net"
    # injections are positive for load, as in every file of this generator
    p_gen, _ = net.injections(pool.vm[solved], pool.va[solved])
    assert np.allclose(clean[:, :, NodeChannel.P_INJ], -p_gen, atol=1e-2)
    # both families are built, the share attacked is near the target, and every labelled bus is New York's
    assert {AT_CODE, AM_CODE} <= set(family.tolist())
    built, requested = attrs[Attr.EPISODES_BUILT].sum(), attrs[Attr.EPISODES_REQUESTED].sum()
    assert built >= 0.8 * requested and 0.2 <= attrs[Attr.ATTACKED_FRAC] <= 0.35
    assert net.case.own_bus[labelled].all()
    # the pretended changes: a ramp within tens of MW at a bus, an overload within a unit's limits
    change = np.abs(seen[:, :, NodeChannel.P_INJ] - benign[:, :, NodeChannel.P_INJ]).max(axis=1)
    largest = {int(code): change[np.isin(seq, np.flatnonzero(family == code))].max() for code in set(family)}
    assert largest[AT_CODE] < 50.0 and largest[AM_CODE] < 1000.0
    if split_by == "week":  # the two days lie in two weeks: each week whole in one split
        week = (time - time[0]) // (7 * 24 * 3600)
        assert all(len(set(split[week == w])) == 1 for w in set(week.tolist()))
    assert len(FdiaGraph(str(out))) == attrs[Attr.T]


def test_a_new_york_source_needs_the_path_of_its_file(grid):
    from fdia_graph.errors import ConfigError
    from fdia_graph.generation.search import SearchSource

    net, pool = grid
    with pytest.raises(ConfigError, match="a timeline needs the path of its file"):
        SearchSource.for_pool(net, pool, _settings()).generate()
