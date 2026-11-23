"""A timeline read back as clients (`dataset.clients`): with the `Ownership` of the buses the source
gives each meter to the client of its bus, and the file reads as one view per client and one for the
operator. On the packaged New York case with made-up clients and the checked-in days' pool: no NYgrid
data, no solver."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from fdia_graph import schema
from fdia_graph.constants.meters import ANGLE_KINDS
from fdia_graph.models.choices import Split

from .inputs import DAYS_POOL


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """(the open file as clients, its source): the original generator on the days' pool, split by weeks,
    with three made-up clients over the New York buses."""
    pytest.importorskip("matpowercaseframes")  # the packaged case file's reader (the [ny] extra)
    from fdia_graph.data.pool import read_pool
    from fdia_graph.dataset.clients import Dataset
    from fdia_graph.generation.search import SearchSource
    from fdia_graph.models.config import TimelineSettings
    from fdia_graph.systems.case import nygrid_case
    from fdia_graph.systems.network import Network
    from fdia_graph.systems.ownership import Ownership

    case = nygrid_case()
    bus_client = np.where(case.is_ny, np.arange(len(case.bus_id)) % 3, -1)
    own = Ownership(np.array(["a", "b", "c"]), bus_client, bus_client[case.f] != bus_client[case.t])
    settings = TimelineSettings.of(families=("At",), ramp_len=12, attacked_frac=0.3, split_by="week")
    source = SearchSource.for_pool(Network(case), read_pool(DAYS_POOL), settings, seed=5, own=own)
    path = source.generate(tmp_path_factory.mktemp("dataset") / "timeline.h5")
    return Dataset(path), source


def test_the_file_records_the_clients_and_the_meter_table_comes_back(written):
    ds, source = written
    plan, own = source.g.plan, source.own
    assert ds.clients == ["a", "b", "c"] and np.array_equal(ds.own.bus_client, own.bus_client)
    # each meter belongs to the client of its bus; a meter at a neighbour's bus belongs to none
    assert np.array_equal(plan.owner, own.bus_client[plan.bus]) and (plan.owner == -1).any()
    # the plan read from the file's masks holds the same meters, whatever their row order
    rows = lambda p: sorted(zip(p.kind.tolist(), p.bus.tolist(), p.branch.tolist(), p.end.tolist()))
    assert rows(ds.plan) == rows(plan)
    assert np.array_equal(ds.plan.owner, own.bus_client[ds.plan.bus])
    # and its flat vector is the file's layers, meter by meter
    frames = np.arange(3)
    back = ds.plan.scatter(ds.z(frames), len(own.bus_client), source.g.n_branch)
    assert np.array_equal(back.node, ds.h[schema.NODE_X][:3])
    assert np.array_equal(back.edge[..., :2], ds.h[schema.EDGE_X][:3])
    assert np.array_equal(back.current, ds.h[schema.PMU_I][:3])


def test_client_view_holds_only_its_own_meters_buses_and_labels(written):
    from fdia_graph.dataset.clients import ClientView

    ds, source = written
    T = source.frames
    z, labels = ds.z(np.arange(T)), ds.h[schema.Y][:].astype(bool)
    seen = np.zeros(len(ds.plan.kind), int)
    for c, name in enumerate(ds.clients):
        v = ds.client(name)
        mine = np.flatnonzero(ds.plan.owner == c)
        assert isinstance(v, ClientView) and (v.meters.owner == c).all() and len(v.meters.kind) == len(mine)
        assert np.array_equal(v.buses, np.flatnonzero(source.own.bus_client == c))
        assert set(v.meters.bus.tolist()) <= set(v.buses.tolist())  # its meters sit at its own buses
        assert np.array_equal(v.z, z[:, mine]) and np.array_equal(v.labels, labels[:, v.buses])
        # the view keeps no file handle and no array wider than its own meters or buses
        for value in vars(v).values():
            assert not hasattr(value, "file")
            if isinstance(value, np.ndarray) and value.ndim == 2:
                assert value.shape[1] in (len(mine), len(v.buses))
        assert v.node_features().shape == (T, len(v.buses), 4) and v.temporal().shape == v.z.shape
        seen[mine] += 1
    assert (seen[ds.plan.owner >= 0] == 1).all() and (seen[ds.plan.owner < 0] == 0).all()
    assert labels.any() and not labels[:, source.own.bus_client < 0].any()  # every label is some client's


def test_no_new_york_ramp_frame_steps_past_its_stealth_bound(written):
    """Every attacked frame of every ramp on the New York days keeps the stealth bound: the ramp is held
    on a support the search found, never run on its whole area past the bound."""
    from tests.helpers import ramp_frames_past_the_bound

    ds, source = written
    devices = ds.h[schema.Group.EPISODES][schema.EPISODE_MIN_DEVICES][:]
    assert len(devices) >= 3 and (devices >= 0).all()
    assert ramp_frames_past_the_bound(source.g, source.X, ds.h.filename) == 0


def test_operator_and_split_views(written):
    ds, source = written
    T = source.frames
    op = ds.operator()
    assert op.z.shape == (T, len(ds.plan.kind)) and op.labels.shape == (T, len(source.own.bus_client))
    sizes = []
    for code, split in enumerate(Split.values()):
        v = ds.client("a", split)
        assert np.array_equal(v.frames, np.flatnonzero(source.split == code))
        gaps = np.diff(v.frames) != 1
        assert (v.temporal()[1:][gaps] == 0).all()  # no change across another split's frames
        sizes.append(len(v.z))
    assert sum(sizes) == T


def test_temporal_wraps_angle_changes(written):
    from fdia_graph.dataset.clients import ClientView

    plan = written[0].plan
    angle = np.isin(plan.kind, ANGLE_KINDS)
    z = np.zeros((3, len(plan.kind)))
    z[:, angle] = [[179.5], [-179.5], [179.0]]  # crosses +-180 twice
    z[:, ~angle] = [[1.0], [3.0], [2.0]]
    view = ClientView(z, np.zeros((3, 0), bool), plan, np.arange(0), np.arange(3))
    d = view.temporal()
    assert angle.any() and np.allclose(d[1:, angle], [[1.0], [-1.5]])  # wrapped to (-180, 180]
    assert np.allclose(d[1:, ~angle], [[2.0], [-1.0]]) and (d[0] == 0).all()
    assert np.array_equal(view.z, z)  # the readings themselves stay raw


def test_a_file_without_clients_is_refused(tmp_path):
    from fdia_graph.dataset.clients import Dataset
    from fdia_graph.errors import NotATimeline

    old = tmp_path / "old.h5"
    with h5py.File(old, "w") as h:
        h.create_dataset("frames/z", data=np.zeros((2, 3), np.float32))
    with pytest.raises(NotATimeline, match="clients"):
        Dataset(old)


def test_a_client_file_in_another_layout_is_refused(written, tmp_path):
    """A file that declares polar currents or flows at both branch ends is refused when it is opened
    as clients, as it is by `FdiaGraph`: no column of it is read as something else."""
    import shutil

    from fdia_graph.dataset.clients import Dataset
    from fdia_graph.errors import NotATimeline

    source_file = written[0].h.filename
    for attr, value, says in (
        (schema.Attr.CURRENT_FORM, "polar", "real and imaginary part"),
        (schema.Attr.EDGE_FEAT, "P_from,Q_from,P_to,Q_to", "flows at the from end"),
    ):
        other = tmp_path / f"{attr}.h5"
        shutil.copy(source_file, other)
        with h5py.File(other, "a") as f:
            f.attrs[attr] = value
        with pytest.raises(NotATimeline, match=says):
            Dataset(other)
