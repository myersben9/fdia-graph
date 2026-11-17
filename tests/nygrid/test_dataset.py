"""A New York timeline read back (`dataset.clients`): the file holds what the source stored, in the layout of
`fdia_graph.schema`; the meter table and the flat reading vector come back from its layers exactly; each client
sees only its own meters, buses and labels; the operator and split views select the right frames; and the
static graph gives the case's admittance matrix. On the synthetic grid of `conftest.py`: no data, no solver."""

from dataclasses import fields

import h5py
import numpy as np
import pytest

from fdia_graph import schema
from fdia_graph.constants.families import FAMILY_CODE
from fdia_graph.models.choices import Split
from tests.helpers import ATTR_KEYS, CURRENT_LAYERS

from .conftest import LENGTHS, UNSOLVED, T


@pytest.fixture(scope="module")
def written(grid, tmp_path_factory):
    """(the open file, its source, what the source stored): the flat vectors the source laid out as layers
    (every value `MeterPlan.scatter` was given, a row per frame), and each frame's family and episode row as
    its designed episodes say."""
    from fdia_graph.dataset.clients import Dataset
    from fdia_graph.engine.attacks.attacker import Attacker
    from fdia_graph.engine.sensing.measure import Measurement
    from fdia_graph.engine.sensing.meters import MeterPlan
    from fdia_graph.generation.nygrid.source import NewYorkSource
    from fdia_graph.models.config import NewYorkSettings

    laid_out = []

    class Recording(MeterPlan):
        def scatter(self, z, buses, branches):
            laid_out.append(np.asarray(z))
            return super().scatter(z, buses, branches)

    pool, att, families, own = grid
    plan = att.meas.plan
    columns = [f.name for f in fields(plan) if f.init]  # `slots` is worked out from them
    meas = Measurement(att.net, Recording(**{name: getattr(plan, name) for name in columns}))
    settings = NewYorkSettings(length=LENGTHS, block=100)
    source = NewYorkSource(pool, Attacker(att.net, meas), families, own, settings)
    path = source.generate(tmp_path_factory.mktemp("dataset") / "timeline.h5")
    # the first value laid out is the meter mask; then, block by block, the scans, their benign twins and
    # the tampered meters
    names = ("z", "z_benign", "tampered")
    stored = {name: np.concatenate(laid_out[1 + i :: 3]) for i, name in enumerate(names)}
    stored["family"], stored["episode"] = np.zeros(T, int), np.full(T, -1)
    for row, ep in enumerate(source._episodes):
        stored["family"][ep.onset : ep.onset + ep.length] = FAMILY_CODE[ep.family]
        stored["episode"][ep.onset : ep.onset + ep.length] = row
    return Dataset(path), source, stored


def _all(ds):
    return np.arange(ds.h[schema.Y].shape[0])


def test_the_meter_table_and_the_flat_vectors_come_back_exactly(written):
    from fdia_graph.engine.sensing.meters import MeterPlan

    ds, source, stored = written
    plan = source.meas.plan
    # each meter has its own place in the layers, or two meters would share a stored number
    place = np.c_[plan.kind, np.where(plan.branch >= 0, plan.branch, plan.bus), plan.end]
    assert len(np.unique(place, axis=0)) == len(plan.kind)
    for f in fields(MeterPlan):  # the same rows in the same order; the bias is not in a file
        if f.init and f.name != "bias":  # `slots` is worked out from the columns
            assert np.array_equal(getattr(ds.plan, f.name), getattr(plan, f.name)), f.name
    assert np.isnan(ds.plan.bias).all()
    frames = _all(ds)
    assert len(frames) == T and np.isnan(ds.z(frames)[UNSOLVED]).all()  # an unsolved frame keeps its place
    assert np.array_equal(ds.z(frames), stored["z"].astype(np.float32), equal_nan=True)
    assert np.array_equal(ds.z(frames, benign=True), stored["z_benign"].astype(np.float32), equal_nan=True)
    assert np.array_equal(ds.tampered(frames), stored["tampered"]) and stored["tampered"].any()


def test_the_file_holds_what_the_source_stored(written):
    ds, source, stored = written
    h, pool, own = ds.h, source.pool, source.own
    assert ds.clients == own.names.tolist() and np.array_equal(ds.own.bus_client, own.bus_client)
    assert np.array_equal(ds.own.tie_lines, own.tie_lines)
    assert np.array_equal(h[schema.TIME][:].astype("datetime64[s]"), pool.time)
    assert np.array_equal(h[schema.SPLIT][:], source._split)
    assert np.array_equal(h[schema.FAMILY][:], stored["family"])
    assert np.array_equal(h[schema.SEQ_ID][:], stored["episode"])
    assert set(np.unique(stored["family"]).tolist()) == {0, FAMILY_CODE["Am"], FAMILY_CODE["At"]}
    # a bus is attacked when a meter at it is tampered
    attacked = np.zeros((T, len(own.bus_client)), bool)
    for j, bus in enumerate(source.meas.plan.bus):
        attacked[:, bus] |= stored["tampered"][:, j]
    assert np.array_equal(h[schema.Y][:].astype(bool), attacked)
    # the true state, float32: the solved voltages with the injections they give
    clean = h[schema.NODE_CLEAN][:]
    assert np.array_equal(clean[:, :, 0], pool.vm.astype(np.float32), equal_nan=True)
    assert np.array_equal(clean[:, :, 3], pool.va.astype(np.float32), equal_nan=True)
    p, _ = source.meas.net.injections(pool.vm[:5], pool.va[:5])
    assert np.array_equal(clean[:5, :, 1], p.astype(np.float32))
    # the episode table: one row per designed episode, in onset order
    e = h[schema.Group.EPISODES]
    episodes = source._episodes
    assert len(episodes) >= 10 and e[schema.EPISODE_ONSET][:].tolist() == [ep.onset for ep in episodes]
    assert e[schema.EPISODE_FAMILY][:].tolist() == [FAMILY_CODE[ep.family] for ep in episodes]
    assert e[schema.EPISODE_TARGET_BRANCH][:].tolist() == [ep.target_branch for ep in episodes]
    ptr, idx = e[schema.EPISODE_BUS_PTR][:], e[schema.EPISODE_BUS_IDX][:]
    for i, ep in enumerate(episodes):
        assert idx[ptr[i] : ptr[i + 1]].tolist() == ep.buses.tolist()


def test_every_name_in_the_file_is_a_schema_name(written):
    ds, _, _ = written
    names = []
    ds.h.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    attrs = set(ds.h.attrs)
    graph_only = (schema.Static.BUS_CLIENT, schema.Static.EDGE_RATING)
    new_york_only = {schema.TIME, schema.path(schema.Group.EPISODES, schema.EPISODE_TARGET_BRANCH)}
    new_york_only |= {schema.path(schema.Group.GRAPH, name) for name in graph_only}
    assert new_york_only <= set(names)  # what only this grid's files carry
    assert {schema.Attr.CURRENT_FORM, schema.Attr.CLIENT_NAMES} <= attrs
    assert attrs <= ATTR_KEYS, sorted(attrs - ATTR_KEYS)
    assert not {schema.TEMPORAL_DELTA, schema.SWING} & set(names)  # left out on this grid for now
    per_frame = set(schema.FIELD_PATH.values()) | set(CURRENT_LAYERS) | {schema.SPLIT, schema.TIME}
    per_frame |= {schema.NODE_TAMPER, schema.EDGE_TAMPER, schema.MAG_PTR, schema.MAG_BUS, schema.MAG}
    graph = {schema.EDGE_INDEX} | {schema.path(schema.Group.GRAPH, k) for k in schema.STATIC_PHYSICS}
    episodes = {n for n in names if n.startswith(schema.Group.EPISODES)}
    assert set(names) - episodes <= per_frame | graph, sorted(set(names) - episodes - per_frame - graph)
    assert ds.h.attrs[schema.Attr.KIND] == schema.KIND_TIMELINE
    assert ds.h.attrs[schema.Attr.CURRENT_FORM] == schema.CURRENT_POLAR
    assert (
        ds.h.attrs[schema.Attr.EDGE_FEAT] == "P_from,Q_from,P_to,Q_to" and ds.h[schema.EDGE_X].shape[2] == 4
    )


def test_the_static_graph_gives_the_cases_admittance_matrix(written):
    """`FdiaGraph` opens a New York file, and the admittance matrix it builds from the file's static graph is
    the one of the case (`Network`)."""
    from fdia_graph.dataset import FdiaGraph

    ds, source, _ = written
    net = source.meas.net
    graph = FdiaGraph(ds.h.filename)
    assert graph.N == len(net.case.bus_id) and graph.E == len(net.case.f) and len(graph) == T
    assert graph.has_currents and graph.current_form == schema.CURRENT_POLAR and not graph.has_temporal
    for ours, theirs in ((graph.ybus_np, net.ybus), (graph.yf_np, net.yf), (graph.yt_np, net.yt)):
        np.testing.assert_allclose(ours, theirs.toarray(), rtol=1e-12, atol=1e-12)


def test_the_graph_holds_the_ratings_and_its_integers_are_int32(written):
    """The line ratings the attacks were designed against are in the file, and the integer arrays have the
    type an IEEE file gives them whatever the platform's own integer is."""
    ds, source, _ = written
    graph = ds.h[schema.Group.GRAPH]
    assert np.array_equal(graph[schema.Static.EDGE_RATING][:], source.meas.net.case.rating)
    for name in (schema.Static.BUS_TYPE, schema.Static.BUS_CLIENT):
        assert graph[name].dtype == np.int32, name
    assert ds.h[schema.EDGE_INDEX].dtype == np.int32


def test_the_settings_and_their_hash_identify_the_file(written):
    """The file records its source's record; a release knob changes it, and so does placing one episode by
    hand, whose record has no attacked share (and its file no `target_attacked_frac`)."""
    import json

    from fdia_graph.generation.nygrid.source import NewYorkSource, SingleEpisodeSource
    from fdia_graph.models.config import NewYorkRelease

    ds, source, _ = written
    attrs = ds.h.attrs
    assert json.loads(attrs[schema.Attr.SETTINGS]) == source.record()
    assert attrs[schema.Attr.SETTINGS_HASH] == schema.config_hash(source.record())
    assert attrs[schema.Attr.TARGET_ATTACKED_FRAC] == source.settings.attacked_frac
    knobs = {"load_cap", "band", "v_margin", "interfaces", "hold_external", "gen_v_tol", "families"}
    hashes = {schema.config_hash(source.record())}  # a source built by hand: no release
    for release in (NewYorkRelease(), NewYorkRelease(load_cap=0.05), NewYorkRelease(hold_external=True)):
        twin = NewYorkSource(source.pool, source.att, source.families, source.own, source.settings)
        twin.release = release.as_record()
        assert set(twin.release) == knobs
        hashes.add(schema.config_hash(twin.record()))
    assert len(hashes) == 4
    one = SingleEpisodeSource(source, 10, "At")
    assert one.record()["episode"] == {"family": "At", "onset": 10, "lag": 0, "lo": 10, "hi": 14}
    assert one.record()["release"] == source.release and "attacked_frac" not in one.record()
    assert schema.Attr.TARGET_ATTACKED_FRAC not in one._recorded() and one._split_frac() == (1.0, 0.0, 0.0)
    others = (SingleEpisodeSource(source, 11, "At"), SingleEpisodeSource(source, 10, "At", lag=1))
    others += (SingleEpisodeSource(source, 10, "At", lo=5), SingleEpisodeSource(source, 10, "Am"))
    hashes |= {schema.config_hash(s.record()) for s in (one, *others)}
    assert len(hashes) == 9


def test_fdia_graph_exports_both_branch_ends_and_converts_polar_angles(written):
    """`FdiaGraph` on a New York file: the pandas table has a column per stored flow (both branch ends),
    and a `units="pu"` view gives the angles of the polar PMU currents in radians, as it gives theta, in a
    record and in an export; the magnitudes and the masks are untouched."""
    pytest.importorskip("pandas")
    pytest.importorskip("torch")
    from fdia_graph.dataset import FdiaGraph

    ds, _, _ = written
    physical, pu = FdiaGraph(ds.h.filename), FdiaGraph(ds.h.filename, units="pu")
    table = physical.export(format="pandas")
    edge_x = ds.h[schema.EDGE_X][:]
    for column, name in enumerate(("Pflow", "Qflow", "P_to", "Q_to")):
        for e in (0, physical.E - 1):
            assert np.array_equal(table[f"{name}_e{e}"].to_numpy(), edge_x[:, e, column], equal_nan=True)
    stored = physical.export(["pmu_i", "pmu_i_benign", "pmu_i_m"])
    converted = pu.export(["pmu_i", "pmu_i_benign", "pmu_i_m", "prev_pmu_i"])
    metered = stored["pmu_i_m"][0, :, 1].astype(bool)
    assert metered.any() and np.array_equal(converted["pmu_i_m"], stored["pmu_i_m"])
    for key in ("pmu_i", "pmu_i_benign"):
        assert np.array_equal(converted[key][..., 0::2], stored[key][..., 0::2], equal_nan=True)
        want = np.deg2rad(stored[key][..., 1::2])
        assert np.array_equal(converted[key][..., 1::2], want, equal_nan=True)
    assert np.array_equal(converted["prev_pmu_i"][1:], converted["pmu_i"][:-1], equal_nan=True)
    record, raw = pu[5], physical[5]
    assert np.array_equal(record["pmu_i"].numpy(), converted["pmu_i"][5])
    assert np.array_equal(record["pmu_i_benign"].numpy(), converted["pmu_i_benign"][5])
    assert np.array_equal(raw["pmu_i"].numpy(), stored["pmu_i"][5])
    assert np.array_equal(record["pmu_i_m"].numpy(), raw["pmu_i_m"].numpy())
    assert np.abs(raw["pmu_i"].numpy()[metered, 1]).max() > np.pi  # degrees as stored, so the test can tell


def test_client_view_holds_only_its_own_meters_buses_and_labels(written):
    from fdia_graph.dataset.clients import ClientView

    ds, source, stored = written
    plan, own = source.meas.plan, source.own
    z, labels = stored["z"].astype(np.float32), ds.h[schema.Y][:].astype(bool)
    seen = np.zeros(len(plan.kind), int)
    for c, name in enumerate(ds.clients):
        v = ds.client(name)
        mine = np.flatnonzero(plan.owner == c)
        assert (v.meters.owner == c).all() and len(v.meters.kind) == len(mine)
        assert np.array_equal(v.buses, np.flatnonzero(own.bus_client == c))
        assert set(v.meters.bus.tolist()) <= set(v.buses.tolist())  # its meters sit at its own buses
        assert np.array_equal(v.z, z[:, mine], equal_nan=True)
        assert np.array_equal(v.labels, labels[:, v.buses]) and v.labels.dtype == bool
        # the view keeps no file handle and no array wider than its own meters or buses
        for value in vars(v).values():
            assert not hasattr(value, "file")
            if isinstance(value, np.ndarray) and value.ndim == 2:
                assert value.shape[1] in (len(mine), len(v.buses))
        assert v.node_features().shape == (T, len(v.buses), 4) and v.temporal().shape == v.z.shape
        seen[mine] += 1
    assert (seen == 1).all()  # the clients' meters partition the plan, none shared
    assert isinstance(ds.client(ds.clients[0]), ClientView)


def test_operator_and_split_views(written):
    ds, source, stored = written
    plan = source.meas.plan
    op = ds.operator()
    assert op.z.shape == (T, len(plan.kind)) and op.labels.shape == (T, len(source.own.bus_client))
    mine = np.flatnonzero(plan.owner == 0)
    sizes = []
    for code, split in enumerate(Split.values()):
        v = ds.client(ds.clients[0], split)
        assert np.array_equal(v.frames, np.flatnonzero(source._split == code))
        assert np.array_equal(v.z, stored["z"].astype(np.float32)[v.frames][:, mine], equal_nan=True)
        gaps = np.diff(v.frames) != 1
        assert (v.temporal()[1:][gaps] == 0).all()  # no change across another split's frames
        sizes.append(len(v.z))
    assert sum(sizes) == T and min(sizes) > 0
    assert (np.diff(ds.client(ds.clients[0], "train").frames) != 1).any()  # the weekly splits interleave


def test_temporal_wraps_angle_changes(written):
    from fdia_graph.dataset.clients import ClientView

    ds, source, _ = written
    from fdia_graph.constants.meters import MeterKind

    plan = source.meas.plan
    angle = np.isin(plan.kind, [MeterKind.VA, MeterKind.IA])
    z = np.zeros((3, len(plan.kind)))
    z[:, angle] = [[179.5], [-179.5], [179.0]]  # crosses +-180 twice
    z[:, ~angle] = [[1.0], [3.0], [2.0]]
    view = ClientView(z, np.zeros((3, 0), bool), plan, np.arange(0), np.arange(3))
    d = view.temporal()
    assert np.allclose(d[1:, angle], [[1.0], [-1.5]])  # wrapped to (-180, 180], not +-359
    assert np.allclose(d[1:, ~angle], [[2.0], [-1.0]]) and (d[0] == 0).all()
    assert np.array_equal(view.z, z)  # the readings themselves stay raw


def test_a_file_in_the_earlier_new_york_layout_is_refused(tmp_path):
    from fdia_graph.dataset.clients import Dataset
    from fdia_graph.errors import NotATimeline

    old = tmp_path / "old.h5"
    with h5py.File(old, "w") as h:
        h.create_dataset("frames/z", data=np.zeros((2, 3), np.float32))
    with pytest.raises(NotATimeline, match="earlier New York layout"):
        Dataset(old)
