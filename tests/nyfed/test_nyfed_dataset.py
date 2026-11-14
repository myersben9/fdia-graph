import json
from dataclasses import fields

import numpy as np
import pytest
from nyfed_inputs import DATA, ROOT, TERRITORIES, needs_data

from fdia_graph.dataset.clients import ClientView, Dataset
from fdia_graph.engine.attacks.family import Episode
from fdia_graph.engine.sensing.meters import MeterPlan, plan_meters
from fdia_graph.federated.utilities import assign_clients
from fdia_graph.generation.nygrid.case import Case
from fdia_graph.generation.nygrid.config import Settings, rng
from fdia_graph.generation.nygrid.fixes import restore_765kv, split_zone_j
from fdia_graph.generation.nygrid.io import TimelineWriter
from fdia_graph.generation.nygrid.network import Network
from fdia_graph.generation.nygrid.npcc import modify_mpc, nygrid_units, read_npcc

pytestmark = needs_data
T = 100


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """A short timeline written block by block: the real case, ownership and meter plan with random
    frames, labels and two episodes, so the round trip and the views are checked without the pool."""
    case = restore_765kv(split_zone_j(nygrid_units(modify_mpc(read_npcc(DATA / "npcc.mat")), DATA)))
    territories = json.load(open(TERRITORIES / "nys_electric_territories.geojson", encoding="utf-8"))
    own = assign_clients(case, territories, TERRITORIES / "ny_bus_locations.csv")
    plan = plan_meters(Network(case), own, rng(Settings(), "meters"))
    g = np.random.default_rng(0)
    M, N = len(plan.kind), len(case.bus_id)
    frames = {
        "time": np.datetime64("2019-01-01T00:00") + np.arange(T) * np.timedelta64(5, "m"),
        # interleaved like the week split (7 train, 3 val, 7 train, 3 test in every 20 frames), so a
        # selection that spans its first to last frame would take in the other splits
        "split": np.tile(np.r_[[0] * 7, [1] * 3, [0] * 7, [2] * 3], T // 20).astype(np.int8),
        "family": g.integers(0, 3, T).astype(np.int8),
        "episode": g.integers(-1, 2, T).astype(np.int32),
        "z": g.normal(size=(T, M)),
        "z_benign": g.normal(size=(T, M)),
        "vm": g.normal(size=(T, N)),
        "va": g.normal(size=(T, N)),
        "bus_attacked": g.random((T, N)) < 0.1,
        "meter_tampered": g.random((T, M)) < 0.1,
    }
    episodes = [
        Episode("Am", np.array([3, 4]), 10, 16, np.array([54, 56, 57]), 59, np.zeros((16, M))),
        Episode("At", np.array([1]), 40, 16, np.array([41]), -1, np.zeros((16, M))),
    ]
    path = tmp_path_factory.mktemp("dataset") / "timeline.h5"
    w = TimelineWriter(path)
    w.write_static(case, own, plan)
    for t0 in range(0, T, 30):  # blocks that do not divide T, so the datasets grow unevenly
        w.write_frames(t0, {k: v[t0 : t0 + 30] for k, v in frames.items()})
    w.write_episodes(episodes)
    w.close()
    return Dataset(path), case, own, plan, frames, episodes


def test_reload_round_trip(written):
    ds, case, own, plan, frames, episodes = written
    for f in fields(Case):  # NaN marks an unknown unit ramp, equal to itself here
        got, want = np.asarray(getattr(ds.case, f.name)), np.asarray(getattr(case, f.name))
        assert np.array_equal(got, want, equal_nan=want.dtype.kind == "f"), f.name
    for f in fields(MeterPlan):
        assert np.array_equal(getattr(ds.plan, f.name), getattr(plan, f.name)), f.name
    assert ds.clients == own.names.tolist() and np.array_equal(ds.own.bus_client, own.bus_client)
    h = ds.h
    assert np.array_equal(h["frames/time"][:].astype("datetime64[s]"), frames["time"].astype("datetime64[s]"))
    for name in ("split", "family", "episode", "vm", "va"):
        assert np.array_equal(h[f"frames/{name}"][:], frames[name])
    np.testing.assert_allclose(h["frames/z"][:], frames["z"], rtol=1e-6)  # stored as float32
    for name in ("bus_attacked", "meter_tampered"):
        assert np.array_equal(h[f"labels/{name}"][:], frames[name])
    e = h["episodes"]
    assert e["family"][:].astype(str).tolist() == ["Am", "At"]
    assert e["onset"][:].tolist() == [10, 40] and e["target_branch"][:].tolist() == [59, -1]
    bp, cp = e["buses_ptr"][:], e["clients_ptr"][:]
    for i, ep in enumerate(episodes):
        assert e["buses"][bp[i] : bp[i + 1]].tolist() == ep.buses.tolist()
        assert e["clients"][cp[i] : cp[i + 1]].tolist() == ep.clients.tolist()


def test_client_view_holds_only_its_own_meters_buses_and_labels(written):
    ds, case, own, plan, frames, _ = written
    seen = np.zeros(len(plan.kind), int)
    for c, name in enumerate(ds.clients):
        v = ds.client(name)
        mine = np.flatnonzero(plan.owner == c)
        assert (v.meters.owner == c).all() and len(v.meters.kind) == len(mine)
        assert np.array_equal(v.buses, np.flatnonzero(own.bus_client == c))
        assert set(v.meters.bus.tolist()) <= set(v.buses.tolist())  # its meters sit at its own buses
        np.testing.assert_allclose(v.z, frames["z"][:, mine], rtol=1e-6)
        assert np.array_equal(v.labels, frames["bus_attacked"][:, v.buses])
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
    ds, case, own, plan, frames, _ = written
    op = ds.operator()
    assert op.z.shape == (T, len(plan.kind)) and op.labels.shape == (T, len(case.bus_id))
    sizes = []
    for split in ("train", "val", "test"):
        v = ds.client(ds.clients[0], split)
        assert np.array_equal(
            v.frames, np.flatnonzero(frames["split"] == ("train", "val", "test").index(split))
        )
        mine = np.flatnonzero(plan.owner == 0)
        np.testing.assert_allclose(v.z, frames["z"][v.frames][:, mine], rtol=1e-6)
        assert (
            v.temporal()[1:][np.diff(v.frames) != 1] == 0
        ).all()  # no change across another split's frames
        sizes.append(len(v.z))
    assert sizes == [70, 15, 15]


def test_temporal_wraps_angle_changes(written):
    ds, case, own, plan, frames, episodes = written
    kinds = ds.kinds
    angle = np.isin(plan.kind, [kinds.index("va"), kinds.index("ia")])
    z = np.zeros((3, len(plan.kind)))
    z[:, angle] = [[179.5], [-179.5], [179.0]]  # crosses +-180 twice
    z[:, ~angle] = [[1.0], [3.0], [2.0]]
    view = ClientView(z, np.zeros((3, 0), bool), plan, np.arange(0), np.arange(3), kinds)
    d = view.temporal()
    assert np.allclose(d[1:, angle], [[1.0], [-1.5]])  # wrapped to (-180, 180], not +-359
    assert np.allclose(d[1:, ~angle], [[2.0], [-1.0]]) and (d[0] == 0).all()
    assert np.array_equal(view.z, z)  # the readings themselves stay raw


V6 = ROOT / "out" / "timeline_2019_v6.h5"
APTEST = ROOT / "out" / "timeline_2019_v6_aptest.h5"


@pytest.mark.skipif(
    not (V6.exists() and APTEST.exists()), reason="v6 timeline or its Ap evaluation set missing"
)
def test_ap_evaluation_set_is_v6_test_frames_with_ap_only():
    """The pattern-aware evaluation set covers v6's test frames with the same times, true states and
    benign scans; its benign frames read exactly v6's benign scans, and it holds Ap episodes only,
    96 scans each, never overlapping, through the same Dataset interface."""
    v6, ev = Dataset(V6), Dataset(APTEST)
    test = slice(*(lambda w: (int(w[0]), int(w[-1]) + 1))(np.flatnonzero(v6.h["frames/split"][:] == 2)))
    assert ev.h.attrs["eval_only"] == 1 and (ev.h["frames/split"][:] == 2).all()
    for name in ("time", "z_benign", "vm", "va"):
        np.testing.assert_array_equal(ev.h[f"frames/{name}"][:], v6.h[f"frames/{name}"][test])
    benign = ev.h["frames/episode"][:] < 0
    np.testing.assert_array_equal(ev.h["frames/z"][:][benign], v6.h["frames/z_benign"][test][benign])
    assert not ev.h["labels/bus_attacked"][:][benign].any()
    fam = ev.h["episodes/family"][:].astype(str)
    assert len(fam) >= 40 and set(fam) == {"Ap"} and (ev.h["episodes/length"][:] == 96).all()
    cover = np.zeros(len(benign), int)
    for onset, length in zip(ev.h["episodes/onset"][:], ev.h["episodes/length"][:]):
        cover[onset : onset + length] += 1
    assert cover.max() == 1 and np.array_equal(cover == 0, benign)
    # the harness reads it as it reads v6: the operator's and a client's test views
    assert ev.operator("test").z.shape == (len(benign), len(v6.plan.kind))
    assert ev.client("ConEd", "test").labels.any()
