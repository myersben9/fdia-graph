import json

import h5py
import numpy as np
import pytest
from nyfed_inputs import DATA, POOL, TERRITORIES

from fdia_graph.nyfed.attacks.attacker import Attacker, generator_ramps, natural_steps
from fdia_graph.nyfed.attacks.msfdia import MsFdia
from fdia_graph.nyfed.attacks.ramp import Ramp
from fdia_graph.nyfed.config import Settings, rng
from fdia_graph.nyfed.grid.clients import assign_clients
from fdia_graph.nyfed.grid.fixes import restore_765kv, split_zone_j
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.grid.nygrid import modify_mpc, nygrid_units, read_npcc
from fdia_graph.nyfed.io import FAMILIES, read_pool
from fdia_graph.nyfed.scenario.pool import SOLVED, Pool, peak_ratings
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.meters import plan_meters
from fdia_graph.nyfed.sensing.noise import NoiseModel
from fdia_graph.nyfed.timeline.builder import TimelineBuilder
from fdia_graph.nyfed.timeline.schedule import WEEK, place_episodes, split_frames, split_weeks

START, FRAMES = 50000, 288  # one day of the 2019 pool (June), short enough to build in seconds


def test_split_frames_contiguous_and_disjoint():
    split = split_frames(1000, (0.7, 0.15, 0.15))
    assert np.bincount(split).tolist() == [700, 150, 150]
    assert (np.diff(split) >= 0).all()  # three blocks in time order, each contiguous, never interleaved


def test_place_episodes_per_family_length_equal_frames():
    lengths = {"Am": 16, "At": 16, "Ap": 96}
    slots = place_episodes(20000, lengths, 0.2, rng(Settings(), "test-place-len"))
    frames = {f: sum(s.length for s in slots if s.family == f) for f in lengths}
    assert all(s.length == lengths[s.family] for s in slots)  # each family at its own window length
    assert abs(sum(frames.values()) - 4000) <= 96  # the attacked share counted in frames
    assert (
        max(frames.values()) - min(frames.values()) <= 96
    )  # families equal in attacked frames, not episodes
    cover = np.zeros(20000, int)
    for s in slots:
        cover[s.onset : s.onset + s.length] += 1
    assert cover.max() == 1


def test_place_episodes_uniform_non_overlapping():
    slots = place_episodes(5000, {"Am": 16, "At": 16}, 0.2, rng(Settings(), "test-place"))
    frames = sum(s.length for s in slots)
    assert abs(frames - 1000) <= 16  # the whole episodes closest to the attacked share
    assert sorted(s.family for s in slots).count("Am") - sorted(s.family for s in slots).count("At") in (0, 1)
    cover = np.zeros(5000, int)
    for s in slots:
        assert 0 <= s.onset and s.onset + s.length <= 5000  # never cut by the split's end
        cover[s.onset : s.onset + s.length] += 1
    assert cover.max() == 1  # never overlapping
    again = place_episodes(5000, {"Am": 16, "At": 16}, 0.2, rng(Settings(), "test-place"))
    assert again == slots  # the generator alone decides the placement


def test_split_weeks_whole_weeks_stratified_on_weight():
    time = np.arange(52 * 2016) * 300  # 52 weeks of five-minute frames
    weight = np.zeros(len(time))
    hot = [1, 2, 3, 26, 27, 28, 29, 30, 31, 32]  # ten loaded weeks, as the summer and January
    for w in hot:
        weight[w * 2016 : w * 2016 + 50] = 1
    split = split_weeks(time, (0.7, 0.15, 0.15), weight, rng(Settings(), "test-weeks"))
    week = time // WEEK
    assert all(len(set(split[week == w].tolist())) == 1 for w in range(52))  # whole weeks, never cut
    per = np.bincount([int(split[week == w][0]) for w in range(52)], minlength=3)
    assert per.tolist() == [36, 8, 8]  # 70/15/15 of the weeks by largest remainder in each stratum
    hot_split = [int(split[week == w][0]) for w in hot]
    assert set(hot_split) == {0, 1, 2}  # the loaded weeks reach every split
    assert (split == split_weeks(time, (0.7, 0.15, 0.15), weight, rng(Settings(), "test-weeks"))).all()


def test_place_episodes_only_at_allowed_onsets_and_capped():
    allowed = np.zeros(5000, bool)
    allowed[[100, 1000, 1005, 3000]] = True  # four allowed Am onsets, two of them overlapping
    slots = place_episodes(5000, {"Am": 16, "At": 16}, 0.2, rng(Settings(), "test-allow"), {"Am": allowed})
    am = sorted(s.onset for s in slots if s.family == "Am")
    assert set(am) <= {100, 1000, 1005, 3000} and len(am) == 3  # capped at what the onsets can hold
    assert sum(s.family == "At" for s in slots) > 3  # the family that does not wait keeps its share


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (DATA.exists() and TERRITORIES.exists() and POOL.exists()):
        pytest.skip("the public inputs or the whole 2019 pool missing (nyfed_inputs.py)")
    case = restore_765kv(split_zone_j(nygrid_units(modify_mpc(read_npcc(DATA / "npcc.mat")), DATA)))
    territories = json.load(open(TERRITORIES / "nys_electric_territories.geojson", encoding="utf-8"))
    own = assign_clients(case, territories, TERRITORIES / "ny_bus_locations.csv")
    full = read_pool(POOL)
    k = slice(START, START + FRAMES)
    pool = Pool(full.time[k], full.vm[k], full.va[k], full.p[k], full.q[k], full.solve_kind[k])
    case = peak_ratings(case, pool)
    net = Network(case)
    plan = plan_meters(net, own, rng(Settings(), "meters"))
    meas = Measurement(net, plan)
    att = Attacker(net, meas)
    noise = NoiseModel(plan)
    steps = natural_steps(full.time, full.p, full.solve_kind == SOLVED)  # Ap: natural steps over the year
    steps[:, att.gen_bus] = generator_ramps(case)[att.gen_bus]  # generators by their units' ramp
    hour = (pool.time.astype(np.int64) // 3600) % 24
    fams = {
        "Am": MsFdia(att, case.rating, budget=4),
        "At": Ramp(att),  # a small search keeps the test fast
        "Ap": MsFdia(att, case.rating, budget=4, step_bound=steps, frame_hour=hour, family="Ap"),
    }
    lengths = {"Am": 16, "At": 16, "Ap": 32}  # Ap's own length (96 in the build; 32 fits one day)
    path = tmp_path_factory.mktemp("timeline") / "timeline.h5"
    summary = TimelineBuilder(
        pool, meas, noise, att, fams, own, Settings(), length=lengths, attacked_frac=0.3, split_by="time"
    ).build(path)  # one day sits in one week, so the contiguous split
    print(summary)
    with h5py.File(path, "r") as h:
        arrays = {
            k: h[k][:]
            for k in (
                "frames/split",
                "frames/family",
                "frames/episode",
                "frames/z",
                "frames/z_benign",
                "frames/vm",
                "frames/va",
                "labels/bus_attacked",
                "labels/meter_tampered",
                "episodes/family",
                "episodes/onset",
                "episodes/length",
            )
        }
    return pool, att, noise, fams, summary, arrays


def test_episodes_stay_inside_their_split(built):
    pool, att, noise, fams, summary, a = built
    assert len(a["episodes/onset"]) >= 2 and sum(map(sum, summary["built"])) == len(a["episodes/onset"])
    assert set(a["episodes/family"].astype(str)) <= {"Am", "Ap", "At"}
    assert (
        summary["families"] == ["Am", "At", "Ap"] and sum(r[2] for r in summary["placed"]) > 0
    )  # Ap is placed
    lengths = {"Am": 16, "At": 16, "Ap": 32}
    for fam, length in zip(a["episodes/family"].astype(str), a["episodes/length"]):
        assert length == lengths[fam]  # each episode spans its family's window
    split = a["frames/split"]
    for onset, length in zip(a["episodes/onset"], a["episodes/length"]):
        assert len(set(split[onset : onset + length].tolist())) == 1  # no window spans two splits
    # every frame of an episode carries its family and row, every other frame is benign
    ep = a["frames/episode"]
    for e, (onset, length, fam) in enumerate(
        zip(a["episodes/onset"], a["episodes/length"], a["episodes/family"])
    ):
        assert (ep[onset : onset + length] == e).all()
        assert (a["frames/family"][onset : onset + length] == FAMILIES.index(fam.decode())).all()
    assert (a["frames/family"][ep < 0] == 0).all()


def test_frames_are_noisy_benign_plus_attack_and_labels_match_tampered(built):
    pool, att, noise, fams, summary, a = built
    meas, plan = att.meas, att.meas.plan
    z, zb = a["frames/z"].astype(float), a["frames/z_benign"].astype(float)
    ep = a["frames/episode"]
    # benign frames: the observed scan is the benign scan, untouched, with no labels
    assert np.array_equal(z[ep < 0], zb[ep < 0])
    assert not a["labels/meter_tampered"][ep < 0].any() and not a["labels/bus_attacked"][ep < 0].any()
    # the benign scan is h(x_true) plus accuracy-class noise: within a few class sigmas of the exact reading
    exact = meas.h(pool.vm, pool.va)
    assert (np.abs(zb - exact) <= 6 * noise.sigmas(exact) + 1e-5).all()
    assert np.abs(zb - exact).max() > 0  # and the noise is really there
    # each episode, re-designed from its key, is the attack z - z_benign, and its tampered meters are the labels
    for e, (onset, length, fam) in enumerate(
        zip(a["episodes/onset"], a["episodes/length"], a["episodes/family"])
    ):
        fam = fam.decode()
        w = slice(onset, onset + length)
        again = fams[fam].design(
            (int(onset), pool.vm[w], pool.va[w]), rng(Settings(), f"design/{fam}/{onset}")
        )
        assert again is not None and again.family == fam
        np.testing.assert_allclose(z[w] - zb[w], again.a, rtol=0, atol=1e-5 + 1e-6 * np.abs(zb[w]).max())
        tam = att.tampered(pool.vm[w], pool.va[w], again.a)
        assert np.array_equal(a["labels/meter_tampered"][w], tam)
        bus = np.zeros((length, len(pool.vm[0])), bool)
        for j in range(len(plan.kind)):
            bus[:, plan.bus[j]] |= tam[:, j]
        assert np.array_equal(a["labels/bus_attacked"][w], bus)  # a bus is attacked when a meter at it is
    assert (pool.solve_kind == SOLVED).all()
