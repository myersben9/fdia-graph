"""The New York timeline: splits cut whole and disjoint, episodes placed at random within their split and only at
allowed onsets, and every frame a noisy benign scan plus its attack with labels that match the meters tampered."""

import json
from dataclasses import replace

import numpy as np
import pytest

from fdia_graph import schema
from fdia_graph.constants.families import FAMILIES, FAMILY_CODE
from fdia_graph.constants.newyork import SolveKind
from fdia_graph.constants.time import WEEK_SECONDS
from fdia_graph.data.config import Settings, rng
from fdia_graph.data.pool import Pool, peak_ratings, read_pool
from fdia_graph.dataset.clients import Dataset
from fdia_graph.engine.attacks.attacker import Attacker
from fdia_graph.engine.attacks.ramp import Ramp
from fdia_graph.generation.newyork import NewYorkSource
from fdia_graph.generation.plan import place_episodes, split_weeks
from fdia_graph.models.config import NewYorkSettings
from fdia_graph.sensing.measure import Measurement
from fdia_graph.sensing.meters import plan_meters
from fdia_graph.systems.case import nygrid_case
from fdia_graph.systems.network import Network
from fdia_graph.systems.ownership import assign_clients

from .inputs import DATA, POOL, TERRITORIES

START, FRAMES = 50000, 288  # one day of the 2019 pool (June), short enough to build in seconds


def _placed(n, lengths, name, allowed=None):
    """One split's placement at an attacked share of 0.2, as the plan stage's slots."""
    placed, _, _ = place_episodes(rng(Settings(), name), lengths, 0.2, n, allowed)
    return placed


def test_place_episodes_per_family_length_equal_frames():
    lengths = {"Am": 16, "At": 16, "long": 96}
    slots = _placed(20000, lengths, "test-place-len")
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
    slots = _placed(5000, {"Am": 16, "At": 16}, "test-place")
    frames = sum(s.length for s in slots)
    assert abs(frames - 1000) <= 16  # the whole episodes closest to the attacked share
    assert sorted(s.family for s in slots).count("Am") - sorted(s.family for s in slots).count("At") in (0, 1)
    cover = np.zeros(5000, int)
    for s in slots:
        assert 0 <= s.onset and s.onset + s.length <= 5000  # never cut by the split's end
        cover[s.onset : s.onset + s.length] += 1
    assert cover.max() == 1  # never overlapping
    again = _placed(5000, {"Am": 16, "At": 16}, "test-place")
    assert again == slots  # the generator alone decides the placement


def test_split_weeks_whole_weeks_stratified_on_weight():
    time = np.arange(52 * 2016) * 300  # 52 weeks of five-minute frames
    weight = np.zeros(len(time))
    hot = [1, 2, 3, 26, 27, 28, 29, 30, 31, 32]  # ten loaded weeks, as the summer and January
    for w in hot:
        weight[w * 2016 : w * 2016 + 50] = 1
    split = split_weeks(time, (0.7, 0.15, 0.15), weight, rng(Settings(), "test-weeks"))
    week = time // WEEK_SECONDS
    assert all(len(set(split[week == w].tolist())) == 1 for w in range(52))  # whole weeks, never cut
    per = np.bincount([int(split[week == w][0]) for w in range(52)], minlength=3)
    assert per.tolist() == [36, 8, 8]  # 70/15/15 of the weeks by largest remainder in each stratum
    hot_split = [int(split[week == w][0]) for w in hot]
    assert set(hot_split) == {0, 1, 2}  # the loaded weeks reach every split
    assert (split == split_weeks(time, (0.7, 0.15, 0.15), weight, rng(Settings(), "test-weeks"))).all()


def test_place_episodes_only_at_allowed_onsets_and_capped():
    allowed = np.zeros(5000, bool)
    allowed[[100, 1000, 1005, 3000]] = True  # four allowed Am onsets, two of them overlapping
    slots = _placed(5000, {"Am": 16, "At": 16}, "test-allow", {"Am": allowed})
    am = sorted(s.onset for s in slots if s.family == "Am")
    assert set(am) <= {100, 1000, 1005, 3000} and len(am) == 3  # capped at what the onsets can hold
    assert sum(s.family == "At" for s in slots) > 3  # the family that does not wait keeps its share


class _RampAs(Ramp):
    """A ramp written under another family's name: a second family for the source that needs no solver."""

    def __init__(self, attacker, family):
        super().__init__(attacker)
        self.family = family

    def design(self, window, rng):
        ep = super().design(window, rng)
        return None if ep is None else replace(ep, family=self.family)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    if not (DATA.exists() and TERRITORIES.exists() and POOL.exists()):
        pytest.skip("the public inputs or the whole 2019 pool missing (inputs.py)")
    case = nygrid_case()
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
    # two families of different window lengths; the release's Am (`InterfaceFdia`) is built and pinned in
    # test_timeline_digests.py, here any second family serves
    fams = {"Am": _RampAs(att, "Am"), "At": Ramp(att)}
    lengths = {"Am": 32, "At": 16}
    path = tmp_path_factory.mktemp("timeline") / "timeline.h5"
    settings = NewYorkSettings(length=lengths, attacked_frac=0.3, split_by="time")
    source = NewYorkSource(pool, att, fams, own, settings)  # one day sits in one week: the contiguous split
    source.generate(path)
    summary = source.summary()
    print(summary)
    ds = Dataset(path)  # the flat vectors of the meter table, gathered from the file's layers
    h, every = ds.h, np.arange(FRAMES)
    episodes = h[schema.Group.EPISODES]
    arrays = {
        "split": h[schema.SPLIT][:],
        "family": h[schema.FAMILY][:],
        "episode": h[schema.SEQ_ID][:],
        "z": ds.z(every),
        "z_benign": ds.z(every, benign=True),
        "bus_attacked": h[schema.Y][:].astype(bool),
        "meter_tampered": ds.tampered(every),
        "episodes/family": np.array([FAMILIES[int(code)] for code in episodes[schema.EPISODE_FAMILY][:]]),
        "episodes/onset": episodes[schema.EPISODE_ONSET][:],
        "episodes/length": episodes[schema.EPISODE_LENGTH][:],
    }
    h.close()
    return pool, att, source.noise, fams, summary, arrays


def test_episodes_stay_inside_their_split(built):
    pool, att, noise, fams, summary, a = built
    assert len(a["episodes/onset"]) >= 2 and sum(map(sum, summary["built"])) == len(a["episodes/onset"])
    assert set(a["episodes/family"].tolist()) <= {"Am", "At"}
    assert summary["families"] == ["Am", "At"] and all(
        sum(r[k] for r in summary["placed"]) > 0 for k in (0, 1)
    )
    lengths = {"Am": 32, "At": 16}
    for fam, length in zip(a["episodes/family"].tolist(), a["episodes/length"]):
        assert length == lengths[fam]  # each episode spans its family's window
    split = a["split"]
    for onset, length in zip(a["episodes/onset"], a["episodes/length"]):
        assert len(set(split[onset : onset + length].tolist())) == 1  # no window spans two splits
    # every frame of an episode carries its family and row, every other frame is benign
    ep = a["episode"]
    for e, (onset, length, fam) in enumerate(
        zip(a["episodes/onset"], a["episodes/length"], a["episodes/family"])
    ):
        assert (ep[onset : onset + length] == e).all()
        assert (a["family"][onset : onset + length] == FAMILY_CODE[str(fam)]).all()
    assert (a["family"][ep < 0] == 0).all()


def test_frames_are_noisy_benign_plus_attack_and_labels_match_tampered(built):
    pool, att, noise, fams, summary, a = built
    meas, plan = att.meas, att.meas.plan
    z, zb = a["z"].astype(float), a["z_benign"].astype(float)
    ep = a["episode"]
    # benign frames: the observed scan is the benign scan, untouched, with no labels
    assert np.array_equal(z[ep < 0], zb[ep < 0])
    assert not a["meter_tampered"][ep < 0].any() and not a["bus_attacked"][ep < 0].any()
    # the benign scan is h(x_true) plus accuracy-class noise: within a few class sigmas of the exact reading
    exact = meas.h(pool.vm, pool.va)
    assert (np.abs(zb - exact) <= 6 * noise.sigmas(exact) + 1e-5).all()
    assert np.abs(zb - exact).max() > 0  # and the noise is really there
    # each episode, re-designed from its key, is the attack z - z_benign, and its tampered meters are the labels
    for e, (onset, length, fam) in enumerate(
        zip(a["episodes/onset"], a["episodes/length"], a["episodes/family"])
    ):
        fam = str(fam)
        w = slice(onset, onset + length)
        again = fams[fam].design(
            (int(onset), pool.vm[w], pool.va[w]), rng(Settings(), f"design/{fam}/{onset}")
        )
        assert again is not None and again.family == fam
        np.testing.assert_allclose(z[w] - zb[w], again.a, rtol=0, atol=1e-5 + 1e-6 * np.abs(zb[w]).max())
        tam = att.tampered(pool.vm[w], pool.va[w], again.a)
        assert np.array_equal(a["meter_tampered"][w], tam)
        bus = np.zeros((length, len(pool.vm[0])), bool)
        for j in range(len(plan.kind)):
            bus[:, plan.bus[j]] |= tam[:, j]
        assert np.array_equal(a["bus_attacked"][w], bus)  # a bus is attacked when a meter at it is
    assert (pool.solve_kind == SolveKind.SOLVED).all()
