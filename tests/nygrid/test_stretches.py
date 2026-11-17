"""The New York source on a synthetic pool, with no NYgrid data and no solver: the frames written in stretches
as the weekly splits are planned are the frames written in one pass, one source generates the same file twice,
a single episode is the design of its (lagged) window on the true scans, and a failed run leaves no file.

The grid is the packaged case file with made-up clients; the pool is eight weeks of hourly made-up states; the
two families write a bias on one meter, so nothing here needs Ipopt."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from ..digests import file_digest

pytest.importorskip("matpowercaseframes")  # the packaged case file's reader (the [ny] extra)
pytest.importorskip("scipy")

WEEK_HOURS = 7 * 24
T = 8 * WEEK_HOURS  # eight weeks of hourly frames: six train, one val, one test
LENGTHS = {"Am": 6, "At": 4}
UNSOLVED = slice(300, 303)


def _family(name, plan):
    from fdia_graph.engine.attacks.family import AttackFamily, Episode

    class Bias(AttackFamily):
        """A growing bias on one meter drawn from the design's stream; no attack at an onset divisible by 7."""

        def design(self, window, rng):
            onset, vm, _ = window
            if onset % 7 == 0:
                self.last_miss = "no_target"
                return None
            meter = int(rng.integers(len(plan.kind)))
            a = np.zeros((len(vm), len(plan.kind)))
            a[:, meter] = 5.0 * (1 + np.arange(len(vm)))
            owner, bus = np.array([plan.owner[meter]]), np.array([plan.bus[meter]])
            return Episode(name, owner, onset, len(vm), bus, -1, a)

    return Bias()


@pytest.fixture(scope="module")
def grid():
    """(pool, attacker, families, ownership): the packaged case, three made-up clients, a made-up pool."""
    from fdia_graph.engine.attacks.attacker import Attacker
    from fdia_graph.engine.sensing.measure import Measurement
    from fdia_graph.engine.sensing.meters import plan_meters
    from fdia_graph.generation.nygrid.case import nygrid_case
    from fdia_graph.generation.nygrid.config import Settings, rng
    from fdia_graph.generation.nygrid.network import Network
    from fdia_graph.generation.nygrid.ownership import Ownership
    from fdia_graph.generation.nygrid.pool import FAILED, SOLVED, Pool

    case = nygrid_case()
    n = len(case.bus_id)
    bus_client = np.where(case.is_ny, np.arange(n) % 3, -1)
    own = Ownership(np.array(["a", "b", "c"]), bus_client, bus_client[case.f] != bus_client[case.t])
    net = Network(case)
    meas = Measurement(net, plan_meters(net, own, rng(Settings(), "meters")))
    draw = np.random.default_rng(7)
    vm, va = 1.0 + 0.01 * draw.standard_normal((T, n)), 2.0 * draw.standard_normal((T, n))
    kind = np.full(T, SOLVED)
    vm[UNSOLVED], va[UNSOLVED], kind[UNSOLVED] = np.nan, np.nan, FAILED
    time = (np.datetime64("2019-01-01T00:00:00") + np.arange(T) * np.timedelta64(1, "h")).astype(
        "datetime64[s]"
    )
    pool = Pool(time, vm, va, np.zeros((T, n)), np.zeros((T, n)), kind)
    families = {name: _family(name, meas.plan) for name in LENGTHS}
    return pool, Attacker(net, meas), families, own


def _source(grid, cls=None, **settings):
    from fdia_graph.generation.nygrid.source import NewYorkSource
    from fdia_graph.models.config import NewYorkSettings

    pool, att, families, own = grid
    return (cls or NewYorkSource)(
        pool, att, families, own, NewYorkSettings(length=LENGTHS, block=100, **settings)
    )


def _datasets(path):
    with h5py.File(path, "r") as h:
        return {k: h[k][:] for k in ("frames/z", "frames/z_benign", "frames/split", "episodes/onset")}


def test_frames_written_in_stretches_are_the_frames_written_in_one_pass(grid, tmp_path):
    """The weekly splits interleave, so the walk runs in several stretches; holding every frame back to one
    pass after the last split is planned writes the same file, dataset by dataset."""
    from fdia_graph.generation.nygrid.source import NewYorkSource

    class Counting(NewYorkSource):
        stretches = 0

        def walk(self, until):
            self.stretches += until > self._cursor
            super().walk(until)

    class OnePass(NewYorkSource):
        def walk(self, until):
            if until == self.stop:  # every split is planned: now write the whole file
                super().walk(until)

    chunked = _source(grid, Counting)
    a = chunked.generate(tmp_path / "stretches.h5")
    b = _source(grid, OnePass).generate(tmp_path / "one_pass.h5")
    assert chunked.stretches >= 2  # the val or test week sits between train weeks
    assert file_digest(a) == file_digest(b)
    got = _datasets(a)
    assert set(np.unique(got["frames/split"]).tolist()) == {0, 1, 2}
    assert len(got["episodes/onset"]) >= 10 and np.isnan(got["frames/z_benign"][UNSOLVED]).all()
    assert (
        chunked.summary()["missed"]["Am"].get("no_target", 0)
        + chunked.summary()["missed"]["At"].get("no_target", 0)
        == sum(map(sum, chunked.summary()["placed"]))
        - len(got["episodes/onset"])
        - chunked.summary()["skipped_unsolved"]
    )


def test_the_benign_scans_do_not_depend_on_the_split_or_the_block(grid, tmp_path):
    """Another split (three blocks in time order: one stretch per split) and another block size place other
    episodes, on the same noisy benign scans."""
    weeks = _datasets(_source(grid).generate(tmp_path / "weeks.h5"))
    blocks = _datasets(_source(grid, split_by="time").generate(tmp_path / "blocks.h5"))
    assert np.array_equal(weeks["frames/z_benign"], blocks["frames/z_benign"], equal_nan=True)
    assert not np.array_equal(weeks["frames/split"], blocks["frames/split"])


def test_one_source_generates_the_same_file_twice(grid, tmp_path):
    source = _source(grid)
    assert file_digest(source.generate(tmp_path / "a.h5")) == file_digest(source.generate(tmp_path / "b.h5"))
    assert not list(tmp_path.glob("*.part"))


def test_a_new_york_timeline_needs_its_path(grid):
    from fdia_graph.errors import ConfigError

    with pytest.raises(ConfigError, match="path"):
        _source(grid).generate()


def test_a_failed_run_leaves_no_file_and_keeps_the_one_that_was_there(grid, tmp_path):
    from fdia_graph.generation.nygrid.source import NewYorkSource

    class Failing(NewYorkSource):
        def finish(self):
            raise RuntimeError("the build failed")

    out = tmp_path / "kept.h5"
    out.write_bytes(b"an earlier timeline")
    with pytest.raises(RuntimeError, match="the build failed"):
        _source(grid, Failing).generate(out)
    assert out.read_bytes() == b"an earlier timeline" and not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("lag", [0, 3])
def test_a_single_episode_is_its_lagged_windows_design_on_the_true_scans(grid, tmp_path, lag):
    from fdia_graph.generation.nygrid.config import Settings, rng
    from fdia_graph.generation.nygrid.source import SingleEpisodeSource

    pool, _, families, _ = grid
    onset, lo, hi, L = 500 + lag, 480, 540, LENGTHS["At"]
    one = SingleEpisodeSource(_source(grid), onset, "At", lo, hi, lag=lag)
    path = one.generate(tmp_path / "one.h5")
    with h5py.File(path, "r") as h:
        z, zb = h["frames/z"][:].astype(float), h["frames/z_benign"][:].astype(float)
        assert (
            h["frames/z"].shape[0] == hi - lo and not h["frames/split"][:].any()
        )  # the file's frames, all train
        assert h["episodes/onset"][:].tolist() == [onset - lo] and h["episodes/family"][:].tolist() == [b"At"]
        inside = h["frames/episode"][:] == 0
    assert inside.sum() == L and inside[onset - lo : onset - lo + L].all()
    known = slice(onset - lag, onset - lag + L)  # the window the attacker knew
    design = families["At"].design(
        (known.start, pool.vm[known], pool.va[known]), rng(Settings(), f"design/At/{known.start}")
    )
    np.testing.assert_allclose((z - zb)[inside], design.a, atol=1e-3)
    assert np.array_equal(z[~inside], zb[~inside])
    assert one.summary()["onset"] == onset - lo and one.summary()["length"] == L


def test_a_single_episode_with_no_attack_or_on_unsolved_frames_is_refused_and_leaves_no_file(grid, tmp_path):
    from fdia_graph.errors import ConfigError, DataConditionError, NoAdmissibleTarget
    from fdia_graph.generation.nygrid.source import SingleEpisodeSource

    source = _source(grid)
    with pytest.raises(NoAdmissibleTarget, match="no_target"):
        SingleEpisodeSource(source, 497, "At").generate(tmp_path / "none.h5")  # 497 = 7 * 71: no attack
    assert not list(tmp_path.iterdir())  # neither the file nor its .part
    with pytest.raises(DataConditionError, match="unsolved"):
        SingleEpisodeSource(source, UNSOLVED.start - 1, "At")
    with pytest.raises(DataConditionError, match="unsolved"):
        SingleEpisodeSource(source, UNSOLVED.stop + 1, "At", lag=3)  # the window the attacker knew
    with pytest.raises(ConfigError, match="lag"):
        SingleEpisodeSource(source, 5, "At", lag=10)
    with pytest.raises(ConfigError, match="inside"):
        SingleEpisodeSource(source, 500, "At", 502, 540)
