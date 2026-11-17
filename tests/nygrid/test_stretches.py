"""The New York source on a synthetic pool, with no NYgrid data and no solver: the frames written in stretches
as the weekly splits are planned are the frames written in one pass, one source generates the same file twice,
a single episode is the design of its (lagged) window on the true scans, and a failed run leaves no file.

The grid is the synthetic one of `conftest.py`: the packaged case file with made-up clients, eight weeks of
hourly made-up states, and two families that write a bias on one meter, so nothing here needs Ipopt."""

from __future__ import annotations

import numpy as np
import pytest

from fdia_graph import schema

from ..digests import file_digest
from .conftest import LENGTHS, UNSOLVED


def _source(grid, cls=None, **settings):
    from fdia_graph.generation.newyork import NewYorkSource
    from fdia_graph.models.config import NewYorkSettings

    pool, att, families, own = grid
    return (cls or NewYorkSource)(pool, att, families, own, NewYorkSettings(length=LENGTHS, **settings))


def _datasets(path):
    """The file at `path` as flat vectors of its meter table, with its split column, episode rows and the
    onsets of its episodes."""
    from fdia_graph.dataset.clients import Dataset

    ds = Dataset(path)
    every = np.arange(ds.h[schema.Y].shape[0])
    episodes = ds.h[schema.Group.EPISODES]
    out = {
        "z": ds.z(every).astype(float),
        "z_benign": ds.z(every, benign=True).astype(float),
        "split": ds.h[schema.SPLIT][:],
        "episode": ds.h[schema.SEQ_ID][:],
        "episodes/onset": episodes[schema.EPISODE_ONSET][:],
        "episodes/family": episodes[schema.EPISODE_FAMILY][:],
    }
    ds.h.close()
    return out


def test_frames_written_in_stretches_are_the_frames_written_in_one_pass(grid, tmp_path):
    """The weekly splits interleave, so the walk runs in several stretches; holding every frame back to one
    pass after the last split is planned writes the same file, dataset by dataset."""
    from fdia_graph.generation.newyork import NewYorkSource

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
    assert set(np.unique(got["split"]).tolist()) == {0, 1, 2}
    assert len(got["episodes/onset"]) >= 10 and np.isnan(got["z_benign"][UNSOLVED]).all()
    # a slot with no attack (an onset divisible by 7) or on an unsolved frame moved, as on an IEEE timeline:
    # every failed attempt is one move (none is given up here), and every placed slot ends with an episode
    summary = chunked.summary()
    misses = sum(sum(summary["missed"][f].values()) for f in ("Am", "At")) + summary["unsolved_attempts"]
    moved, given_up = sum(map(sum, summary["moved"])), sum(map(sum, summary["given_up"]))
    assert given_up == 0 and misses == moved > 0 and summary["missed"]["At"].get("no_target", 0) > 0
    assert len(got["episodes/onset"]) == sum(map(sum, summary["placed"])) == sum(map(sum, summary["built"]))
    assert not (got["episodes/onset"] % 7 == 0).any()


def test_the_benign_scans_do_not_depend_on_the_split_or_the_attacks(grid, tmp_path):
    """Another split (three blocks in time order: one stretch per split) places other episodes, on the same
    noisy benign scans: a frame's jitter comes from its own stream."""
    weeks = _datasets(_source(grid).generate(tmp_path / "weeks.h5"))
    blocks = _datasets(_source(grid, split_by="time").generate(tmp_path / "blocks.h5"))
    assert np.array_equal(weeks["z_benign"], blocks["z_benign"], equal_nan=True)
    assert not np.array_equal(weeks["split"], blocks["split"])
    assert not np.array_equal(weeks["episodes/onset"], blocks["episodes/onset"])


def test_one_source_generates_the_same_file_twice(grid, tmp_path):
    source = _source(grid)
    assert file_digest(source.generate(tmp_path / "a.h5")) == file_digest(source.generate(tmp_path / "b.h5"))
    assert not list(tmp_path.glob("*.part"))


def test_a_new_york_timeline_needs_its_path(grid):
    from fdia_graph.errors import ConfigError

    with pytest.raises(ConfigError, match="path"):
        _source(grid).generate()


def test_a_failed_run_leaves_no_file_and_keeps_the_one_that_was_there(grid, tmp_path):
    from fdia_graph.generation.newyork import NewYorkSource

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
    from fdia_graph.constants.families import FAMILY_CODE
    from fdia_graph.data.config import Settings, rng
    from fdia_graph.generation.newyork import SingleEpisodeSource

    pool, _, families, _ = grid
    onset, lo, hi, L = 500 + lag, 480, 540, LENGTHS["At"]
    one = SingleEpisodeSource(_source(grid), onset, "At", lo, hi, lag=lag)
    got = _datasets(one.generate(tmp_path / "one.h5"))
    z, zb = got["z"], got["z_benign"]
    assert len(z) == hi - lo and not got["split"].any()  # the file's frames, all train
    assert got["episodes/onset"].tolist() == [onset - lo]
    assert got["episodes/family"].tolist() == [FAMILY_CODE["At"]]
    inside = got["episode"] == 0
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
    from fdia_graph.generation.newyork import SingleEpisodeSource

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
