"""The one path that generates a timeline (`TimelineSource.generate`), checked on the parts both grids share:
how far the walk may run once a split is planned, where an episode fits in a split that lists its frames, the
worker pool the designs run on, and what a run leaves on disk when it fails."""

import operator
import os

import numpy as np
import pytest

from fdia_graph.generation.design import DesignPool
from fdia_graph.generation.plan import planned_until
from fdia_graph.models.plan import SplitSpan

from .digests import file_digest


def _weeks(codes, week=10):
    """The three splits of a timeline whose consecutive weeks (of `week` frames) carry the split `codes`."""
    split = np.repeat(np.asarray(codes), week)
    spans = []
    for code in range(3):
        where = np.flatnonzero(split == code)
        start, stop = (int(where[0]), int(where[-1]) + 1) if len(where) else (0, 0)
        spans.append(SplitSpan(code, start, stop, where))
    return spans


# ---- how far the walk may run -------------------------------------------------------------------
def test_block_splits_are_walked_one_at_a_time():
    spans = [SplitSpan(0, 0, 60), SplitSpan(1, 60, 80), SplitSpan(2, 80, 100)]
    assert [planned_until(spans, k) for k in range(3)] == [60, 80, 100]  # each split's own end


def test_interleaved_weeks_are_walked_up_to_the_first_frame_of_a_split_still_to_plan():
    spans = _weeks([0, 1, 0, 2, 1, 0])  # train weeks 0, 2, 5; val weeks 1, 4; test week 3
    # with train planned the walk stops where val begins, with val planned where test begins, then the end
    assert [planned_until(spans, k) for k in range(3)] == [10, 30, 60]


def test_a_split_planned_late_holds_the_walk_from_its_first_frame():
    spans = _weeks([2, 0, 1, 0])  # the test split holds the first week
    assert [planned_until(spans, k) for k in range(3)] == [0, 0, 40]  # nothing is written until it is planned


def test_an_empty_split_never_holds_the_walk():
    spans = _weeks([0, 2, 0, 2])  # no val week at all
    assert spans[1].size == 0
    assert [planned_until(spans, k) for k in range(3)] == [10, 10, 40]
    blocks = [SplitSpan(0, 0, 3), SplitSpan(1, 3, 3), SplitSpan(2, 3, 5)]  # a block split too short for val
    assert [planned_until(blocks, k) for k in range(3)] == [3, 3, 5]


# ---- where an episode fits ----------------------------------------------------------------------
def test_an_episode_fits_only_inside_one_run_of_a_splits_listed_frames():
    span = _weeks([0, 1, 0, 0])[0]  # train: frames 0..9 and 20..39
    fits = span.onsets_fit(4)
    assert span.size == 30 and fits.shape == (30,)
    onsets = span.index()[fits]
    assert onsets.tolist() == [*range(0, 7), *range(20, 37)]  # never across the gap, never past the end
    assert not span.onsets_fit(41).any()  # longer than the split: nowhere


def test_a_block_split_fits_an_episode_at_every_onset_that_leaves_it_room():
    fits = SplitSpan(1, 100, 110).onsets_fit(4)
    assert fits.tolist() == [True] * 7 + [False] * 3


def test_spans_compare_and_hash_whether_or_not_they_list_their_frames():
    a, b = _weeks([0, 1, 0])[0], _weeks([0, 1, 0])[0]
    assert a == b and hash(a) == hash(b) and a != SplitSpan(1, 0, 30)
    assert len({a, b, SplitSpan(0, 0, 30)}) == 1  # the same code and stretch


# ---- the worker pool ----------------------------------------------------------------------------
@pytest.mark.parametrize("workers", [1, 2])
def test_the_design_pool_gives_the_designs_in_task_order_from_what_each_worker_made(workers):
    """`start(setup)` runs once per process (here in this one, or in each of two spawned ones) and every
    design gets what it made; the results come in the order of the tasks."""
    pool = DesignPool(workers, int, "100")  # each worker makes the number 100 from its setup
    try:
        assert list(pool.map(operator.add, range(20))) == [100 + k for k in range(20)]
        assert list(pool.map(operator.add, [])) == []
    finally:
        pool.close()


def test_a_failed_run_stops_the_pool_without_waiting_for_designs_in_flight():
    pool = DesignPool(2, int, "7")
    results = pool.map(operator.add, range(5000))  # handed out, none of them read
    pool.close(finished=False)  # the run failed elsewhere: the processes are stopped, nothing is waited for
    with pytest.raises(ValueError, match="not running"):
        list(pool.map(operator.add, [1]))  # the pool is gone
    del results


# ---- what a run leaves on disk ------------------------------------------------------------------
@pytest.fixture(scope="module")
def pool():
    pytest.importorskip("pandapower")
    from fdia_graph.generation import _load_states

    return _load_states(14, None)[:60]


KW = dict(families=("At",), ramp_len=10, min_tamper=False)


def test_a_bus_count_from_numpy_names_the_system_like_a_plain_int(pool, tmp_path):
    from fdia_graph.timeline import generate_timeline

    plain = generate_timeline(14, states=pool, out=str(tmp_path / "plain.h5"), seed=1, **KW)
    numpy = generate_timeline(np.int64(14), states=pool, out=tmp_path / "numpy.h5", seed=1, **KW)
    assert numpy == os.fspath(tmp_path / "numpy.h5")  # a path object is taken, the path comes back as text
    assert file_digest(plain) == file_digest(numpy)
    assert not list(tmp_path.glob("*.part"))  # the file took its name once it was whole


def test_a_failed_run_leaves_no_file_and_keeps_the_one_that_was_there(pool, tmp_path, monkeypatch):
    from fdia_graph.generation.ieee import IeeeSource
    from fdia_graph.models.config import TimelineSettings

    def fail(self):
        raise RuntimeError("the walk failed")

    monkeypatch.setattr(IeeeSource, "finish", fail)
    out = tmp_path / "kept.h5"
    for before in (None, b"an earlier timeline"):
        if before is not None:
            out.write_bytes(before)
        with pytest.raises(RuntimeError, match="the walk failed"):
            IeeeSource(14, pool, TimelineSettings.of(**KW), 1).generate(out)
        assert (out.read_bytes() if out.exists() else None) == before
        assert not list(tmp_path.glob("*.part"))


def test_an_ieee_source_generates_once(pool, tmp_path):
    from fdia_graph.errors import ConfigError
    from fdia_graph.generation.ieee import IeeeSource
    from fdia_graph.models.config import TimelineSettings

    source = IeeeSource(14, pool, TimelineSettings.of(**KW), 1)
    first = source.generate(tmp_path / "first.h5")
    kept = file_digest(first)
    with pytest.raises(ConfigError, match="one timeline"):
        source.generate(tmp_path / "second.h5")
    assert file_digest(first) == kept and not (tmp_path / "second.h5").exists()
