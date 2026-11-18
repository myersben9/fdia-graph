"""The per-case choices are settings of the one source: each works on any case whose data allows it,
and is refused with the reason where the data is missing. The New York grid's side of these tests is
in tests/nygrid/test_any_case.py."""

import datetime
import re
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from fdia_graph import schema
from fdia_graph.errors import ConfigError
from fdia_graph.models.config import TimelineSettings

HOUR = np.timedelta64(1, "h")
WEEK_FRAMES = 7 * 24  # hourly frames


@pytest.fixture(scope="module")
def hourly_pool():
    """Six weeks of hourly IEEE-14 operating points and their times."""
    pytest.importorskip("pandapower")
    from fdia_graph.generation import _load_states

    X = _load_states(14, None)[: 6 * WEEK_FRAMES]
    return X, np.datetime64("2019-01-07T00:00:00") + np.arange(len(X)) * HOUR


# ---- the split rule -----------------------------------------------------------------------------
def test_an_ieee_timeline_is_split_by_whole_weeks_when_its_pool_has_time(hourly_pool, tmp_path):
    import fdia_graph as fg

    X, time = hourly_pool
    kw = dict(states=X, time=time, families=("At",), ramp_len=10, min_tamper=False, seed=5)
    out = fg.generate("ieee14", "weeks_ieee14", out=str(tmp_path / "weeks.h5"), split_by="week", **kw)
    with h5py.File(out, "r") as f:
        split, stored = f[schema.SPLIT][:], f[schema.TIME][:]
        record = f.attrs[schema.Attr.SETTINGS]
    weeks = split.reshape(6, WEEK_FRAMES)
    assert (weeks == weeks[:, :1]).all()  # a week is in one split, whole
    assert np.bincount(weeks[:, 0], minlength=3).tolist() == [4, 1, 1]  # 0.6, 0.2, 0.2 of six weeks
    assert (np.diff(weeks[:, 0]) != 0).sum() > 2 or weeks[0, 0] != 0  # weeks, not three blocks in time order
    assert (stored == time.astype("datetime64[s]").astype(np.int64)).all()
    assert '"split_by": "week"' in record
    ds = fg.load("weeks_ieee14")
    assert len(ds) == len(X)


def test_weeks_are_refused_on_a_pool_without_time(hourly_pool):
    import fdia_graph as fg

    with pytest.raises(ConfigError, match="this pool carries no time"):
        fg.generate("ieee14", "weeks_refused", states=hourly_pool[0], split_by="week")


def test_time_must_match_the_pool(hourly_pool):
    import fdia_graph as fg

    X, time = hourly_pool
    with pytest.raises(ConfigError, match="one entry per frame"):
        fg.generate("ieee14", "time_refused", states=X, time=time[:-1])


def test_a_setting_at_its_legacy_value_is_not_recorded():
    """A file made before `split_by` existed means "time": that value is never recorded, so the
    record and hash of such a file stand. The hash is the one `main` gave the default settings
    before the field was added."""
    from fdia_graph.models.validation import WhenSet

    record = TimelineSettings().as_record()
    assert "split_by" not in record
    assert TimelineSettings(split_by="week").as_record()["split_by"] == "week"
    assert schema.config_hash(record) == "d916af37352e"
    assert WhenSet(legacy="time").legacy == "time"


# ---- the frames' times --------------------------------------------------------------------------
NOT_TIMES = (
    "SplitRule.time must be the frames' times as a 1-d array of numpy datetime64 values, none of them "
    "NaT (numbers, strings, Python datetime objects and timezone-aware times are not read), got "
)


def _times(n=4):
    return np.datetime64("2019-01-07T00:00:00") + np.arange(n) * HOUR


def test_a_list_of_times_is_read_once_as_an_array():
    from fdia_graph.models.inputs import SplitRule

    rule = SplitRule("week", 4, list(_times()))
    assert rule.time.dtype.kind == "M" and (rule.times == _times()).all()


@pytest.mark.parametrize(
    "bad",
    [
        [0.0, 3600.0, 7200.0, 10800.0],  # floats
        [0, 3600, 7200, 10800],  # integer seconds
        ["2019-01-07", "2019-01-08", "2019-01-09", "2019-01-10"],  # strings
        [datetime.datetime(2019, 1, 7, h) for h in range(4)],  # Python datetimes
        [datetime.datetime(2019, 1, 7, h, tzinfo=datetime.timezone.utc) for h in range(4)],  # tz-aware
        np.r_[_times(3), np.datetime64("NaT")],  # NaT
        _times(4).reshape(2, 2),  # two dimensions
    ],
    ids=["float", "integer", "string", "datetime", "tz-aware", "NaT", "2-d"],
)
def test_times_that_are_not_datetime64_are_refused(bad):
    from fdia_graph.models.inputs import SplitRule

    with pytest.raises(ConfigError, match=re.escape(NOT_TIMES)):
        SplitRule("time", 4, bad)


def test_times_out_of_order_are_refused():
    from fdia_graph.models.inputs import SplitRule

    says = "SplitRule: time must be strictly increasing, and frame 1 is not later than the frame before it"
    with pytest.raises(ConfigError, match=re.escape(says)):
        SplitRule("time", 4, _times()[::-1])
    repeated = _times()[[0, 1, 1, 3]]
    with pytest.raises(ConfigError, match="frame 2 is not later than the frame before it"):
        SplitRule("time", 4, repeated)


def test_bad_times_are_refused_when_the_source_is_built(hourly_pool):
    """Before any design or walk: building the source raises, and `generate` is never reached."""
    from fdia_graph.generation.ieee import IeeeSource

    X, time = hourly_pool
    with pytest.raises(ConfigError, match="time must be strictly increasing"):
        IeeeSource(14, X, TimelineSettings.of(families=("At",)), 5, time[::-1])


# ---- no episode across a gap in time ------------------------------------------------------------
def test_an_onset_fits_only_inside_one_run_of_regular_steps():
    from fdia_graph.generation.plan import time_runs
    from fdia_graph.models.plan import SplitSpan

    time = _times(10).copy()
    time[6:] += 5 * HOUR  # a gap between frames 5 and 6; the regular step is the median, one hour
    runs = time_runs(time)
    assert runs.tolist() == [0] * 6 + [1] * 4
    assert time_runs(_times(1)).tolist() == [0]
    whole = np.arange(10)
    assert SplitSpan(0, 0, 10, whole).onsets_fit(3).tolist() == [True] * 8 + [False] * 2
    gapped = SplitSpan(0, 0, 10, whole, runs).onsets_fit(3)
    assert gapped.tolist() == [True] * 4 + [False] * 2 + [True] * 2 + [False] * 2


def test_no_episode_of_a_file_crosses_a_gap_in_time(hourly_pool, tmp_path):
    import fdia_graph as fg

    X, time = hourly_pool
    X, time = X[:400], time[:400].copy()
    gaps = np.arange(25, 400, 25)  # a five-hour gap every 25 frames, so an 18-frame episode has little room
    for at in gaps:
        time[at:] += 5 * HOUR
    kw = dict(states=X, time=time, families=("At",), ramp_len=18, min_tamper=False, seed=5)
    out = fg.generate("ieee14", "gaps_ieee14", out=str(tmp_path / "gaps.h5"), attacked_frac=0.4, **kw)
    with h5py.File(out, "r") as f:
        onset = f[schema.path(schema.Group.EPISODES, schema.EPISODE_ONSET)][:]
        length = f[schema.path(schema.Group.EPISODES, schema.EPISODE_LENGTH)][:]
    assert len(onset) >= 3
    for a, n in zip(onset, length):
        assert not ((gaps > a) & (gaps < a + n)).any(), f"the episode at {a} crosses a gap"


# ---- the split's random stream ------------------------------------------------------------------
def test_the_split_stream_is_the_one_the_new_york_build_keys_by_its_seed():
    from fdia_graph.data.config import Settings, rng
    from fdia_graph.generation.source import TimelineSource

    for seed in (123, 7):
        ours = TimelineSource.stream(SimpleNamespace(seed=seed), "split")
        theirs = rng(Settings(seed=seed), "split")
        assert ours.bit_generator.state == theirs.bit_generator.state
