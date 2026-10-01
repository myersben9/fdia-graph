"""Shared fixtures. The suite builds its own tiny IEEE-14 timeline (about 20 s, needs the [generate]
extra and the operating-point pool download) inside a throwaway cache, so it never touches a user's
~/.cache/fdia_graph and never downloads a full timeline. Old releases are read from the checked-in
files under tests/data (tests/data/README.md)."""

import atexit
import os
import shutil
import tempfile

# The cache dir is read when fdia_graph is imported, so it must be set before any test module
# imports the package. A fresh temp dir per session keeps the local-dataset registry clean, and
# atexit removes it even when the run is interrupted.
_CACHE = tempfile.mkdtemp(prefix="fdia_graph_test_cache_")
os.environ["FDIA_GRAPH_CACHE"] = _CACHE
atexit.register(shutil.rmtree, _CACHE, ignore_errors=True)

import pytest  # noqa: E402

TINY = "tiny_ieee14"
# The v0.7.2 record-shard layout the loader still reads: 12 benign + 3 per family on a 200-timestep
# pool slice, written by the pre-0.18 writer (deleted in 0.18) and checked in.
SHARD_V072 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tiny_shard_v072.h5")


# The shared tiny timeline: new generation's families (At and the overload Am) on the hybrid meters,
# 1000 frames with 20-frame episodes. The fewest-tamper search is off for At (min_tamper=False) and
# the overload Am searches at most 16 supports, so the fixture builds in about 20 s; the searched
# paths have their own small fixtures (test_minimize, test_overload, test_search_speed). Seed 2 puts
# both families in every split.
TIMELINE_KW = dict(frames=1000, ramp_len=20, seed=2, min_tamper=False, min_budget=16)


@pytest.fixture(scope="session")
def timeline(tmp_path_factory):
    """Name of a small generated IEEE-14 timeline: 1000 frames, At and Am, 20-frame episodes."""
    pytest.importorskip("pandapower")
    import fdia_graph as fg

    out = tmp_path_factory.mktemp("timeline") / "tiny.h5"
    # The string system name is deliberate: it is the documented public form and once crashed generate().
    fg.generate("ieee14", TINY, out=str(out), **TIMELINE_KW)
    return TINY


@pytest.fixture(scope="session")
def splits(timeline):
    import fdia_graph as fg

    return {s: fg.load(timeline, split=s) for s in ("train", "val", "test")}
