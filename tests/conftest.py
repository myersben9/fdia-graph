"""Shared fixtures. The suite builds its own tiny IEEE-14 shard (about 15 s, needs the [generate]
extra and the 13 MB operating-point pool download) inside a throwaway cache, so it never touches a
user's ~/.cache/fdia_graph and never downloads a full shard."""

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
TINY_TL = "tiny_ieee14_timeline"


@pytest.fixture(scope="session")
def shard(tmp_path_factory):
    """Name of a small generated IEEE-14 record shard (the v0.7.2 layout): 80 benign + 12 per
    family (At ramps expand). The frozen references are built from it until step 3 of
    docs/plans/ONE_DATASET_PLAN.md re-freezes them on a timeline."""
    pytest.importorskip("pandapower")
    from fdia_graph.generation import generate_shard

    out = tmp_path_factory.mktemp("shard") / "tiny.h5"
    # The string system name is deliberate: it is the documented public form and once crashed generate().
    generate_shard("ieee14", TINY, per_family=12, n_benign=80, out=str(out), seed=1)
    return TINY


@pytest.fixture(scope="session")
def timeline(tmp_path_factory):
    """Name of a small generated IEEE-14 timeline: 1000 frames, every family, 20-frame ramps."""
    pytest.importorskip("pandapower")
    import fdia_graph as fg

    out = tmp_path_factory.mktemp("timeline") / "tiny_tl.h5"
    fg.generate("ieee14", TINY_TL, frames=1000, ramp_len=20, out=str(out), seed=3)
    return TINY_TL


@pytest.fixture(scope="session")
def splits(shard):
    import fdia_graph as fg

    return {s: fg.load(shard, split=s) for s in ("train", "val", "test")}
