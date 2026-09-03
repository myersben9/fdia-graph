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


@pytest.fixture(scope="session")
def shard(tmp_path_factory):
    """Name of a small generated IEEE-14 shard: 80 benign + 12 per family (At ramps expand)."""
    pytest.importorskip("pandapower")
    import fdia_graph as fg

    out = tmp_path_factory.mktemp("shard") / "tiny.h5"
    # The string system name is deliberate: it is the documented public form and once crashed generate().
    fg.generate("ieee14", TINY, per_family=12, n_benign=80, out=str(out), seed=1)
    return TINY


@pytest.fixture(scope="session")
def splits(shard):
    import fdia_graph as fg

    return {s: fg.load(shard, split=s) for s in ("train", "val", "test")}
