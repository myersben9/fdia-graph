"""The test configuration of both toolkits (`tests/` and `analysis/tests/`, run from the repository
root): the throwaway cache, the fixtures the two trees share, and the rule that leaves the New York
tests out where their extra is not installed. A fixture only one test file uses stays in that file;
the plain helpers the two trees share are in `tests/helpers.py`.

The suite builds its own tiny IEEE-14 timeline (about 20 s, needs the [generate] extra and the
operating-point pool download) inside the throwaway cache, so it never touches a user's
~/.cache/fdia_graph and never downloads a full timeline. Old releases are read from the checked-in
files under tests/data (tests/data/README.md)."""

import atexit
import importlib.util
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

from tests.helpers import TIMELINE_KW, TIMELINE_V083, TINY  # noqa: E402

# tests/nygrid needs the [ny] extra (Pyomo, pandas, pandapower, SciPy, matpowercaseframes, imported when the New York grid's
# modules load): without it the folder is left out of collection rather than failing on import. With it,
# each test still skips when the NYgrid checkout or the built pool is missing under NYGRID_ROOT.
if any(
    importlib.util.find_spec(m) is None
    for m in ("pyomo", "pandas", "pandapower", "scipy", "matpowercaseframes")
):
    collect_ignore_glob = ["tests/nygrid/*"]


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


@pytest.fixture(scope="session")
def timeline_file(tmp_path_factory):
    """Path of a timeline written by `generate_timeline` itself: At and the overload Am, 1000 frames,
    20-frame episodes, the search capped at 16 supports. Seed 2 puts At and Am in every split."""
    from fdia_graph.generation import _load_states
    from fdia_graph.timeline import generate_timeline

    out = tmp_path_factory.mktemp("timeline") / "all.h5"
    return generate_timeline(
        14,
        states=_load_states(14, None)[:1000],
        seed=2,
        ramp_len=20,
        min_tamper=False,
        min_budget=16,
        out=str(out),
    )


@pytest.fixture(scope="module")
def pmu_files(tmp_path_factory):
    """(a v0.8.3-meter file of the old release, read only; a hybrid-meter file generated now)"""
    from fdia_graph.generation import _load_states
    from fdia_graph.timeline import generate_timeline

    d = tmp_path_factory.mktemp("pmu")
    kw = dict(states=_load_states(14, None)[:200], seed=1, families=("At",), ramp_len=10, min_budget=32)
    return TIMELINE_V083, generate_timeline(14, out=str(d / "hybrid.h5"), **kw)


@pytest.fixture(scope="module")
def case():
    """The IEEE-14 generator, its operating-point pool and the knobs of the fewest-tamper search."""
    pytest.importorskip("pandapower")
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.generation import _load_states
    from fdia_graph.models.frames import FrameKnobs
    from fdia_graph.systems.network import Network

    g = FdiaGenerator(Network.from_ieee(14), seed=1)
    X = _load_states(14, None)
    k = FrameKnobs(2, g.false_state.operating_limits(X), True, 4096)
    return g, X, k
