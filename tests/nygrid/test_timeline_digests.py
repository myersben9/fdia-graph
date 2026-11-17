"""The New York timeline, whole: every dataset and attribute of the files the release's source writes on the
checked-in days' pool, against stored digests (`tests/digests.py` says what is compared where).

Three files, each written by the path every timeline takes (`TimelineSource.generate`): the timeline of the release's source (Am
against the interface, and At), one At episode alone, and the same episode lagged by one frame.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from fdia_graph import schema
from fdia_graph.constants.families import FAMILY_CODE

from ..digests import check_file
from .inputs import DAYS_POOL, ROOT, needs_data, needs_territories

pytestmark = [needs_data, needs_territories]
GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def source():
    pytest.importorskip("pyomo")
    from fdia_graph.generation.newyork import NewYorkSource
    from fdia_graph.models.config import NewYorkRelease

    release = NewYorkRelease(families=("Am", "At"), workers=1, survey=0, root=ROOT)
    return NewYorkSource.production(str(DAYS_POOL), release)


def test_the_days_timeline_is_the_stored_one(source, tmp_path):
    out = tmp_path / "days.h5"
    source.generate(out)
    with h5py.File(out, "r") as h:  # a file with both families in it, or the digest pins nothing
        families = h[schema.path(schema.Group.EPISODES, schema.EPISODE_FAMILY)][:]
        assert set(families.tolist()) == {FAMILY_CODE["Am"], FAMILY_CODE["At"]}
    check_file("timeline_newyork_days", out, GOLDEN)


@pytest.mark.parametrize("lag", (0, 1))
def test_a_single_episode_is_the_stored_one(source, tmp_path, lag):
    from fdia_graph.constants.newyork import SolveKind
    from fdia_graph.generation.newyork import SingleEpisodeSource

    pool = source.pool
    solved = pool.solve_kind == SolveKind.SOLVED
    # the first onset whose window, and the frame its lagged twin starts from, are all solved
    onset = next(t for t in range(40, len(solved) - 40) if solved[t - 1 : t + 20].all())
    out = tmp_path / f"episode_lag{lag}.h5"
    SingleEpisodeSource(source, onset + lag, "At", lag=lag).generate(out)
    check_file(f"episode_newyork_at_lag{lag}", out, GOLDEN)
