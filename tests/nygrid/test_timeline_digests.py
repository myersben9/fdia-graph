"""The New York timeline, whole: every dataset and attribute of the files the release builder writes on the
checked-in days' pool, against stored digests (`tests/digests.py` says what is compared where).

Three files: the timeline `build` writes (Am against the interface, and At), one At episode built alone, and the
same episode lagged by one frame. The one-timeline-path refactor moves the stages behind them.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from ..digests import check_file
from .inputs import DAYS_POOL, ROOT, needs_data, needs_territories

pytestmark = [needs_data, needs_territories]
GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def builder():
    pytest.importorskip("pyomo")
    from fdia_graph.generation.nygrid.build import production_builder

    return production_builder(str(DAYS_POOL), ("Am", "At"), workers=1, survey=0, root=ROOT)


def test_the_days_timeline_is_the_stored_one(builder, tmp_path):
    out = tmp_path / "days.h5"
    builder.build(out)
    with h5py.File(out, "r") as h:  # a file with both families in it, or the digest pins nothing
        assert set(np.unique(h["episodes/family"][:]).tolist()) == {b"Am", b"At"}
    check_file("timeline_newyork_days", out, GOLDEN)


@pytest.mark.parametrize("lag", (0, 1))
def test_a_single_episode_is_the_stored_one(builder, tmp_path, lag):
    from fdia_graph.generation.nygrid.pool import SOLVED

    pool = builder.pool
    solved = pool.solve_kind == SOLVED
    # the first onset whose window, and the frame its lagged twin starts from, are all solved
    onset = next(t for t in range(40, len(solved) - 40) if solved[t - 1 : t + 20].all())
    out = tmp_path / f"episode_lag{lag}.h5"
    builder.build_episode(out, onset + lag, "At", lag=lag)
    check_file(f"episode_newyork_at_lag{lag}", out, GOLDEN)
