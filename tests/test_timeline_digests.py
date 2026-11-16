"""The generated IEEE timeline, whole: every dataset and attribute of the file against a stored digest.

The one-timeline-path refactor moves the stages that write this file; these snapshots say whether a step
changed anything in it. Two files: the shared tiny timeline of the suite (the search off for At, a small
budget for Am) and a searched one (the defaults: the fewest-tamper search for both families). `digests.py`
says what is compared where, and how to rewrite a snapshot.
"""

from pathlib import Path

import pytest

from .digests import check_file

GOLDEN = Path(__file__).parent / "golden"
# the defaults on a short pool slice: At held on its fewest-tamper support, Am searched in full
SEARCHED_KW = dict(frames=240, ramp_len=12, seed=3)


def test_the_suites_timeline_is_the_stored_one(timeline):
    import fdia_graph as fg

    check_file("timeline_ieee14_suite", fg.load(timeline).path, GOLDEN)


def test_a_searched_timeline_is_the_stored_one(tmp_path):
    pytest.importorskip("pandapower")
    import fdia_graph as fg

    out = tmp_path / "searched.h5"
    fg.generate("ieee14", "digest_searched_ieee14", out=str(out), **SEARCHED_KW)
    check_file("timeline_ieee14_searched", out, GOLDEN)
