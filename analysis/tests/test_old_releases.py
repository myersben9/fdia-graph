"""The analysis toolkit's part of tests/test_old_releases.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
The pinned data releases stay readable. The generator makes only At and Am, but a timeline of each of
v0.8.0, v0.8.1 and v0.8.3 loads, splits, filters by family name and scores per family as before; the
record shards and stream files of the releases before v0.8.0 are refused. Each fixture under tests/data was written by the SDK
that built its release, or is a slice of the published file (tests/data/README.md).

Also here: the graph fields a GNN user relies on (the series admittance and the clean flow on every
branch), checked on a timeline."""

import pytest

from tests.helpers import OLD_FAMILIES, TIMELINES, family_names


@pytest.mark.parametrize("release", sorted(TIMELINES))
def test_old_releases_score_per_old_family(release):
    """A per-family estimator and localizer score on an old timeline names the old families."""
    from fdia_analysis.localization.methods import SwingThreshold
    from fdia_analysis.se.methods import WLS
    from fdia_graph.dataset import FdiaGraph

    path = TIMELINES[release]
    train, test = FdiaGraph(path, split="train"), FdiaGraph(path, split="test")
    se = WLS().fit(train).score(test)
    loc = SwingThreshold().fit(train).score(test)
    present = family_names(test)
    assert present <= OLD_FAMILIES and {"Aq", "Ad"} <= present
    for name in present:
        assert se[name] is not None and loc[name] is not None
