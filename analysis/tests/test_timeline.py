"""The analysis toolkit's part of tests/test_timeline.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
The one-file timeline writer: its layout, its episodes and the At and Am frames.

One tiny IEEE-14 timeline is generated once per session (1000 frames, At and the overload Am,
20-frame episodes, the search capped at 16 supports: the fixture `timeline_file` of the root
conftest.py), and read here with h5py against
the layout the module docstring promises."""

import numpy as np
import pytest

pytest.importorskip("pandapower")


def test_stealthy_families_pass_the_residual_test(timeline_file):
    # [WU26 eqs. 13-20]
    """Every stealthy frame is an exact local AC state plus the true scan's own meter noise: a WLS
    residual test at the 1% benign alarm level flags a stealthy frame no more often than it flags
    that frame's own benign twin (a noisy stretch of true states raises both alike)."""
    pytest.importorskip("torch")
    from fdia_analysis.se import WLS
    from fdia_graph.dataset import FdiaGraph

    train, test = (
        FdiaGraph(timeline_file, split="train"),
        FdiaGraph(timeline_file, split="test", order="time"),
    )
    est = WLS().fit(train)
    d = test.export(["node_x", "edge_x", "family", "benign", "edge_benign"])

    def alarm(nx, ex):
        z = est._z_of(nx, ex)
        return np.abs(est._nres(est._solve(z), z)).max(axis=1)

    r, twin = alarm(d["node_x"], d["edge_x"]), alarm(d["benign"], d["edge_benign"])
    level = np.quantile(r[d["family"] == 0], 0.99)
    for fid in (5, 7):
        rows = d["family"] == fid
        if rows.sum() >= 10:
            assert (r[rows] > level).mean() <= (twin[rows] > level).mean() + 0.05, fid


def test_score_bundles_accept_the_seventh_family():
    """`SEBase.score` and `LocalizerBase.score` build their bundles from `FAMILIES`, so both carry
    an `Am` slot once a scored view holds Am frames."""
    from fdia_analysis.models import (
        ErrorPair,
        EstimatorScores,
        FamilyMetrics,
        LocalizerScores,
        OverallMetrics,
    )

    se = EstimatorScores(geo=ErrorPair(1.0, 0.1), Am=ErrorPair(2.0, 0.2))
    assert list(se) == ["Am", "geo"] and se["Am"].angle_mae_deg == 2.0
    fam = FamilyMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    loc = LocalizerScores(all=OverallMetrics(1.0, 1.0, 0.0, 1.0), Am=fam)
    assert list(loc) == ["all", "Am"] and loc.Am is fam
