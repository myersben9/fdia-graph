"""The analysis toolkit's part of tests/test_public_bundles.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
The public dual-access bundles (data models step 3): each is the dict it used to be, with the
same keys in the same order, and its attribute view agrees with the dict view."""

import json

from fdia_analysis.models.config import LocalizerConfig
from tests.helpers import agree


def test_estimator_and_localizer_scores(timeline):
    import fdia_graph as fg
    from fdia_analysis.localization.methods import SwingThreshold
    from fdia_analysis.models.scores import ErrorPair, EstimatorScores, LocalizerScores, OverallMetrics
    from fdia_analysis.se.methods import WLS

    train, test = fg.load(timeline, split="train"), fg.load(timeline, split="test")
    se = WLS().fit(train).score(test)
    assert isinstance(se, EstimatorScores) and isinstance(se.geo, ErrorPair)
    agree(se)
    assert se["geo"]["angle_mae_deg"] == se.geo.angle_mae_deg
    assert list(se)[-1] == "geo" and list(se)[0] == "benign"
    # one block per family present in the view, and the geometric mean
    assert set(se) == {"benign", "At", "Am", "geo"} and all(
        set(se[k]) == {"angle_mae_deg", "voltage_mae_pu"} for k in se
    )

    loc = SwingThreshold(localizer=LocalizerConfig(fa_target=0.01)).fit(train).score(test)
    assert isinstance(loc, LocalizerScores) and isinstance(loc.all, OverallMetrics)
    agree(loc)
    assert loc["all"]["macro_f1"] == loc.all.macro_f1
    assert set(loc) == {"all", "benign", "At", "Am"}
    assert set(loc["At"]) == {
        "strict_acc",
        "node_precision",
        "node_recall",
        "node_f1",
        "macro_f1",
        "sample_f1",
        "detection_rate",
    }
    json.dumps(loc)


def test_jacobian_outputs(timeline):
    import fdia_graph as fg
    from fdia_analysis.models.scores import JacobianOutputs
    from fdia_analysis.se.jacobian import JacobianFeatures

    train, test = fg.load(timeline, split="train"), fg.load(timeline, split="test")
    out = (
        JacobianFeatures()
        .fit(train)
        .transform(test.export(["node_x", "edge_x", "prev_node_x", "prev_edge_x", "prev_timestep"]))
    )
    assert isinstance(out, JacobianOutputs)
    agree(out)
    assert list(out) == ["bus", "global", "dx_hat", "r_perp"]
    assert out["global"] is out.global_ and out.bus.shape[:2] == (len(test), test.N)
