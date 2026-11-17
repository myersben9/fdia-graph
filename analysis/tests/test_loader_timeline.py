"""The analysis toolkit's part of tests/test_loader_timeline.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
`fg.load` on a timeline file: the record table and the timeline from one file, `order`, the
benign layer on records, batches and exports, `ds.windows` and `ds.episodes`. Uses the `timeline`
fixture of conftest."""

import pytest

pytest.importorskip("pandapower")
pytest.importorskip("torch")

import fdia_graph as fg  # noqa: E402
from fdia_analysis.models import LocalizerConfig


def test_estimator_and_localizer_score_a_timeline_with_am(timeline):
    from fdia_analysis.localization import SwingThreshold
    from fdia_analysis.se import WLS

    train, test = fg.load(timeline, split="train"), fg.load(timeline, split="test")
    assert 7 in set(test.export(["family"])["family"].tolist())
    se = WLS().fit(train).score(test)
    assert se.Am is not None and se["Am"].angle_mae_deg > 0 and list(se)[-1] == "geo"
    loc = SwingThreshold(localizer=LocalizerConfig(fa_target=0.01)).fit(train).score(test)
    assert loc.Am is not None and 0 <= loc["Am"].detection_rate <= 1
