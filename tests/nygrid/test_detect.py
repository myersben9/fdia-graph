"""The detection building block: the persistence rule and its calibration (no data needed)."""

from __future__ import annotations

import numpy as np

from fdia_graph.se.detect.rules import AlarmRule, PersistenceRule


def test_persistence_rule_needs_k_of_n():
    rule = PersistenceRule(theta=10.0, k=2, n=3)
    assert rule.first_alarm(np.array([11.0, 0.0, 0.0, 11.0])) == -1  # two exceedances, but four scans apart
    assert rule.first_alarm(np.array([0.0, 11.0, 0.0, 11.0])) == 3  # two in the last three
    assert rule.first_alarm(np.array([11.0, np.nan, 11.0])) == 2  # an unevaluated scan counts as below


def test_fit_takes_the_percentile_of_the_benign_statistic():
    benign = np.arange(1001.0)
    assert PersistenceRule.fit(benign, q=99.9).theta == np.percentile(benign, 99.9)
    assert isinstance(PersistenceRule(1.0), AlarmRule)  # the rule satisfies the protocol a partner implements


def test_rule_edge_cases():
    """No scans is no alarm; a NaN never exceeds, even a negative threshold; nothing to fit on is an error."""
    import pytest

    from fdia_graph.errors import NoBenignRecords

    assert PersistenceRule(1.0).first_alarm(np.array([])) == -1
    assert PersistenceRule(-1.0).first_alarm(np.array([np.nan, np.nan, np.nan])) == -1
    with pytest.raises(NoBenignRecords):
        PersistenceRule.fit(np.array([np.nan, np.nan]))
    with pytest.raises(NoBenignRecords):
        PersistenceRule.fit(np.array([]))
