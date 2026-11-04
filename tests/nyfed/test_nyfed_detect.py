"""The detection building blocks that need no grid: the persistence rule and its calibration, and the meter fleets
(the estimator itself is checked on real days by the golden snapshots)."""

from __future__ import annotations

import numpy as np

from fdia_graph.nyfed.detect import FLEET_STREAM, AlarmRule, PersistenceRule, fleet_bias, with_fleet
from fdia_graph.nyfed.sensing.meters import MeterPlan


def test_persistence_rule_needs_k_of_n():
    rule = PersistenceRule(theta=10.0, k=2, n=3)
    assert rule.first_alarm(np.array([11.0, 0.0, 0.0, 11.0])) == -1  # two exceedances, but four scans apart
    assert rule.first_alarm(np.array([0.0, 11.0, 0.0, 11.0])) == 3  # two in the last three
    assert rule.first_alarm(np.array([11.0, np.nan, 11.0])) == 2  # an unevaluated scan counts as below


def test_fit_takes_the_percentile_of_the_benign_statistic():
    benign = np.arange(1001.0)
    assert PersistenceRule.fit(benign, q=99.9).theta == np.percentile(benign, 99.9)
    assert isinstance(PersistenceRule(1.0), AlarmRule)  # the rule satisfies the protocol a partner implements


def _plan(m: int = 6) -> MeterPlan:
    rel = np.array([0.017, 0.0] * (m // 2))
    return MeterPlan(
        kind=np.zeros(m, int),
        bus=np.zeros(m, int),
        branch=np.zeros(m, int),
        end=np.zeros(m, int),
        owner=np.zeros(m, int),
        sigma_rel=rel,
        sigma_abs=np.where(rel > 0, 0.0, 0.3),
        bias=np.zeros(m),
    )


def test_fleets_are_reproducible_and_draw_zero_is_the_plan():
    plan = _plan()
    assert np.array_equal(fleet_bias(plan, 2), fleet_bias(plan, 2))
    assert not np.array_equal(fleet_bias(plan, 1), fleet_bias(plan, 2))
    assert with_fleet(plan, 0) is plan and np.array_equal(with_fleet(plan, 3).bias, fleet_bias(plan, 3))
    assert FLEET_STREAM == (20190101, 4219)  # the stream the published fleets were drawn from
