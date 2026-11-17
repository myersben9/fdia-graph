"""The detection building block: the persistence rule and its calibration, the package's public names,
the rule's golden snapshot (`golden/persistence_rule.json`: a change of behavior fails here and has to
say why), and the alarm-rule example of docs/guides/extending_new_york.md run as published (no data
needed)."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np

import fdia_analysis.se.detect as detect
from fdia_analysis.se.detect.rules import AlarmRule, PersistenceRule
from tests.helpers import ANALYSIS_SRC, GENERATION_SRC, REPO, interpreter_env

GOLDEN = Path(__file__).parent / "golden" / "persistence_rule.json"


def test_the_alarm_rule_example_runs(capsys):
    runpy.run_path(os.path.join(REPO, "examples", "nygrid", "custom_alarm_rule.py"), run_name="__main__")
    assert capsys.readouterr().out.strip()  # it printed its result


def test_every_public_name_resolves():
    assert detect.__doc__ and len(detect.__doc__) > 200  # the package says what its stage does
    assert not [n for n in detect.__all__ if getattr(detect, n, None) is None]
    assert set(detect.__all__) <= set(dir(detect))


def test_the_rules_need_neither_pandapower_nor_pyomo():
    code = (
        "import sys, fdia_analysis.se.detect.rules; "
        "print(sorted(m for m in ('pandapower', 'pyomo') if m in sys.modules))"
    )
    env = interpreter_env(GENERATION_SRC, ANALYSIS_SRC)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert out.stdout.strip() == "[]"


def test_persistence_rule_needs_k_of_n():
    rule = PersistenceRule(theta=10.0, k=2, n=3)
    assert rule.first_alarm(np.array([11.0, 0.0, 0.0, 11.0])) == -1  # two exceedances, but four scans apart
    assert rule.first_alarm(np.array([0.0, 11.0, 0.0, 11.0])) == 3  # two in the last three
    assert rule.first_alarm(np.array([11.0, np.nan, 11.0])) == 2  # an unevaluated scan counts as below


def test_persistence_rule():
    """The residual test's persistence rule (428.7, 2 of 3) on a fixed residual series."""
    rule = PersistenceRule(428.7, 2, 3)
    j = np.array([400.0, 430.0, 410.0, 431.0, np.nan, 440.0, 441.0, 300.0])
    got = {"first": rule.first_alarm(j), "none": rule.first_alarm(j[:3])}
    assert got == json.loads(GOLDEN.read_text())


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
