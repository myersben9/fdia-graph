"""The operator's choices are one validated record, and an operator is anything that returns a Response
(no data needed)."""

from __future__ import annotations

import pytest

pytest.importorskip("pandapower")  # the operator package imports the [ny] extra


def test_policy_checks_every_choice():
    from fdia_graph.errors import ConfigError
    from fdia_graph.nyfed.operator import OperatorPolicy

    p = OperatorPolicy(mode="accounting", delay=2, entry="west", closed_loop=True, after="freeze_cs")
    assert (p.mode, p.delay, p.entry, p.closed_loop, p.after) == ("accounting", 2, "west", True, "freeze_cs")
    for bad in ({"mode": "trust"}, {"delay": -1}, {"delay": 1.5}, {"entry": "pjmx"}, {"after": "frozen"}):
        with pytest.raises(ConfigError, match="OperatorPolicy"):
            OperatorPolicy(**bad)


def test_the_choices_and_their_tables_agree():
    from fdia_graph.nyfed.operator import AFTER, ENTRY, MODES, AlarmAnswer, Entry, OperatorMode

    assert MODES == tuple(OperatorMode.values()) and AFTER == tuple(AlarmAnswer.values())
    assert set(ENTRY) == set(Entry.values())


def test_an_operator_is_anything_that_responds():
    """The extension point: a partner's operator only has to return a Response."""
    from fdia_graph.nyfed.operator import NyisoOperator, Operator, OperatorPolicy

    class DoesNothing:
        def respond(self, day, *, calib=None, alarm_at=None):
            raise NotImplementedError

    assert isinstance(NyisoOperator(OperatorPolicy(mode="oracle")), Operator)
    assert isinstance(DoesNothing(), Operator)
