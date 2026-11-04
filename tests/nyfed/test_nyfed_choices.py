"""The operator's choices are checked before anything runs: a typo is an error, never a silent fall back to
the default run (no data needed)."""

from __future__ import annotations

import pytest

pytest.importorskip("pandapower")  # the operator's true grid needs the [ny] extra


@pytest.mark.parametrize(
    "kw", [{"mode": "trust"}, {"after": "frozen"}, {"entry": "pjmx"}], ids=["mode", "after", "entry"]
)
def test_respond_rejects_an_unknown_choice(kw):
    from fdia_graph.errors import ConfigError
    from fdia_graph.nyfed.operator.response import respond

    with pytest.raises(ConfigError, match="must be one of"):
        respond(None, **kw)  # type: ignore[arg-type]  # checked before the day is read
