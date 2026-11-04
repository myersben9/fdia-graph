"""The extension examples of docs/nyfed/EXTENDING.md run as published, so the guide never shows code that broke
(no data needed)."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "nyfed"


@pytest.mark.parametrize("name", ["custom_alarm_rule", "custom_attack_family", "custom_operator"])
def test_example_runs(name, capsys):
    if name == "custom_operator":
        pytest.importorskip("pandapower")  # the operator package imports the [ny] extra
    runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name="__main__")
    assert capsys.readouterr().out.strip()  # it printed its result
