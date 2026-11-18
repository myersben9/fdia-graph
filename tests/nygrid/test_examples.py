"""The extension examples of docs/guides/extending.md run as published, so the guide never shows code that broke
(no data needed)."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "nygrid"


@pytest.mark.parametrize("name", ["custom_attack_family"])
def test_example_runs(name, capsys):
    runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert out.strip()  # it printed its result
    # the episode goes through the path a timeline takes it by, which reads its family's code
    assert "family code 2" in out
