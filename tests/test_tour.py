"""The tour of docs/guides/tour.md prints what the page shows: `examples/tour.py` runs as published and
its output is compared, line for line, with the block the page quotes, so the page cannot go stale."""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import pytest

pytest.importorskip("pandapower")

REPO = Path(__file__).resolve().parents[1]
START, END = "<!-- tour:start -->", "<!-- tour:end -->"


def quoted_output() -> list[str]:
    """The lines of the output block of the page, between its two markers, the code fence left out."""
    page = (REPO / "docs" / "guides" / "tour.md").read_text(encoding="utf-8")
    block = page.split(START)[1].split(END)[0].strip().splitlines()
    return [line.rstrip() for line in block[1:-1]]


def test_the_page_quotes_an_output_block():
    lines = quoted_output()
    assert lines[0].startswith("1. ") and any(line.startswith("8. ") for line in lines)


@pytest.mark.skipif(not os.environ.get("FDIA_SLOW"), reason="set FDIA_SLOW=1: a one-minute IEEE-14 search")
def test_the_tour_prints_what_the_page_shows(capsys):
    runpy.run_path(str(REPO / "examples" / "tour.py"), run_name="__main__")
    printed = [line.rstrip() for line in capsys.readouterr().out.strip().splitlines()]
    assert printed == quoted_output()
    assert "of the file: True" in printed[-3]  # the frame built by hand is the file's
