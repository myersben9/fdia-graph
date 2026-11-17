"""The docs' results blocks render from the results store (`tools/results_docs.py`): the repository's
docs match it, a block edited by hand fails the check, and the lint reports a number typed outside a block."""

from __future__ import annotations

import os
import subprocess
import sys

from tests.helpers import ANALYSIS_SRC, GENERATION_SRC, interpreter_env
from tests.helpers import REPO as ROOT


def _tool(*args: str) -> subprocess.CompletedProcess[str]:
    env = interpreter_env(GENERATION_SRC, ANALYSIS_SRC)
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "results_docs.py"), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_the_repository_docs_match_the_store() -> None:
    out = _tool("--check")
    assert out.returncode == 0, out.stdout + out.stderr


def test_the_check_fails_on_a_hand_edited_block_and_lint_flags_a_typed_number(tmp_path: object) -> None:
    doc = os.path.join(ROOT, "docs", "analysis", "estimation", "README.md")
    text = open(doc, encoding="utf-8").read()
    start = text.index("<!-- results: red se ieee14 geo angle_mae_deg wls prior+huber -->")
    end = text.index("<!-- /results -->", start)
    edited = text[:start] + "<!-- results: red se ieee14 geo angle_mae_deg wls prior+huber -->99" + text[end:]
    path = os.path.join(str(tmp_path), "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(edited)
    assert _tool("--check", path).returncode == 1
    typed = os.path.join(str(tmp_path), "typed.md")
    with open(typed, "w", encoding="utf-8") as f:
        f.write(
            "A block <!-- results: v se ieee14 wls geo angle_mae_deg --><!-- /results --> and a typed 0.123 here.\n"
        )
    lint = _tool("--lint", typed).stdout
    assert "1 number(s)" in lint and "0.123" in lint
