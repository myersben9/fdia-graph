"""The dependency between the two toolkits points one way: the analysis toolkit `fdia_analysis` builds
on the generation toolkit `fdia_graph`, and `fdia_graph` never imports it. Checked in the source (no
import, no module name in a lazy table) and by running: a fresh interpreter in which `fdia_analysis`
is blocked (so the check is the same whether or not it is installed) loads every module of
`fdia_graph`, generates a timeline and reads it back."""

import ast
import os
import subprocess
import sys

import pytest

from tests.helpers import GENERATION_SRC, interpreter_env

ANALYSIS = "fdia_analysis"
PACKAGE = os.path.join(GENERATION_SRC, "fdia_graph")

# run in the fresh interpreter: the analysis toolkit is blocked (None in sys.modules makes any import
# of it an ImportError, installed or not), every module of fdia_graph imports (a missing optional
# extra aside), and with pandapower a small timeline is generated and read
ALONE = """
import importlib, importlib.util, pkgutil, sys

sys.modules["fdia_analysis"] = None
import fdia_graph as fg

for info in pkgutil.walk_packages(fg.__path__, "fdia_graph."):
    try:
        importlib.import_module(info.name)
    except ImportError as e:
        if (e.name or "").startswith("fdia_"):
            raise
if importlib.util.find_spec("pandapower") is not None:
    fg.generate("ieee14", "alone", out=sys.argv[1], frames=200, ramp_len=20, seed=2, min_tamper=False, min_budget=16)
    assert len(fg.load("alone", split="train")) > 0
assert sys.modules["fdia_analysis"] is None, "something replaced the block"
assert not [m for m in sys.modules if m.startswith("fdia_analysis.")]
print("ok")
"""


def _docstrings(tree: ast.AST) -> set[int]:
    return {
        id(n.body[0].value)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and n.body
        and isinstance(n.body[0], ast.Expr)
        and isinstance(n.body[0].value, ast.Constant)
    }


def _names_analysis(tree: ast.AST) -> list[int]:
    """The lines of a module that import the analysis toolkit, or spell one of its modules in a string
    that is not a docstring (a lazy table, an `importlib.import_module` argument)."""
    docs = _docstrings(tree)
    lines = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(a.name.split(".")[0] == ANALYSIS for a in n.names):
            lines.append(n.lineno)
        elif isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] == ANALYSIS:
            lines.append(n.lineno)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            if n.value.split(".")[0] == ANALYSIS:
                lines.append(n.lineno)
    return lines


def test_nothing_in_the_generation_toolkit_imports_the_analysis_toolkit():
    found = []
    for dirpath, _, files in os.walk(PACKAGE):
        for f in sorted(files):
            if f.endswith(".py"):
                path = os.path.join(dirpath, f)
                with open(path, encoding="utf8") as fh:
                    tree = ast.parse(fh.read())
                found += [f"{os.path.relpath(path, PACKAGE)}:{line}" for line in _names_analysis(tree)]
    assert not found, f"fdia_graph must not import fdia_analysis: {found}"


def test_the_check_sees_an_import_and_a_lazy_name():
    seen = _names_analysis(
        ast.parse('"""fdia_analysis.se in prose."""\nimport fdia_analysis.se\nT = {"x": "fdia_analysis.se"}')
    )
    assert seen == [2, 3]


def test_the_generation_toolkit_runs_without_the_analysis_toolkit(tmp_path):
    """Imports every module of fdia_graph, generates a small IEEE-14 timeline and loads it, in an
    interpreter that reads this checkout's `src` and in which `fdia_analysis` is blocked."""
    pytest.importorskip("h5py")
    run = subprocess.run(
        [sys.executable, "-c", ALONE, str(tmp_path / "alone.h5")],
        capture_output=True,
        text=True,
        env=interpreter_env(GENERATION_SRC),
    )
    assert run.returncode == 0 and run.stdout.strip().endswith("ok"), run.stdout + run.stderr
