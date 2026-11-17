"""The two toolkits are released together at one version: both `pyproject.toml` files and both
`__version__` values say the same number, and `fdia-analysis` requires exactly that `fdia-graph`
(a released `fdia-graph` of another version may lack a name `fdia_analysis` imports). No data needed;
`tools/release.py` refuses to tag a commit where this does not hold."""

import os
import re

from tests.helpers import ANALYSIS_SRC, GENERATION_SRC, REPO

ANALYSIS = os.path.join(REPO, "analysis")


def _find(pattern: str, *path: str) -> str:
    with open(os.path.join(*path), encoding="utf8") as f:
        found = re.search(pattern, f.read(), re.M)
    assert found, f"{pattern} not found in {os.path.join(*path)}"
    return found.group(1)


def test_the_two_toolkits_share_one_version():
    version = _find(r'^version = "([^"]+)"', REPO, "pyproject.toml")
    assert _find(r'^version = "([^"]+)"', ANALYSIS, "pyproject.toml") == version
    assert _find(r'^__version__ = "([^"]+)"', GENERATION_SRC, "fdia_graph", "__init__.py") == version
    assert _find(r'^__version__ = "([^"]+)"', ANALYSIS_SRC, "fdia_analysis", "__init__.py") == version
    assert _find(r'^\s*"(fdia-graph[^"]*)"', ANALYSIS, "pyproject.toml") == f"fdia-graph=={version}"
