"""Where the repository's two toolkits live, said once for every tool: the generation toolkit
`fdia_graph` and the analysis toolkit `fdia_analysis`, each a package under its own source root.

    from sources import PACKAGES, import_this_checkout   # tools/ is a script's own folder, so on sys.path

A tool that imports the packages calls `import_this_checkout()` first, so it reads this checkout's
source and not an installed copy of another one.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# the two package names, then package name -> its folder, relative to the repository root with
# forward slashes; the order is the reading order (the analysis toolkit depends on the generation
# toolkit, never the reverse)
GENERATION = "fdia_graph"
ANALYSIS = "fdia_analysis"
PACKAGES = {GENERATION: "src/fdia_graph", ANALYSIS: "analysis/src/fdia_analysis"}
SOURCE_ROOTS = tuple(os.path.dirname(folder) for folder in PACKAGES.values())  # "src", "analysis/src"
TESTS = ("tests", "analysis/tests")  # each toolkit's test tree, in the same order; pytest runs from the root
CONFTEST = "conftest.py"  # the fixtures the two test trees share, at the repository root


def package_dir(package: str) -> str:
    """The absolute folder of one of the two packages."""
    return os.path.join(ROOT, *PACKAGES[package].split("/"))


def python_path() -> str:
    """The two source roots as a PYTHONPATH value, for a subprocess that imports the packages."""
    return os.pathsep.join(os.path.join(ROOT, *root.split("/")) for root in SOURCE_ROOTS)


def import_this_checkout() -> None:
    """Put the two source roots first on sys.path."""
    sys.path[:0] = python_path().split(os.pathsep)
