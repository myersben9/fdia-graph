"""The public surface of the New York code: every module of its packages imports, a package exports no name of
its own, and reading a dataset stays light (no data needed)."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import subprocess
import sys

import pytest

PACKAGES = (
    "fdia_graph.data",
    "fdia_graph.sensing",
    "fdia_graph.systems",
)


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_module_imports_and_the_package_exports_no_name(pkg):
    pytest.importorskip("pandapower")  # importing every module imports the [ny] extra
    m = importlib.import_module(pkg)
    assert m.__doc__ and len(m.__doc__) > 200  # the package says what its stage does
    for info in pkgutil.iter_modules(m.__path__):
        importlib.import_module(f"{pkg}.{info.name}")
    # a name is imported from the module that defines it: the package holds its modules and nothing else
    assert not [n for n, o in vars(m).items() if not n.startswith("_") and not inspect.ismodule(o)]


@pytest.mark.parametrize(
    ("families", "interfaces", "says"),
    [(("Am", "Ap"), ("SPR/DUN-SOUTH",), "Am, At"), (("Am",), (), "interface")],
    ids=["a family nothing builds", "Am with no interface"],
)
def test_the_release_source_refuses_what_it_does_not_build(families, interfaces, says):
    """Before it reads a file: a family other than Am and At, or Am with no interface to attack."""
    pytest.importorskip("pandapower")
    from fdia_graph.errors import ConfigError
    from fdia_graph.generation.newyork import NewYorkSource
    from fdia_graph.models.config import NewYorkRelease

    with pytest.raises(ConfigError, match=says):
        NewYorkSource.production("no_pool.h5", NewYorkRelease(families=families, interfaces=interfaces))


def test_reading_a_dataset_needs_neither_pandapower_nor_pyomo():
    """Runs without the [ny] extra too: there above all, reading a dataset must not need it."""
    pytest.importorskip("h5py")  # the dataset reader's own dependency
    code = (
        "import sys, fdia_graph.data, fdia_graph.generation, fdia_graph.dataset.clients; "
        "print(sorted(m for m in ('pandapower', 'pyomo') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
