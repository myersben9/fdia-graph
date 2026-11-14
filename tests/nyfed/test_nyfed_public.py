"""The public surface of the New York code: every package's names resolve, reading a dataset stays light, and the
names that became public keep their old spelling for one release (no data needed)."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

PACKAGES = (
    "fdia_graph.generation.nygrid",
    "fdia_graph.engine.sensing",
    "fdia_graph.operator",
    "fdia_graph.se.detect",
)


@pytest.mark.parametrize("pkg", PACKAGES)
def test_every_public_name_resolves(pkg):
    pytest.importorskip("pandapower")  # resolving every name imports the [ny] extra
    m = importlib.import_module(pkg)
    assert m.__doc__ and len(m.__doc__) > 200  # the package says what its stage does
    missing = [n for n in m.__all__ if getattr(m, n, None) is None]
    assert not missing
    assert set(m.__all__) <= set(dir(m))


def test_reading_a_dataset_needs_neither_pandapower_nor_pyomo():
    """Runs without the [ny] extra too: there above all, reading a dataset must not need it."""
    pytest.importorskip("h5py")  # the dataset reader's own dependency
    code = (
        "import sys, fdia_graph.generation.nygrid, fdia_graph.dataset.clients, fdia_graph.operator; "
        "print(sorted(m for m in ('pandapower', 'pyomo') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


@pytest.mark.parametrize(("old", "new"), [("_TrueGrid", "TrueGrid"), ("_entry_share", "entry_share")])
def test_renamed_names_still_import_with_a_warning(old, new):
    pytest.importorskip("pandapower")  # the operator's module imports the [ny] extra
    from fdia_graph.operator import response

    with pytest.warns(DeprecationWarning, match=new):
        assert getattr(response, old) is getattr(response, new)
