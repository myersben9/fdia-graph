"""The public surface of fdia_graph.nyfed: every package's names resolve, reading a dataset stays light, and the
names that became public keep their old spelling for one release (no data needed)."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

pytest.importorskip("pandapower")  # resolving every name imports the [ny] extra

PACKAGES = ("", ".grid", ".scenario", ".sensing", ".attacks", ".timeline", ".operator")


@pytest.mark.parametrize("pkg", PACKAGES, ids=lambda p: p.strip(".") or "nyfed")
def test_every_public_name_resolves(pkg):
    m = importlib.import_module("fdia_graph.nyfed" + pkg)
    assert m.__doc__ and len(m.__doc__) > 200  # the package says what its stage does
    missing = [n for n in m.__all__ if getattr(m, n, None) is None]
    assert not missing
    assert set(m.__all__) <= set(dir(m))


def test_reading_a_dataset_needs_neither_pandapower_nor_pyomo():
    code = (
        "import sys, fdia_graph.nyfed, fdia_graph.nyfed.dataset, fdia_graph.nyfed.operator; "
        "print(sorted(m for m in ('pandapower', 'pyomo') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"


@pytest.mark.parametrize(("old", "new"), [("_TrueGrid", "TrueGrid"), ("_entry_share", "entry_share")])
def test_renamed_names_still_import_with_a_warning(old, new):
    from fdia_graph.nyfed.operator import response

    with pytest.warns(DeprecationWarning, match=new):
        assert getattr(response, old) is getattr(response, new)
