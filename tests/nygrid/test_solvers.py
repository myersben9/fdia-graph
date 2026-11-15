"""The checked Ipopt every nygrid OPF runs on: a solver that is missing, or found but unable to start, raises
`SolverUnavailable` with the fix, instead of reading as an infeasible model downstream."""

from __future__ import annotations

import pytest

from fdia_graph.errors import SolverUnavailable

pe = pytest.importorskip("pyomo.environ")


class _FakeSolver:
    def __init__(self, found: bool, starts: bool, version: tuple = (3, 14, 19, 0)):
        self.found, self.starts, self.options, self._version = found, starts, {}, version

    def available(self, exception_flag: bool = False) -> bool:
        return self.found

    def version(self):
        if not self.starts:
            raise RuntimeError("the program did not start")
        return self._version


@pytest.mark.parametrize(
    ("found", "starts"), [(False, False), (True, False)], ids=["missing", "does-not-start"]
)
def test_unusable_ipopt_raises_with_the_fix(monkeypatch, found, starts):
    from fdia_graph.engine import solvers

    monkeypatch.setattr(pe, "SolverFactory", lambda *a, **k: _FakeSolver(found, starts))
    with pytest.raises(SolverUnavailable, match="conda install -c conda-forge ipopt"):
        solvers.ipopt(tol=1e-8)


def test_working_ipopt_takes_the_options(monkeypatch):
    from fdia_graph.engine import solvers

    monkeypatch.setattr(pe, "SolverFactory", lambda *a, **k: _FakeSolver(True, True))
    assert solvers.ipopt(tol=1e-8, max_iter=300).options == {"tol": 1e-8, "max_iter": 300}


def test_solver_unavailable_is_not_a_data_condition():
    """A broken environment is not a property of the data: callers catching data conditions must not swallow it."""
    from fdia_graph.errors import DataConditionError

    assert not issubclass(SolverUnavailable, (DataConditionError, ValueError))


def test_ipopt_that_prints_no_version_raises(monkeypatch):
    """Pyomo reads a program that printed nothing as version (0, 0, 0, 0): it did not really start."""
    from fdia_graph.engine import solvers

    monkeypatch.setattr(pe, "SolverFactory", lambda *a, **k: _FakeSolver(True, True, (0, 0, 0, 0)))
    with pytest.raises(SolverUnavailable):
        solvers.ipopt()


def test_the_dll_folder_is_added_as_a_whole_path_entry(monkeypatch, tmp_path):
    """A PATH entry that merely contains the folder's name as text does not count as the folder."""
    import os
    import sys

    from fdia_graph.engine import solvers

    lib = tmp_path / "Library" / "bin"
    lib.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setenv("PATH", str(lib) + "-extra")
    monkeypatch.setattr(pe, "SolverFactory", lambda *a, **k: _FakeSolver(True, True))
    solvers.ipopt()
    assert os.environ["PATH"].split(os.pathsep)[0] == str(lib)
