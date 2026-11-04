"""The checked Ipopt every nyfed OPF runs on: a solver that is missing, or found but unable to start, raises
`SolverUnavailable` with the fix, instead of reading as an infeasible model downstream."""

from __future__ import annotations

import pytest

from fdia_graph.errors import SolverUnavailable

pe = pytest.importorskip("pyomo.environ")


class _FakeSolver:
    def __init__(self, found: bool, starts: bool):
        self.found, self.starts, self.options = found, starts, {}

    def available(self, exception_flag: bool = False) -> bool:
        return self.found

    def version(self):
        if not self.starts:
            raise RuntimeError("the program did not start")
        return (3, 14, 19, 0)


@pytest.mark.parametrize(
    ("found", "starts"), [(False, False), (True, False)], ids=["missing", "does-not-start"]
)
def test_unusable_ipopt_raises_with_the_fix(monkeypatch, found, starts):
    from fdia_graph.nyfed import solvers

    monkeypatch.setattr(pe, "SolverFactory", lambda *a, **k: _FakeSolver(found, starts))
    with pytest.raises(SolverUnavailable, match="conda install -c conda-forge ipopt"):
        solvers.ipopt(tol=1e-8)


def test_working_ipopt_takes_the_options(monkeypatch):
    from fdia_graph.nyfed import solvers

    monkeypatch.setattr(pe, "SolverFactory", lambda *a, **k: _FakeSolver(True, True))
    assert solvers.ipopt(tol=1e-8, max_iter=300).options == {"tol": 1e-8, "max_iter": 300}


def test_solver_unavailable_is_not_a_data_condition():
    """A broken environment is not a property of the data: callers catching data conditions must not swallow it."""
    from fdia_graph.errors import DataConditionError

    assert not issubclass(SolverUnavailable, (DataConditionError, ValueError))
