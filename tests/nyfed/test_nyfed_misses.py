"""Every design that finds no attack says why (`MISSES`), so a timeline with few attacks can be read: nothing to
attack, nothing feasible, or a solver that failed (no data needed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")  # the timeline builder writes HDF5


def _family(miss: str | None, solver_errors: int, found: bool):
    from fdia_graph.nyfed.attacks.episode import AttackFamily, Episode

    class Toy(AttackFamily):
        def design(self, window, rng):
            self.solver_errors += solver_errors
            if found:
                return Episode("Am", np.array([0]), window[0], 1, np.array([0]), -1, np.zeros((1, 1)))
            if miss:
                self.last_miss = miss
            return None

    return Toy()


@pytest.mark.parametrize(
    ("miss", "errors", "found", "want"),
    [
        (None, 0, True, None),
        ("no_target", 0, False, "no_target"),
        (None, 0, False, "infeasible"),
        (None, 2, False, "solver_error"),
    ],
    ids=["found", "no-target", "infeasible", "solver-error"],
)
def test_design_says_why_it_found_no_attack(miss, errors, found, want):
    from fdia_graph.nyfed.attacks.episode import MISSES
    from fdia_graph.nyfed.config import Settings
    from fdia_graph.nyfed.timeline import builder

    builder._init({"Am": _family(miss, errors, found)}, Settings())
    _, ep, _, why = builder._design((0, "Am", 5, np.ones((1, 2)), np.zeros((1, 2))))
    assert (ep is not None) is found
    assert why == want and (why is None or why in MISSES)
