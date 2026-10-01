"""[WU26]'s row-reduction attack (p. 655 and Sec. IV-D1 steps 1-5, after [YAN17]) as the overload
attack's support method (`OverloadSettings.support_method="rref"`): the column-exchange reduction, the
supports it pins on the paper's IEEE-14 scenarios, and that the attack it builds is feasible.

On the IEEE-14 scenarios (20 pool frames, ratings 1.1 times each target line's peak flow) the reduction
names buses 3 and 11 for lines 3-4 and 6-11 at every snapshot, the sparsest attacks that move each line;
that pair cannot reach both ratings under the operating limits, so the support grows by the next sparsest
rows (our fallback) to the whole area, where the AC solve tampers 7 devices against the search's 5."""

import numpy as np
import pytest

from fdia_graph.engine.attacks.rref import RrefSupport

pytest.importorskip("pandapower")

from test_trusted_pmus import ORDER, SLOTS, _setup  # noqa: E402

from fdia_graph.engine.attacks.minimize import _Window  # noqa: E402
from fdia_graph.formulas.trust import sparsest_rows  # noqa: E402
from fdia_graph.models import TrustSchedule  # noqa: E402
from fdia_graph.models.config import OverloadSettings  # noqa: E402
from fdia_graph.models.grid import NODE  # noqa: E402
from fdia_graph.models.validation import ConfigError  # noqa: E402

_TOL = 1e-9


def _nnz(a: np.ndarray) -> int:
    return int((np.abs(a) > _TOL).sum())


# ---- the reduction ----------------------------------------------------------------------------------
def test_the_exchanges_find_a_sparse_vector_the_first_reduction_hides():
    """A 2-dimensional row space holding a 2-sparse vector: the plain reduction's rows have 3 or more
    nonzeros, and the column exchanges bring the 2-sparse vector out, in the space's own coordinates."""
    sparse, dense = np.array([0.0, 0.0, 0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    M = np.stack([sparse + dense, dense])
    rows = sparsest_rows(M)
    assert len(rows) == 2 and _nnz(rows[0]) == 2
    assert np.linalg.matrix_rank(np.vstack([M, rows])) == 2  # the rows span the same space
    assert np.allclose(rows[0] / rows[0][3], sparse)


def test_eligible_rows_come_first_and_steer_the_exchanges():
    """With `eligible` the chase ignores the rows it refuses: here only rows with a nonzero first entry
    count, so the sparsest such row leads even though a sparser refused row exists."""
    sparse, dense = np.array([0.0, 0.0, 0.0, 1.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    rows = sparsest_rows(np.stack([sparse + dense, dense]), eligible=lambda r: abs(r[0]) > _TOL)
    assert abs(rows[0][0]) > _TOL and _nnz(rows[1]) == 2


def test_the_support_method_is_a_validated_choice():
    assert OverloadSettings(support_method="rref").support_method == "rref"
    with pytest.raises(ConfigError):
        OverloadSettings(support_method="greedy")


# ---- the IEEE-14 case studies ------------------------------------------------------------------------
def _reaches_the_goal(g, window, k, goal, result, trust=None) -> None:
    """Every snapshot has its false state on the support (the goal met inside the limits and the D16
    bounds, `goal_state`), the last one at both ratings, and a trusted PMU's bus is true from its slot."""
    w = _Window(g, window, goal, k, trust=trust)
    for t in range(len(window)):
        S = w.support_at(t, result.support)
        Xa = g.goal_state(goal, t, window[t], S, k)
        assert Xa is not None, f"no false state at snapshot {t}"
        for b in w.pinned[t]:
            assert np.array_equal(Xa[b, [NODE.v, NODE.theta]], window[t][b, [NODE.v, NODE.theta]])
    flows = g.clean_flows_from_states(Xa[None])[0, list(goal.lines)]
    ratings = g.line_ratings()[list(goal.lines)]
    assert np.allclose(np.hypot(flows[:, 0], flows[:, 1]), ratings, rtol=1e-6)


def test_the_reduction_names_buses_3_and_11_for_lines_3_4_and_6_11():
    """The sparsest attacks that move lines 3-4 and 6-11 act at buses 3 and 11 (MATPOWER numbers), at
    every snapshot: the paper's reduction, before our fallback grows the support."""
    g, window, k, goal = _setup(0, 1.1)
    number = g.base.bus["name"].astype(int).to_numpy()
    seeds, _, _ = g._goal_seeds(goal)
    area = np.asarray(g.local_region(seeds, k.hops))
    w = _Window(g, window, goal, k)
    for t in (0, 7, 19):
        assert sorted(int(number[b]) for b in RrefSupport(g).ladder(w, t, area)[0]) == [3, 11]


@pytest.mark.parametrize(
    "scenario, margin, trusted, counts",
    [(0, 1.1, False, (7, 14)), (0, 1.1, True, (6, 11)), (1, 1.1, False, (8, 24)), (1, 1.1, True, (9, 28))],
)
def test_the_rref_attack_is_feasible_and_pinned(scenario, margin, trusted, counts):
    """The row-reduction attack reaches both ratings at every snapshot inside the limits, with and
    without the paper's trust schedule, and its device and channel counts are pinned."""
    g, window, k, goal = _setup(scenario, margin)
    trust = TrustSchedule([int(b) for b in g.wu26_buses(ORDER[scenario])], SLOTS) if trusted else None
    k = k._replace(support_method="rref")
    r = g.min_tamper(window, goal, k, trust=trust)
    assert (r.devices, r.channels) == counts
    _reaches_the_goal(g, window, k, goal, r, trust)


@pytest.mark.parametrize("scenario", [0, 1])
def test_the_schedule_stops_the_rref_attack_at_k_1_2(scenario):
    """At k = 1.2 no support the reduction ranks reaches both ratings with the four PMUs trusted, as for
    the search."""
    g, window, k, goal = _setup(scenario, 1.2)
    trust = TrustSchedule([int(b) for b in g.wu26_buses(ORDER[scenario])], SLOTS)
    r = g.min_tamper(window, goal, k._replace(support_method="rref"), trust=trust)
    assert r.devices == -1


def test_a_line_no_free_bus_moves_leaves_the_snapshot_nothing_to_rank(monkeypatch):
    """When no attack on the free buses moves a target line, the reduction has no eligible row: the
    snapshot's ladder is empty and the search reports the window infeasible instead of failing."""
    import fdia_graph.engine.attacks.rref as rref

    assert sparsest_rows(np.eye(3), eligible=lambda r: False).shape == (0, 3)
    monkeypatch.setattr(rref, "_moved_lines", lambda G, c: set())
    g, window, k, goal = _setup(0, 1.1)
    seeds, _, _ = g._goal_seeds(goal)
    area = np.asarray(g.local_region(seeds, k.hops))
    assert RrefSupport(g).ladder(_Window(g, window, goal, k), 0, area) == []
    assert g.min_tamper(window, goal, k._replace(support_method="rref")).devices == -1
