"""The fewest-tamper search's speed-ups change nothing it finds: exhaustive searches on fixed IEEE-14
windows return the answers the search returned before them (pinned), and each shortcut equals the
computation it replaced (the cached boundary, the reused solve blocks, the attack vector over the
support's branches only)."""

import numpy as np
import pytest

pytest.importorskip("pandapower")

from fdia_graph.engine.attacks.episodes import draw_ramp  # noqa: E402
from fdia_graph.engine.core import FdiaGenerator  # noqa: E402
from fdia_graph.formulas.network import (  # noqa: E402
    _flow_blocks,
    _row_block,
    bus_injections,
    complex_voltages,
    local_flow_solve,
    subnetwork,
)
from fdia_graph.generation import NOISE_FLOOR, _load_states  # noqa: E402
from fdia_graph.models.config import OverloadSettings  # noqa: E402
from fdia_graph.models.frames import FrameKnobs, LoadGoal  # noqa: E402
from fdia_graph.models.grid import NODE  # noqa: E402

WINDOW = 8
MARGIN = 1.2  # S_max = MARGIN x each branch's peak true flow over the window, as tests/test_wu_scenarios.py

# (meter model, load_cap, kind, onset, target lines or ramp draw) ->
# (support, devices, channels, proven, evaluated, unsolved), every search exhaustive (evaluated below
# the budget), recorded by the search before the speed-ups. With the load cap (D16) several goals have
# no false state at all (devices -1, every candidate unsolved); without it the same goals are solved.
PINNED = {
    ("v083", 0.5, "Am", 0, (0,)): ([1, 2, 3, 4, 5, 6, 8], 8, 33, False, 18, 14),
    ("v083", 0.5, "Am", 150, (0,)): ([1, 2, 3, 4, 5, 6, 8], -1, -1, False, 18, 18),
    ("v083", 0.5, "Am", 300, (0,)): ([1, 2, 3, 4, 5, 6], 6, 19, False, 18, 12),
    ("v083", 0.5, "At", 20, 0): ([3, 4, 6, 7, 9, 10, 12, 13], 1, 1, True, 128, 0),
    ("v083", 0.5, "At", 200, 1): ([4, 6, 7, 8, 10, 11, 13], 1, 1, True, 112, 0),
    ("hybrid", 0.5, "Am", 0, (0,)): ([1, 2, 3, 4, 5, 6, 8], 8, 43, False, 18, 14),
    ("hybrid", 0.5, "Am", 150, (0,)): ([1, 2, 3, 4, 5, 6, 8], -1, -1, False, 18, 18),
    ("hybrid", 0.5, "Am", 300, (0,)): ([1, 2, 3, 4, 5, 6], 7, 25, False, 18, 12),
    ("hybrid", 0.5, "At", 20, 0): ([3, 4, 6, 7, 9, 10, 11, 12, 13], 1, 1, True, 128, 0),
    ("hybrid", 0.5, "At", 200, 1): ([2, 3, 4, 6, 7, 8, 9, 10, 11, 13], 1, 1, True, 112, 0),
    ("hybrid", None, "Am", 0, (0,)): ([1, 4], 7, 23, False, 18, 4),
    ("hybrid", None, "Am", 150, (1,)): ([4], 6, 14, True, 99, 0),
    ("hybrid", None, "Am", 300, (0,)): ([1, 4], 7, 25, False, 18, 2),
    ("hybrid", None, "Am", 0, (1, 4)): ([1, 4], 7, 24, False, 99, 45),
    ("hybrid", None, "Am", 150, (4, 5)): ([1, 2, 4], 7, 30, False, 707, 267),
}


@pytest.fixture(scope="module")
def pool():
    return _load_states(14, None)[:400]


@pytest.fixture(scope="module", params=["v083", "hybrid"])
def model(request):
    return request.param


@pytest.fixture(scope="module")
def gen(model):
    return FdiaGenerator(14, seed=123, meter_model=model)


def _knobs(g, pool, load_cap=0.5, budget=100000):
    return FrameKnobs(
        0.2, NOISE_FLOOR, 6, None, False, True, 2, g.operating_limits(pool), True, budget, 1.0, load_cap
    )


def _goal(g, pool, kind, t, which):
    """The window at onset t and its goal; an overload goal's ratings are MARGIN times the window's
    peak flows."""
    window = [pool[u] for u in range(t, t + WINDOW)]
    if kind == "Am":
        g.use_line_ratings(OverloadSettings(rating_margin=MARGIN), np.stack(window))
        return window, g.overload_goal(window, *which)
    design = draw_ramp(np.random.default_rng(100 + which), g.stealthy_pos, WINDOW)
    return window, LoadGoal(tuple(g.ramp_step(design, i, 0.002) for i in range(WINDOW)))


def test_exhaustive_searches_return_their_pinned_answers(model, gen, pool):
    """The optimum (support, devices, channels), whether it is proven, and how many candidates were
    solved and failed: all as before the speed-ups, the search running to completion. The overload
    cases cover the D16 bounds (generator limits on the support's edge, the load cap) and two-line
    goals (D17)."""
    cases = {key: want for key, want in PINNED.items() if key[0] == model}
    for (_, cap, kind, t, which), want in cases.items():
        k = _knobs(gen, pool, cap)
        window, goal = _goal(gen, pool, kind, t, which)
        if kind == "Am":
            assert set(which) <= set(gen.eligible_lines(window, 2).tolist())
        r = gen.min_tamper(window, goal, k)
        got = (r.support.tolist(), r.devices, r.channels, bool(r.proven), r.evaluated, r.unsolved)
        assert got == want, (cap, kind, t, which)
        assert r.evaluated < k.min_budget  # exhaustive: the answer is the search's full optimum


def test_the_cached_boundary_is_the_subnetwork_boundary(gen):
    rng = np.random.default_rng(0)
    live = gen._live_edges()
    for _ in range(50):
        interior = rng.choice(gen.C, int(rng.integers(1, 6)), replace=False)
        want = subnetwork(live, interior, 0, gen.C)
        got = gen._boundary(interior)
        assert all(np.array_equal(a, b) for a, b in zip(got, want))


def test_the_row_block_product_equals_the_full_one(gen, pool):
    """The branch rows the search's attack vector reads, cut to their columns, give the full
    product's values (to rounding: the sum runs over fewer terms)."""
    lut = gen._ppc_row[np.arange(gen.C)]
    V = np.zeros(gen._n_ppc_buses, complex)
    V[lut] = complex_voltages(pool[0][:, NODE.v], pool[0][:, NODE.theta])
    for rows in (np.array([0]), np.array([2, 3, 8]), np.arange(gen.E)):
        for Y in (gen._Yf, gen._Yt, gen._dense_admittances()[1]):
            block, cols = _row_block(Y, rows)
            full = np.asarray(Y @ V)[rows]
            assert np.allclose(block @ V[cols], full, rtol=1e-13, atol=1e-13)


def test_the_reused_blocks_solve_as_the_fresh_ones(gen, pool):
    """`local_flow_solve` with the blocks a caller cut once, for a larger set of holdable rows (the
    support and its edge), returns bit for bit the voltages it returns cutting them itself, with an
    edge bus held as well as interior ones."""
    Xt = pool[0]
    lut = gen._ppc_row[np.arange(gen.C)]
    V = np.zeros(gen._n_ppc_buses, complex)
    V[lut] = complex_voltages(Xt[:, NODE.v], Xt[:, NODE.theta])
    Yb, Yf = gen._dense_admittances()
    line, S = 4, np.array([2, 3, 4, 8])
    touched = gen.touched_buses(S)
    edge = np.setdiff1d(touched, np.r_[S, gen.slack_bus])  # the slack's injection is the balance
    interior, fixed = lut[S], lut[np.r_[S[:2], edge[:1]]]
    S0 = bus_injections(V, Yb)
    f = int(gen._from_bus_ppc[line])
    flow0 = abs(V[f] * np.conj(Yf[line] @ V))
    fresh = local_flow_solve(Yb, Yf[line], f, V, interior, fixed, S0[fixed], 1.1 * flow0)
    blocks = _flow_blocks(Yb, Yf[line], interior, lut[touched])
    reused = local_flow_solve(Yb, Yf[line], f, V, interior, fixed, S0[fixed], 1.1 * flow0, blocks=blocks)
    assert fresh is not None and reused is not None and np.array_equal(fresh, reused)
    assert np.allclose(bus_injections(reused, Yb)[fixed], S0[fixed], atol=1e-8)  # the held buses hold


def test_the_support_attack_vector_equals_the_full_one(gen, pool):
    """The search's attack vector over the support's branches equals `_attack_vector` and
    `_current_attack` over every branch (float32 flows: to their rounding)."""
    from fdia_graph.engine.attacks.minimize import _Window

    k = _knobs(gen, pool, None)
    window, goal = _goal(gen, pool, "Am", 150, (1,))
    win = _Window(gen, window, goal, k)
    S = np.array([4, 5])
    Xa, _ = gen.goal_state(goal, 0, window[0], S, k)
    assert Xa is not None
    near = win.near(S)
    a_edge, a_cur = win._branch_attack(0, Xa, near)
    _, want_edge = gen._attack_vector(Xa, window[0])
    assert np.allclose(a_edge, want_edge, rtol=1e-5, atol=1e-4)
    far = np.setdiff1d(np.arange(gen.E), near.lines)
    assert not a_edge[far].any()  # a branch off the support reads the same on both sides
    want_cur = gen._current_attack(Xa, window[0])
    assert (a_cur is None) == (want_cur is None)
    if a_cur is not None and want_cur is not None:
        assert np.allclose(a_cur, want_cur, rtol=1e-10, atol=1e-12)


def test_search_holds_blas_to_one_thread_and_restores_it(monkeypatch):
    """`min_tamper` runs its search with every BLAS pool on one thread and gives the caller's
    setting back after, returning the search's answer unchanged."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    from fdia_graph.engine.attacks.minimize import MinimizeMixin

    def blas() -> list[int]:
        return [p["num_threads"] for p in threadpoolctl.threadpool_info() if p["user_api"] == "blas"]

    if not blas():
        pytest.skip("no BLAS pool loaded")
    seen: list[list[int]] = []

    def search(self, states, goal, k, prev):
        seen.append(blas())
        return "answer"

    monkeypatch.setattr(MinimizeMixin, "_min_tamper", search)
    with threadpoolctl.threadpool_limits(limits=2, user_api="blas"):
        outside = blas()
        g = object.__new__(MinimizeMixin)  # no grid needed: the search itself is replaced
        assert g.min_tamper([], None, None) == "answer"  # type: ignore[arg-type]
        assert seen == [[1] * len(outside)]
        assert blas() == outside


def test_overlapping_searches_share_one_blas_limit():
    """Two searches that overlap (A enters, B enters, A leaves first) both run on one thread, and the
    caller's setting comes back only when the last one leaves, not when the first does."""
    threadpoolctl = pytest.importorskip("threadpoolctl")
    from fdia_graph.engine.attacks.minimize import _one_blas_thread

    def blas() -> list[int]:
        return [p["num_threads"] for p in threadpoolctl.threadpool_info() if p["user_api"] == "blas"]

    if not blas():
        pytest.skip("no BLAS pool loaded")
    with threadpoolctl.threadpool_limits(limits=2, user_api="blas"):
        outside = blas()
        a, b = _one_blas_thread(), _one_blas_thread()
        a.__enter__()
        b.__enter__()
        a.__exit__(None, None, None)
        assert blas() == [1] * len(outside)  # B is still searching
        b.__exit__(None, None, None)
        assert blas() == outside
