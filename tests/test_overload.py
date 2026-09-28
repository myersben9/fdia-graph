"""`Am` as the overload attack of [WU26]: the PGLib-OPF line ratings, the flow-goal false state, the
eligible target branches, the fewest-tamper search on a flow goal, and new generation's families."""

import warnings

import h5py
import numpy as np
import pytest

pytest.importorskip("pandapower")

from fdia_graph import schema  # noqa: E402
from fdia_graph.engine.core import FdiaGenerator  # noqa: E402
from fdia_graph.errors import NoLineRatings  # noqa: E402
from fdia_graph.formulas.attacks import branch_ratings  # noqa: E402
from fdia_graph.formulas.network import bus_injections, complex_voltages, local_flow_solve  # noqa: E402
from fdia_graph.generation import NOISE_FLOOR, _load_states  # noqa: E402
from fdia_graph.models.frames import FrameKnobs  # noqa: E402
from fdia_graph.models.grid import NODE  # noqa: E402
from fdia_graph.timeline import DEFAULT_FAMILIES, LEGACY_FAMILIES, generate_timeline  # noqa: E402

WIDE = 256.0  # a stealth scale at which IEEE-14's overload windows are feasible (1, the plan's D7, is not)


@pytest.fixture(scope="module")
def g():
    return FdiaGenerator(14, seed=123)


@pytest.fixture(scope="module")
def pool():
    return _load_states(14, None)[:400]


def _knobs(g, pool, scale=WIDE):
    return FrameKnobs(0.2, NOISE_FLOOR, 6, None, False, True, 2, g.operating_limits(pool), True, 256, scale)


def test_ratings_match_by_end_buses_in_either_order():
    rows = [(1, 2, 100.0), (2, 1, 80.0), (3, 4, 9900.0), (5, 6, 50.0)]
    got = branch_ratings([(2, 1), (4, 3), (6, 5), (7, 8)], rows)
    assert got[0] == 80.0  # parallel rows: the binding (smallest) rating
    assert np.isnan(got[1])  # MATPOWER's 9,900 MVA placeholder is no rating
    assert got[2] == 50.0
    assert np.isnan(got[3])  # no row


def test_every_ieee14_branch_has_a_pglib_rating_and_other_cases_refuse(g):
    rating = g.line_ratings()
    assert len(rating) == g.E and np.isfinite(rating).all()
    assert 50 < np.nanmin(rating) and np.nanmax(rating) < 700  # PGLib-OPF v23.07: 53 to 664 MVA
    with pytest.raises(NoLineRatings):
        FdiaGenerator(30, seed=1).line_ratings()


def test_the_flow_solve_reaches_its_target_and_holds_the_fixed_buses(g, pool):
    Xt = pool[0]
    lut = g._ppc_row[np.arange(g.C)]
    V = np.zeros(g._n_ppc_buses, complex)
    V[lut] = complex_voltages(Xt[:, NODE.v], Xt[:, NODE.theta])
    Yb, Yf = g._dense_admittances()
    line = 4
    interior = lut[np.array([2, 3, 4, 8])]
    fixed = interior[:2]  # the other two are free
    S0 = bus_injections(V, Yb)
    f = int(g._from_bus_ppc[line])
    flow0 = abs(V[f] * np.conj(Yf[line] @ V))
    Vf = local_flow_solve(Yb, Yf[line], f, V, interior, fixed, S0[fixed], 1.1 * flow0)
    assert Vf is not None
    assert abs(abs(Vf[f] * np.conj(Yf[line] @ Vf)) - 1.1 * flow0) < 1e-8
    assert np.max(np.abs(bus_injections(Vf, Yb)[fixed] - S0[fixed])) < 1e-8
    outside = np.setdiff1d(np.arange(len(V)), interior)
    assert np.array_equal(Vf[outside], V[outside])  # every voltage outside the interior held true


def test_the_flow_solve_takes_the_least_norm_step(g, pool):
    """For a small target change the solution's voltage change matches the minimum-norm solution of
    the linearized equations, the smallest change that meets them to first order."""
    from fdia_graph.formulas.network import _flow_jacobian

    Xt = pool[0]
    lut = g._ppc_row[np.arange(g.C)]
    V = np.zeros(g._n_ppc_buses, complex)
    V[lut] = complex_voltages(Xt[:, NODE.v], Xt[:, NODE.theta])
    Yb, Yf = g._dense_admittances()
    line, interior = 4, lut[np.array([2, 3, 4, 8])]
    fixed = interior[:2]
    S0 = bus_injections(V, Yb)
    f = int(g._from_bus_ppc[line])
    flow0 = abs(V[f] * np.conj(Yf[line] @ V))
    eps = 1e-5
    Vf = local_flow_solve(Yb, Yf[line], f, V, interior, fixed, S0[fixed], flow0 + eps)
    k = len(interior)
    got = np.concatenate(
        [np.angle(Vf[interior]) - np.angle(V[interior]), np.abs(Vf[interior]) - np.abs(V[interior])]
    )
    J = _flow_jacobian(Yb, Yf[line], f, V, interior, np.arange(2))
    r = np.zeros(J.shape[0])
    r[-1] = -eps
    want = np.linalg.lstsq(J, -r, rcond=None)[0]
    assert len(got) == 2 * k and np.allclose(got, want, rtol=1e-3, atol=1e-9)


def test_eligible_lines_are_rated_metered_and_below_their_rating(g, pool):
    window = [pool[u] for u in range(10)]
    lines = set(g.eligible_lines(window, 2).tolist())
    metered = np.asarray(g.meters.flow, bool)
    assert lines and all(metered[b] for b in lines)
    saved = g.line_ratings().copy()
    try:
        flows = g.clean_flows_from_states(np.stack(window))
        S = np.hypot(flows[..., 0], flows[..., 1])
        b = next(iter(lines))
        g._line_ratings = saved.copy()
        g._line_ratings[b] = float(S[:, b].max())  # at its rating at one snapshot: not a target
        assert b not in set(g.eligible_lines(window, 2).tolist())
        g._line_ratings[b] = np.nan  # unrated: not a target
        assert b not in set(g.eligible_lines(window, 2).tolist())
    finally:
        g._line_ratings = saved


def test_the_search_on_a_flow_goal_equals_brute_force(g, pool):
    """Every candidate support solved: the fewest-tamper result is the cheapest among the converged."""
    from fdia_graph.engine.attacks.minimize import _Window

    k = _knobs(g, pool)
    window = [pool[u] for u in range(30)]
    line = int(g.eligible_lines(window, 2)[0])
    goal = g.overload_goal(window, line)
    result = g.min_tamper(window, goal, k)
    seeds, starts, must_hold = g._goal_seeds(goal)
    area = g.local_region(seeds, k.hops)
    inside = {int(b) for b in area}
    starts = [s for s in starts if s <= inside]
    win = _Window(g, window, goal, k)
    costs = [c for S in [*g._supports(starts, area, must_hold)] if (c := win.cost(S, None)) is not None]
    if result is None or result.devices < 0:
        assert not costs
    else:
        assert (result.devices, result.channels, len(result.support)) == min(costs)


def test_am_reaches_the_rating_on_the_held_support(tmp_path, pool):
    """New generation's Am on IEEE-14 (the bound widened so a window is feasible): each episode's
    noiseless reported flow on its target branch reaches the branch's rating at the last frame, on a
    support that moves at least one device beyond noise."""
    out = generate_timeline(
        14,
        states=pool[:200],
        seed=3,
        families=("Am",),
        am_len=30,
        attacked_frac=0.3,
        stealth_scale=WIDE,
        out=str(tmp_path / "am.h5"),
    )
    with h5py.File(out, "r") as f:
        eg = f[schema.Group.EPISODES]
        assert schema.EPISODE_AM_LINE in eg, "no overload episode was built"
        rating, reached = eg[schema.EPISODE_AM_RATING][()], eg[schema.EPISODE_AM_REACHED][()]
        full = (
            eg[schema.EPISODE_LENGTH][()][eg[schema.EPISODE_AM_EPISODE][()]] == 30
        )  # not clipped at the end
        assert full.any() and np.allclose(reached[full], rating[full], rtol=1e-4)
        assert (eg[schema.EPISODE_MIN_DEVICES][()] >= 1).all()
        assert f.attrs[schema.Attr.AM_ATTACK] == "overload"
        assert f.attrs[schema.Attr.STEALTH_SCALE] == WIDE


def test_the_paper_noise_is_in_the_stored_units(g):
    """D8: [WU26]'s case-study noise, 0.03 pu on SCADA channels and 0.01 pu on PMU channels, in the
    units a scan stores (MW and MVAr on the case base, |V| in pu, the angle in degrees)."""
    from fdia_graph.formulas.noise import WU26_NOISE, paper_sigma

    pmu = np.zeros(g.C, bool)
    pmu[[0, 3]] = True
    node, edge = paper_sigma((g.C, 4), (g.E, 2), pmu, g._base_mva)
    assert WU26_NOISE == {"scada": 0.03, "pmu": 0.01}
    assert np.all(node[:, NODE.p_inj] == 0.03 * g._base_mva) and np.all(edge == 0.03 * g._base_mva)
    assert node[0, NODE.v] == 0.01 and node[1, NODE.v] == 0.03  # a PMU bus against a SCADA voltmeter
    assert np.allclose(node[:, NODE.theta], np.degrees(0.01))


def test_am_counts_against_the_paper_noise_and_at_against_the_rated_accuracy(g, pool):
    """The window of a flow goal (Am) takes [WU26]'s noise (D8); a load goal (At) keeps the meters'
    rated accuracy (D7)."""
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.engine.base import POWER_NOISE_FLOOR_MW
    from fdia_graph.formulas.noise import accuracy_sigma
    from fdia_graph.models.frames import AttackDesign, LoadGoal

    k = _knobs(g, pool, 1.0)
    window = [pool[u] for u in range(5)]
    flow = _Window(g, window, g.overload_goal(window, int(g.eligible_lines(window, 2)[0])), k)
    assert np.all(flow.sigma[0][1] == 0.03 * g._base_mva)
    design = AttackDesign(g.stealthy_pos[:1], 1.01)
    load = _Window(g, window, LoadGoal(tuple([design] * 5)), k)
    flows = g.clean_flows_from_states(np.stack(window))
    want = accuracy_sigma(window[0], flows[0], g.SD, POWER_NOISE_FLOOR_MW)
    assert np.allclose(load.sigma[0][0], want[0]) and np.allclose(load.sigma[0][1], want[1])


def test_the_goal_rides_on_the_true_flow_and_ends_at_the_rating(g, pool):
    """D9: S_{l,t} = S_true_{l,t} + (t - kappa)/T (S_max - S_true_{l,kappa+T}): the attack's own
    share grows linearly from zero, the natural drift is left in, and the last target is the rating."""
    window = [pool[u] for u in range(12)]
    line = int(g.eligible_lines(window, 2)[0])
    goal = g.overload_goal(window, line)
    flows = g.clean_flows_from_states(np.stack(window))[:, line]
    true = np.hypot(flows[:, 0], flows[:, 1])
    added = np.array(goal.targets) - true
    rating = float(g.line_ratings()[line])
    assert added[0] == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(np.diff(added), (rating - true[-1]) / 11)
    assert goal.targets[-1] == pytest.approx(rating)


def test_new_generation_makes_the_multi_snapshot_families_and_the_old_ones_warn(tmp_path, pool):
    assert DEFAULT_FAMILIES == ("At", "Am")
    assert LEGACY_FAMILIES == ("Aq", "Ad", "As", "Ar", "At", "Al", "Am")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        generate_timeline(
            14, states=pool[:40], seed=1, families=("At",), ramp_len=10, out=str(tmp_path / "a.h5")
        )
    assert not any("refused from 0.22" in str(w.message) for w in seen)
    for old in ("Aq", "Ad", "As", "Ar", "Al"):
        with pytest.warns(DeprecationWarning, match=f"generating {old} is deprecated and refused from 0.22"):
            generate_timeline(
                14,
                states=pool[:40],
                seed=1,
                families=(old,),
                min_tamper=False,
                out=str(tmp_path / f"{old}.h5"),
            )


def test_the_overload_attack_is_refused_on_a_case_without_ratings(tmp_path):
    with pytest.raises(NoLineRatings, match="IEEE-30 has no line ratings"):
        generate_timeline(30, families=("Am",), attacked_frac=0.5, out=str(tmp_path / "x.h5"))
