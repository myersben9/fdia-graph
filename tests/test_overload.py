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
from fdia_graph.generation import _load_states  # noqa: E402
from fdia_graph.models.frames import FrameKnobs  # noqa: E402
from fdia_graph.models.grid import NODE  # noqa: E402
from fdia_graph.timeline import DEFAULT_FAMILIES, generate_timeline  # noqa: E402


@pytest.fixture(scope="module")
def g():
    return FdiaGenerator(14, seed=123)


@pytest.fixture(scope="module")
def pool():
    return _load_states(14, None)[:400]


def _knobs(g, pool, scale=1.0):
    return FrameKnobs(2, g.operating_limits(pool), True, 256, scale)


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
    from fdia_graph.formulas.network import _flow_blocks, _flow_jacobian, _held

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
    # one goal branch; the two fixed buses hold P and Q
    blocks = _flow_blocks(Yb, Yf[[line]], interior, interior)
    held = _held(blocks, interior, (fixed, None), np.zeros(0, int), np.array([f]))
    J = _flow_jacobian(blocks, held, np.array([f]), V, interior, (fixed, Yb @ V))
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


def test_an_out_of_service_branch_is_never_a_target(g, pool, monkeypatch):
    """An outage keeps the branch's row with a zero admittance: its flow reads zero, so it would look
    rated, metered and below its rating, but no false state can drive it to its rating."""
    window = [pool[u] for u in range(10)]
    b = int(g.eligible_lines(window, 2)[0])
    status = np.ones(g.ei.shape[1]) if g.branch.status is None else np.array(g.branch.status, float)
    status[b] = 0.0
    monkeypatch.setattr(g, "branch", g.branch._replace(status=status))
    assert b not in set(g.eligible_lines(window, 2).tolist())


def test_the_search_on_a_flow_goal_equals_brute_force(g, pool):
    # [WU26 eq. 12]
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
    """New generation's Am on IEEE-14 (no stealth bound, as in [WU26], [D11]): each episode's
    noiseless reported flow on its target branch reaches the branch's rating at the last frame, on a
    support that moves at least one device beyond noise."""
    out = generate_timeline(
        14,
        states=pool[:200],
        seed=3,
        families=("Am",),
        am_len=30,
        attacked_frac=0.3,
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
    """The window of a flow goal (Am) takes [WU26]'s noise [D8]; a load goal (At) keeps the meters'
    rated accuracy [D7]."""
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
    # [WU26 eqs. 24-25] [D9] [D5]
    """D9: S_{l,t} = S_true_{l,t} + (t - kappa)/T (S_max - S_true_{l,kappa+T}) for t = kappa+1 ...
    kappa+T [WU26 eq. 25]: the window's first snapshot already carries 1/T of the way to the rating
    (kappa, the reference, is the frame before), the share grows linearly, the natural drift is left
    in, and the last target is the rating."""
    window = [pool[u] for u in range(12)]
    line = int(g.eligible_lines(window, 2)[0])
    goal = g.overload_goal(window, line)
    flows = g.clean_flows_from_states(np.stack(window))[:, line]
    true = np.hypot(flows[:, 0], flows[:, 1])
    added = np.array(goal.targets) - true
    rating = float(g.line_ratings()[line])
    step = (rating - true[-1]) / 12
    assert added[0] == pytest.approx(step) and step > 0  # every snapshot of the window moves the flow
    assert np.allclose(np.diff(added), step)
    assert goal.targets[-1] == pytest.approx(rating)


def test_new_generation_makes_the_multi_snapshot_families(tmp_path, pool):
    """The generator makes At and Am; an older family is refused before any frame is walked."""
    assert DEFAULT_FAMILIES == ("At", "Am")
    generate_timeline(14, states=pool[:40], seed=1, families=("At",), ramp_len=10, out=str(tmp_path / "a.h5"))
    for old in ("Aq", "Ad", "As", "Ar", "Al"):
        with pytest.raises(ValueError, match=f"{old} are single-snapshot"):
            generate_timeline(14, states=pool[:40], seed=1, families=(old,), out=str(tmp_path / f"{old}.h5"))


def test_the_pglib_ratings_are_refused_on_a_case_without_them(tmp_path):
    """D15: rating_source="pglib" keeps the previous behaviour, IEEE-14, 118 and 300 only."""
    from fdia_graph.models.config import OverloadSettings

    with pytest.raises(NoLineRatings, match="IEEE-30 has no line ratings"):
        generate_timeline(
            30,
            families=("Am",),
            attacked_frac=0.5,
            am_attack=OverloadSettings(rating_source="pglib"),
            out=str(tmp_path / "x.h5"),
        )


def test_pool_ratings_are_the_margin_times_the_pool_peak_flow(g, pool):
    """D15: each branch's rating is rating_margin times its peak true apparent flow over the pool, on
    every branch, metered or not; "pglib" restores PGLib-OPF's rate_a."""
    from fdia_graph.formulas.attacks import branch_ratings
    from fdia_graph.models.config import OverloadSettings
    from fdia_graph.ratings import pglib_branches

    fresh = FdiaGenerator(14, seed=123)
    fresh.use_line_ratings(OverloadSettings(rating_margin=1.3), pool)
    peak = np.abs(fresh.all_flows_from_states(pool)).max(axis=0)
    assert np.allclose(fresh.line_ratings(), 1.3 * peak) and (peak > 0).all()
    fresh.use_line_ratings(OverloadSettings(rating_source="pglib"), pool)
    number = fresh.base.bus["name"].astype(int).to_numpy()
    want = branch_ratings([(int(number[a]), int(number[b])) for a, b in fresh.ei.T], pglib_branches(14))
    assert np.array_equal(fresh.line_ratings(), want, equal_nan=True)


def test_the_rating_margin_must_exceed_one():
    from fdia_graph.models.config import OverloadSettings

    for bad in (1.0, 0.9, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            OverloadSettings(rating_margin=bad)
    with pytest.raises(ValueError):
        OverloadSettings(rating_source="nameplate")
    for bad in (0.0, -0.1, 1.5, float("nan")):  # the load cap tau is in (0, 1]
        with pytest.raises(ValueError):
            OverloadSettings(load_cap=bad)
    assert OverloadSettings(load_cap=1.0).load_cap == 1.0
    assert OverloadSettings.of("overload")[1] == OverloadSettings()
    assert OverloadSettings.of("redistribution") == ("redistribution", None)  # TimelineKnobs refuses it
    assert OverloadSettings.of({"rating_margin": 1.5}) == ("overload", OverloadSettings(rating_margin=1.5))


def test_a_system_without_pglib_ratings_makes_am_on_pool_ratings(tmp_path):
    """D15: the pool ratings work on every system of the ladder; IEEE-30 has no PGLib-OPF ratings."""
    out = generate_timeline(
        30,
        states=_load_states(30, None)[:120],
        seed=2,
        families=("Am",),
        am_len=10,
        ramp_len=10,
        attacked_frac=0.3,
        out=str(tmp_path / "am30.h5"),
    )
    with h5py.File(out, "r") as f:
        assert f.attrs[schema.Attr.RATING_SOURCE] == "pool" and f.attrs[schema.Attr.RATING_MARGIN] == 1.25
        assert f.attrs[schema.Attr.LOAD_CAP] == 0.5
        eg = f[schema.Group.EPISODES]
        assert schema.EPISODE_AM_LINE in eg, "no overload episode was built"
        assert (eg[schema.EPISODE_MIN_DEVICES][()] >= 1).all()


def test_a_voltmeter_angle_is_not_a_channel_the_attack_is_charged_for(g, pool):
    """A SCADA voltmeter reads |V| only; the angle channel exists only at a PMU bus, so the search's
    masks carry an angle at PMU buses alone."""
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.models.grid import NODE

    window = [pool[u] for u in range(3)]
    line = int(g.eligible_lines(window, 2)[0])
    w = _Window(g, window, g.overload_goal(window, line), FrameKnobs())
    pmu = np.zeros(g.C, bool)
    pmu[sorted(g.meters.pmu)] = True
    assert not w.node_m[~pmu, NODE.theta].any()


def test_the_deprecated_stream_makes_new_generation(monkeypatch):
    """generate_stream is deprecated and passes new generation's families and defaults through."""

    from fdia_graph import streams, timeline
    from fdia_graph.models.choices import GENERATED_FAMILIES

    seen = {}

    def fake(system, **kw):
        seen.update(kw)
        raise RuntimeError("stop before any work")

    monkeypatch.setattr(timeline, "generate_timeline", fake)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(RuntimeError, match="stop before any work"):
            streams.generate_stream(30)
    assert tuple(seen["families"]) == tuple(GENERATED_FAMILIES)
    assert "am_attack" not in seen and "min_tamper" not in seen  # generate_timeline's own defaults


def test_the_overload_am_is_not_checked_against_a_target_count(monkeypatch):
    """The overload Am picks its lines per window, so the admissibility check before the walk asks
    only for At's targets."""
    from fdia_graph import timeline
    from fdia_graph.engine.core import FdiaGenerator

    seen = {}

    def stop(*a, **k):
        seen["reached"] = True
        raise RuntimeError("stop after the admissibility check")

    monkeypatch.setattr(FdiaGenerator, "target_counts", lambda self: {})  # no At target at all
    monkeypatch.setattr(FdiaGenerator, "operating_limits", stop)
    with pytest.raises(RuntimeError, match="after the admissibility check"):
        timeline.generate_timeline(14, families=("Am",), am_attack="overload")
    assert seen["reached"]


def test_am_has_no_stealth_bound_and_at_keeps_its_own(g, pool):
    # [D11] [D7] [D8]
    """D11: [WU26]'s model bounds nothing between snapshots; its noise only thresholds the l0 count.
    The overload window's cost does not depend on the stealth scale, At's window is bounded."""
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.models.frames import AttackDesign, LoadGoal

    window = [pool[u] for u in range(8)]
    goal = g.overload_goal(window, int(g.eligible_lines(window, 2)[0]))
    area = np.asarray(g.local_region(np.unique(g.ei[:, goal.line]), 2))
    tight, loose = (_Window(g, window, goal, _knobs(g, pool, s)) for s in (1e-9, 1e9))
    assert not tight.stealth_bound and tight.cost(area, None) == loose.cost(area, None)
    design = AttackDesign(g.stealthy_pos[:1], 1.01)
    assert _Window(g, window, LoadGoal(tuple([design] * 8)), _knobs(g, pool)).stealth_bound


def test_a_generator_only_support_is_valid_and_its_output_stays_in_limits(g, pool):
    """D14: a generator's injection is free within its limits [WU26 eqs. 13-14, 22-23]; a support
    whose only free injection is a generator meets a small flow goal, and the implied generator output
    passes the limit check."""
    from fdia_graph.formulas.attacks import generator_output, within_limits

    Xt = pool[0]
    gens = g.generator_buses()
    assert set(gens.tolist()) <= set(g.free_injection_buses().tolist()) and g.slack_bus not in gens
    k = _knobs(g, pool)
    for line in range(g.E):
        S = np.intersect1d(g.ei[:, line], gens)
        if len(S) != 1 or len(np.intersect1d(g.ei[:, line], g.zero_inj)):
            continue
        flow = g.clean_flows_from_states(Xt[None])[0, line]
        Xa, converged, dload = g.solve_flow_local(Xt, S, line, 1.03 * float(np.hypot(*flow)))
        if Xa is None:
            continue
        assert converged and not dload.any()  # a generator's change is its output, not a load
        reached = g.clean_flows_from_states(Xa[None])[0, line]
        assert np.hypot(*reached) == pytest.approx(1.03 * float(np.hypot(*flow)), rel=1e-6)
        gen = generator_output(Xt, g.load_base, g.gen_base)
        assert within_limits(Xa, Xt, gen, dload, k.limits, S)
        return
    pytest.fail("no generator-only support met a 3% flow goal")


def test_a_goal_drives_two_lines_at_once(g, pool):
    """D14: one held support, each line's noiseless flow reaching its own target."""
    window = [pool[u] for u in range(6)]
    lines = [int(b) for b in g.eligible_lines(window, 2)[:2]]
    goal = g.overload_goal(window, *lines)
    assert goal.lines == tuple(lines) and len(goal.targets_at(0)) == 2
    assert goal.targets_at(5) == pytest.approx(tuple(float(g.line_ratings()[b]) for b in lines))
    Xt = pool[0]
    area = np.asarray(g.local_region(np.unique(g.ei[:, lines]), 2))
    flows = g.clean_flows_from_states(Xt[None])[0]
    want = [1.02 * float(np.hypot(*flows[b])) for b in lines]
    Xa, converged, _ = g.solve_flow_local(Xt, area, lines, want)
    assert converged and Xa is not None
    got = g.clean_flows_from_states(Xa[None])[0]
    assert np.allclose([np.hypot(*got[b]) for b in lines], want, rtol=1e-6)


def _generator_only_case(g, pool):
    """A branch with a generator at one end (not the slack, no zero-injection end), that generator as
    the only support bus, and a 3% flow goal on the branch."""
    Xt = pool[0]
    gens = g.generator_buses()
    for line in range(g.E):
        S = np.intersect1d(g.ei[:, line], gens)
        if len(S) == 1 and not len(np.intersect1d(g.ei[:, line], g.zero_inj)):
            flow = g.clean_flows_from_states(Xt[None])[0, line]
            return Xt, S, line, 1.03 * float(np.hypot(*flow))
    pytest.fail("IEEE-14 has no generator-ended branch")


def test_a_generator_at_its_p_limit_keeps_its_q_free(g, pool):
    # [WU26 eqs. 22-23] [D14]
    """Review fix: pinning is per component. With the generator's P range shut to its true output, the
    solve pins P alone and still meets the goal by moving Q; pinning both would leave no free injection."""
    from fdia_graph.formulas.attacks import generator_output

    Xt, S, line, target = _generator_only_case(g, pool)
    b = int(S[0])
    gen = generator_output(Xt, g.load_base, g.gen_base)
    limits = _knobs(g, pool).limits
    # every other generator (the edge's too, D16) unbounded, so only this one's P limit binds
    p_lo, p_hi = np.full(g.C, -np.inf), np.full(g.C, np.inf)
    p_lo[b] = p_hi[b] = gen[b, 0]
    q_lo, q_hi = np.full(g.C, -np.inf), np.full(g.C, np.inf)
    tight = limits._replace(p_lo=p_lo, p_hi=p_hi, q_lo=q_lo, q_hi=q_hi)
    Xa, converged, _ = g.solve_flow_local(Xt, S, line, target, tight)
    assert converged and Xa is not None
    reached = g.clean_flows_from_states(Xa[None])[0, line]
    assert np.hypot(*reached) == pytest.approx(target, rel=1e-6)
    out = generator_output(Xt, g.load_base, g.gen_base)[b] - (
        Xa[b, NODE.p_inj : NODE.q_inj + 1] - Xt[b, NODE.p_inj : NODE.q_inj + 1]
    )
    assert out[0] == pytest.approx(gen[b, 0], abs=1e-6)  # P held at its (shut) limit
    assert abs(out[1] - gen[b, 1]) > 1e-3  # Q moved to meet the goal


def test_a_generator_label_is_relative_to_its_output(g, pool):
    """Review fix: the frame magnitude at a generator bus is its apparent change against the true
    generator output, not against the bus's net injection."""
    from fdia_graph.formulas.attacks import generator_output

    Xt, S, line, target = _generator_only_case(g, pool)
    Xa, _, _ = g.solve_flow_local(Xt, S, line, target)
    b = int(S[0])
    gen = generator_output(Xt, g.load_base, g.gen_base)[b]
    d = Xa[b] - Xt[b]
    want = np.hypot(d[NODE.p_inj], d[NODE.q_inj]) / np.hypot(gen[0], gen[1])
    assert g.pretended_change(Xt, Xa, S)[0] == pytest.approx(want)


def _edge_cases(g, pool):
    """Supports of one free load bus at a branch end whose edge holds a non-slack generator, each with
    a 3% flow goal on that branch, the edge generator, and the free state without bounds."""
    Xt = pool[0]
    gens, loads = set(g.generator_buses().tolist()), set(g.free_load_buses().tolist())
    for line in range(g.E):
        for end in g.ei[:, line]:
            S = np.array([int(end)])
            if int(end) not in loads:
                continue
            edge_gens = sorted(set(g.touched_buses(S).tolist()) & gens)
            flow = g.clean_flows_from_states(Xt[None])[0, line]
            target = 1.03 * float(np.hypot(*flow))
            free, _, _ = g.solve_flow_local(Xt, S, line, target)
            if edge_gens and free is not None:
                yield Xt, S, line, target, edge_gens, free


def test_an_edge_generator_stays_inside_its_limits(g, pool):
    # [D16]
    """D16: (22)-(23) bound every generator whose reported output the attack changes, a generator on
    the support's edge included: shut its P range and the solve holds its P there or refuses."""
    from fdia_graph.formulas.attacks import generator_output

    base = _knobs(g, pool).limits
    gen = None
    checked = 0
    for Xt, S, line, target, edge_gens, free in _edge_cases(g, pool):
        gen = generator_output(Xt, g.load_base, g.gen_base)
        b = edge_gens[0]
        if abs(free[b, NODE.p_inj] - Xt[b, NODE.p_inj]) < 1e-3:
            continue  # the unbounded solve leaves this generator's P alone: nothing to test
        p_lo, p_hi = base.p_lo.copy(), base.p_hi.copy()
        p_lo[b] = p_hi[b] = gen[b, 0]
        Xa, _, _ = g.solve_flow_local(Xt, S, line, target, base._replace(p_lo=p_lo, p_hi=p_hi))
        if Xa is not None:
            out = gen[b, 0] - (Xa[b, NODE.p_inj] - Xt[b, NODE.p_inj])
            assert out == pytest.approx(gen[b, 0], abs=1e-6)
            reached = g.clean_flows_from_states(Xa[None])[0, line]
            assert np.hypot(*reached) == pytest.approx(target, rel=1e-6)
            checked += 1
    assert checked, "no edge generator case solved"


def test_the_load_cap_bounds_every_load_the_attack_moves(g, pool):
    """D16: with load_cap tau, no load bus the attack moves (in the support or on its edge) shows a
    change beyond tau times its true load [YUA11]; the goal is still met."""
    from fdia_graph.formulas.attacks import bus_load

    checked = 0
    for Xt, S, line, target, _, _ in _edge_cases(g, pool):
        Xa, _, _ = g.solve_flow_local(Xt, S, line, target, None, 0.05)
        if Xa is None:
            continue
        load = bus_load(Xt, g.load_base, g.gen_base)
        buses = np.setdiff1d(g.touched_buses(S), np.r_[g.generator_buses(), g.slack_bus])
        buses = buses[g.load_base[buses, 0] != 0]
        change = np.abs(Xa[buses, NODE.p_inj] - Xt[buses, NODE.p_inj])
        assert (change <= 0.05 * np.abs(load[buses]) + 1e-6).all()
        reached = g.clean_flows_from_states(Xa[None])[0, line]
        assert np.hypot(*reached) == pytest.approx(target, rel=1e-6)
        checked += 1
    assert checked, "no capped case solved"


def test_two_lines_by_default_and_the_pairs_share_one_area(g, pool):
    # [D17]
    """D17: an episode overloads n_lines at once (2 by default, 1 allowed); each pair's second line
    lies inside the first one's attack area, so one held support can reach both."""
    from fdia_graph.models.config import OverloadSettings

    assert OverloadSettings().n_lines == 2 and OverloadSettings(n_lines=1).n_lines == 1
    for bad in (0, 3, 1.5):
        with pytest.raises(ValueError):
            OverloadSettings(n_lines=bad)
    window = [pool[u] for u in range(6)]
    order = g.rng.permutation(g.eligible_lines(window, 2))
    pairs = g._target_sets(order, 2, 2)
    assert pairs and all(len(p) == 2 and p[0] != p[1] for p in pairs)
    for a, b in pairs:
        area = {int(x) for x in g.local_region(np.unique(g.ei[:, a]), 2)}
        assert {int(e) for e in g.ei[:, b]} <= area
    assert g._target_sets(order, 1, 2) == [(int(b),) for b in order[:8]]


def test_a_two_line_episode_records_both_lines(tmp_path, pool):
    """D17: the file records n_lines and one am_* row per target line, each with its rating, the
    goal at the window's end (the rating) and the noiseless flow reached."""
    out = generate_timeline(
        14,
        states=pool[:200],
        seed=3,
        families=("Am",),
        am_len=10,
        attacked_frac=0.3,
        out=str(tmp_path / "am2.h5"),
    )
    with h5py.File(out, "r") as f:
        assert f.attrs[schema.Attr.N_LINES] == 2
        eg = f[schema.Group.EPISODES]
        assert schema.EPISODE_AM_LINE in eg, "no overload episode was built"
        ep = eg[schema.EPISODE_AM_EPISODE][()]
        assert (np.unique(ep, return_counts=True)[1] == 2).all()
        rating, target = eg[schema.EPISODE_AM_RATING][()], eg[schema.EPISODE_AM_TARGET][()]
        reached = eg[schema.EPISODE_AM_REACHED][()]
        assert np.allclose(target, rating) and np.allclose(reached, rating, rtol=1e-4)


def test_a_voltage_is_held_at_its_limit_while_the_goal_is_met(g, pool):
    # [WU26 eq. 21]
    """The voltage active set of the flow solve: shut one support bus's voltage range to its true
    magnitude; the unbounded solve moves it, the bounded one holds it at the limit (21) and still
    meets the flow goal."""
    Xt = pool[0]
    k = _knobs(g, pool)
    free_box = k.limits._replace(
        p_lo=np.full(g.C, -np.inf),
        p_hi=np.full(g.C, np.inf),
        q_lo=np.full(g.C, -np.inf),
        q_hi=np.full(g.C, np.inf),
    )
    for line in g.eligible_lines([Xt], 2):
        S = np.asarray(g.local_region(np.unique(g.ei[:, line]), 2))
        flow = g.clean_flows_from_states(Xt[None])[0, line]
        target = 1.03 * float(np.hypot(*flow))
        free, _, _ = g.solve_flow_local(Xt, S, int(line), target)
        if free is None:
            continue
        b = int(S[np.argmax(np.abs(free[S, NODE.v] - Xt[S, NODE.v]))])
        if abs(free[b, NODE.v] - Xt[b, NODE.v]) < 1e-4:
            continue  # the goal barely moves any voltage here: nothing to hold
        v_lo, v_hi = np.zeros(g.C), np.full(g.C, 10.0)
        v_lo[b] = v_hi[b] = Xt[b, NODE.v]  # the allowed range at b is its true magnitude alone
        Xa, converged, _ = g.solve_flow_local(
            Xt, S, int(line), target, free_box._replace(v_lo=v_lo, v_hi=v_hi)
        )
        if Xa is None:
            continue
        assert converged and Xa[b, NODE.v] == pytest.approx(Xt[b, NODE.v], abs=1e-9)
        reached = g.clean_flows_from_states(Xa[None])[0, line]
        assert np.hypot(*reached) == pytest.approx(target, rel=1e-6)
        return
    pytest.fail("no line's flow solve moved and then held a voltage")
