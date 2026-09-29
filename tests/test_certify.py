"""The relaxation that certifies the fewest-tamper search (docs/plans/RELAX_CERTIFIER_PLAN.md): its
pieces, its channel maps against the search's own attack vector, and its bound never above the
search's count. The solver tests need the [certify] extra; the cut families' validity cases and the
full solve take minutes, so they run with FDIA_SLOW=1."""

import os

import numpy as np
import pytest

from fdia_graph.formulas.relax import big_m, cone_gap, sector_cuts, voltage_box
from fdia_graph.models.grid import NODE

WINDOW = 2  # snapshots: enough for a shared device binary, still fast
RAMP_RATE = 0.002  # the At ramp's rate per frame, as generation's default
SLOW = pytest.mark.skipif(not os.environ.get("FDIA_SLOW"), reason="set FDIA_SLOW=1: minutes of SCIP solves")


def test_the_voltage_box_widens_to_the_true_value():
    lo, hi = voltage_box(np.array([0.93, 1.0]), np.array([0.94, 0.94]), np.array([1.06, 1.06]), 1e-3)
    assert np.allclose(np.sqrt(lo), [0.929, 0.939]) and np.allclose(np.sqrt(hi), [1.061, 1.061])


def test_big_m_bounds_every_point_of_the_box():
    rng = np.random.default_rng(0)
    A, r, sig = rng.normal(size=(5, 4)), rng.uniform(0.1, 1, 4), np.full(5, 0.05)
    M = big_m(A, r, sig)
    for _ in range(200):
        dx = rng.uniform(-1, 1, 4) * r
        assert np.all(np.abs(A @ dx) <= sig + M + 1e-12)


def test_the_sectors_cover_the_exterior_of_the_circle():
    U, cos_k = sector_cuts()
    for phi in np.linspace(0, 2 * np.pi, 97):
        s = 3.0 * np.array([np.cos(phi), np.sin(phi)])
        assert np.max(U @ s) >= 3.0 * cos_k - 1e-12


def test_a_rank_one_point_has_no_cone_gap():
    V = np.array([1.02 * np.exp(0.1j), 0.97 * np.exp(-0.2j)])
    W = V[0] * np.conj(V[1])
    assert cone_gap(np.abs(V[:1]) ** 2, np.abs(V[1:]) ** 2, np.array([W.real]), np.array([W.imag])) < 1e-12
    assert cone_gap(np.array([1.0]), np.array([1.0]), np.array([0.5]), np.array([0.0])) == pytest.approx(0.75)


@pytest.fixture(scope="module")
def window():
    """A generated overload episode on IEEE-14's hybrid meters with new generation's defaults (pool
    ratings, the D16 bounds, two lines) and the search's attack on it."""
    pytest.importorskip("pandapower")
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.generation import NOISE_FLOOR, _load_states
    from fdia_graph.models.config import OverloadSettings
    from fdia_graph.models.frames import FrameKnobs

    g = FdiaGenerator(14, seed=1, meter_model="hybrid")
    X = _load_states(14, None)
    g.use_line_ratings(OverloadSettings(), X)
    k = FrameKnobs(0.2, NOISE_FLOOR, 6, None, False, True, 2, g.operating_limits(X), True, 256, 1.0, 0.5, 2)
    for t in range(0, len(X) - WINDOW, 37):
        design = g.am_overload_design(X, t, WINDOW, k)
        if design is not None and design.tamper is not None and design.tamper.devices > 0:
            assert len(design.goal.lines) == 2  # the two-line goal is what the relaxation must hold
            return g, k, list(X[t : t + WINDOW]), design.goal, design.tamper
    pytest.skip("no overload window with an attack")


def test_the_channel_maps_reproduce_the_search_attack_vector(window):
    """The relaxation's linear maps, evaluated at the search's false state, give the attack vector
    a = h(x_false) - h(x_true) the search counts, channel by channel."""
    from fdia_graph.engine.attacks.certify import _Relaxation
    from fdia_graph.engine.attacks.minimize import _Window

    g, k, states, goal, res = window
    seeds, _, _ = g._goal_seeds(goal)
    w = _Window(g, states, goal, k)
    relax = _Relaxation(g, w, np.asarray(g.local_region(seeds, k.hops)))
    for t, Xt in enumerate(states):
        Xa, _ = g.goal_state(goal, t, Xt, res.support, k)
        a_node, a_edge = g._attack_vector(Xa, Xt)
        a_cur = g._current_attack(Xa, Xt)
        ref = [a_node[b, c] for b in range(g.C) for c in (NODE.p_inj, NODE.q_inj) if w.node_m[b, c] > 0]
        ref += [a_edge[e, c] for e in range(g.E) for c in range(2) if w.edge_m[e, c] > 0]
        if w.i_m is not None:
            ref += [a_cur[e, c] for e in range(g.E) for c in range(4) if w.i_m[e, c] > 0]
        got = relax.lin[t].A @ (relax.layout.point(relax._voltages(Xa)) - relax.x0[t])
        assert np.allclose(got, ref, atol=1e-6)


@pytest.fixture(scope="module")
def ramp_window():
    """An At window that starts mid-ramp: frames t+1 and t+2 of an accepted ramp on IEEE-14's hybrid
    meters, their first step measured from `prev`, the attack vector of frame t on the ramp's
    support (a non-zero onset), and the search's attack on the window from that `prev`."""
    pytest.importorskip("pandapower")
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.generation import NOISE_FLOOR, _load_states
    from fdia_graph.models.frames import FrameKnobs, LoadGoal

    g = FdiaGenerator(14, seed=1, meter_model="hybrid")
    X = _load_states(14, None)
    k = FrameKnobs(0.2, NOISE_FLOOR, 6, None, False, True, 2, g.operating_limits(X), True, 256, 1.0, 0.5, 2)
    for t in range(0, len(X) - 3, 53):
        design = g.ramp_design(X, t, (3, RAMP_RATE), k)
        if design is None or design.tamper is None or design.tamper.devices < 1:
            continue
        first = _Window(g, [X[t]], LoadGoal((g.ramp_step(design, 0, RAMP_RATE),)), k)
        onset = first._snapshot(0, design.tamper.support, first.prev)
        if onset is None:
            continue
        prev = onset[3]
        states = [X[t + 1], X[t + 2]]
        goal = LoadGoal(tuple(g.ramp_step(design, i, RAMP_RATE) for i in (1, 2)))
        res = g.min_tamper(states, goal, k, prev)
        if res is not None and res.devices > 0:
            assert np.abs(prev.node).max() > 0  # the onset is measured from a real attack vector
            return g, k, states, goal, res, prev
    pytest.skip("no ramp window with an attack")


def _assert_search_attack_in_relaxation(case, cuts, prev=None):
    """Put the search's attack of `case` (its rank-one W, r = |dV|, its devices' binaries at 1, its
    support, each goal line in the sector it lies in, its polar quantities for the QC variables) into
    the relaxation built with `cuts` from the window's `prev`, after bound tightening at the search's
    own count, and check that no constraint is violated."""
    from fdia_graph.engine.attacks import relax_cuts
    from fdia_graph.engine.attacks.certify import _Relaxation
    from fdia_graph.engine.attacks.minimize import _union, _Window
    from fdia_graph.formulas.attacks import tampered_devices
    from fdia_graph.formulas.relax import sector_cuts

    g, k, states, goal, res = case
    seeds, _, _ = g._goal_seeds(goal)
    w = _Window(g, states, goal, k, prev=prev)
    relax = _Relaxation(g, w, np.asarray(g.local_region(seeds, k.hops)))
    relax.cuts = cuts
    if "bounds" in cuts:
        rho_before = relax.rho[0].copy()
        assert relax_cuts.tighten(relax, 0, res.devices, time_limit=0.5)  # any stop gives a valid bound
        assert np.all(relax.rho[0] <= rho_before)
    prob, X, b = relax.build()
    points, before = [], w.prev
    union = (np.zeros(w.node_m.shape, bool), np.zeros(w.edge_m.shape, bool), None)
    for t, Xt in enumerate(states):
        Xa, _ = g.goal_state(goal, t, Xt, res.support, k)
        V = relax._voltages(Xa)
        points.append(relax.layout.point(V, relax.V[t]))
        relax_cuts.lift(relax, t, V)
        node, edge, current, before = w._snapshot(t, res.support, before)
        union = _union(union, (node, edge, current))
    devices = tampered_devices(union[0], union[1], w.pmu, g.ei[0], union[2], g.ei[1])
    X.value = np.array(points)
    relax.in_support.value = np.isin(relax.held, res.support).astype(float)
    b.value = np.isin(relax.devices, sorted(int(d) for d in devices)).astype(float)
    U, _ = sector_cuts()
    for z, s in relax.sectors:
        z.value = (np.arange(len(U)) == np.argmax(U @ s.value)).astype(float)
    assert b.value.sum() == res.devices
    assert relax.mismatch(np.array(points), list(range(len(states)))) < 1e-6  # a rank-one point is AC
    assert max(float(np.max(c.violation())) for c in prob.constraints) < 1e-5


CUT_CASES = pytest.mark.parametrize(
    "cuts",
    [
        (),
        pytest.param(("bounds",), marks=SLOW),
        pytest.param(("bounds", "qc"), marks=SLOW),
        pytest.param(("bounds", "cycle"), marks=SLOW),
    ],
    ids=["socp", "bounds", "qc", "cycle"],
)


@CUT_CASES
def test_the_search_attack_is_a_point_of_the_relaxation(window, cuts, monkeypatch):
    """Validity, per cut family, on a two-line overload window under the D16 bounds: the search's
    attack satisfies every constraint of the relaxation, so the relaxation's optimum cannot exceed
    its count."""
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    pytest.importorskip("cvxpy")
    _assert_search_attack_in_relaxation(window, cuts)


@CUT_CASES
def test_the_search_ramp_attack_is_a_point_of_the_relaxation(ramp_window, cuts, monkeypatch):
    """Validity, per cut family, on an At window that starts mid-ramp: the search's attack satisfies
    every constraint of the relaxation, the stealth bound's onset step from `prev` included."""
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    pytest.importorskip("cvxpy")
    *case, prev = ramp_window
    _assert_search_attack_in_relaxation(tuple(case), cuts, prev)


def test_the_qc_envelopes_hold_on_their_interval():
    from fdia_graph.formulas.relax import cos_envelope, mccormick, sin_envelope

    for m in (0.01, 0.3, 1.2):
        c, cos_m = cos_envelope(m)
        slope, offset = sin_envelope(m)
        phi = np.linspace(-m, m, 201)
        assert np.all(np.cos(phi) <= 1 - c * phi**2 + 1e-12) and np.all(np.cos(phi) >= cos_m - 1e-12)
        assert np.all(np.sin(phi) <= slope * (phi - m / 2) + offset + 1e-12)
        assert np.all(np.sin(phi) >= slope * (phi + m / 2) - offset - 1e-12)
    a, b = np.meshgrid(np.linspace(0.9, 1.1, 11), np.linspace(-0.3, 0.5, 11))
    for sense, ca, cb, c0 in mccormick((0.9, 1.1), (-0.3, 0.5)):
        rhs = ca * a + cb * b + c0
        assert np.all(a * b >= rhs - 1e-12) if sense == ">=" else np.all(a * b <= rhs + 1e-12)


@SLOW
def test_the_bound_never_exceeds_the_search(window, monkeypatch):
    # torch (imported by other test modules) and SCIP each load an OpenMP runtime; on Windows the
    # second one aborts the process at SCIP's first solve unless the duplicate is allowed
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    pytest.importorskip("cvxpy")
    pytest.importorskip("pyscipopt")
    from fdia_graph.engine.attacks.certify import certify
    from fdia_graph.models.config import CertifyOptions

    g, k, states, goal, res = window
    c = certify(g, states, goal, k, options=CertifyOptions(time_limit=120.0))
    assert c is not None and c.upper == res.devices
    assert 1 <= c.lower <= c.upper
    assert c.verdict in ("certified", "gap", "uncertain")
    assert c.certified == (c.verdict == "certified")
    assert not c.certified or c.lower == c.upper
    assert not c.certified or c.cone_lower <= c.lower  # a contradiction between levels never certifies


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_limit": 0.0},
        {"time_limit": -5.0},
        {"tighten_limit": 0.0},
        {"time_limit": float("inf")},
        {"snapshots": ()},
        {"snapshots": (-1,)},
        {"snapshots": (0.5,)},
        {"snapshots": "0"},
        {"snapshots": (0, 2), "window": 2},
        {"cuts": ("bounds", "sdp")},
        {"cuts": "qc"},
        {"window": 0},
    ],
)
def test_the_certify_options_refuse_bad_values_on_construction(kwargs):
    from fdia_graph.models.config import CertifyOptions
    from fdia_graph.models.validation import ConfigError

    with pytest.raises(ConfigError):
        CertifyOptions(**kwargs)


def test_the_certify_options_store_canonical_values():
    from fdia_graph.models.config import CertifyOptions

    opts = CertifyOptions(snapshots=[1, 0, 1], cuts=["qc", "bounds", "qc"], window=2)
    assert opts.snapshots == (0, 1) and opts.cuts == ("qc", "bounds")
    assert CertifyOptions().tighten_limit == 5.0  # the plan's section 2.1


def _claim(lower, status="optimal", doubt=""):
    from fdia_graph.models.frames import BoundClaim

    return BoundClaim(lower, status, None, np.zeros(3), doubt)


def test_a_cut_bound_below_the_cone_bound_is_uncertain_not_certified():
    """Cuts only shrink the relaxation, so a cut bound below the cone bound (IEEE-14 At episode 1:
    10 against 9) is a numerical contradiction: the smaller bound is kept and nothing is certified."""
    from fdia_graph.engine.attacks.certify import _verdict

    lower, verdict, reason = _verdict(12, _claim(10), _claim(9), 3)
    assert (lower, verdict) == (9, "uncertain") and "below the cone relaxation's 10" in reason
    # even when the smaller bound would meet the search's count
    assert _verdict(9, _claim(10), _claim(9, "infeasible"), 0)[1] == "uncertain"


def test_the_verdict_certifies_only_clear_claims():
    from fdia_graph.engine.attacks.certify import _verdict

    assert _verdict(9, _claim(7), _claim(9, "infeasible"), 0) == (9, "certified", "")
    assert _verdict(9, _claim(9), None, 0) == (9, "certified", "")  # the cone bound alone meets it
    assert _verdict(9, _claim(6), _claim(7), 0) == (7, "gap", "")
    assert _verdict(9, _claim(6), None, 8) == (8, "gap", "")  # the search's forced devices are a floor
    doubted = _verdict(9, _claim(7), _claim(9, "optimal", "not confirmed"), 0)
    assert doubted == (7, "uncertain", "not confirmed")
    # IEEE-14 At episode 3: the cut level's infeasibility is not confirmed and the loosened problem,
    # a larger set, bounds 6; that is doubt, not a contradiction with the cone's 9 at default tolerance
    loosened = _verdict(9, _claim(9), _claim(6, "optimal", "not confirmed"), 0)
    assert loosened == (6, "uncertain", "not confirmed")


def test_a_bound_is_rounded_down_with_a_margin():
    from fdia_graph.engine.attacks.certify import _rounded

    assert _rounded(9.005, 0.01, 12) == 9  # just above an integer by the tolerances: not rounded up
    assert _rounded(9.02, 0.01, 12) == 10
    assert _rounded(12.5, 0.01, 12) == 12  # never above the search's count
    assert _rounded(float("-inf"), 0.01, 12) == 0


class _FakeRelaxation:
    """Stands in for `_Relaxation` in the claim logic: infeasible at SCIP's default tolerances and,
    per `loose`, infeasible or feasible once numerics/feastol is loosened."""

    devices = np.arange(3)
    cuts = ()

    def __init__(self, loose):
        self.loose, self.calls = loose, []

    def solve(self, time_limit, snapshots=None, cutoff=None, feastol=None):
        self.calls.append(feastol)
        if feastol is None or self.loose == "infeasible":
            return "infeasible", float("inf"), None, np.zeros(3)
        return "optimal", 7.3, np.zeros((1, 4)), np.array([1.0, 0.0, 1.0])


def test_an_infeasibility_is_accepted_only_after_the_loosened_re_solve():
    from fdia_graph.engine.attacks.certify import solve_claim
    from fdia_graph.models.config import CertifyOptions

    opts = CertifyOptions()
    held = _FakeRelaxation("infeasible")
    claim = solve_claim(held, [0], (8, 9), opts)
    assert held.calls == [None, opts.robust_feastol]  # the default solve, then the loosened re-solve
    assert (claim.lower, claim.doubt) == (9, "")
    gone = _FakeRelaxation("feasible")
    claim = solve_claim(gone, [0], (8, 9), opts)
    assert claim.lower == 8 and "feastol 0.0001" in claim.doubt  # the loosened bound, and the doubt
    tightened = _FakeRelaxation("infeasible")
    solve_claim(tightened, [0], (8, 9), opts, feasible=False)  # bound tightening found it infeasible
    assert tightened.calls == [opts.robust_feastol]


def test_the_robust_re_solve_runs_on_scip(window, monkeypatch):
    """The loosened re-solve on SCIP itself: at a cutoff of zero devices the relaxation (which asks
    for at least one) is infeasible at any tolerance, and the claim is the search's count."""
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    pytest.importorskip("cvxpy")
    pytest.importorskip("pyscipopt")
    from fdia_graph.engine.attacks.certify import _Relaxation, solve_claim
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.models.config import CertifyOptions

    g, k, states, goal, res = window
    seeds, _, _ = g._goal_seeds(goal)
    relax = _Relaxation(g, _Window(g, states, goal, k), np.asarray(g.local_region(seeds, k.hops)))
    calls = []
    solve = relax.solve
    monkeypatch.setattr(relax, "solve", lambda *a, **kw: calls.append(kw.get("feastol")) or solve(*a, **kw))
    opts = CertifyOptions(time_limit=60.0)
    claim = solve_claim(relax, [0], (0, res.devices), opts)
    assert calls == [None, opts.robust_feastol]
    assert (claim.lower, claim.status, claim.doubt) == (res.devices, "infeasible", "")
