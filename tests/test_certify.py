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


@pytest.mark.parametrize(
    "cuts",
    [
        (),
        pytest.param(("bounds",), marks=SLOW),
        pytest.param(("bounds", "qc"), marks=SLOW),
        pytest.param(("bounds", "cycle"), marks=SLOW),
    ],
    ids=["socp", "bounds", "qc", "cycle"],
)
def test_the_search_attack_is_a_point_of_the_relaxation(window, cuts, monkeypatch):
    """Validity, per cut family: the search's attack (its rank-one W, r = |dV|, its devices' binaries
    at 1, its support, each goal line in the sector it lies in, its polar quantities for the QC
    variables) satisfies every constraint of the relaxation, after bound tightening at the search's
    own count, so the relaxation's optimum cannot exceed that count."""
    monkeypatch.setenv("KMP_DUPLICATE_LIB_OK", "TRUE")
    pytest.importorskip("cvxpy")
    from fdia_graph.engine.attacks import relax_cuts
    from fdia_graph.engine.attacks.certify import _Relaxation
    from fdia_graph.engine.attacks.minimize import _union, _Window
    from fdia_graph.formulas.attacks import tampered_devices
    from fdia_graph.formulas.relax import sector_cuts

    g, k, states, goal, res = window
    seeds, _, _ = g._goal_seeds(goal)
    w = _Window(g, states, goal, k)
    relax = _Relaxation(g, w, np.asarray(g.local_region(seeds, k.hops)))
    relax.cuts = cuts
    if "bounds" in cuts:
        rho_before = relax.rho[0].copy()
        assert relax_cuts.tighten(relax, 0, res.devices, time_limit=0.5)  # any stop gives a valid bound
        assert np.all(relax.rho[0] <= rho_before)
    prob, X, b = relax.build()
    points, prev = [], w.prev
    union = (np.zeros(w.node_m.shape, bool), np.zeros(w.edge_m.shape, bool), None)
    for t, Xt in enumerate(states):
        Xa, _ = g.goal_state(goal, t, Xt, res.support, k)
        V = relax._voltages(Xa)
        points.append(relax.layout.point(V, relax.V[t]))
        relax_cuts.lift(relax, t, V)
        node, edge, current, prev = w._snapshot(t, res.support, prev)
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
    from fdia_graph.models.frames import CertifyOptions

    g, k, states, goal, res = window
    c = certify(g, states, goal, k, options=CertifyOptions(time_limit=120.0))
    assert c is not None and c.upper == res.devices
    assert 1 <= c.lower <= c.upper
    assert c.certified == (c.lower == c.upper)
