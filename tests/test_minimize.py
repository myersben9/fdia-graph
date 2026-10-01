"""The fewest-tamper support search [WU26 eq. 12]: exact against brute force on IEEE-14, a lower
bound that never exceeds a cost, the two sigmas (the emitter's per-scan jitter, the meters'
accuracy class the search counts against), and a continuous reweighted-l1 cross-check the search is
never beaten by."""

import numpy as np
import pytest

from fdia_graph.formulas.attacks import tampered_channels, tampered_devices
from fdia_graph.formulas.noise import accuracy_sigma, jitter_sigma
from fdia_graph.models.grid import NODE

WINDOW = 6  # snapshots per test window: a ramp's loads move above their noise by then, still fast


@pytest.fixture(scope="module")
def case():
    pytest.importorskip("pandapower")
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.generation import _load_states
    from fdia_graph.models.frames import FrameKnobs

    g = FdiaGenerator(14, seed=1)
    X = _load_states(14, None)
    k = FrameKnobs(2, g.operating_limits(X), True, 4096)
    return g, X, k


def _windows(case, n, held=True):
    """n ramp windows accepted on today's region, each with its goal (one design per snapshot) from the
    ramp's first frame; with `held`, only windows where some support held for the window meets every
    constraint (the stealth bound rejects most: a reactive flow near zero has a noise floor of 1e-3)."""
    from fdia_graph.engine.attacks.episodes import probe_frames
    from fdia_graph.models.frames import LoadGoal

    g, X, k = case
    rng = np.random.default_rng(3)
    out = []
    while len(out) < n:
        t = int(rng.integers(0, len(X) - WINDOW))
        d = g.ramp_design(X, t, (WINDOW, 0.002), k._replace(min_tamper=False))
        if d is None:
            continue
        frames = probe_frames(len(X), t, WINDOW)
        goal = LoadGoal(tuple(g.ramp_step(d, u - t, 0.002) for u in frames))
        states = [X[u] for u in frames]
        if held and g.min_tamper(states, goal, k).devices <= 0:
            continue  # no held support, or nothing moves above noise: nothing for the search to choose
        out.append((states, goal, d))
    return out


def test_a_window_with_no_held_support_says_so(case):
    """Brute force finds no feasible support exactly when the search reports -1."""
    from fdia_graph.engine.attacks.minimize import _Window

    g, _, k = case
    for states, goal, d in _windows(case, 4, held=False):
        result = g.min_tamper(states, goal, k)
        buses = np.unique(g.load_bus[d.targets])
        area = g.local_region(buses, k.hops)
        window = _Window(g, states, goal, k)
        feasible = [
            c
            for S in g._supports([frozenset(int(b) for b in buses)], area)
            if (c := window.cost(S, None)) is not None
        ]
        assert (result.devices < 0) == (not feasible and window.cost(area, None) is None)


def test_the_search_equals_brute_force(case):
    from fdia_graph.engine.attacks.minimize import _Window

    g, _, k = case
    for states, goal, d in _windows(case, 3):
        result = g.min_tamper(states, goal, k)
        buses = np.unique(g.load_bus[d.targets])
        area = g.local_region(buses, k.hops)
        window = _Window(g, states, goal, k)
        feasible = []
        for S in [*g._supports([frozenset(int(b) for b in buses)], area), np.asarray(area)]:
            c = window.cost(S, None)
            if c is not None:
                feasible.append(c)
        # every candidate solved within the budget: the search's answer is the cheapest of them
        assert (result.devices, result.channels, len(result.support)) == min(feasible)


def test_the_emitter_draws_with_the_shared_noise_rule(case):
    g, X, _ = case
    drawn = []

    class _Recorder:  # stands in for the emission's stream: records every std, in draw order
        def normal(self, loc, scale):
            drawn.extend(float(s) for s in np.ravel(scale))
            return np.zeros(np.shape(scale))

    g._jitter_stream = lambda: _Recorder()
    try:
        scan = g.emit_from_state(X[0])
    finally:
        del g._jitter_stream
    from fdia_graph.formulas.network import branch_flows, complex_voltages

    Vc = np.zeros(g._n_ppc_buses, complex)  # the emitter's own float64 flows
    Vc[g._ppc_row[np.arange(g.C)]] = complex_voltages(X[0][:, NODE.v], X[0][:, NODE.theta])
    Sf = branch_flows(Vc, g._Yf, g._from_bus_ppc, g._base_mva)
    flows = np.stack([Sf.real, Sf.imag], axis=1)
    sig, sig_f = jitter_sigma(X[0], flows, g.SDj, 1e-3)
    expected = []
    for b in range(g.C):
        if scan.node_m[b, NODE.v]:
            expected.append(sig[b, NODE.v])
        if scan.node_m[b, NODE.theta]:  # the hybrid meters: an angle at the PMU buses only
            expected.append(sig[b, NODE.theta])
        if scan.node_m[b, NODE.p_inj]:
            expected += [sig[b, NODE.p_inj], sig[b, NODE.q_inj]]
    for e in range(g.E):
        if scan.edge_m[e, 0]:
            expected += [sig_f[e, 0], sig_f[e, 1]]
    # then the PMU branch-current channels, after the flows
    assert drawn[: len(expected)] == pytest.approx(expected, rel=0, abs=0)
    assert len(drawn) == len(expected) + int(scan.i_m.sum())
    # and the rule is the emitter's former one, relative power noise with the floor
    b = int(np.argmax(np.abs(X[0][:, NODE.p_inj])))
    assert sig[b, NODE.p_inj] == abs(X[0][b, NODE.p_inj]) * g.SDj["pi"] + 1e-3


def test_the_search_counts_against_the_calibrations_accuracy_sigma(case):
    """accuracy_sigma is the measured calibration's sigma (SEBase._class_sigma through
    accuracy_class_sigma) at one scan's readings, in physical units; it is larger than the jitter."""
    from fdia_graph.engine.base import ACCURACY_CLASS
    from fdia_graph.formulas.estimation import accuracy_class_sigma

    g, X, _ = case
    x, base = X[0], g._base_mva
    flows = g.clean_flows_from_states(X[:1])[0].astype(np.float64)
    sig, sig_f = accuracy_sigma(x, flows, ACCURACY_CLASS, 1e-3)
    # the calibration's rule, in its units (per unit on baseMVA, radians), on the same readings
    pu_p = accuracy_class_sigma(
        np.abs(x[:, NODE.p_inj]) / base, np.full(g.C, ACCURACY_CLASS["pi"]), np.ones(g.C, bool), 1e-3 / base
    )
    va = accuracy_class_sigma(
        np.abs(np.radians(x[:, NODE.theta])),
        np.full(g.C, ACCURACY_CLASS["va"]),
        np.zeros(g.C, bool),
        1e-3 / base,
    )
    assert np.allclose(sig[:, NODE.p_inj], pu_p * base) and np.allclose(sig[:, NODE.theta], np.degrees(va))
    assert np.allclose(sig[:, NODE.v], ACCURACY_CLASS["v"])
    # the stored angle is in degrees, the class in radians: the sigma is returned in the scan's units
    assert np.allclose(sig[:, NODE.theta], np.degrees(ACCURACY_CLASS["va"]))
    assert np.allclose(sig_f[:, 1], ACCURACY_CLASS["qf"] * np.abs(flows[:, 1]) + 1e-3)
    jit, jit_f = jitter_sigma(x, flows, g.SDj, 1e-3)
    assert (sig >= jit).all() and (sig_f >= jit_f).all()


def test_devices_group_channels_by_bus_terminal_and_pmu():
    node = np.zeros((4, 4), bool)
    edge = np.zeros((3, 2), bool)
    node[0, NODE.p_inj] = True  # SCADA terminal of bus 0
    node[1, NODE.v] = True  # a PMU bus: its PMU
    node[2, NODE.theta] = True  # a voltage meter without PMU: bus 2's SCADA terminal
    edge[1, 0] = True  # flow metered at the from end, bus 3
    pmu = np.array([False, True, False, False])
    devices = tampered_devices(node, edge, pmu, np.array([0, 3, 2]))
    assert devices.tolist() == [0, 2, 3, 4 + 1]
    moved, _ = tampered_channels(
        np.array([[0.0, 2.0, 0.0, 0.0]]),
        np.zeros((1, 2)),
        np.ones((1, 4)),
        np.ones((1, 2)),
        np.ones((1, 4)),
        np.ones((1, 2)),
    )
    assert moved.tolist() == [[False, True, False, False]]  # only the change beyond its noise counts


def _irls_support(g, states, goal, area, iters=20):
    """A support from iteratively reweighted l1 on the linearized local AC equations: the area's
    voltage moves that realize the target buses' load change while penalizing, reweighted, the
    injection change they cause at every other bus (the tamper proxy)."""
    from fdia_graph.formulas.network import ac_jacobian

    Xt = states[0]
    N = g.C
    lut = g._ppc_row[np.arange(N)]
    H = ac_jacobian(Xt[:, NODE.v], np.radians(Xt[:, NODE.theta]), g._Ybus, g._Yf, g._Yt, g._ppc_branch, lut)
    P, Q = H[N * NODE.p_inj : N * (NODE.p_inj + 1)], H[N * NODE.q_inj : N * (NODE.q_inj + 1)]
    cols = np.r_[area, N + area]  # theta and |V| of the area buses
    design = goal.designs[0]
    targets = np.unique(g.load_bus[design.targets])
    dload = np.zeros(N)
    np.add.at(
        dload, g.load_bus[design.targets], g.true_load(Xt)[design.targets] * (np.asarray(design.mult) - 1.0)
    )
    A = np.vstack([P[targets][:, cols], Q[targets][:, cols]])
    rhs = np.r_[dload[targets], np.zeros(len(targets))] / g._base_mva
    others = [b for b in range(N) if b not in set(targets.tolist())]
    w = np.ones(len(others))
    x = np.zeros(len(cols))
    for _ in range(iters):
        G = sum(
            wi * (np.outer(P[b, cols], P[b, cols]) + np.outer(Q[b, cols], Q[b, cols]))
            for wi, b in zip(w, others)
        )
        G = G + 1e-9 * np.eye(len(cols))
        K = np.block([[G, A.T], [A, np.zeros((len(A), len(A)))]])
        x = np.linalg.lstsq(K, np.r_[np.zeros(len(cols)), rhs], rcond=None)[0][: len(cols)]
        change = np.array([np.hypot(P[b, cols] @ x, Q[b, cols] @ x) for b in others])
        w = 1.0 / (change + 1e-6)
    move = np.abs(x[: len(area)]) + np.abs(x[len(area) :])
    S = np.union1d(area[move > 5e-2 * move.max()], targets)
    S, _ = g._grow_over_zero_injection(S)  # the region's rule, as every candidate follows it
    return S


def test_reweighted_l1_never_beats_the_search(case):
    """The continuous relaxation proposes a support; under the same constraints it never tampers fewer
    devices than exhaustive search over the area. The relaxation cannot see the per-snapshot stealth
    bound, so both sides are costed without it."""
    from fdia_graph.engine.attacks.minimize import _Window

    g, _, k = case
    compared = 0
    for states, goal, d in _windows(case, 3, held=False):
        buses = np.unique(g.load_bus[d.targets])
        area = g.local_region(buses, k.hops)
        window = _Window(g, states, goal, k, stealth_bound=False)
        costs = [
            c
            for S in g._supports([frozenset(int(b) for b in buses)], area)
            if (c := window.cost(S, None)) is not None
        ]
        S = _irls_support(g, states, goal, np.asarray(area))
        cost = window.cost(S, None)
        if costs and cost is not None:
            compared += 1
            assert cost >= min(costs)
    assert compared >= 1, "the reweighted-l1 support should be feasible on at least one window"


def test_new_generation_searches_by_default():
    """New generation holds every At episode on its fewest-tamper support [D2]; the frame
    knobs default off, so a walk with min_tamper=False holds the region within `hops`."""
    from fdia_graph.models.config import TimelineKnobs
    from fdia_graph.models.frames import FrameKnobs, RampDesign

    assert FrameKnobs().min_tamper is False
    assert TimelineKnobs().min_tamper is True
    assert RampDesign(np.array([0]), 1.0, 1, 0).support is None


def test_supports_keep_the_region_rules(case):
    """Every candidate excludes the slack and takes in the zero-injection buses of its boundary, the
    rules `local_region` applies to the region."""
    g, _, k = case
    zero = set(int(b) for b in g.zero_inj) - {g.slack_bus}
    edges = g._live_edges()
    for _, _, d in _windows(case, 2):
        buses = np.unique(g.load_bus[d.targets])
        area = g.local_region(buses, k.hops)
        for S in g._supports([frozenset(int(b) for b in buses)], area):
            assert g.slack_bus not in S
            boundary = set(np.unique(edges[:, np.isin(edges, S).any(axis=0)]).tolist()) - set(S.tolist())
            assert not boundary & zero


def test_generate_with_the_knob_records_each_ramp_search(tmp_path):
    pytest.importorskip("pandapower")
    import h5py

    import fdia_graph as fg
    from fdia_graph import schema

    out = tmp_path / "min.h5"
    fg.generate(
        "ieee14", "min_tiny", out=str(out), frames=200, ramp_len=10, families=["At"], seed=1, min_tamper=True
    )
    with h5py.File(out, "r") as f:
        eg = f[schema.Group.EPISODES]
        n = len(eg[schema.EPISODE_MIN_EPISODE])
        assert n >= 1 and n == len(eg[schema.EPISODE_MIN_DEVICES]) == len(eg[schema.EPISODE_MIN_EVALUATED])
        devices = eg[schema.EPISODE_MIN_DEVICES][()]
        assert ((devices >= 1) | (devices == -1)).all()  # an attack tampers a device, or none was found
        assert f.attrs[schema.Attr.MIN_TAMPER] == 1
        assert f.attrs[schema.Attr.STEALTH_SCALE] == 1.0  # an At-only timeline records its bound too


def test_the_stealth_bound_covers_the_onset(case):
    """A window whose first frame jumps (a 15 percent load step from the attack-free frame before it)
    solves without the bound and is refused with it: the increment is measured from a zero attack."""
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.models.frames import AttackDesign, LoadGoal

    g, X, k = case
    for states, _, d in _windows(case, 1, held=False):
        area = np.asarray(g.local_region(np.unique(g.load_bus[d.targets]), k.hops))
        jump = LoadGoal(
            tuple(AttackDesign(d.targets, 1.15) for _ in states)
        )  # held at 1.15: only the onset jumps
        assert _Window(g, states, jump, k, stealth_bound=False).cost(area, None) is not None
        assert _Window(g, states, jump, k).cost(area, None) is None


def test_a_support_within_noise_is_not_an_attack(case):
    """A window whose loads never move a device beyond its accuracy sigma costs nothing, so it is not an
    attack: no support is feasible and the search records -1, never a 0-device result."""
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.models.frames import AttackDesign, LoadGoal

    g, _, k = case
    for states, _, d in _windows(case, 1, held=False):
        area = np.asarray(g.local_region(np.unique(g.load_bus[d.targets]), k.hops))
        still = LoadGoal(tuple(AttackDesign(d.targets, 1.0 + 1e-9) for _ in states))
        assert _Window(g, states, still, k).cost(area, None) is None
        assert g.min_tamper(states, still, k).devices == -1


def test_the_onset_is_bounded_against_the_frame_before(case):
    """Episodes may be adjacent: a window after an attacked frame starts its stealth bound from that
    frame's attack vector. A held 15 percent step is refused after a benign frame (it jumps at onset)
    and accepted right after a frame carrying the same step (no increment at onset)."""
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.models.frames import AttackDesign, LoadGoal

    g, _, k = case
    for states, _, d in _windows(case, 1, held=False):
        area = np.asarray(g.local_region(np.unique(g.load_bus[d.targets]), k.hops))
        held = LoadGoal(tuple(AttackDesign(d.targets, 1.15) for _ in states))
        Xa = g.goal_state(held, 0, states[0], area, k)
        assert Xa is not None
        # the frame before carried the same step, its branch currents included (hybrid meters)
        from fdia_graph.models.frames import AttackVector

        before = AttackVector(*g._attack_vector(Xa, states[0]), g._current_attack(Xa, states[0]))
        assert _Window(g, states, held, k).cost(area, None) is None
        assert _Window(g, states, held, k, prev=before).cost(area, None) is not None


def test_the_search_stops_early_only_once_a_smaller_support_is_the_cheapest():
    """One device on one channel is the least an attack tampers, so the loop stops there, but not on the
    area itself (solved first, the largest candidate): a smaller support of the same counts follows."""
    from fdia_graph.engine.attacks.minimize import _cheapest

    area, failing, small = np.arange(13), np.arange(9), np.arange(7)
    costs = {13: (1, 1, 13), 9: None, 7: (1, 1, 7)}

    class Table:
        def cost(self, S, beat):
            c = costs[len(S)]
            return c if c is not None and (beat is None or c < beat) else None

    best, evaluated = _cheapest(Table(), iter([area, failing, small]), area, 256)  # type: ignore[arg-type]
    assert best is not None and len(best[1]) == 7 and evaluated == 3  # the area alone does not stop it
