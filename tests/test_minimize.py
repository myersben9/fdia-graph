"""The fewest-tamper support search [WU26 eq. 12]: exact against brute force on IEEE-14, a lower
bound that never exceeds a cost, the two sigmas (the emitter's per-scan jitter, the meters'
accuracy class the search counts against), and a continuous reweighted-l1 cross-check the search is
never beaten by."""

import numpy as np
import pytest

from fdia_graph.formulas.attacks import tampered_channels, tampered_devices
from fdia_graph.formulas.noise import accuracy_sigma, jitter_sigma
from fdia_graph.models.grid import NODE
from tests.helpers import windows


def test_a_window_with_no_held_support_says_so(case):
    """Brute force finds no feasible support exactly when the search reports -1."""
    from fdia_graph.engine.attacks.minimize import _Window

    g, _, k = case
    for states, goal, d in windows(case, 4, held=False):
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
    for states, goal, d in windows(case, 3):
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
    from fdia_graph.formulas.noise import ACCURACY_CLASS, accuracy_class_sigma

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


def test_new_generation_searches_by_default():
    """New generation holds every At episode on its fewest-tamper support the frame
    knobs default off, so a walk with min_tamper=False holds the region within `hops`."""
    from fdia_graph.models.config import TimelineSettings
    from fdia_graph.models.frames import FrameKnobs, RampDesign

    assert FrameKnobs().min_tamper is False
    assert TimelineSettings().search.min_tamper is True
    assert RampDesign(np.array([0]), 1.0, 1, 0).support is None


def test_supports_keep_the_region_rules(case):
    """Every candidate excludes the slack and takes in the zero-injection buses of its boundary, the
    rules `local_region` applies to the region."""
    g, _, k = case
    zero = set(int(b) for b in g.zero_inj) - {g.slack_bus}
    edges = g._live_edges()
    for _, _, d in windows(case, 2):
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
    for states, _, d in windows(case, 1, held=False):
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
    for states, _, d in windows(case, 1, held=False):
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
    for states, _, d in windows(case, 1, held=False):
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
