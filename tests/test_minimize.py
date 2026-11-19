"""The fewest-tamper support search [WU26 eq. 12]: exact against brute force on IEEE-14, a lower
bound that never exceeds a cost, the two sigmas (the emitter's per-scan jitter, the meters'
accuracy class the search counts against), and a continuous reweighted-l1 cross-check the search is
never beaten by."""

import numpy as np
import pytest

from fdia_graph.constants.channels import NodeChannel
from fdia_graph.formulas.attacks import tampered_channels, tampered_devices
from tests.helpers import FixedDraws, windows


def test_a_window_with_no_held_support_says_so(case):
    """Brute force finds no feasible support exactly when the search reports -1."""
    from fdia_graph.engine.attacks.window import AttackWindow

    g, _, knobs = case
    for states, goal, d in windows(case, 4, held=False):
        result = g.minimizer.min_tamper(states, goal, knobs)
        buses = np.unique(g.load_bus[d.targets])
        area = g.area.hop_area(buses, knobs.hops)
        window = AttackWindow(g, states, goal, knobs)
        feasible = [
            c
            for support in g.minimizer._supports([frozenset(int(b) for b in buses)], area)
            if (c := window.cost(support, None)) is not None
        ]
        assert (result.devices < 0) == (not feasible and window.cost(area, None) is None)


def test_the_search_equals_brute_force(case):
    from fdia_graph.engine.attacks.window import AttackWindow

    g, _, knobs = case
    for states, goal, d in windows(case, 3):
        result = g.minimizer.min_tamper(states, goal, knobs)
        buses = np.unique(g.load_bus[d.targets])
        area = g.area.hop_area(buses, knobs.hops)
        window = AttackWindow(g, states, goal, knobs)
        feasible = []
        for support in [*g.minimizer._supports([frozenset(int(b) for b in buses)], area), np.asarray(area)]:
            c = window.cost(support, None)
            if c is not None:
                feasible.append(c)
        # every candidate solved within the budget: the search's answer is the cheapest of them
        assert (result.devices, result.channels, len(result.support)) == min(feasible)


def test_the_search_minimises_channels_first_when_that_is_the_objective(case):
    """With `objective="channels"` the search compares the channel count first, the devices next:
    its answer is the cheapest candidate in that order."""
    from fdia_graph.engine.attacks.window import AttackWindow

    g, _, knobs = case
    by_channels = knobs._replace(objective="channels")
    for states, goal, d in windows(case, 4):
        result = g.minimizer.min_tamper(states, goal, by_channels)
        buses = np.unique(g.load_bus[d.targets])
        area = g.area.hop_area(buses, knobs.hops)
        window = AttackWindow(g, states, goal, by_channels)
        candidates = [*g.minimizer._supports([frozenset(int(b) for b in buses)], area), np.asarray(area)]
        feasible = [c for support in candidates if (c := window.cost(support, None)) is not None]
        want = min(c.key(True) for c in feasible)
        assert (result.channels, result.devices, len(result.support)) == want


def test_the_emitter_draws_with_the_shared_noise_rule(case):
    """A scan takes one draw per meter, bus by bus (|V|, angle, P, Q), then the flows, then the PMU
    currents, and scales each by the meter's jitter: the class's jitter share, relative on powers
    and currents with an unscaled floor."""
    from fdia_graph.constants.meters import (
        ACCURACY_CLASS,
        CURRENT_FLOOR_PU,
        JITTER_FRAC,
        PMU_CURRENT_CLASS,
        POWER_NOISE_FLOOR_MW,
    )
    from fdia_graph.formulas.network import branch_currents, branch_flows, complex_voltages
    from fdia_graph.formulas.noise import current_magnitude

    g, X, _ = case
    sizes = []

    class _Recorder:  # stands in for the emission's stream: records every draw's size
        def normal(self, loc, scale, size):
            sizes.append(size)
            return np.random.default_rng(5).normal(loc, scale, size)

    g.emitter._jitter_stream = lambda: _Recorder()
    try:
        scan = g.emitter.emit_from_state(X[0])
    finally:
        del g.emitter._jitter_stream
    z = g.emitter.exact(X[0])
    assert sizes == [z.shape]  # one vector draw, a normal per meter
    drawn = g.emitter.layers(g.noise.sample(z, _Recorder()).astype(np.float32))
    assert np.array_equal(scan.node_x, drawn.node) and np.array_equal(scan.edge_x, drawn.edge)
    assert scan.i_x is not None and np.array_equal(scan.i_x, drawn.current)

    x, N = np.asarray(X[0], np.float64), NodeChannel
    Vc = complex_voltages(x[:, N.V], x[:, N.THETA])  # the emitter's own float64 flows and currents
    Sf = branch_flows(Vc, g.net.yf, g.case.f, g.case.base_mva)
    flows = np.stack([Sf.real, Sf.imag], axis=1)
    cur = JITTER_FRAC * PMU_CURRENT_CLASS * current_magnitude(branch_currents(Vc, g.net.yf, g.net.yt))
    cur = cur + CURRENT_FLOOR_PU
    jitter = {k: JITTER_FRAC * c for k, c in ACCURACY_CLASS.items()}
    expected = []
    for b in range(g.n_bus):
        if scan.node_m[b, N.V]:
            expected.append(jitter["v"])
        if scan.node_m[b, N.THETA]:  # the hybrid meters: an angle at the PMU buses only
            expected.append(JITTER_FRAC * np.degrees(ACCURACY_CLASS["va"]))
        if scan.node_m[b, N.P_INJ]:
            expected.append(abs(x[b, N.P_INJ]) * jitter["pi"] + POWER_NOISE_FLOOR_MW)
            expected.append(abs(x[b, N.Q_INJ]) * jitter["qi"] + POWER_NOISE_FLOOR_MW)
    for e in range(g.n_branch):
        if scan.edge_m[e, 0]:
            expected.append(abs(flows[e, 0]) * jitter["pf"] + POWER_NOISE_FLOOR_MW)
            expected.append(abs(flows[e, 1]) * jitter["qf"] + POWER_NOISE_FLOOR_MW)
    # then the PMU branch-current channels, after the flows, branch by branch
    expected += [cur[e, c] for e, c in zip(*np.nonzero(scan.i_m))]
    assert g.noise.jitter(z).tolist() == expected  # exactly: the same products in the same order
    # draw i moves meter i alone
    one = np.zeros(len(z))
    one[3] = 1.0

    class _Third:
        def normal(self, loc, scale, size):
            return one

    moved = g.noise.sample(z, _Third()) - g.noise.sample(z, FixedDraws(0.0))
    assert np.flatnonzero(moved).tolist() == [3] and moved[3] == pytest.approx(expected[3])


def test_the_search_counts_against_the_calibrations_accuracy_sigma(case):
    """The rated accuracy (`NoiseModel.sigmas`) is the measured calibration's sigma (SEBase._class_sigma
    through accuracy_class_sigma) at one scan's readings, in physical units; it is larger than the jitter."""
    from fdia_graph.constants.meters import ACCURACY_CLASS
    from fdia_graph.formulas.noise import accuracy_class_sigma

    g, X, _ = case
    x, base = X[0], g.case.base_mva
    flows = g.emitter.clean_flows_from_states(X[:1])[0].astype(np.float64)
    z = g.emitter.exact(x)
    rated = g.emitter.layers(g.noise.sigmas(z))
    sig, sig_f = rated.node, rated.edge
    node, edge = g.masks.node > 0, g.masks.edge > 0  # a sigma is kept where a meter sits
    volt, pmu = node[:, NodeChannel.V], node[:, NodeChannel.THETA]
    assert node[:, NodeChannel.P_INJ].all() and volt.any() and pmu.any() and edge[:, 1].any()
    # the calibration's rule, in its units (per unit on baseMVA, radians), on the same readings
    pu_p = accuracy_class_sigma(
        np.abs(x[:, NodeChannel.P_INJ]) / base,
        np.full(g.n_bus, ACCURACY_CLASS["pi"]),
        np.ones(g.n_bus, bool),
        1e-3 / base,
    )
    va = accuracy_class_sigma(
        np.abs(np.radians(x[:, NodeChannel.THETA])),
        np.full(g.n_bus, ACCURACY_CLASS["va"]),
        np.zeros(g.n_bus, bool),
        1e-3 / base,
    )
    assert np.allclose(sig[:, NodeChannel.P_INJ], pu_p * base) and np.allclose(
        sig[pmu, NodeChannel.THETA], np.degrees(va)[pmu]
    )
    assert np.allclose(sig[volt, NodeChannel.V], ACCURACY_CLASS["v"])
    # the stored angle is in degrees, the class in radians: the sigma is returned in the scan's units
    assert np.allclose(sig[pmu, NodeChannel.THETA], np.degrees(ACCURACY_CLASS["va"]))
    on = edge[:, 1]
    assert np.allclose(sig_f[on, 1], ACCURACY_CLASS["qf"] * np.abs(flows[on, 1]) + 1e-3)
    jit = g.emitter.layers(g.noise.jitter(z))
    assert (sig >= jit.node).all() and (sig_f >= jit.edge).all() and (rated.current >= jit.current).all()
    assert (sig[volt, NodeChannel.V] > jit.node[volt, NodeChannel.V]).all()


def test_devices_group_channels_by_bus_terminal_and_pmu():
    node = np.zeros((4, 4), bool)
    edge = np.zeros((3, 2), bool)
    node[0, NodeChannel.P_INJ] = True  # SCADA terminal of bus 0
    node[1, NodeChannel.V] = True  # a PMU bus: its PMU
    node[2, NodeChannel.THETA] = True  # a voltage meter without PMU: bus 2's SCADA terminal
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
    knobs default off, so a walk with min_tamper=False holds the area within `hops`."""
    from fdia_graph.models.config import TimelineSettings
    from fdia_graph.models.frames import FrameKnobs, RampDesign

    assert FrameKnobs().min_tamper is False
    assert TimelineSettings().search.min_tamper is True
    assert RampDesign(np.array([0]), 1.0, 1, 0).support is None


def test_supports_keep_the_area_rules(case):
    """Every candidate excludes the slack and takes in the zero-injection buses of its boundary, the
    rules `hop_area` applies to the area."""
    g, _, knobs = case
    zero = set(int(b) for b in g.zero_inj) - {g.slack_bus}
    edges = g.edge_index
    for _, _, d in windows(case, 2):
        buses = np.unique(g.load_bus[d.targets])
        area = g.area.hop_area(buses, knobs.hops)
        for support in g.minimizer._supports([frozenset(int(b) for b in buses)], area):
            assert g.slack_bus not in support
            boundary = set(np.unique(edges[:, np.isin(edges, support).any(axis=0)]).tolist()) - set(
                support.tolist()
            )
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
    from fdia_graph.engine.attacks.window import AttackWindow
    from fdia_graph.models.frames import AttackDesign, LoadGoal

    g, X, knobs = case
    for states, _, d in windows(case, 1, held=False):
        area = np.asarray(g.area.hop_area(np.unique(g.load_bus[d.targets]), knobs.hops))
        jump = LoadGoal(
            tuple(AttackDesign(d.targets, 1.15) for _ in states)
        )  # held at 1.15: only the onset jumps
        assert AttackWindow(g, states, jump, knobs, stealth_bound=False).cost(area, None) is not None
        assert AttackWindow(g, states, jump, knobs).cost(area, None) is None


def test_a_support_within_noise_is_not_an_attack(case):
    """A window whose loads never move a device beyond its accuracy sigma costs nothing, so it is not an
    attack: no support is feasible and the search records -1, never a 0-device result."""
    from fdia_graph.engine.attacks.window import AttackWindow
    from fdia_graph.models.frames import AttackDesign, LoadGoal

    g, _, knobs = case
    for states, _, d in windows(case, 1, held=False):
        area = np.asarray(g.area.hop_area(np.unique(g.load_bus[d.targets]), knobs.hops))
        still = LoadGoal(tuple(AttackDesign(d.targets, 1.0 + 1e-9) for _ in states))
        assert AttackWindow(g, states, still, knobs).cost(area, None) is None
        assert g.minimizer.min_tamper(states, still, knobs).devices == -1


def test_the_onset_is_bounded_against_the_frame_before(case):
    """Episodes may be adjacent: a window after an attacked frame starts its stealth bound from that
    frame's attack vector. A held 15 percent step is refused after a benign frame (it jumps at onset)
    and accepted right after a frame carrying the same step (no increment at onset)."""
    from fdia_graph.engine.attacks.window import AttackWindow
    from fdia_graph.models.frames import AttackDesign, LoadGoal

    g, _, knobs = case
    for states, _, d in windows(case, 1, held=False):
        area = np.asarray(g.area.hop_area(np.unique(g.load_bus[d.targets]), knobs.hops))
        held = LoadGoal(tuple(AttackDesign(d.targets, 1.15) for _ in states))
        x_false = g.false_state.goal_state(held, 0, states[0], area, knobs)
        assert x_false is not None
        # the frame before carried the same step, its branch currents included (hybrid meters)
        from fdia_graph.models.frames import AttackVector

        before = AttackVector(
            *g.attack_frames.attack_vector(x_false, states[0]),
            g.attack_frames.current_attack(x_false, states[0]),
        )
        assert AttackWindow(g, states, held, knobs).cost(area, None) is None
        assert AttackWindow(g, states, held, knobs, prev=before).cost(area, None) is not None


def test_the_search_stops_early_only_once_a_smaller_support_is_the_cheapest():
    """One device on one channel is the least an attack tampers, so the loop stops there, but not on the
    area itself (solved first, the largest candidate): a smaller support of the same counts follows."""
    from fdia_graph.engine.attacks.minimize import _cheapest
    from fdia_graph.models.frames import SupportCost

    area, failing, small = np.arange(13), np.arange(9), np.arange(7)
    costs = {13: SupportCost(1, 1, 13), 9: None, 7: SupportCost(1, 1, 7)}

    class Table:
        def cost(self, support, beat):
            c = costs[len(support)]
            return c if c is not None and (beat is None or c.key(False) < beat.key(False)) else None

    best, evaluated = _cheapest(Table(), iter([area, failing, small]), area, 256)  # type: ignore[arg-type]
    assert best is not None and len(best[1]) == 7 and evaluated == 3  # the area alone does not stop it
