"""[WU26]'s attack as the paper states it (`trust.wu26`): the meter plan, the objective's form, the
incremental trust and the reproduction's settings."""

import os

import numpy as np
import pytest

from fdia_graph.models.config import MeterSettings, OverloadSettings, Wu26Attack
from fdia_graph.models.validation import ConfigError

pp = pytest.importorskip("pandapower")

from fdia_graph.trust.wu26 import (  # noqa: E402
    _PF,
    WindowAttack,
    _WindowProblem,
    incremental_freeze,
    wu26_network,
)
from fdia_graph.trust.wu26_defense import wu26_solution1  # noqa: E402


@pytest.fixture(scope="module")
def net14():
    return wu26_network("case14", [1, 4, 6, 13])


def test_every_branch_is_metered_at_both_ends_by_the_bus_at_that_end(net14):
    """[29] Sec. 2.2: each node's SCADA reads every incident flow at its end, so line 6-11's bus-11 end
    belongs to SCADA 11 (Fig. 4 tampers it)."""
    for k in range(len(net14.f)):
        ends = net14.end[(net14.br == k) & (net14.kind == _PF)]
        assert sorted(ends.tolist()) == [0, 1]
    k, e = net14.branch_of(11, 6)
    rows = np.flatnonzero((net14.br == k) & (net14.end == e) & (net14.dev_type == 0))
    assert net14.devices(rows) == {("SCADA", 11)}
    devices = net14.devices(np.arange(len(net14.kind)))
    assert len(devices) == 18  # Fig. 4's axes: SCADA 1-14, PMU 1, 4, 6, 13
    assert not ((net14.dev_type == 0) & (net14.kind == 4)).any()  # no SCADA voltmeter


def test_readings_match_the_power_flow(net14):
    """Injections and from-end flows equal pandapower's results (generation positive, per unit)."""
    case = net14.case
    Vm, Va = case.res_bus.vm_pu.values, np.radians(case.res_bus.va_degree.values)
    z = net14.measure(Vm, Va)
    p_rows = np.flatnonzero(net14.kind == 0)
    np.testing.assert_allclose(z[p_rows] * 100, -case.res_bus.p_mw.values[net14.bus[p_rows]], atol=1e-6)
    line0 = case.line.iloc[0]
    k, e = net14.branch_of(int(line0.from_bus) + 1, int(line0.to_bus) + 1)
    assert e == 0
    np.testing.assert_allclose(
        net14.flow(Vm, Va, k, 0).real * 100, case.res_line.p_from_mw.iloc[0], atol=1e-6
    )


def test_the_jacobian_matches_finite_differences(net14):
    case = net14.case
    Vm, Va = case.res_bus.vm_pu.values.copy(), np.radians(case.res_bus.va_degree.values)
    cols = np.array([3, 5, 10])
    J = net14.jacobian(Vm, Va, cols)
    h = 1e-7
    for j, b in enumerate(cols):
        for off, (dvm, dva) in ((0, (h, 0.0)), (len(cols), (0.0, h))):
            Vm2, Va2 = Vm.copy(), Va.copy()
            Vm2[b] += dvm
            Va2[b] += dva
            fd = (net14.measure(Vm2, Va2) - net14.measure(Vm, Va)) / h
            np.testing.assert_allclose(J[:, off + j], fd, atol=1e-5)


def test_the_objective_scores_the_net_attack_of_the_window(net14):
    """Eq. (12) puts the sum over snapshots inside the norm: two snapshots moving a bus in opposite
    directions nearly cancel in the objective, while the per-snapshot term counts both."""
    case = net14.case
    state = (case.res_bus.vm_pu.values.copy(), np.radians(case.res_bus.va_degree.values))
    area = np.array([3, 4, 5])
    flat = _WindowProblem(net14, [state, state], area, [(4, 5)], Wu26Attack(tau=0.0), {})
    each = _WindowProblem(net14, [state, state], area, [(4, 5)], Wu26Attack(tau=1.0), {})
    x = flat.x0.copy()
    x[flat.nb + 1] += 0.01  # snapshot 0: bus 5's angle up
    x[2 * flat.nb + flat.nb + 1] -= 0.01  # snapshot 1: down
    base = flat.objective(flat.x0)
    assert flat.objective(x) - base < 0.05 * (each.objective(x) - each.objective(each.x0))


def test_a_trusted_pmu_keeps_the_offset_it_had_before_its_slot():
    offsets = np.arange(3 * 2 * 2, dtype=float).reshape(3, 2, 2)
    attack = WindowAttack(None, (3, 5), np.zeros((3, 4)), offsets, True, 0)  # type: ignore[arg-type]
    freeze = incremental_freeze([5, 3], [0, 2], attack)
    assert freeze[0] == {5: (0.0, 0.0)}
    assert freeze[2] == {5: (0.0, 0.0), 3: tuple(offsets[1, 0])}


def test_the_reproduction_settings_are_validated():
    with pytest.raises(ConfigError):
        Wu26Attack(vmin=1.1, vmax=1.0)
    with pytest.raises(ConfigError):
        Wu26Attack(meter_model="hybrid")
    with pytest.raises(ConfigError):
        MeterSettings(meter_model="wu26")  # not a generator plan
    with pytest.raises(ConfigError):
        OverloadSettings(support_method="wu_l1")


def test_solution1_trusts_the_four_ieee14_pmus(net14):
    assert sorted(wu26_solution1(net14, [(3, 4), (6, 11)], n=4)) == [1, 4, 6, 13]


@pytest.mark.skipif(not os.environ.get("FDIA_SLOW"), reason="set FDIA_SLOW=1: a 20-snapshot IPOPT solve")
def test_scenario1_attack_reaches_fig4(net14):
    """IEEE-14 scenario 1 (lines 3-4 and 6-11): the attack tampers every device of Fig. 4's set."""
    pytest.importorskip("cyipopt")
    from fdia_graph.trust.wu26 import solve_window, wu26_snapshots

    states = wu26_snapshots(net14.case, 20, seed=123)
    area = [b - 1 for b in (2, 3, 4, 5, 6, 11, 12, 13)]
    attack = Wu26Attack(rho=1.5, tau=0.5, dv=0.03, da=0.1)
    result = solve_window(net14, states, area, [(3, 4), (6, 11)], attack)
    assert result.feasible
    fig4 = {("SCADA", b) for b in (3, 4, 5, 6, 11)} | {("PMU", 4), ("PMU", 6)}
    assert fig4 <= result.devices_any
    assert result.max_magnitude < 0.5
