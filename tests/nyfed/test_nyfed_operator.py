from dataclasses import replace

import numpy as np
import pytest
from nyfed_inputs import DAMAGE_DAYS, DATA, DAYS, DAYS_POOL, SINGLE_DAY, needs_data

from fdia_graph.nyfed.grid.case import REF
from fdia_graph.nyfed.grid.fixes import restore_765kv, split_zone_j
from fdia_graph.nyfed.grid.interfaces import hourly_interfaces
from fdia_graph.nyfed.grid.nygrid import modify_mpc, nygrid_units, read_npcc
from fdia_graph.nyfed.operator.sced import DcSced, loss_factors, outage_factors, shift_factors, shortage_usd

pytestmark = needs_data


@pytest.fixture(scope="module")
def case():
    return restore_765kv(split_zone_j(nygrid_units(modify_mpc(read_npcc(DATA / "npcc.mat")), DATA)))


def _dc_flows(case, p):
    """Branch flows of a DC power flow solved directly, the reference bus balancing `p`."""
    n = len(case.bus_id)
    ref = int(np.flatnonzero(case.bus_type == REF)[0])
    b = 1.0 / (case.x * np.where(case.tap == 0, 1.0, case.tap))
    inc = np.zeros((len(case.f), n))
    inc[np.arange(len(case.f)), case.f], inc[np.arange(len(case.f)), case.t] = 1.0, -1.0
    keep = np.arange(n) != ref
    theta = np.zeros(n)
    theta[keep] = np.linalg.solve((inc.T @ (b[:, None] * inc))[np.ix_(keep, keep)], p[keep])
    return b * (inc @ theta)


def test_shortage_is_priced_at_the_2019_curve():
    """350 $/MWh for the first 5 MW over the limit, 1,175 for the next 15, 4,000 beyond, over five minutes."""
    assert shortage_usd(0.0) == 0.0
    assert np.isclose(shortage_usd(20.0), (5 * 350 + 15 * 1175) / 12)
    assert np.isclose(shortage_usd(30.0), (5 * 350 + 15 * 1175 + 10 * 4000) / 12)


def test_shift_factors_reproduce_a_dc_power_flow(case):
    p = np.random.default_rng(0).normal(0, 50, len(case.bus_id))
    np.testing.assert_allclose(shift_factors(case) @ p, _dc_flows(case, p), atol=1e-6)


def test_outage_factors_reproduce_the_flows_without_the_branch(case):
    p = np.random.default_rng(1).normal(0, 50, len(case.bus_id))
    ptdf = shift_factors(case)
    before = ptdf @ p
    k = int(np.argmax(np.abs(before) * (case.kv[case.f] >= 345)))  # a loaded 345 kV branch
    lodf = outage_factors(case, ptdf, np.array([k]))
    after = _dc_flows(replace(case, x=np.where(np.arange(len(case.x)) == k, 1e9, case.x)), p)
    np.testing.assert_allclose(before + lodf[:, 0] * before[k], after, atol=1e-3)


@pytest.fixture(scope="module")
def sced(case):
    """The dispatch over two scans with every unit at the middle of its range (the base case has most
    thermal units off), at a cost rising with the unit index."""
    unit_p = np.tile(np.where(case.pmax > 1.0, 0.5 * (case.pmin + case.pmax), case.unit_p), (2, 1))
    cost = np.tile(np.linspace(10.0, 100.0, len(case.unit_p)), (2, 1))
    time = np.array(["2019-07-05T11:00:00", "2019-07-05T11:05:00"], dtype="datetime64[s]").astype(np.int64)
    return DcSced(case, unit_p, cost, hourly_interfaces(case, time, DATA, ["SPR/DUN-SOUTH"]))


def test_the_operator_moves_nothing_when_it_sees_nothing_wrong(sced):
    s = sced.step(1, np.array([1000.0]), np.array([1500.0]), 0.0, np.zeros(len(sced.free)))
    assert np.abs(s.dev).max() < 1e-6 and s.unrelieved[0] < 1e-6 and abs(s.unmatched) < 1e-6


def test_the_operator_relieves_an_interface_it_sees_over_its_limit(sced):
    s = sced.step(1, np.array([1550.0]), np.array([1500.0]), 0.0, np.zeros(len(sced.free)))
    moved = sced.sf[0, sced.case.unit_bus] @ s.dev
    assert s.unrelieved[0] < 1e-6 and moved <= -50.0 + 1e-6  # the seen flow brought back to the limit
    assert abs(s.dev.sum()) < 1e-6  # balance kept: the load it sees did not change
    assert (s.dev[~sced.free] == 0).all()  # nuclear, hydro and imports follow their schedule


def test_the_operator_follows_the_load_it_sees(sced):
    s = sced.step(1, np.array([1000.0]), np.array([1500.0]), -100.0, np.zeros(len(sced.free)))
    assert abs(s.dev.sum() + 100.0) < 1e-6 and abs(s.unmatched) < 1e-6


def test_recovery_after_an_attack_is_ramp_limited(sced):
    """A unit the operator moved 40 MW up cannot return to its basepoint in one scan: its deviation at the
    next scan stays within one five-minute ramp of where it was, even with nothing seen wrong."""
    c = sced.case
    room = (c.pmax - sced.unit_p[1]) > 60
    g = int(np.flatnonzero(sced.free & (sced.unit_p[1] > 0.5) & room & (c.ramp * 5 < 30))[0])
    prev = np.zeros(len(sced.free))
    prev[g] = 40.0
    s = sced.step(1, np.array([1000.0]), np.array([1500.0]), 40.0, prev)
    assert s.dev[g] >= 40.0 - c.ramp[g] * 5.0 - 1e-6


def test_the_operator_sheds_load_only_when_redispatch_cannot_relieve(case, sced):
    """A seen overload far beyond what redispatch can move is relieved by shedding load, only at buses where
    a MW shed takes at least 0.1 MW off the interface; a small overload is relieved by redispatch alone."""
    from fdia_graph.nyfed.operator.sced import SHED_FACTOR

    load = np.tile(np.where(case.is_ny, 200.0, 0.0), (2, 1))
    op = DcSced(case, sced.unit_p, sced.cost, [], load)
    op.sf = sced.sf
    small = op.step(1, np.array([1550.0]), np.array([1500.0]), 0.0, np.zeros(len(op.free)))
    assert not small.shed.any() and small.unrelieved[0] < 1e-6
    held = op.step(1, np.array([9000.0]), np.array([1500.0]), 0.0, np.zeros(len(op.free)), shed_ok=False)
    assert not held.shed.any() and held.unrelieved[0] > 0  # before 15 minutes over the limit, no relief
    big = op.step(1, np.array([9000.0]), np.array([1500.0]), 0.0, np.zeros(len(op.free)))
    assert big.shed.sum() > 0 and (sced.sf[0, big.shed > 0] <= SHED_FACTOR).all()


def test_a_day_under_attack_against_its_benign_twin():
    """On the 5 July file the benign twin needs nothing, while the attacked run follows the pretended load
    drop, and regulation with what the neighbours supply makes up the generation the dispatch left out."""
    from fdia_graph.nyfed.operator.response import operator_day, respond

    r = respond(operator_day(SINGLE_DAY, DAYS_POOL, DATA))
    a, b = r.attacked, r.benign
    assert np.abs(b.dev).max() < 1e-3 and np.abs(b.reg).max() < 1e-3
    w = slice(r.onset, r.onset + r.length)
    assert -a.dev[w].sum(axis=1).min() > 300  # the pretended drop reaches about 370 MW
    # regulation and the inadvertent interchange together make up the generation the dispatch removed
    # beyond the load actually shed
    np.testing.assert_allclose(
        a.reg.sum(axis=1) + a.uncovered, -(a.dev.sum(axis=1) + a.shed.sum(axis=1)), atol=1e-6
    )


def test_the_three_operators_on_a_day_under_attack():
    """The 5 July attack moves New York's net interchange (its change is taken up outside New York), so the
    read interchange leaves the neighbours' meters beyond their noise inside the window (N62). The trusting
    operator follows the false load drop and its AGC, reading the same falsified ties, hides the shortfall,
    which the neighbours supply as inadvertent interchange. The accounting operator takes its load from
    generation plus the agreed interchange, so it does not follow the false load (N66)."""
    from fdia_graph.nyfed.operator.response import operator_day, respond

    day = operator_day(SINGLE_DAY, DAYS_POOL, DATA)
    trusting, accounting = respond(day), respond(day, mode="accounting")
    assert np.isfinite(trusting.alarm) and 0 <= trusting.alarm < trusting.length
    w = slice(trusting.onset, trusting.onset + trusting.length)
    assert -trusting.attacked.dev[w].sum(axis=1).min() > 300  # it cuts generation for the false drop
    assert np.abs(trusting.attacked.uncovered[w]).max() > 200  # left to the neighbours
    assert np.abs(accounting.attacked.dev[w].sum(axis=1)).max() < 50  # no false load followed
    assert np.abs(accounting.attacked.uncovered).sum() < np.abs(trusting.attacked.uncovered).sum()


def test_a_failed_power_flow_is_unknown_not_safe(monkeypatch):
    """Scans whose AC power flow fails are flagged in `pf_failed`, have no margin, are counted in the summary and
    never read as within the limit; every other scan stays unflagged."""
    from fdia_graph.nyfed.operator import response
    from fdia_graph.nyfed.operator.response import operator_day, respond, summarize

    day = operator_day(SINGLE_DAY, DAYS_POOL, DATA)
    fail = set(range(day.onset + 2, day.onset + 5))
    real = response.TrueGrid.change

    def change(self, t, units, shed=None):
        if t in fail:
            return np.full(len(self.day.interfaces), np.nan), np.nan
        return real(self, t, units, shed)

    monkeypatch.setattr(response.TrueGrid, "change", change)
    r = respond(day)
    a, b = r.attacked, r.benign
    flagged = set(np.flatnonzero(a.pf_failed).tolist())
    assert flagged and flagged <= fail  # only scans the dispatch moved reach the true grid
    assert np.isnan(a.margin[sorted(flagged)]).all() and np.isfinite(a.margin[sorted(fail - flagged)]).all()
    s = summarize(r)
    assert s["pf_failed_scans"] == len(flagged) and s["pf_failed_benign_scans"] == int(b.pf_failed.sum())
    assert not a.pf_failed[[t for t in range(len(a.pf_failed)) if t not in fail]].any()


def test_a_pool_without_the_days_frames_is_a_named_error(tmp_path):
    """A pool that ends before the day (here the checked-in pool cut to its first day, 17 June, for the 5 July
    file) is a `DataConditionError` naming what is missing, not an index error."""
    from fdia_graph.errors import DataConditionError
    from fdia_graph.nyfed.io import read_pool, write_pool
    from fdia_graph.nyfed.operator.response import operator_day
    from fdia_graph.nyfed.scenario.pool import Pool

    p = read_pool(DAYS_POOL)
    june = p.time.astype("datetime64[D]") == np.datetime64("2019-06-17")
    write_pool(tmp_path / "june.h5", Pool(**{k: getattr(p, k)[june] for k in Pool.__dataclass_fields__}))
    with pytest.raises(DataConditionError, match="does not hold every frame"):
        operator_day(SINGLE_DAY, tmp_path / "june.h5", DATA)


def test_the_dispatch_around_the_ac_point_needs_the_ac_state(case, sced):
    from fdia_graph.errors import ConfigError

    with pytest.raises(ConfigError, match="AC state"):
        DcSced(case, sced.unit_p, sced.cost, [], objective="cost_lf")


def test_lossless_prices_are_the_energy_price_less_congestion(sced):
    """Under the least-cost dispatch without loss factors a bus's price is the energy price less each interface's
    price times the bus's shift factor, and what a dispatch delivers is its plain sum."""
    prices = sced.prices(1, 30.0, np.array([5.0]))
    np.testing.assert_allclose(prices, 30.0 - 5.0 * sced.sf[0])
    dev = np.random.default_rng(2).normal(0, 5, len(sced.free))
    assert np.isclose(sced.delivered(1, dev, np.zeros(len(sced.case.bus_id))), dev.sum())


def test_loss_factors_match_an_ac_power_flow():
    """The delivery factors at a day's AC state are what the true grid's power flow gives: five more MW at a unit
    take five times its bus's factor off the reference bus (the factors run from about 0.47 in the west to 1.02 in
    New York City on this scan)."""
    from fdia_graph.nyfed.grid.network import Network
    from fdia_graph.nyfed.operator.response import operator_day
    from fdia_graph.nyfed.operator.truegrid import TrueGrid

    day = operator_day(DAYS / DAMAGE_DAYS[0], DAYS_POOL, DATA)
    c, t = day.case, day.onset
    beta = loss_factors(c, day.vm[t], day.va[t])
    grid, net = TrueGrid(day), Network(c)
    ref = int(np.flatnonzero(c.bus_type == REF)[0])
    assert beta[ref] == 1.0
    state = grid.state(t, day.unit_p[t])
    assert state is not None
    base = net.injections(*state)[0][ref]
    free = ~np.isin(c.fuel.astype(str), ("Nuclear", "Hydro", "Import")) & c.is_ny[c.unit_bus]
    for k in np.argsort(-np.where(free, day.unit_p[t], 0.0))[:3]:  # the three largest free units on line
        units = day.unit_p[t].copy()
        units[k] += 5.0
        moved = grid.state(t, units)
        assert moved is not None
        assert np.isclose((base - net.injections(*moved)[0][ref]) / 5.0, beta[c.unit_bus[k]], atol=2e-3)


def test_the_dispatch_around_the_ac_point_keeps_the_benign_twin_at_the_ac_points():
    """The least-cost dispatch on its own moves the benign twin about a gigawatt off the AC points (the DC model
    does not see the losses, voltage limits and schedules they were dispatched against); around the AC point the
    benign twin stays there, so the attack's consequence is the attack's. Bus prices are reported every scan."""
    from fdia_graph.nyfed.operator.response import operator_day, respond

    day = operator_day(DAYS / DAMAGE_DAYS[0], DAYS_POOL, DATA)
    r = respond(day, mode="accounting", closed_loop=True, dispatch="cost_lf")
    b = r.benign
    assert np.abs(b.dev).sum(axis=1).max() < 0.1
    assert np.nanmax(np.abs(b.iface_change[:, 0])) < 1.0
    assert b.price is not None and r.attacked.price is not None
    assert np.isfinite(r.attacked.price[day.onset : day.onset + day.length]).all()


# The AC operator (`AcSced`): it reads the pool's own inputs for the day, the reactive loads from the checked-in slice
AC_LOADS = DAYS / "loads_2019_v2_days.npz"


@pytest.fixture(scope="module")
def ac_day():
    from fdia_graph.nyfed.operator.day import pool_frames
    from fdia_graph.nyfed.operator.response import operator_day
    from fdia_graph.nyfed.operator.truegrid import TrueGrid

    day = operator_day(DAYS / DAMAGE_DAYS[0], DAYS_POOL, DATA)
    return day, TrueGrid(day, ac=True), pool_frames(day, AC_LOADS)


def _ac_sced(ac_day, objective):
    from fdia_graph.nyfed.operator.acsced import AcSced

    day, grid, frames = ac_day
    return AcSced(
        day.case,
        day.unit_p,
        day.cost,
        day.interfaces,
        (day.vm, day.va),
        (grid.pd, frames.qd),
        frames,
        objective=objective,
    )


def test_ac_sensitivities_match_an_ac_power_flow(ac_day):
    """The AC sensitivity of the attacked interface to a bus injection is what the true grid's power flow gives for
    five more MW at a unit there, the reference bus balancing."""
    from fdia_graph.nyfed.grid.network import Network
    from fdia_graph.nyfed.operator.acsced import ac_shift

    day, grid, _ = ac_day
    c, t = day.case, day.onset
    shift = ac_shift(c, day.vm[t], day.va[t], day.interfaces)
    net = Network(c)
    base = grid.state(t, day.unit_p[t])
    assert base is not None
    flow0 = day.interfaces[0].model_flow(net.flows(*base)[0])
    free = ~np.isin(c.fuel.astype(str), ("Nuclear", "Hydro", "Import")) & c.is_ny[c.unit_bus]
    for k in np.argsort(-np.where(free, day.unit_p[t], 0.0))[:3]:
        units = day.unit_p[t].copy()
        units[k] += 5.0
        moved = grid.state(t, units)
        assert moved is not None
        got = (day.interfaces[0].model_flow(net.flows(*moved)[0]) - flow0) / 5.0
        assert np.isclose(got, shift[0, c.unit_bus[k]], atol=5e-3)


def test_the_outage_power_flow_matches_pandapower(ac_day):
    """The N-1 screen's Newton power flow of the grid without a branch reaches the state pandapower reaches on the same
    case, from the same injections and unit voltages."""
    from fdia_graph.nyfed.grid.network import Network
    from fdia_graph.nyfed.operator.acsced import ac_power_flow, open_branch
    from fdia_graph.nyfed.scenario.solver import AcSolver

    day, grid, _ = ac_day
    c, t = day.case, day.onset
    k = int(grid.outages[0])
    opened = open_branch(c, k)
    p, q = Network(c).injections(day.vm[t], day.va[t])
    mine = ac_power_flow(c, Network(opened), day.vm[t], day.va[t], p, q)
    assert mine is not None
    theirs = AcSolver(opened).power_flow(
        grid.pd[t],
        grid.qd[t],
        day.unit_p[t],
        np.ones(len(c.unit_bus), bool),
        day.vm[t, c.unit_bus],
        np.zeros(len(c.dc_f)),
        start=(day.vm[t], day.va[t]),
    )
    assert theirs is not None
    np.testing.assert_allclose(mine[0], theirs["vm"], atol=1e-6)
    np.testing.assert_allclose(mine[1], theirs["va"], atol=1e-4)


def test_the_least_cost_ac_dispatch_keeps_the_ac_points_and_prices_them(ac_day):
    """The least-cost AC dispatch solves the problem the pool solved, so on the true loads it stays at the stored AC
    points, and a unit inside its bounds and not ramp limited is priced within the movement charge of its cost: the
    balance duals are the AC LMPs in $/MWh."""
    from fdia_graph.nyfed.operator.sced import MOVE_USD

    day, grid, frames = ac_day
    sced = _ac_sced(ac_day, "cost")
    limit = sced.flow0 + np.array([it.limit - it.flow for it in day.interfaces]).T
    c, prev = day.case, None
    interior = 0
    for t in range(day.onset, day.onset + 3):
        step = sced.step(t, grid.pd[t], frames.qd[t], np.zeros(len(day.interfaces)), limit[t], prev)
        assert np.abs(step.dev).sum() < 0.5
        assert step.price is not None and np.isfinite(step.price).all()
        prev = step.dev
        p0, before = day.unit_p[t], day.unit_p[t - 1]
        inside = (
            sced.free
            & (p0 > frames.p_lo[t] + 1.0)
            & (p0 < frames.p_hi[t] - 1.0)
            & (np.abs(p0 - before) < sced.ramp - 1.0)
        )
        gap = np.abs(step.price[c.unit_bus] - day.cost[t])[inside]
        interior += int(inside.sum())
        assert (gap <= MOVE_USD + 0.5).all()
    assert interior > 0


def test_the_ac_operator_on_a_damaging_day(monkeypatch):
    """The least-movement AC operator keeps the benign twin at the AC points, and against the trusting operator the
    damaging attack cuts the attacked interface's true transfer."""
    from fdia_graph.nyfed.operator.response import operator_day, respond

    monkeypatch.setenv("NYFED_LOADS", str(AC_LOADS))
    day = operator_day(DAYS / DAMAGE_DAYS[1], DAYS_POOL, DATA)
    r = respond(day, mode="trusting", closed_loop=True, dispatch="ac")
    assert np.abs(r.benign.dev).max() == 0.0 and np.nanmax(np.abs(r.benign.iface_change)) == 0.0
    assert np.nanmin(r.attacked.iface_change[:, 0] - r.benign.iface_change[:, 0]) < -50.0
