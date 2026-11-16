"""The operator's real-time dispatch (`operator.acsced`, an AC optimal power flow each scan): the 2019 shortage
price, the delivery factors and AC sensitivities, the outage power flow, and the operator on days under attack."""

import numpy as np
import pytest

from fdia_graph.generation.nygrid.case import REF, nygrid_case
from fdia_graph.operator.acsced import loss_factors, shortage_usd

from .inputs import DAMAGE_DAYS, DATA, DAYS, DAYS_POOL, SINGLE_DAY, needs_data

pytestmark = needs_data


@pytest.fixture(scope="module")
def case():
    return nygrid_case()


def test_shortage_is_priced_at_the_2019_curve():
    """350 $/MWh for the first 5 MW over the limit, 1,175 for the next 15, 4,000 beyond, over five minutes."""
    assert shortage_usd(0.0) == 0.0
    assert np.isclose(shortage_usd(20.0), (5 * 350 + 15 * 1175) / 12)
    assert np.isclose(shortage_usd(30.0), (5 * 350 + 15 * 1175 + 10 * 4000) / 12)


def test_a_day_under_attack_against_its_benign_twin():
    """On the 5 July file the benign twin needs nothing, while the attacked run follows the pretended load
    drop, and regulation with what the neighbours supply makes up what the dispatch no longer delivers: its
    change and its shed, each weighted by its bus's delivery factor at the true AC state."""
    from fdia_graph.operator.day import operator_day
    from fdia_graph.operator.run import respond

    day = operator_day(SINGLE_DAY, DAYS_POOL, DATA)
    r = respond(day)
    a, b = r.attacked, r.benign
    assert np.abs(b.dev).max() < 1e-3 and np.abs(b.reg).max() < 1e-3
    w = slice(r.onset, r.onset + r.length)
    assert -a.dev[w].sum(axis=1).min() > 300  # the pretended drop reaches about 370 MW
    # regulation and the inadvertent interchange together make up what the dispatch took off the reference
    # bus: a MW at a bus is worth its delivery factor there (the plain sums differ by up to 66 MW of losses)
    beta = loss_factors(day.case, day.vm, day.va)
    delivered = (beta[:, day.case.unit_bus] * a.dev).sum(axis=1) + (beta * a.shed).sum(axis=1)
    np.testing.assert_allclose(a.reg.sum(axis=1) + a.uncovered, -delivered, atol=1e-6)


def test_the_three_operators_on_a_day_under_attack():
    """The 5 July attack moves New York's net interchange (its change is taken up outside New York), so the
    read interchange leaves the neighbours' meters beyond their noise inside the window (N62). The trusting
    operator follows the false load drop and its AGC, reading the same falsified ties, hides the shortfall,
    which the neighbours supply as inadvertent interchange. The accounting operator takes its load from
    generation plus the agreed interchange, so it leaves nothing to the neighbours and moves its units a fraction
    of what the trusting one does (N66)."""
    from fdia_graph.operator.day import operator_day
    from fdia_graph.operator.run import respond

    day = operator_day(SINGLE_DAY, DAYS_POOL, DATA)
    trusting, accounting = respond(day), respond(day, mode="accounting")
    assert np.isfinite(trusting.alarm) and 0 <= trusting.alarm < trusting.length
    w = slice(trusting.onset, trusting.onset + trusting.length)
    assert -trusting.attacked.dev[w].sum(axis=1).min() > 300  # it cuts generation for the false drop
    assert np.abs(trusting.attacked.uncovered[w]).max() > 200  # left to the neighbours
    followed = -trusting.attacked.dev[w].sum(axis=1).min()
    # the false drop is not followed: what moves (about 74 MW against 357) is the redispatch for the load put back
    assert np.abs(accounting.attacked.dev[w].sum(axis=1)).max() < followed / 4
    assert np.abs(accounting.attacked.uncovered).sum() < 1e-6 * np.abs(trusting.attacked.uncovered).sum()


def test_a_failed_power_flow_is_unknown_not_safe(monkeypatch):
    """Scans whose AC power flow fails are flagged in `pf_failed`, have no margin, are counted in the summary and
    never read as within the limit; every other scan stays unflagged."""
    from fdia_graph.operator import truegrid
    from fdia_graph.operator.day import operator_day
    from fdia_graph.operator.run import respond
    from fdia_graph.operator.summary import summarize

    day = operator_day(SINGLE_DAY, DAYS_POOL, DATA)
    fail = set(range(day.onset + 2, day.onset + 5))
    real = truegrid.TrueGrid.change

    def change(self, t, units, shed=None):
        if t in fail:
            return np.full(len(self.day.interfaces), np.nan), np.nan
        return real(self, t, units, shed)

    monkeypatch.setattr(truegrid.TrueGrid, "change", change)
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
    from fdia_graph.generation.nygrid.io import read_pool, write_pool
    from fdia_graph.generation.nygrid.pool import Pool
    from fdia_graph.operator.day import operator_day

    p = read_pool(DAYS_POOL)
    june = p.time.astype("datetime64[D]") == np.datetime64("2019-06-17")
    write_pool(tmp_path / "june.h5", Pool(**{k: getattr(p, k)[june] for k in Pool.__dataclass_fields__}))
    with pytest.raises(DataConditionError, match="does not hold every frame"):
        operator_day(SINGLE_DAY, tmp_path / "june.h5", DATA)


def test_loss_factors_match_an_ac_power_flow():
    """The delivery factors at a day's AC state are what the true grid's power flow gives: five more MW at a unit
    take five times its bus's factor off the reference bus (the factors run from about 0.47 in the west to 1.02 in
    New York City on this scan)."""
    from fdia_graph.generation.nygrid.network import Network
    from fdia_graph.operator.day import operator_day
    from fdia_graph.operator.truegrid import TrueGrid

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


# The AC operator (`AcSced`): it reads the pool's own inputs for the day, the reactive loads from the checked-in slice
AC_LOADS = DAYS / "loads_2019_v2_days.npz"


@pytest.fixture(scope="module")
def ac_day():
    from fdia_graph.operator.day import operator_day, pool_frames
    from fdia_graph.operator.truegrid import TrueGrid

    day = operator_day(DAYS / DAMAGE_DAYS[0], DAYS_POOL, DATA)
    return day, TrueGrid(day), pool_frames(day, AC_LOADS)


def _ac_sced(ac_day, objective):
    from fdia_graph.operator.acsced import AcSced

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
    from fdia_graph.generation.nygrid.network import Network
    from fdia_graph.operator.acsced import ac_shift

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
    from fdia_graph.generation.nygrid.network import Network
    from fdia_graph.generation.nygrid.solver import AcSolver
    from fdia_graph.operator.acsced import ac_power_flow, open_branch

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
    """The least-cost AC dispatch solves the problem the pool solved, so on the true loads (solved, not answered by the
    basepoint) it stays at the stored AC
    points, and a unit inside its bounds and not ramp limited is priced within the movement charge of its cost: the
    balance duals are the AC LMPs in $/MWh."""
    from fdia_graph.operator.acsced import MOVE_USD

    day, grid, frames = ac_day
    sced = _ac_sced(ac_day, "cost")
    limit = sced.flow0 + np.array([it.limit - it.flow for it in day.interfaces]).T
    c, prev = day.case, None
    interior = 0
    for t in range(day.onset, day.onset + 3):
        step = sced.step(
            t, grid.pd[t], frames.qd[t], np.zeros(len(day.interfaces)), limit[t], prev, force=True
        )
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


def test_the_operator_on_a_damaging_day():
    """The least-movement AC operator keeps the benign twin at the AC points, and against the trusting operator the
    damaging attack cuts the attacked interface's true transfer."""
    from fdia_graph.operator.day import operator_day
    from fdia_graph.operator.run import respond

    day = operator_day(DAYS / DAMAGE_DAYS[1], DAYS_POOL, DATA)
    r = respond(day, mode="trusting", closed_loop=True)
    assert np.abs(r.benign.dev).max() == 0.0 and np.nanmax(np.abs(r.benign.iface_change)) == 0.0
    assert np.nanmin(r.attacked.iface_change[:, 0] - r.benign.iface_change[:, 0]) < -50.0


def test_the_least_movement_ac_model_flows_are_the_true_grids(ac_day):
    """Solved on the true loads, the least-movement AC dispatch holds the units' voltages at the AC point's schedule,
    as the true grid's power flow does, so its model's interface flows are the true ones and the closed loop reads no
    difference that is not there."""
    day, grid, frames = ac_day
    sced = _ac_sced(ac_day, "movement")
    limit = sced.flow0 + np.array([it.limit - it.flow for it in day.interfaces]).T
    for t in range(day.onset, day.onset + 3):
        step = sced.step(
            t, grid.pd[t], frames.qd[t], np.zeros(len(day.interfaces)), limit[t], None, force=True
        )
        assert np.abs(step.dev).sum() < 0.5
        assert step.flow is not None
        np.testing.assert_allclose(step.flow, sced.flow0[t], atol=0.1)


def test_the_operator_relieves_an_interface_it_reads_over_its_limit(ac_day):
    """Read 30 MW over its limit, the attacked interface is brought back inside it by moving free units; nuclear
    units hold their schedule (N39)."""
    day, grid, frames = ac_day
    sced = _ac_sced(ac_day, "movement")
    t = day.onset
    limit = sced.flow0[t].copy()
    corr = np.zeros(len(day.interfaces))
    corr[0] = 30.0  # the reading sits 30 MW above the model's flow, which is at the limit
    step = sced.step(t, grid.pd[t], frames.qd[t], corr, limit, None)
    assert step.flow is not None
    assert step.flow[0] + corr[0] <= limit[0] + step.unrelieved[0] + 0.1
    assert step.unrelieved[0] < 1.0 and np.abs(step.dev[sced.free]).sum() > 1.0
    assert np.abs(step.dev[sced.nuclear]).max(initial=0.0) < 1e-3


def test_recovery_after_a_dispatch_is_ramp_limited(ac_day):
    """With nothing left to answer the dispatch returns to the basepoint, but a unit 40 MW above it comes back no
    faster than its five-minute ramp."""
    day, grid, frames = ac_day
    sced = _ac_sced(ac_day, "movement")
    t = day.onset
    room = (day.case.pmax - day.unit_p[t]) > 60
    slow = sced.free & (day.unit_p[t] > 0.5) & (day.unit_p[t - 1] > 0.5) & room & (sced.ramp < 30)
    g = int(np.flatnonzero(slow)[0])
    prev = np.zeros(len(sced.free))
    prev[g] = 40.0
    limit = np.full(len(day.interfaces), np.nan)
    step = sced.step(t, grid.pd[t], frames.qd[t], np.zeros(len(day.interfaces)), limit, prev)
    was = day.unit_p[t - 1][g] + 40.0
    assert day.unit_p[t][g] + step.dev[g] >= was - sced.ramp[g] - 1e-3
    assert step.dev[g] > 40.0 - sced.ramp[g] - abs(day.unit_p[t][g] - day.unit_p[t - 1][g]) - 1e-3


def test_the_operator_sheds_load_only_when_allowed_and_only_where_it_relieves(ac_day):
    """An interface read far beyond what redispatch can relieve stays over it while shedding is not yet allowed
    (`SHED_AFTER` scans); once allowed, load is shed only at buses whose shedding takes flow off the interface."""
    from fdia_graph.operator.acsced import SHED_FACTOR, AcSced

    day, grid, frames = ac_day
    load = np.clip(grid.pd, 0.0, None) * day.case.is_ny
    sced = AcSced(
        day.case,
        day.unit_p,
        day.cost,
        day.interfaces,
        (day.vm, day.va),
        (grid.pd, frames.qd),
        frames,
        bus_load=load,
    )
    t = day.onset
    limit = sced.flow0[t].copy()
    corr = np.zeros(len(day.interfaces))
    corr[0] = 3000.0
    held = sced.step(t, grid.pd[t], frames.qd[t], corr, limit, None, shed_ok=False)
    assert held.shed.sum() < 1e-6 and held.unrelieved[0] > 100.0
    shed = sced.step(t, grid.pd[t], frames.qd[t], corr, limit, None, shed_ok=True)
    at = shed.shed > 1e-3
    assert at.any() and (sced._sensitivity(t)[0][at] <= SHED_FACTOR + 1e-9).all()
    assert shed.unrelieved[0] < held.unrelieved[0]


def test_an_ipopt_that_ends_abnormally_is_a_failed_solve(ac_day, monkeypatch):
    """An Ipopt that does not exit normally (killed, out of memory) is a solve without an optimum: `OperatorOpf.solve`
    returns None and the AC operator tries once more from the AC point, rather than ending the study."""
    from pyomo.common.errors import ApplicationError

    day, grid, frames = ac_day
    sced = _ac_sced(ac_day, "cost")
    limit = sced.flow0 + np.array([it.limit - it.flow for it in day.interfaces]).T
    calls = {"n": 0}
    real = sced.opf._ipopt.solve

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ApplicationError("Solver (ipopt) did not exit normally")
        return real(*a, **k)

    monkeypatch.setattr(sced.opf._ipopt, "solve", flaky)
    t = day.onset
    sced._last = (day.vm[t], day.va[t], day.unit_p[t])  # a warm start, so the retry from the AC point runs
    step = sced.step(t, grid.pd[t], frames.qd[t], np.zeros(len(day.interfaces)), limit[t], None, force=True)
    assert calls["n"] >= 2 and np.abs(step.dev).sum() < 0.5
