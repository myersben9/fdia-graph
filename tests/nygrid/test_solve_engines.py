"""The two OPF engines against each other: the package's (Pyomo and Ipopt) and pandapower's, on the same
case, loads and costs. Also the choices the New York pool refuses. The comparisons read no NYgrid data
and are skipped where Ipopt cannot start."""

import time

import numpy as np
import pytest

from .inputs import DATA, needs_data

pytest.importorskip("pyomo.environ")
pytest.importorskip("pandapower")

FUEL_PRICE = {"Nuclear": 10.0, "Hydro": 5.0, "Coal": 25.0, "Natural Gas": 35.0, "Import": 30.0}  # $/MWh
OTHER_FUEL_PRICE = 60.0
NEW_YORK_LOAD = 0.7  # of the case's base load: where pandapower's OPF converges (see the test)


@pytest.fixture(scope="module")
def ipopt_runs():
    """Skip a comparison where Ipopt is not installed or cannot start."""
    from fdia_graph.data.ipopt import ipopt
    from fdia_graph.errors import SolverUnavailable

    try:
        ipopt()
    except SolverUnavailable as error:
        pytest.skip(str(error))


def both_engines(case, pd, qd, costs):
    """The OPF of each engine as (point, seconds), the package's first."""
    from fdia_graph.data.solver import Solver

    out = []
    for engine in ("ipopt", "pandapower"):
        solver = Solver.from_engine(engine, case)
        clock = time.perf_counter()
        out.append((solver.opf(pd, qd, costs), time.perf_counter() - clock))
    return out


def report(name, load, ours, theirs):
    """Print the comparison and return (relative cost difference, loss difference MW)."""
    (a, ta), (b, tb) = ours, theirs
    loss_a, loss_b = a.unit_p.sum() - load, b.unit_p.sum() - load
    print(
        f"{name}: cost {a.cost:.3f} (Ipopt) vs {b.cost:.3f} (pandapower) $/h, generation {a.unit_p.sum():.3f} vs "
        f"{b.unit_p.sum():.3f} MW, losses {loss_a:.4f} vs {loss_b:.4f} MW, {ta:.2f} s vs {tb:.2f} s"
    )
    return abs(a.cost - b.cost) / b.cost, abs(loss_a - loss_b)


def ieee(buses):
    """(case, costs) of the IEEE case pandapower ships: its limits and its linear and quadratic cost."""
    import pandapower as pp
    import pandapower.networks as pn

    from fdia_graph.constants.grid import IEEE_CASES
    from fdia_graph.systems.case import Case

    net = getattr(pn, IEEE_CASES[buses])()
    pp.runpp(net)
    return Case.from_pandapower(net), Case.pandapower_costs(net)


@pytest.mark.parametrize("buses", [14, 30, 118])
def test_the_engines_agree_on_an_ieee_case(buses, ipopt_runs):
    """Least cost at the case's base loads and its own cost curve: the objective within 1e-4 (relative),
    the losses within 0.01 MW, every unit within 0.1 MW and every voltage within 0.005 p.u."""
    case, costs = ieee(buses)
    ours, theirs = both_engines(case, case.pd, case.qd, costs)
    assert ours[0] is not None and theirs[0] is not None
    cost_gap, loss_gap = report(f"IEEE-{buses}", case.pd.sum(), ours, theirs)
    assert cost_gap < 1e-4 and loss_gap < 0.01
    assert np.abs(ours[0].unit_p - theirs[0].unit_p).max() < 0.1
    assert np.abs(ours[0].vm - theirs[0].vm).max() < 5e-3


def test_ieee_14_costs_what_pandapower_finds_on_its_own_net(ipopt_runs):
    """IEEE-14 has no branch rating, so pandapower's OPF of the net it ships is the same problem."""
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case14()
    pp.runopp(net, numba=False)
    case, costs = ieee(14)
    ours, theirs = both_engines(case, case.pd, case.qd, costs)
    print(
        f"IEEE-14: {ours[0].cost:.3f} (Ipopt), {theirs[0].cost:.3f} (pandapower), {net.res_cost:.3f} (shipped net)"
    )
    assert abs(ours[0].cost - net.res_cost) < 1e-4 * net.res_cost
    assert abs(theirs[0].cost - net.res_cost) < 1e-4 * net.res_cost


def test_the_engines_agree_on_the_cost_where_units_tie(ipopt_runs):
    """IEEE-89's costs are linear and several units cost the same, so the optimum is not one point: the
    engines agree on the cost and the losses, and may split the output among equal units differently."""
    case, costs = ieee(89)
    assert not costs.quadratic.any()
    ours, theirs = both_engines(case, case.pd, case.qd, costs)
    cost_gap, loss_gap = report("IEEE-89", case.pd.sum(), ours, theirs)
    assert cost_gap < 1e-4 and loss_gap < 0.01


def test_an_ieee_opf_pool_is_the_same_in_both_engines(ipopt_runs):
    """Twelve steps of IEEE-14. Where a unit sits near its lower limit the cost is nearly flat in its output,
    and the engines stop up to 0.3 MW apart on it at a cost equal to 1e-6: hence 0.5 MW, not 0.1."""
    from fdia_graph.data.ieee_pool import generate_states
    from fdia_graph.models.config import SolveSettings

    profile = np.sin(np.arange(12) / 3.0)
    ours = generate_states(14, profile, seed=3, solve=SolveSettings("opf", "ipopt"))
    theirs = generate_states(14, profile, seed=3, solve=SolveSettings("opf", "pandapower"))
    assert ours.shape == theirs.shape == (12, 14, 4)
    gap = np.abs(ours - theirs).max(axis=(0, 1))  # per column: |V| pu, P MW, Q MVAr, theta degrees
    print(f"IEEE-14 OPF pool, largest difference per column: {gap}")
    assert gap[0] < 1e-3 and gap[1] < 0.5 and gap[2] < 0.2 and gap[3] < 0.05


def test_an_ieee_opf_pool_does_not_depend_on_the_workers(ipopt_runs):
    """Every step is solved from the same start, so cutting the steps into chunks changes no byte."""
    from fdia_graph.data.ieee_pool import generate_states
    from fdia_graph.models.config import SolveSettings

    profile, solve = np.sin(np.arange(12) / 3.0), SolveSettings("opf", "ipopt")
    one = generate_states(14, profile, seed=3, solve=solve)
    two = generate_states(14, profile, seed=3, solve=solve, workers=2)
    assert one.tobytes() == two.tobytes()


def test_an_opf_does_not_depend_on_the_solves_before_it(ipopt_runs):
    from fdia_graph.data.solver import Solver

    case, costs = ieee(14)
    solver = Solver.from_engine("ipopt", case)
    first = solver.opf(case.pd, case.qd, costs)
    solver.opf(1.2 * case.pd, 1.2 * case.qd, costs)
    again = solver.opf(case.pd, case.qd, costs)
    assert first.vm.tobytes() == again.vm.tobytes() and first.unit_p.tobytes() == again.unit_p.tobytes()


def test_the_engines_agree_on_the_new_york_case(ipopt_runs):
    """One operating point of the 140-bus case at plain least cost. pandapower's OPF is fragile on this
    case (at the base load and at 0.8, 0.9 and 1.05 of it, it does not converge where Ipopt does), so
    the point is 0.7 of the base load, where it converges."""
    pytest.importorskip("matpowercaseframes")
    from fdia_graph.models.inputs import UnitCosts
    from fdia_graph.systems.case import nygrid_case

    case = nygrid_case()
    linear = np.array([FUEL_PRICE.get(str(fuel), OTHER_FUEL_PRICE) for fuel in case.fuel])
    costs = UnitCosts(linear=linear, quadratic=np.zeros(len(linear)))
    pd, qd = NEW_YORK_LOAD * case.pd, NEW_YORK_LOAD * case.qd
    ours, theirs = both_engines(case, pd, qd, costs)
    assert ours[0] is not None
    assert theirs[0] is not None, "pandapower's OPF did not converge on the New York case at this load"
    cost_gap, loss_gap = report("New York", pd.sum(), ours, theirs)
    assert cost_gap < 1e-4 and loss_gap < 0.01
    assert np.abs(ours[0].unit_p - theirs[0].unit_p).max() < 0.1


REFUSALS = {
    ("power_flow", "pandapower"): (
        "NewYorkSolve: method='power_flow' cannot build the New York pool. PoolBuilder builds by OPF only: the "
        "source model gives hourly generator outputs, at which a power flow leaves voltages far below the case's "
        "limits. A power flow at given outputs is PandapowerSolver(case).power_flow."
    ),
    ("opf", "pandapower"): (
        "NewYorkSolve: method='opf' with engine='pandapower' cannot build the New York pool. The dispatch limits "
        "each unit's change between frames by its ramp and holds the hydro and external units near their schedule "
        "by a penalty, and pandapower's OPF solves plain least cost only: use engine='ipopt'. One operating point "
        "at plain least cost is PandapowerSolver(case).opf."
    ),
}


@pytest.mark.parametrize("choice", list(REFUSALS))
def test_the_new_york_pool_refuses_what_it_cannot_be_solved_with(choice):
    pytest.importorskip("matpowercaseframes")
    from fdia_graph.data.loads import LoadSeries
    from fdia_graph.data.pool import PoolBuilder
    from fdia_graph.errors import ConfigError
    from fdia_graph.models.config import SolveSettings
    from fdia_graph.systems.case import nygrid_case

    case, none = nygrid_case(), np.zeros(0)
    loads = LoadSeries(time=none, p=none, q=none)
    with pytest.raises(ConfigError) as refused:
        PoolBuilder.from_settings(case, loads, None, none, none, SolveSettings(*choice))
    assert str(refused.value) == REFUSALS[choice]


@needs_data
def test_the_new_york_pool_from_settings_is_the_pool_of_the_constructor(ipopt_runs):
    """`PoolBuilder.from_settings` with no setting, and with the builder's own named, builds what
    `PoolBuilder(IpoptSolver(case), ...)` builds, array for array."""
    from fdia_graph.data.dispatch import hourly_dispatch
    from fdia_graph.data.inputs import read_nygrid
    from fdia_graph.data.ipopt_solver import IpoptSolver
    from fdia_graph.data.loads import LoadSeries
    from fdia_graph.data.pool import PoolBuilder
    from fdia_graph.models.config import SolveSettings
    from fdia_graph.systems.case import nygrid_case

    case, hours = nygrid_case(), read_nygrid(DATA, 2019)[:2]
    d = hourly_dispatch(case, hours)
    renew = np.array([h.renew_mw for h in hours])
    half = np.timedelta64(30, "m")
    loads = LoadSeries(
        time=np.array([d.hour[0] + half, d.hour[0] + 2 * half]),
        p=np.stack([d.pd[0] + renew[0], 0.5 * (d.pd[0] + renew[0] + d.pd[1] + renew[1])]),
        q=np.stack([d.qd[0], 0.5 * (d.qd[0] + d.qd[1])]),
    )
    ramp = np.where(case.fuel == "Nuclear", 0.0, np.nan_to_num(case.ramp, nan=np.inf)) * 5
    built = PoolBuilder(IpoptSolver(case), loads, d, renew, ramp).build()
    for solve in (None, SolveSettings("opf", "ipopt")):
        pool = PoolBuilder.from_settings(case, loads, d, renew, ramp, solve).build()
        for name in ("vm", "va", "p", "q", "solve_kind", "unit_p"):
            assert getattr(pool, name).tobytes() == getattr(built, name).tobytes(), name
