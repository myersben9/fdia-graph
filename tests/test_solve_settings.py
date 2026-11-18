"""The solve setting of the pool-building step (`SolveSettings`) on the IEEE cases, with pandapower alone:
what the model refuses, the default as the released power flow, and a least-cost OPF pool."""

import numpy as np
import pytest

from fdia_graph.errors import ConfigError, DataConditionError
from fdia_graph.models.config import SolveSettings

pytest.importorskip("pandapower")

PROFILE = np.sin(np.arange(12) / 3.0)


def test_the_setting_takes_a_method_and_an_engine():
    solve = SolveSettings(method="opf", engine="ipopt")
    assert (solve.method, solve.engine) == ("opf", "ipopt")
    with pytest.raises(ConfigError, match="SolveSettings.method must be one of"):
        SolveSettings(method="dc", engine="ipopt")
    with pytest.raises(ConfigError, match="SolveSettings.engine must be one of"):
        SolveSettings(method="opf", engine="matpower")


def test_a_power_flow_in_the_ipopt_engine_is_refused():
    with pytest.raises(ConfigError, match="a power flow runs in pandapower"):
        SolveSettings(method="power_flow", engine="ipopt")


def test_the_default_is_the_power_flow_in_pandapower():
    from fdia_graph.data.ieee_pool import generate_states

    default = generate_states(14, PROFILE, seed=3)
    named = generate_states(14, PROFILE, seed=3, solve=SolveSettings("power_flow", "pandapower"))
    assert default.tobytes() == named.tobytes()


def test_an_opf_pool_in_pandapower():
    """Loads scaled as in the power-flow pool, outputs chosen at least cost: a bus without a unit has
    the same injection, every voltage is within the case's limits, and generation covers the load."""
    from fdia_graph.data.ieee_pool import generate_states
    from fdia_graph.systems.case import Case

    case = Case.from_ieee(14)
    flow = generate_states(14, PROFILE, seed=3)
    opf = generate_states(14, PROFILE, seed=3, solve=SolveSettings("opf", "pandapower"))
    assert opf.shape == flow.shape == (len(PROFILE), 14, 4)
    no_unit = np.setdiff1d(np.arange(14), case.unit_bus)
    assert np.allclose(opf[:, no_unit, 1:3], flow[:, no_unit, 1:3], atol=1e-4)  # MW, MVAr
    assert (opf[..., 0] >= case.vmin - 1e-6).all() and (opf[..., 0] <= case.vmax + 1e-6).all()
    losses = -opf[..., 1].sum(axis=1)  # the columns are consumer-positive
    assert (losses > 0).all() and (losses < 0.1 * case.pd.sum()).all()


def test_the_net_built_from_a_case_solves_the_case():
    """`PandapowerSolver`'s net (one load per bus, one generator per unit) at the case's own loads and
    outputs has the case's own power flow."""
    from fdia_graph.data.pandapower_solver import PandapowerSolver
    from fdia_graph.systems.case import Case

    case = Case.from_ieee(14)
    on = np.ones(len(case.unit_bus), bool)
    point = PandapowerSolver(case).power_flow(case.pd, case.qd, case.unit_p, on, case.unit_vg, np.zeros(0))
    assert np.allclose(point.vm, case.vm, atol=1e-8) and np.allclose(point.va, case.va, atol=1e-6)


def test_unit_costs_of_a_pandapower_net():
    import pandapower.networks as pn

    from fdia_graph.systems.case import Case

    net = pn.case14()
    costs = Case.pandapower_costs(net)
    # four generators, then the external grid: MATPOWER's case14 cost curves
    assert costs.linear.tolist() == [20.0, 40.0, 40.0, 40.0, 20.0]
    assert np.allclose(costs.quadratic, [0.25, 0.01, 0.01, 0.01, 0.0430293])
    assert np.isclose(costs.of(np.full(5, 10.0)), 10 * 160.0 + 100 * costs.quadratic.sum())
    net.poly_cost.drop(net.poly_cost.index[net.poly_cost["et"] == "gen"][:2], inplace=True)
    with pytest.raises(DataConditionError) as refused:
        Case.pandapower_costs(net)
    assert str(refused.value) == (
        "an OPF needs a cost for every generator, and 2 have none in the net's poly_cost table: gen 0, gen 1"
    )


def test_an_opf_pool_of_a_case_with_no_optimum_is_refused():
    """IEEE-57 has no optimum within its own limits at any of these loads."""
    from fdia_graph.data.ieee_pool import generate_states

    with pytest.raises(DataConditionError) as refused:
        generate_states(57, PROFILE[:3], seed=3, solve=SolveSettings("opf", "pandapower"))
    assert str(refused.value) == (
        "IEEE-57: none of the 3 steps has an optimum within the case's limits (each unit's P and Q range, each "
        "bus's voltage range), so an OPF builds no pool of this case at these loads; the power flow (the default "
        "solve setting) does"
    )


def test_a_solver_is_made_for_an_engine_by_name():
    from fdia_graph.data.pandapower_solver import PandapowerSolver
    from fdia_graph.data.solver import Solver
    from fdia_graph.systems.case import Case

    case = Case.from_ieee(14)
    assert type(Solver.from_engine("pandapower", case)) is PandapowerSolver
    with pytest.raises(ValueError, match="solve_engine must be one of"):
        Solver.from_engine("matpower", case)


def test_a_power_flow_that_does_not_converge_is_no_point():
    """Twenty times the load has no power-flow solution: `power_flow` returns None, from a flat start."""
    from fdia_graph.data.pandapower_solver import PandapowerSolver
    from fdia_graph.systems.case import Case

    case = Case.from_ieee(14)
    on = np.ones(len(case.unit_bus), bool)
    solver = PandapowerSolver(case)
    assert solver.power_flow(20 * case.pd, 20 * case.qd, case.unit_p, on, case.unit_vg, np.zeros(0)) is None
