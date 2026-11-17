"""An IEEE case as a `Case`: `Case.from_pandapower` against the pandapower net it reads, what it
refuses, and the `Network` admittances against pandapower's `makeYbus` on the same net."""

import numpy as np
import pytest

pytest.importorskip("pandapower")

import pandapower as pp  # noqa: E402
import pandapower.networks as pn  # noqa: E402
from pandapower.pypower.makeYbus import makeYbus  # noqa: E402

from fdia_graph.constants.grid import IEEE_CASES, BusType  # noqa: E402
from fdia_graph.errors import DataConditionError  # noqa: E402
from fdia_graph.systems.case import Case  # noqa: E402
from fdia_graph.systems.network import Network  # noqa: E402

# makeYbus sums a bus's from-side and to-side branch admittances apart and then adds them; a `Network`
# adds every term in one pass unless it is asked for MATPOWER's order. The two orders round differently
# on a few diagonal entries (IEEE-300: 27 of 1,118, at most 5.7e-14 pu).
YBUS_TOLERANCE = 1e-11


def solved(system: int):
    net = getattr(pn, IEEE_CASES[system])()
    pp.runpp(net)
    return net


@pytest.mark.parametrize("system", sorted(IEEE_CASES))  # all eight: pandapower ships every case
def test_the_network_of_an_ieee_case_has_makeybus_admittances(system):
    net = solved(system)
    case = Case.from_pandapower(net)
    ybus, yf, yt = makeYbus(net._ppc["baseMVA"], net._ppc["bus"], net._ppc["branch"])
    ours = Network(case, matpower_order=True)  # the generator's
    assert np.array_equal(ours.yf.toarray(), yf.toarray())  # bit for bit: the flows and currents of a state
    assert np.array_equal(ours.yt.toarray(), yt.toarray())
    assert np.array_equal(ours.ybus.toarray(), ybus.toarray())  # and every local solve of an attack
    assert ours.ybus.format == ybus.format == "csc"  # held by column, as a product over it runs
    one_pass = Network(case)
    assert np.abs(one_pass.ybus.toarray() - ybus.toarray()).max() < YBUS_TOLERANCE
    assert np.array_equal(one_pass.yf.toarray(), yf.toarray())


@pytest.mark.parametrize("system", [14, 118, 300])
def test_a_case_keeps_the_nets_buses_branches_loads_and_units(system):
    net = solved(system)
    case = Case.from_pandapower(net)
    lines, trafos = net.line, net.trafo
    assert case.bus_id.tolist() == net.bus["name"].astype(int).tolist()
    assert case.f.tolist() == [*lines.from_bus, *trafos.hv_bus]  # lines, then transformers
    assert case.t.tolist() == [*lines.to_bus, *trafos.lv_bus]
    loaded = np.flatnonzero((case.pd != 0) | (case.qd != 0))
    assert loaded.tolist() == net.load.bus.tolist()  # one load element per loaded bus, in bus order
    assert case.pd[loaded].tolist() == net.load.p_mw.tolist()
    assert np.flatnonzero(case.bus_type == BusType.REF).tolist() == net.ext_grid.bus.tolist()
    assert np.flatnonzero(case.bus_type == BusType.PV).tolist() == sorted(net.gen.bus.tolist())
    assert case.unit_bus.tolist() == [*net.gen.bus, *net.sgen.bus, *net.ext_grid.bus]
    assert np.array_equal(case.vm, net.res_bus.vm_pu.to_numpy())
    assert (case.g != 0).sum() == {14: 0, 118: 4, 300: 18}[system]  # transformers with iron losses


# ---- what a Case does not model is refused, by the element's name ----------------------------------
def _switch_off(table: str):
    def edit(net):
        net[table].loc[net[table].index[1], "in_service"] = False

    return edit


def _scale_loads(net):
    net.load["scaling"] = 0.5


def _live_sgen_off(net):
    pp.create_sgen(net, 3, p_mw=5.0, in_service=False)


REFUSED = {
    "a bus out of service": (_switch_off("bus"), "has bus 1 out of service"),
    "a line out of service": (_switch_off("line"), "has line 1 out of service"),
    "a transformer out of service": (_switch_off("trafo"), "has trafo 1 out of service"),
    "a load out of service": (_switch_off("load"), "has load 1 out of service"),
    "a generator out of service": (_switch_off("gen"), "has gen 1 out of service"),
    "a producing static generator out of service": (_live_sgen_off, "has sgen 0 out of service"),
    "a load scaling": (_scale_loads, r"scales load 0, load 1, .* \(`scaling` is not 1\)"),
    "an impedance element": (
        lambda net: pp.create_impedance(net, 0, 1, 0.01, 0.05, 100.0),
        "has rows in impedance, which a Case does not model",
    ),
    "an extended ward": (
        lambda net: pp.create_xward(net, 3, 1.0, 0.5, 0.1, 0.1, 0.01, 0.05, 1.0),
        "has rows in xward, which a Case does not model",
    ),
    "a ward": (lambda net: pp.create_ward(net, 3, 1.0, 0.5, 0.1, 0.1), "has rows in ward"),
    "a three-winding transformer table": (
        lambda net: net.trafo3w.__setitem__("hv_bus", [0]),
        "has rows in trafo3w",
    ),
    "a DC line": (
        lambda net: pp.create_dcline(net, 0, 1, 1.0, 0.0, 0.0, 1.0, 1.0),
        "has rows in dcline",
    ),
    "a switch": (lambda net: pp.create_switch(net, 0, 0, et="l", closed=True), "has rows in switch"),
}
# a row in the table is enough for these two: the power flow is not run again on it
ROWS_ONLY = ("a three-winding transformer table", "a DC line")


@pytest.mark.parametrize("what", sorted(REFUSED))
def test_a_net_holding_what_a_case_does_not_model_is_refused(what):
    edit, message = REFUSED[what]
    net = solved(14)
    edit(net)
    if what not in ROWS_ONLY:
        pp.runpp(net)
    with pytest.raises(DataConditionError, match=f"^the pandapower net {message}"):
        Case.from_pandapower(net)


def test_a_branch_switched_off_in_the_solved_case_is_refused():
    net = solved(14)
    net._ppc["branch"][2, 10] = 0  # BR_STATUS
    with pytest.raises(DataConditionError, match=r"has 1 branches switched off \(status 0\)"):
        Case.from_pandapower(net)


def test_a_solved_case_of_another_size_is_refused():
    net = solved(14)
    net._ppc["bus"] = np.vstack([net._ppc["bus"], net._ppc["bus"][-1:]])  # an auxiliary bus
    with pytest.raises(DataConditionError, match="has 14 buses and its solved case 15$"):
        Case.from_pandapower(net)


def test_what_a_case_models_is_read():
    """Two loads on one bus are one bus load, and an idle static generator out of service is an idle
    unit (IEEE-200 has eleven)."""
    net = pn.case14()
    pp.create_load(net, 1, 5.0, 1.0)
    pp.create_sgen(net, 3, p_mw=0.0, in_service=False)
    pp.runpp(net)
    case = Case.from_pandapower(net)
    assert case.pd.sum() == net.load.p_mw.sum() and case.pd[1] == net.load.p_mw[net.load.bus == 1].sum()
    assert len(case.unit_bus) == len(net.gen) + 1 + len(net.ext_grid)
    assert (solved(200).sgen.in_service == False).sum() == 11  # noqa: E712
    Case.from_pandapower(solved(200))
