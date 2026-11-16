"""The New York grid's case file: what it holds, its exact round trip through the MATPOWER reader and writer, the
recipe that made it from NYgrid, the two case variants and the utility of each bus."""

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from fdia_graph.errors import DataConditionError
from fdia_graph.federated.utilities import assign_clients
from fdia_graph.generation.nygrid.case import CASE_FILE, HQ_BUS, PQ, PV, REF, Case, nygrid_case
from fdia_graph.generation.nygrid.fixes import externals_at_base, hvdc_links
from fdia_graph.generation.nygrid.network import Network

from .inputs import DATA, TERRITORIES, needs_data, needs_territories
from .test_case_file import differing

RECIPE = Path(__file__).resolve().parents[2] / "pipelines" / "newyork" / "make_case.py"


@pytest.fixture(scope="module")
def nygrid():
    return nygrid_case()


@pytest.fixture(scope="module")
def make_case():
    """The recipe script, loaded as a module."""
    spec = importlib.util.spec_from_file_location("make_case", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- what the file holds ----


def test_case_sizes(nygrid):
    assert len(nygrid.bus_id) == 140 and len(nygrid.f) == 227 and len(nygrid.unit_bus) == 271
    assert len(nygrid.dc_f) == 0  # the release keeps the AC branch 81-125, no HVDC link
    assert nygrid.is_ny.sum() == 46
    assert set(nygrid.zone[nygrid.is_ny]) == set("ABCDEFGHIJK")
    assert (nygrid.zone[~nygrid.is_ny] == "").all()


def test_case_has_nygrid_modifications(nygrid):
    # modifyMPC.m: the bus types, positive loads, three PJM-IESO lines out, two E-G lines in, ten tie ratings
    pos = {b: i for i, b in enumerate(nygrid.bus_id.tolist())}
    assert nygrid.bus_type[pos[74]] == REF and nygrid.bus_type[pos[78]] == PQ
    assert nygrid.bus_type[pos[39]] == PV and nygrid.bus_type[pos[72]] == PQ
    assert (nygrid.pd >= 0).all()
    assert (nygrid.rating > 0).sum() == 10
    ends = set(zip(nygrid.bus_id[nygrid.f].tolist(), nygrid.bus_id[nygrid.t].tolist()))
    assert (84, 116) not in ends and (38, 77) in ends and (43, 38) in ends


def test_unit_list_matches_nygrid_hourly_case(nygrid):
    # updateOpCond.m builds 271 units for 2019-01-01 00:00 (checked in MATLAB): 227 thermal, 6 nuclear,
    # 10 hydro, 27 external and the HQ import.
    fuel = nygrid.fuel.tolist()
    assert fuel.count("Nuclear") == 6 and fuel.count("Hydro") == 10 and fuel.count("Import") == 28
    assert nygrid.is_ny[nygrid.unit_bus[: 271 - 28]].all()
    assert not nygrid.is_ny[nygrid.unit_bus[243:-1]].any()
    assert nygrid.bus_id[nygrid.unit_bus[-1]] == HQ_BUS and (nygrid.pmin[-1], nygrid.pmax[-1]) == (
        -1100,
        1500,
    )
    # ramps: every thermal unit has one, nuclear 0, hydro and imports none
    assert np.isnan(nygrid.ramp).sum() == 38 and (nygrid.ramp[nygrid.fuel == "Nuclear"] == 0).all()
    assert nygrid.unit_on.all()


def test_case_has_the_ac_fixes(nygrid):
    pos = {b: i for i, b in enumerate(nygrid.bus_id.tolist())}
    n81, n82 = (nygrid.unit_bus == pos[81]).sum(), (nygrid.unit_bus == pos[82]).sum()
    print(f"zone J units at 81/82: {n81}/{n82}")
    assert n81 + n82 == 100 and n81 == 50 and nygrid.bus_type[pos[81]] == PV  # zone J is split
    assert (nygrid.f[-1], nygrid.t[-1]) == (pos[47], pos[43])  # the 765 kV Massena-Marcy line
    assert (nygrid.r[-1], nygrid.x[-1]) == (0.004, 0.0085)


# ---- the reader, and the writer of the recipe script ----


def test_round_trip_is_exact(make_case, nygrid, tmp_path):
    make_case.write_case(nygrid, tmp_path / "again.m", "the New York grid")
    assert differing(nygrid, Case.from_matpower(tmp_path / "again.m")) == []


@pytest.mark.parametrize("name", ["2nd.m", "my-case.m", "case nygrid.m"])
def test_a_file_name_that_is_not_a_matlab_function_is_refused(make_case, nygrid, tmp_path, name):
    with pytest.raises(DataConditionError, match="MATLAB function"):
        make_case.write_case(nygrid, tmp_path / name, "a title")


def test_round_trip_keeps_hvdc_links_nan_inf_and_quotes(make_case, nygrid, tmp_path):
    odd = hvdc_links(nygrid)
    ramp, qmax, name = odd.ramp.copy(), odd.qmax.copy(), odd.name.copy()
    ramp[0], qmax[0], qmax[1], name[0] = np.inf, np.inf, -np.inf, "O'NEILL"
    tap, pd = odd.tap.copy(), odd.pd.copy()
    tap[3], pd[1] = 0.975, 0.1 + 0.2
    odd = dataclasses.replace(odd, ramp=ramp, qmax=qmax, name=name, tap=tap, pd=pd)
    make_case.write_case(odd, tmp_path / "odd.m", "two lines\n\nof title")
    back = Case.from_matpower(tmp_path / "odd.m")
    assert differing(odd, back) == []
    assert len(back.dc_f) == 4 and np.isnan(back.ramp).any() and np.isinf(back.qmax[:2]).all()
    text = (tmp_path / "odd.m").read_text(encoding="utf-8")
    assert text.startswith("function mpc = odd\n% two lines\n%\n% of title\n") and "mpc.dcline = [" in text
    assert "NaN" in text and "-Inf" in text and "nan" not in text


@needs_data
def test_recipe_reproduces_the_file(make_case, nygrid, tmp_path):
    # pipelines/newyork/make_case.py builds the case from NYgrid's Data/ folder: the same Case, and the same file
    built = make_case.recipe(DATA)
    assert differing(built, nygrid) == []
    make_case.write_case(built, tmp_path / CASE_FILE.name, make_case.TITLE)
    assert (tmp_path / CASE_FILE.name).read_text(encoding="utf-8") == CASE_FILE.read_text(encoding="utf-8")


@needs_data
def test_recipe_steps(make_case):
    raw = make_case.read_npcc(DATA / "npcc.mat")
    assert len(raw.bus_id) == 140 and len(raw.f) == 227 and len(raw.unit_bus) == 48
    modified = make_case.modify_mpc(raw)
    assert len(modified.f) == 227 - 3 + 2
    units = make_case.nygrid_units(modified, DATA)
    assert len(units.unit_bus) == 271
    pos = {b: i for i, b in enumerate(units.bus_id.tolist())}
    assert (make_case.split_zone_j(units).unit_bus == pos[81]).sum() == 50
    assert len(make_case.restore_765kv(units).f) == len(units.f) + 1


# ---- the network, the variants, the clients and the ratings ----


@needs_data
def test_network_matches_matpower_power_flow(make_case):
    # The stored npcc state is not a solved point (it carries HVDC transfers the .mat lacks), so solve
    # the case with PYPOWER and check the admittances reproduce the solved injections and flows.
    import pandapower as pp
    import scipy.io
    from pandapower.converter.pypower import from_ppc

    npcc = DATA / "npcc.mat"
    mpc = scipy.io.loadmat(npcc, squeeze_me=True, struct_as_record=False)["mpc"]
    branch = mpc.branch.astype(float)
    branch[branch[:, 5] == 0, 5] = 9999.0  # from_ppc fails on zero ratings; ratings do not enter a power flow
    ppc = {
        "version": "2",
        "baseMVA": float(mpc.baseMVA),
        "bus": mpc.bus.astype(float),
        "branch": branch,
        "gen": mpc.gen.astype(float),
    }
    pnet = from_ppc(ppc, f_hz=60, validate_conversion=False)
    pp.runpp(pnet, tolerance_mva=1e-9)
    vm, va = pnet.res_bus.vm_pu.to_numpy(), pnet.res_bus.va_degree.to_numpy()
    p, q = Network(make_case.read_npcc(npcc)).injections(vm, va)
    dp = np.abs(p + pnet.res_bus.p_mw.to_numpy()).max()  # pandapower reports consumption as positive
    dq = np.abs(q + pnet.res_bus.q_mvar.to_numpy()).max()
    print(f"max injection mismatch vs pandapower: P {dp:.2e} MW, Q {dq:.2e} MVAr")
    assert dp < 1e-3 and dq < 1e-3


def test_neighbors(nygrid):
    net = Network(nygrid)
    one = net.neighbors([0], 1)
    assert 0 in one and len(one) > 1
    assert set(one) <= set(net.neighbors([0], 2))


def test_variants(nygrid):
    dc = hvdc_links(nygrid)
    assert len(dc.dc_f) == 4 and len(dc.f) == len(nygrid.f) - 1

    hour = dataclasses.replace(nygrid, pd=nygrid.pd * 1.3)
    fixed = externals_at_base(hour, nygrid)
    ext = ~fixed.is_ny
    assert np.allclose(fixed.pd[ext], nygrid.pd[ext])
    assert np.isclose(fixed.unit_p[fixed.unit_on].sum(), fixed.pd.sum())


@needs_territories
def test_assign_clients(nygrid):
    territories = json.load(open(TERRITORIES / "nys_electric_territories.geojson", encoding="utf-8"))
    own = assign_clients(nygrid, territories, TERRITORIES / "ny_bus_locations.csv")
    counts = np.bincount(own.bus_client[own.bus_client >= 0], minlength=len(own.names))
    print(dict(zip(own.names.tolist(), counts.tolist())))
    assert (own.bus_client[nygrid.is_ny] >= 0).all() and (own.bus_client[~nygrid.is_ny] == -1).all()
    assert len(own.names) == 7 and (counts >= 1).all()  # every utility is a client (N10)

    def owner(name):
        (k,) = set(own.bus_client[nygrid.name == name])
        return own.names[k]

    for name, utility in [
        ("MILLWOOD", "ConEd"),
        ("CE UG", "ConEd"),
        ("GOETHALS", "ConEd"),
        ("NORTHPORT", "LIPA"),
        ("NIAGARA W", "NGRID"),
        ("NIAGARA E", "NGRID"),
    ]:
        assert owner(name) == utility, name
    assert owner("CE UG") == owner("GOETHALS")
    assert (own.tie_lines == (own.bus_client[nygrid.f] != own.bus_client[nygrid.t])).all()
    ny_ties = own.tie_lines & nygrid.is_ny[nygrid.f] & nygrid.is_ny[nygrid.t]
    assert ny_ties.any()
