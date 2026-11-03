import json
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.io

from fdia_graph.nyfed.grid.case import PQ, PV, REF
from fdia_graph.nyfed.grid.clients import assign_clients
from fdia_graph.nyfed.grid.fixes import externals_at_base, hvdc_links, restore_765kv, split_zone_j
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.grid.nygrid import modify_mpc, nygrid_units, read_npcc
from fdia_graph.nyfed.grid.ratings import line_ratings

NPCC = (
    Path(os.environ.get("NYFED_ROOT", Path(__file__).resolve().parents[2]))
    / "data"
    / "NYgrid"
    / "Data"
    / "npcc.mat"
)
TERRITORIES = NPCC.parents[2] / "territories"
pytestmark = pytest.mark.skipif(not NPCC.exists(), reason="NYgrid checkout missing at data/NYgrid")


@pytest.fixture(scope="module")
def raw():
    return read_npcc(NPCC)


def test_read_npcc_sizes(raw):
    assert len(raw.bus_id) == 140 and len(raw.f) == 227 and len(raw.unit_bus) == 48
    assert raw.is_ny.sum() == 46
    assert set(raw.zone[raw.is_ny]) == set("ABCDEFGHIJK")
    assert (raw.zone[~raw.is_ny] == "").all()


def test_network_matches_matpower_power_flow(raw):
    # The stored npcc state is not a solved point (it carries HVDC transfers the .mat lacks), so solve
    # the case with PYPOWER and check the admittances reproduce the solved injections and flows.
    import pandapower as pp
    from pandapower.converter.pypower import from_ppc

    mpc = scipy.io.loadmat(NPCC, squeeze_me=True, struct_as_record=False)["mpc"]
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
    p, q = Network(raw).injections(vm, va)
    dp = np.abs(p + pnet.res_bus.p_mw.to_numpy()).max()  # pandapower reports consumption as positive
    dq = np.abs(q + pnet.res_bus.q_mvar.to_numpy()).max()
    print(f"max injection mismatch vs pandapower: P {dp:.2e} MW, Q {dq:.2e} MVAr")
    assert dp < 1e-3 and dq < 1e-3


def test_modify_mpc(raw):
    case = modify_mpc(raw)
    pos = {b: i for i, b in enumerate(case.bus_id.tolist())}
    assert len(case.f) == 227 - 3 + 2
    assert case.bus_type[pos[74]] == REF and case.bus_type[pos[78]] == PQ
    assert case.bus_type[pos[39]] == PV and case.bus_type[pos[72]] == PQ
    assert (case.pd >= 0).all()
    assert (case.rating > 0).sum() == 10
    ends = set(zip(case.bus_id[case.f].tolist(), case.bus_id[case.t].tolist()))
    assert (84, 116) not in ends and (38, 77) in ends and (43, 38) in ends


def test_neighbors(raw):
    net = Network(raw)
    one = net.neighbors([0], 1)
    assert 0 in one and len(one) > 1
    assert set(one) <= set(net.neighbors([0], 2))


@pytest.fixture(scope="module")
def nygrid(raw):
    return nygrid_units(modify_mpc(raw), NPCC.parent)


def test_unit_list_matches_nygrid_hourly_case(nygrid):
    # updateOpCond.m builds 271 units for 2019-01-01 00:00 (checked in MATLAB): 227 thermal, 6 nuclear,
    # 10 hydro, 27 external and the HQ import.
    fuel = nygrid.fuel.tolist()
    assert len(fuel) == 271
    assert fuel.count("Nuclear") == 6 and fuel.count("Hydro") == 10 and fuel.count("Import") == 28
    assert nygrid.is_ny[nygrid.unit_bus[: 271 - 28]].all()
    assert not nygrid.is_ny[nygrid.unit_bus[243:-1]].any()


def test_fixes(nygrid):
    pos = {b: i for i, b in enumerate(nygrid.bus_id.tolist())}
    j = split_zone_j(nygrid)
    n81, n82 = (j.unit_bus == pos[81]).sum(), (j.unit_bus == pos[82]).sum()
    print(f"zone J units at 81/82: {n81}/{n82}")
    assert n81 + n82 == 100 and n81 == 50 and j.bus_type[pos[81]] == PV

    line = restore_765kv(nygrid)
    assert len(line.f) == len(nygrid.f) + 1 and (line.f[-1], line.t[-1]) == (pos[47], pos[43])

    dc = hvdc_links(nygrid)
    assert len(dc.dc_f) == 4 and len(dc.f) == len(nygrid.f) - 1

    hour = nygrid.__class__(**{**nygrid.__dict__, "pd": nygrid.pd * 1.3})
    fixed = externals_at_base(hour, nygrid)
    ext = ~fixed.is_ny
    assert np.allclose(fixed.pd[ext], nygrid.pd[ext])
    assert np.isclose(fixed.unit_p[fixed.unit_on].sum(), fixed.pd.sum())


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


def test_line_ratings(nygrid):
    case = line_ratings(restore_765kv(split_zone_j(nygrid)))
    ends = list(zip(case.bus_id[case.f].tolist(), case.bus_id[case.t].tolist()))
    rated = {e: float(r) for e, r in zip(ends, case.rating) if r > 0}
    print(f"{len(rated)} of {len(ends)} branches rated: {rated}")
    # the ten tie ratings of modifyMPC.m and the two single-branch interfaces with a fixed limit
    assert len(rated) == 12
    assert rated[(37, 39)] == 5400 and rated[(74, 78)] == 8450 and rated[(75, 124)] == 2000
    assert (38, 77) not in rated and (47, 43) not in rated  # hourly E-G limit, restored line: no rating
    assert (line_ratings(hvdc_links(nygrid)).rating > 0).sum() == 11  # 81-125 goes with the HVDC links
