import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fdia_graph.nyfed.grid.case import REF
from fdia_graph.nyfed.grid.fixes import restore_765kv, split_zone_j
from fdia_graph.nyfed.grid.interfaces import hourly_interfaces
from fdia_graph.nyfed.grid.nygrid import modify_mpc, nygrid_units, read_npcc
from fdia_graph.nyfed.operator.sced import DcSced, outage_factors, shift_factors

ROOT = Path(
    os.environ.get("NYFED_ROOT", Path(__file__).resolve().parents[2])
)  # the nyfed inputs, data/ and out/
DATA = ROOT / "data" / "NYgrid" / "Data"
DAY = ROOT / "out" / "single_episode" / "sd_south_2019-07-05.h5"
POOL = ROOT / "out" / "pool_2019_v3.h5"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="NYgrid checkout missing at data/NYgrid")


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


@pytest.mark.skipif(not (DAY.exists() and POOL.exists()), reason="single-episode file or pool missing")
def test_a_day_under_attack_against_its_benign_twin():
    """On the 5 July file the benign twin needs nothing, while the attacked run follows the pretended load
    drop and regulation covers exactly the generation the dispatch left out."""
    from fdia_graph.nyfed.operator.response import operator_day, respond

    r = respond(operator_day(DAY, POOL, DATA))
    a, b = r.attacked, r.benign
    assert np.abs(b.dev).max() < 1e-3 and np.abs(b.reg).max() < 1e-3
    w = slice(r.onset, r.onset + r.length)
    assert -a.dev[w].sum(axis=1).min() > 300  # the pretended drop reaches about 370 MW
    np.testing.assert_allclose(a.reg.sum(axis=1), -a.dev.sum(axis=1), atol=1e-6)
