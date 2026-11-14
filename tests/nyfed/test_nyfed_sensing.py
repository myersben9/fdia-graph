import json

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp
from nyfed_inputs import DATA, TERRITORIES, needs_data

from fdia_graph.engine.sensing.measure import Measurement
from fdia_graph.engine.sensing.meters import (
    ACCURACY,
    IA,
    JITTER_FRAC,
    KINDS,
    PF,
    VA,
    I,
    P,
    V,
    _min_pmus,
    plan_meters,
)
from fdia_graph.engine.sensing.noise import NoiseModel
from fdia_graph.federated.utilities import assign_clients
from fdia_graph.generation.nygrid.config import Settings, rng
from fdia_graph.generation.nygrid.fixes import restore_765kv, split_zone_j
from fdia_graph.generation.nygrid.network import Network
from fdia_graph.generation.nygrid.npcc import modify_mpc, nygrid_units, read_npcc

pytestmark = needs_data


@pytest.fixture(scope="module")
def grid():
    case = restore_765kv(split_zone_j(nygrid_units(modify_mpc(read_npcc(DATA / "npcc.mat")), DATA)))
    territories = json.load(open(TERRITORIES / "nys_electric_territories.geojson", encoding="utf-8"))
    own = assign_clients(case, territories, TERRITORIES / "ny_bus_locations.csv")
    net = Network(case)
    return net, own, plan_meters(net, own, rng(Settings(), "meters"))


def _identifiable(h: np.ndarray, cols: np.ndarray, free: np.ndarray | None = None) -> bool:
    """True when every null direction of `h` leaves the state columns `cols` unchanged, apart from the
    direction `free` over them (a common angle shift, when no meter reads an absolute angle)."""
    sub = scipy.linalg.null_space(h, rcond=1e-10)[cols]
    if free is not None:
        u = free / np.linalg.norm(free)
        sub = sub - np.outer(u, u @ sub)
    return bool((np.linalg.svd(sub, compute_uv=False) < 1e-6).all()) if sub.size else True


def test_min_pmus_small_graphs():
    path = sp.diags([1, 1, 1, 1, 1], 1, shape=(6, 6))  # six buses in a line need two PMUs
    assert _min_pmus((path + path.T).tocsr()).tolist() == [1, 4]
    star = sp.csr_matrix(([1] * 4, ([0] * 4, [1, 2, 3, 4])), shape=(5, 5))
    assert _min_pmus((star + star.T).tocsr()).tolist() == [0]


def test_meter_plan(grid):
    net, own, plan = grid
    case = net.case
    tab = np.zeros((len(own.names), len(KINDS)), int)
    np.add.at(tab, (plan.owner, plan.kind), 1)
    print({n: dict(zip(KINDS, row.tolist())) for n, row in zip(own.names.tolist(), tab)})
    ny = np.flatnonzero(own.bus_client >= 0)
    # every meter on a NY bus, owned by that bus's client
    assert (plan.owner >= 0).all() and (plan.owner == own.bus_client[plan.bus]).all()
    assert sorted(plan.bus[plan.kind == P].tolist()) == ny.tolist()
    # a flow meter at every branch end on a NY bus, at that end's bus, so each side of a tie meters its own end
    ends = plan.kind == PF
    assert (
        plan.bus[ends] == np.where(plan.end[ends] == 0, case.f[plan.branch[ends]], case.t[plan.branch[ends]])
    ).all()
    n_ends = (own.bus_client[case.f] >= 0).sum() + (own.bus_client[case.t] >= 0).sum()
    assert ends.sum() == n_ends
    tie = (
        own.tie_lines[plan.branch[ends]]
        & (own.bus_client[case.f] >= 0)[plan.branch[ends]]
        & (own.bus_client[case.t] >= 0)[plan.branch[ends]]
    )
    assert len(set(plan.owner[ends][tie].tolist())) > 1
    # PMUs see every NY bus through their own bus or a branch current
    pmu = plan.bus[plan.kind == VA]
    seen = np.zeros(len(case.bus_id), bool)
    seen[net.neighbors(pmu, 1)] = True
    assert seen[ny].all() and np.isin(pmu, plan.bus[plan.kind == V]).all()
    cur = plan.kind == I
    assert np.isin(plan.bus[cur], pmu).all() and cur.sum() == (plan.kind == IA).sum()
    # one owner per meter, one row per (kind, bus, branch, end)
    keys = set(zip(plan.kind.tolist(), plan.bus.tolist(), plan.branch.tolist(), plan.end.tolist()))
    assert len(keys) == len(plan.kind)


def test_jacobian_matches_finite_differences(grid):
    net, _, plan = grid
    meas, case = Measurement(net, plan), net.case
    vm, va, n = case.vm.astype(float), case.va.astype(float), len(case.vm)
    jac = meas.jacobian(vm, va)
    pick = np.r_[np.flatnonzero(case.is_ny)[:8], n + np.flatnonzero(case.is_ny)[:8]]
    eps = 1e-6
    for j in pick:
        x = np.r_[vm, va]
        x[j] += eps
        up = meas.h(x[:n], x[n:])
        x[j] -= 2 * eps
        fd = (up - meas.h(x[:n], x[n:])) / (2 * eps)
        assert np.allclose(jac[:, j], fd, rtol=1e-5, atol=1e-3), j


def test_observability(grid):
    net, own, plan = grid
    case, meas, noise = net.case, Measurement(net, plan), NoiseModel(plan)
    n, ny = len(case.vm), np.flatnonzero(own.bus_client >= 0)
    z = meas.h(case.vm, case.va)
    hw = meas.jacobian(case.vm, case.va) / noise.sigmas(z)[:, None]
    assert _identifiable(hw, np.r_[ny, n + ny])
    for c, name in enumerate(own.names):
        rows, buses = plan.owner == c, np.flatnonzero(own.bus_client == c)
        angle = np.isin(plan.kind[rows], [VA, IA]).any()
        free = None if angle else np.r_[np.zeros(len(buses)), np.ones(len(buses))]
        print(
            f"{name}: {rows.sum()} meters, locally observable {_identifiable(hw[rows], np.r_[buses, n + buses], free)}"
        )
    # a client's view holds only its own meters
    sub = meas.for_client(0)
    assert np.allclose(sub.h(case.vm, case.va), z[plan.owner == 0])


def test_noise(grid):
    net, _, plan = grid
    case, noise = net.case, NoiseModel(plan)
    z = np.tile(Measurement(net, plan).h(case.vm, case.va), (4000, 1))
    scans = noise.sample(z, rng(Settings(), "test-noise"))
    err = scans - z
    # the bias is the same every scan, the jitter is JITTER_FRAC of the class
    bias = np.where(noise.relative, z[0] * plan.bias, plan.bias)
    assert np.allclose(err.mean(0), bias, atol=4 * err.std(0).max() / np.sqrt(len(z)))
    jitter = JITTER_FRAC * plan.sigma_rel * np.abs(z[0]) + np.where(
        noise.relative, plan.sigma_abs, JITTER_FRAC * plan.sigma_abs
    )
    assert np.allclose(err.std(0) / jitter, 1.0, atol=0.06)
    # heteroscedastic: a relative meter's class grows with its reading
    flows = plan.kind == PF
    assert np.allclose(noise.sigmas(z[0])[flows], ACCURACY[PF][0] * np.abs(z[0, flows]) + ACCURACY[PF][1])
    # across meters of one kind, bias and jitter add to the class
    total = (err[0] / noise.sigmas(z[0]))[flows]
    print(f"flow error / class sigma: std {total.std():.2f} over {flows.sum()} meters")
    assert 0.7 < total.std() < 1.3
