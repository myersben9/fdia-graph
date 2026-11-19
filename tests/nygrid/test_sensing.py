"""The meter plan on the New York case and its readings: each meter owned by the client of its bus, the
measurement Jacobian against finite differences, observability and the meter noise. On the packaged
case file with made-up clients: no NYgrid data."""

from dataclasses import replace

import numpy as np
import pytest
import scipy.linalg

from fdia_graph.constants.meters import JITTER_FRAC, KIND_ACCURACY, MeterKind
from fdia_graph.data.config import Settings, rng
from fdia_graph.models.config import MeterSettings
from fdia_graph.sensing.measure import Measurement
from fdia_graph.sensing.meters import MeterPlan
from fdia_graph.sensing.noise import NoiseModel


@pytest.fixture(scope="module")
def grid():
    """(the network, each bus's client, a meter at every bus and branch with its owner)"""
    pytest.importorskip("matpowercaseframes")  # the packaged case file's reader (the [ny] extra)
    from fdia_graph.systems.case import nygrid_case
    from fdia_graph.systems.network import Network

    case = nygrid_case()
    bus_client = np.where(case.is_ny, np.arange(len(case.bus_id)) % 3, -1)
    plan = MeterPlan.from_coverage(case, MeterSettings(1.0, 1.0, 1.0), rng(Settings(), "meters"))
    return Network(case), bus_client, replace(plan, owner=bus_client[plan.bus])


def _identifiable(h: np.ndarray, cols: np.ndarray) -> bool:
    """True when every null direction of `h` leaves the state columns `cols` unchanged."""
    sub = scipy.linalg.null_space(h, rcond=1e-10)[cols]
    return bool((np.linalg.svd(sub, compute_uv=False) < 1e-6).all()) if sub.size else True


def test_meter_plan(grid):
    net, bus_client, plan = grid
    case = net.case
    # an injection meter at every bus, a flow meter at every branch's from end, at that end's bus
    assert sorted(plan.bus[plan.kind == MeterKind.P].tolist()) == list(range(len(case.bus_id)))
    flows = plan.kind == MeterKind.PF
    assert (plan.end[flows] == 0).all() and (plan.bus[flows] == case.f[plan.branch[flows]]).all()
    # a PMU's current phasors sit at its own bus, real and imaginary part alike
    cur = plan.kind == MeterKind.IR
    assert np.isin(plan.bus[cur], plan.bus[plan.kind == MeterKind.VA]).all()
    assert cur.sum() == (plan.kind == MeterKind.II).sum()
    # one row per (kind, bus, branch, end), owned by the client of its bus
    keys = set(zip(plan.kind.tolist(), plan.bus.tolist(), plan.branch.tolist(), plan.end.tolist()))
    assert len(keys) == len(plan.kind) and np.array_equal(plan.owner, bus_client[plan.bus])


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
    net, bus_client, plan = grid
    case, meas, noise = net.case, Measurement(net, plan), NoiseModel(plan)
    n = len(case.vm)
    z = meas.h(case.vm, case.va)
    hw = meas.jacobian(case.vm, case.va) / noise.sigmas(z)[:, None]
    assert _identifiable(hw, np.arange(2 * n))
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
    # heteroscedastic: a relative meter's class grows with its reading
    flows = plan.kind == MeterKind.PF
    assert np.allclose(
        noise.sigmas(z[0])[flows],
        KIND_ACCURACY[MeterKind.PF][0] * np.abs(z[0, flows]) + KIND_ACCURACY[MeterKind.PF][1],
    )
    jitter = JITTER_FRAC * plan.sigma_rel * np.abs(z[0]) + np.where(
        noise.relative, plan.sigma_abs, JITTER_FRAC * plan.sigma_abs
    )
    assert np.allclose(err.std(0)[flows] / jitter[flows], 1.0, atol=0.06)
    # across meters of one kind, bias and jitter add to the class
    total = (err[0] / noise.sigmas(z[0]))[flows]
    assert 0.7 < total.std() < 1.3
