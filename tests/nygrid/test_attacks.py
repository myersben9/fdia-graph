"""The New York attacker on the real grid: its area and edge rules, the false state of a pretended load, the
ramp built on it (within the natural load swing, quiet to the residual test), and the bounds it reads from the pool."""

import json

import numpy as np
import pytest
from scipy.stats import chi2

from fdia_graph.constants.meters import MeterKind
from fdia_graph.constants.newyork import SolveKind
from fdia_graph.data.config import Settings, rng
from fdia_graph.data.pool import read_pool
from fdia_graph.engine.attacks.attacker import ABSORB_CAP, Attacker, natural_steps
from fdia_graph.engine.attacks.ramp import Ramp
from fdia_graph.sensing.measure import Measurement
from fdia_graph.sensing.noise import NoiseModel, wrap_angles
from fdia_graph.systems.case import nygrid_case
from fdia_graph.systems.network import Network
from fdia_graph.systems.ownership import assign_clients

from .inputs import POOL_V1, TERRITORIES, needs_data, needs_pool_v1

LENGTH = 16
MARCH = 18284  # a frame of 2019-03-05 11:40, where `natural_steps` is checked by hand
pytestmark = [needs_data, needs_pool_v1]


@pytest.fixture(scope="module")
def pool():
    return read_pool(POOL_V1)


@pytest.fixture(scope="module")
def setup(pool):
    """The case, ownership, meter plan, an attacker, a short window of the real 2019 pool (the longest
    solved run, in summer), and the operator's observable state (the buses any meter sees), for the WLS
    residual test."""
    case = nygrid_case()
    territories = json.load(open(TERRITORIES / "nys_electric_territories.geojson", encoding="utf-8"))
    own = assign_clients(case, territories, TERRITORIES / "ny_bus_locations.csv")
    net = Network(case)
    plan = plan_meters(net, own)
    meas = Measurement(net, plan)
    att = Attacker(net, meas)
    sol = np.flatnonzero(pool.solve_kind == SolveKind.SOLVED)
    w = max(np.split(sol, np.flatnonzero(np.diff(sol) != 1) + 1), key=len)[:LENGTH]
    vm, va = pool.vm[w].copy(), pool.va[w].copy()
    n = len(vm[0])
    mb = plan.branch[plan.branch >= 0]
    obs = np.unique(np.r_[plan.bus, case.f[mb], case.t[mb]])  # the buses a meter sees
    free_cols = np.array(sorted({int(b) for b in obs} | {n + int(b) for b in obs if b != att.slack}))
    return att, own, plan, meas, NoiseModel(plan), (int(w[0]), vm, va), n, free_cols


def plan_meters(net, own):
    from fdia_graph.sensing.meters import plan_meters as pm

    return pm(net, own, rng(Settings(), "meters"))


def _wls_j(meas, noise, free_cols, n, z, vm0, va0, iters=40):
    """The weighted residual J after a WLS state estimate over the observable island (external buses,
    which no meter sees, are held true). A consistent measurement leaves J at the noise level; an
    inconsistent one leaves it far larger. Gauss-Newton from the true state, the slack angle fixed."""
    vmx, va_x = vm0.copy(), va0.copy()
    w_inv = 1.0 / noise.sigmas(z) ** 2
    angle = np.isin(meas.plan.kind, (MeterKind.VA, MeterKind.IA))

    def resid(vmx, va_x):  # angle readings wrap at +-180 degrees, so their residuals are wrapped too
        r = z - meas.h(vmx, va_x)
        r[angle] = (r[angle] + 180.0) % 360.0 - 180.0
        return r

    for _ in range(iters):
        r = resid(vmx, va_x)
        h = meas.jacobian(vmx, va_x)[:, free_cols]
        dx = np.linalg.lstsq((h.T * w_inv) @ h, h.T @ (w_inv * r), rcond=None)[0]
        step = np.zeros(2 * n)
        step[free_cols] = dx
        vmx, va_x = vmx + step[:n], va_x + step[n:]
        if np.max(np.abs(dx)) < 1e-10:
            break
    r = resid(vmx, va_x)
    return float((r**2 * w_inv).sum())


def _threshold(meas, free_cols):
    return float(chi2.ppf(0.999, len(meas.plan.kind) - len(free_cols)))


def _changed(att, vm, va, a):
    """The meters an episode tampers: moved beyond their noise in some frame."""
    return np.flatnonzero(att.tampered(vm, va, a).any(axis=0))


def test_area_spans_utilities_with_a_boundary(setup):
    att, own, plan, meas, noise, (_, vm, va), n, _ = setup
    spanning = 0
    for b in att.load_buses(vm[0], va[0]):
        area = att.area(int(b), 3)
        if area is None:
            continue
        assert att.slack not in area.tolist() and len(att.boundary(area))  # a boundary to hold true
        if len(set(own.bus_client[area][att.own[area]].tolist())) > 1:
            spanning += 1
    assert spanning > 0  # the local subnetwork may span utilities (not one client)


def test_false_state_load_goal_ac_equations(setup):
    att, own, plan, meas, noise, (_, vm, va), n, _ = setup
    load = att.bus_load(vm[0], va[0])
    state, area, center = None, None, -1
    for c in att.load_buses(vm[0], va[0]):  # the first load whose area can absorb a 5% extra load
        area = att.area(int(c), 2)
        change = np.zeros(n, complex)
        change[c] = 0.05 * load[c]
        state = att.false_state(vm[0], va[0], area, change)
        if state is not None:
            center = int(c)
            break
    assert state is not None
    vf, af = state
    v = vf * np.exp(1j * np.deg2rad(af))
    s = v * np.conj(att.ybus @ v) * att.base_mva
    p, q = att.net.injections(vm[0], va[0])
    true = p + 1j * q
    assert abs(s[center] - (true[center] - change[center])) < 1e-4  # the pretended load is read
    # every New York bus of the grown area (N25) and its edge that is not an attackable load or
    # generator, a zero-injection bus among them, keeps its true injection; every voltage outside the
    # grown area stays true
    grown = att.absorb(vm[0], va[0], area)
    touched = np.union1d(grown, att.boundary(grown))
    free = att.free_injection_buses(vm[0], va[0], touched)
    held = np.setdiff1d(touched, np.r_[free, center])
    held = held[att.own[held]]
    assert len(held) and np.abs(s[held] - true[held]).max() < 1e-4
    outside = np.setdiff1d(np.arange(n), grown)
    assert np.allclose(vf[outside], vm[0][outside]) and np.allclose(af[outside], va[0][outside])
    # an absorbing load moves within the plausibility cap: its active change within ABSORB_CAP of its true
    # load, its reactive change within ABSORB_CAP max(|Q|, 0.2 P) (N24)
    absorbing = np.setdiff1d(free[~att.gen_bus[free]], [center])
    moved = s[absorbing] - true[absorbing]
    cap_q = ABSORB_CAP * np.maximum(np.abs(q[absorbing]), 0.2 * load[absorbing])
    assert len(absorbing) and (np.abs(moved.real) <= ABSORB_CAP * load[absorbing] + 1e-4).all()
    assert (np.abs(moved.imag) <= cap_q + 1e-4).all()


def test_ramp_is_global_stealthy_and_within_natural(setup):
    att, own, plan, meas, noise, (onset, vm, va), n, free_cols = setup
    ep = Ramp(att).design((onset, vm, va), rng(Settings(), "at"))
    assert ep is not None and ep.family == "At" and ep.target_branch == -1 and len(ep.buses)
    changed = _changed(att, vm, va, ep.a)
    assert len(changed) and np.array_equal(ep.clients, np.unique(plan.owner[changed]))
    # every per-step change of a pretended load stays under that bus's natural five-minute swing
    load = np.stack([att.bus_load(vm[k], va[k]) for k in range(LENGTH)])
    natural = np.abs(np.diff(load, axis=0)).max(axis=0)
    for bus in ep.buses:
        row = np.flatnonzero((plan.kind == MeterKind.P) & (plan.bus == bus))[0]
        assert (np.abs(np.diff(ep.a[:, row])) <= natural[bus] + 1e-6).all()
    # the ramp clears the noise floor on at least one channel at its peak
    peak = int(np.argmax(np.abs(ep.a).sum(axis=1)))
    assert (np.abs(ep.a[peak]) > att.sigma(meas.h(vm[peak], va[peak]))).any()
    # global stealth: the WLS residual of every attacked frame stays at the benign level, below threshold
    thr = _threshold(meas, free_cols)
    for k in range(LENGTH):
        z_true = meas.h(vm[k], va[k])
        z_att = noise.sample(z_true, rng(Settings(), f"at-n{k}")) + ep.a[k]
        assert _wls_j(meas, noise, free_cols, n, z_att, vm[k], va[k]) < thr


def test_natural_steps_by_hour(setup, pool):
    att = setup[0]
    solved = pool.solve_kind == SolveKind.SOLVED
    steps = natural_steps(pool.time, pool.p, solved)
    assert steps.shape == (24, pool.p.shape[1]) and (steps >= 0).all()
    # one bus and hour against the percentile taken by hand
    b = int(att.load_buses(pool.vm[MARCH], pool.va[MARCH])[0])
    hours = (pool.time.astype(np.int64) // 3600) % 24
    pair = solved[1:] & solved[:-1] & (hours[1:] == 18)
    assert np.isclose(steps[18, b], np.percentile(np.abs(np.diff(pool.p[:, b]))[pair], 95))


def test_angle_changes_wrap(setup):
    att, own, plan, meas, noise, (_, vm, va), *_ = setup
    angle = np.flatnonzero(np.isin(plan.kind, (MeterKind.VA, MeterKind.IA)))
    d = np.zeros(len(plan.kind))
    d[angle] = 359.99  # a reading that crossed +-180 by 0.01 degree
    w = wrap_angles(d, plan.kind)
    assert np.allclose(w[angle], -0.01) and (w[np.setdiff1d(np.arange(len(d)), angle)] == 0).all()
    assert wrap_angles(np.full(len(d), -180.0), plan.kind)[angle[0]] == 180.0  # (-180, 180]
    # a crossing that moves the angle by less than its noise is not tampering
    assert not att.tampered(vm[:1], va[:1], d[None, :]).any()


def test_zero_injection_edge_bus_is_absorbed_and_reads_zero(setup):
    """A zero-injection New York bus on an area's edge joins the interior (N25): its voltage is free
    while its injection stays at zero, and the grown interior's edge holds only buses the attack may
    change, the slack, or buses outside New York."""
    att, own, plan, meas, noise, (_, vm, va), n, _ = setup
    p, q = att.net.injections(vm[0], va[0])
    zero = ~att.gen_bus & att.own & (np.abs(p) < 1e-3) & (np.abs(q) < 1e-3)
    free = np.zeros(n, bool)
    free[att.free_injection_buses(vm[0], va[0], np.arange(n))] = True
    load = att.bus_load(vm[0], va[0])
    checked = 0
    for c in att.load_buses(vm[0], va[0]):
        area = att.area(int(c), 1)
        edge_zero = [int(b) for b in att.boundary(area) if zero[b]]
        if not edge_zero:
            continue
        grown = att.absorb(vm[0], va[0], area)
        assert set(edge_zero) <= set(grown.tolist())  # absorbed into the interior
        e = att.boundary(grown)
        assert (free[e] | ~att.own[e] | (e == att.slack)).all()  # the edge moved out
        change = np.zeros(n, complex)
        change[c] = 0.05 * load[c]
        state = att.false_state(vm[0], va[0], area, change)
        if state is None:
            continue
        vf, af = state
        pf, qf = att.net.injections(vf, af)
        assert np.abs(pf[edge_zero]).max() < 1e-4 and np.abs(qf[edge_zero]).max() < 1e-4  # still zero
        outside = np.setdiff1d(np.arange(n), grown)  # only the grown interior's voltages may move
        assert np.allclose(vf[outside], vm[0][outside]) and np.allclose(af[outside], va[0][outside])
        checked += 1
        break
    assert checked == 1


def test_the_interchange_consistent_attacker_holds_every_bus_outside_new_york(setup):
    """N62: with `hold_external` every bus outside New York keeps its true injection and no unit there takes
    up the change, and an external bus stays on the edge of the attack's interior instead of joining it,
    so the solve holds every tie at its true flow. Without it only dead external buses are held (N44)."""
    att, meas, (_, vms, vas) = setup[0], setup[3], setup[5]
    net, vm, va = att.net, vms[0], vas[0]
    case = net.case
    held = Attacker(net, meas, hold_external=True)
    ext = ~held.own
    assert held._held_bus(vm, va)[ext].all() and not att._held_bus(vm, va)[ext].all()
    every = np.arange(len(held.own))
    assert held.own[held.free_injection_buses(vm, va, every)].all()
    for b in np.flatnonzero(case.is_ny)[:40]:
        inside = held.absorb(vm, va, np.array([b]))
        assert held.own[inside].all()  # never grown across the border
