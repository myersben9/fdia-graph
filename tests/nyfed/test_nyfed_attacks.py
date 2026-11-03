import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from fdia_graph.nyfed.attacks.attacker import Attacker, generator_ramps, natural_steps, wrap_angles
from fdia_graph.nyfed.attacks.msfdia import MsFdia
from fdia_graph.nyfed.attacks.ramp import Ramp
from fdia_graph.nyfed.config import Settings, rng
from fdia_graph.nyfed.grid.clients import assign_clients
from fdia_graph.nyfed.grid.fixes import restore_765kv, split_zone_j
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.grid.nygrid import modify_mpc, nygrid_units, read_npcc
from fdia_graph.nyfed.io import read_pool
from fdia_graph.nyfed.scenario.pool import SOLVED, Pool, peak_ratings
from fdia_graph.nyfed.sensing.measure import Measurement
from fdia_graph.nyfed.sensing.meters import IA, PF, QF, VA, P, Q
from fdia_graph.nyfed.sensing.noise import NoiseModel

ROOT = Path(
    os.environ.get("NYFED_ROOT", Path(__file__).resolve().parents[2])
)  # the nyfed inputs, data/ and out/
DATA = ROOT / "data" / "NYgrid" / "Data"
TERRITORIES = ROOT / "data" / "territories"
POOL = ROOT / "out" / "pool_2019.h5"
LENGTH = 16
pytestmark = pytest.mark.skipif(
    not (DATA.exists() and POOL.exists()), reason="NYgrid checkout or out/pool_2019.h5 missing"
)


@pytest.fixture(scope="module")
def pool():
    return read_pool(POOL)


@pytest.fixture(scope="module")
def setup(pool):
    """The case, ownership, meter plan, an attacker, a short window of the real 2019 pool (the longest
    solved run, in summer, so loads are high enough to overload a branch), its peak ratings over the
    window, and the operator's observable state (the buses any meter sees), for the WLS residual test."""
    case = restore_765kv(split_zone_j(nygrid_units(modify_mpc(read_npcc(DATA / "npcc.mat")), DATA)))
    territories = json.load(open(TERRITORIES / "nys_electric_territories.geojson", encoding="utf-8"))
    own = assign_clients(case, territories, TERRITORIES / "ny_bus_locations.csv")
    net = Network(case)
    plan = plan_meters(net, own)
    meas = Measurement(net, plan)
    att = Attacker(net, meas)
    sol = np.flatnonzero(pool.solve_kind == SOLVED)
    w = max(np.split(sol, np.flatnonzero(np.diff(sol) != 1) + 1), key=len)[:LENGTH]
    vm, va = pool.vm[w].copy(), pool.va[w].copy()
    # rated over this short window alone, so a margin above its own peak keeps lines below their rating
    # (the production factor 1.0 of N47 rates over the whole year, where only the peak scan touches it)
    window = Pool(pool.time[w], vm, va, *net.injections(vm, va), np.full(LENGTH, SOLVED, np.int8))
    ratings = peak_ratings(case, window, factor=1.05).rating
    n = len(vm[0])
    mb = plan.branch[plan.branch >= 0]
    obs = np.unique(np.r_[plan.bus, case.f[mb], case.t[mb]])  # the buses a meter sees
    free_cols = np.array(sorted({int(b) for b in obs} | {n + int(b) for b in obs if b != att.slack}))
    return att, own, plan, meas, NoiseModel(plan), (int(w[0]), vm, va), ratings, n, free_cols


def plan_meters(net, own):
    from fdia_graph.nyfed.sensing.meters import plan_meters as pm

    return pm(net, own, rng(Settings(), "meters"))


def _wls_j(meas, noise, free_cols, n, z, vm0, va0, iters=40):
    """The weighted residual J after a WLS state estimate over the observable island (external buses,
    which no meter sees, are held true). A consistent measurement leaves J at the noise level; an
    inconsistent one leaves it far larger. Gauss-Newton from the true state, the slack angle fixed."""
    vmx, va_x = vm0.copy(), va0.copy()
    w_inv = 1.0 / noise.sigmas(z) ** 2
    angle = np.isin(meas.plan.kind, (VA, IA))

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


def _flow_rows(plan, b):
    pf = np.flatnonzero((plan.kind == PF) & (plan.branch == b) & (plan.end == 0))[0]
    qf = np.flatnonzero((plan.kind == QF) & (plan.branch == b) & (plan.end == 0))[0]
    return pf, qf


def test_area_spans_utilities_with_a_boundary(setup):
    att, own, plan, meas, noise, (_, vm, va), ratings, n, _ = setup
    spanning = 0
    for b in att.load_buses(vm[0], va[0]):
        area = att.area(int(b), 3)
        if area is None:
            continue
        assert att.slack not in area.tolist() and len(att.boundary(area))  # a boundary to hold true
        if len(set(att.owner[area][att.owner[area] >= 0].tolist())) > 1:
            spanning += 1
    assert spanning > 0  # the local subnetwork may span utilities (not one client)


def test_false_flow_state_meets_goal_within_limits(setup):
    att, own, plan, meas, noise, (_, vm, va), ratings, n, _ = setup
    case = att.net.case
    f = case.f
    metered = np.zeros(len(f), bool)
    metered[plan.branch[(plan.kind == PF) & (plan.end == 0)]] = True
    pf, qf, pt, qt = att.net.flows(vm[0], va[0])
    peak = np.maximum(np.hypot(pf, qf), np.hypot(pt, qt))
    internal = (att.owner[f] >= 0) & (att.owner[case.t] >= 0) & metered & (ratings > 0) & (peak < ratings)
    # the first eligible branch whose area can carry its rating under the cap (many cannot, as the
    # search expects); the solve is correct where a solution exists
    res, area, b = None, None, -1
    for x in np.flatnonzero(internal):
        area = att.area(int(f[x]), 3)
        if area is None:
            continue
        res = att.false_flow_state(vm[0], va[0], area, [x], [ratings[x]], load_cap=0.5)
        if res is not None:
            b = int(x)
            break
    assert res is not None
    vf, af, dload = res
    reported = np.hypot(*att.net.flows(vf, af)[:2])[b]
    assert abs(reported - ratings[b]) < 1e-2  # the from-end flow reaches the rating
    # outside the support, grown over its non-free New York edge buses (N25), every voltage stays true
    outside = np.setdiff1d(np.arange(n), att._absorb(vm[0], va[0], area))
    assert np.allclose(vf[outside], vm[0][outside]) and np.allclose(af[outside], va[0][outside])
    # a pretended load change never exceeds the 0.5 cap of its true load
    load = att.bus_load(vm[0], va[0])
    moved = np.flatnonzero((np.abs(dload) > 1e-6) & ~att.gen_bus)
    assert (np.abs(dload[moved]) <= 0.5 * load[moved] + 1e-6).all()


def test_false_state_load_goal_ac_equations(setup):
    att, own, plan, meas, noise, (_, vm, va), ratings, n, _ = setup
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
    grown = att._absorb(vm[0], va[0], area)
    touched = np.union1d(grown, att.boundary(grown))
    held = np.setdiff1d(touched, np.r_[att.free_injection_buses(vm[0], va[0], touched), center])
    held = held[att.owner[held] >= 0]
    assert len(held) and np.abs(s[held] - true[held]).max() < 1e-4
    outside = np.setdiff1d(np.arange(n), grown)
    assert np.allclose(vf[outside], vm[0][outside]) and np.allclose(af[outside], va[0][outside])


def test_msfdia_is_sparse_overloads_target_and_evades_bdd(setup):
    att, own, plan, meas, noise, (onset, vm, va), ratings, n, free_cols = setup
    ep = MsFdia(att, ratings).design((onset, vm, va), rng(Settings(), "am"))
    assert ep is not None and ep.family == "Am" and ep.target_branch >= 0
    b = ep.target_branch
    changed = _changed(att, vm, va, ep.a)
    # Episode.clients are the owners of the tampered meters, possibly several utilities (G5)
    assert np.array_equal(ep.clients, np.unique(plan.owner[changed]))
    # the attack is sparse: it tampers the fewest meters (eq. 12), far below the whole plan
    assert 1 <= len(changed) < len(plan.kind) // 3
    # the reported apparent flow on the target branch ends at or above its rating
    pf, qf = _flow_rows(plan, b)
    z_last = meas.h(vm[-1], va[-1])
    reported = np.hypot(z_last[pf] + ep.a[-1, pf], z_last[qf] + ep.a[-1, qf])
    assert reported >= ratings[b] - 1e-3 and np.hypot(*att.net.flows(vm[-1], va[-1])[:2])[b] < ratings[b]
    # a pretended load stays within the 0.5 plausibility cap
    load = att.bus_load(vm[-1], va[-1])
    for bus in ep.buses[~att.gen_bus[ep.buses]]:
        row = np.flatnonzero((plan.kind == P) & (plan.bus == bus))
        if len(row) and load[bus] > 1e-6:
            assert abs(ep.a[-1, row[0]]) <= 0.5 * load[bus] + 1e-6
    # global stealth: a WLS over all meters refits to the false state, leaving the benign residual;
    # a detector that assumed the true state (no refit) would see a huge residual
    thr = _threshold(meas, free_cols)
    for k in (0, LENGTH - 1):
        z_true = meas.h(vm[k], va[k])
        z_att = noise.sample(z_true, rng(Settings(), f"am-n{k}")) + ep.a[k]
        j_att = _wls_j(meas, noise, free_cols, n, z_att, vm[k], va[k])
        j_benign = _wls_j(
            meas, noise, free_cols, n, noise.sample(z_true, rng(Settings(), f"am-b{k}")), vm[k], va[k]
        )
        assert j_att < thr and abs(j_att - j_benign) < 0.4 * j_benign
    naive = float((((z_att - z_true) / noise.sigmas(z_true)) ** 2).sum())
    assert naive > thr  # the attack is substantive, not sub-noise


def test_msfdia_tries_the_most_loaded_line_first(setup):
    att, own, plan, meas, noise, (onset, vm, va), ratings, n, free_cols = setup
    am = MsFdia(att, ratings)
    lines = am._eligible(vm, va)
    assert len(lines) > 1
    order = am._order(lines, vm, va, rng(Settings(), "am"))
    pf, qf, _, _ = att.net.flows(vm[-1:], va[-1:])
    loading = np.hypot(pf[0], qf[0])[order] / ratings[order]
    # N26: every eligible line once, by true loading at the last scan, highest first
    assert sorted(order.tolist()) == sorted(lines.tolist()) and np.all(np.diff(loading) <= 0)
    shuffled = MsFdia(att, ratings, target_order="random")._order(lines, vm, va, rng(Settings(), "am"))
    assert sorted(shuffled.tolist()) == sorted(lines.tolist())
    with pytest.raises(ValueError):
        MsFdia(att, ratings, target_order="nearest")


def test_ramp_is_global_stealthy_and_within_natural(setup):
    att, own, plan, meas, noise, (onset, vm, va), ratings, n, free_cols = setup
    ep = Ramp(att).design((onset, vm, va), rng(Settings(), "at"))
    assert ep is not None and ep.family == "At" and ep.target_branch == -1 and len(ep.buses)
    changed = _changed(att, vm, va, ep.a)
    assert len(changed) and np.array_equal(ep.clients, np.unique(plan.owner[changed]))
    # every per-step change of a pretended load stays under that bus's natural five-minute swing
    load = np.stack([att.bus_load(vm[k], va[k]) for k in range(LENGTH)])
    natural = np.abs(np.diff(load, axis=0)).max(axis=0)
    for bus in ep.buses:
        row = np.flatnonzero((plan.kind == P) & (plan.bus == bus))[0]
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


AP_ONSET = 18284  # a March window (2019-03-05 11:40) where Am and Ap (P-only and P+Q) all find an overload
# under the bounds of N27 to N34 and the N47 ratings with target lines tried at random, and no pretended
# load change of Am or Ap-P changes sign (PLAN: sign changes remain in about 45% of Am episodes, so
# that check holds on this window, not in general); the tests on it pin the random order to test the
# bounds, not N26


@pytest.fixture(scope="module")
def pattern(setup, pool):
    """Ap's inputs on the production ratings (1.25 times each branch's 2019 peak, N8): each bus's natural
    five-minute step by hour from the whole pool, each generator bus's units' five-minute ramp, the
    clock hour of every pool frame, and the window at AP_ONSET."""
    att = setup[0]
    ratings = peak_ratings(att.net.case, pool).rating
    steps = natural_steps(pool.time, pool.p, pool.solve_kind == SOLVED)
    steps[:, att.gen_bus] = generator_ramps(att.net.case)[att.gen_bus]
    hour = (pool.time.astype(np.int64) // 3600) % 24
    w = slice(AP_ONSET, AP_ONSET + LENGTH)
    return ratings, steps, hour, pool.vm[w].copy(), pool.va[w].copy()


def test_natural_steps_by_hour(setup, pool):
    att = setup[0]
    solved = pool.solve_kind == SOLVED
    steps = natural_steps(pool.time, pool.p, solved)
    assert steps.shape == (24, pool.p.shape[1]) and (steps >= 0).all()
    # one bus and hour against the percentile taken by hand
    b = int(att.load_buses(pool.vm[AP_ONSET], pool.va[AP_ONSET])[0])
    hours = (pool.time.astype(np.int64) // 3600) % 24
    pair = solved[1:] & solved[:-1] & (hours[1:] == 18)
    assert np.isclose(steps[18, b], np.percentile(np.abs(np.diff(pool.p[:, b]))[pair], 95))


def test_ap_pq_ramps_within_natural_steps_overloads_and_evades_bdd(setup, pattern):
    """The P+Q variant of Ap: active and reactive steps of every pretended load bounded."""
    att, own, plan, meas, noise, _, _, n, free_cols = setup
    ratings, steps, hour, vm, va = pattern
    ap = MsFdia(
        att,
        ratings,
        target_order="random",
        step_bound=steps,
        frame_hour=hour,
        reactive_step=True,
        family="Ap",
    )
    ep = ap.design((AP_ONSET, vm, va), rng(Settings(), f"Am{AP_ONSET}"))
    assert ep is not None and ep.family == "Ap" and ep.target_branch >= 0
    bound_p = steps[hour[AP_ONSET : AP_ONSET + LENGTH]]  # kappa 1
    p_true, q_true = att.net.injections(vm, va)
    load = np.stack([att.bus_load(vm[k], va[k]) for k in range(LENGTH)])
    # every pretended load, at every scan, moves from the previous scan's by at most its natural step,
    # starting from zero at the scan before the window (so the attack ramps in); Q in proportion
    checked = 0
    for b in np.flatnonzero(~att.gen_bus & (load > 1e-6).any(axis=0)):
        rows_p = np.flatnonzero((plan.kind == P) & (plan.bus == b))
        rows_q = np.flatnonzero((plan.kind == Q) & (plan.bus == b))
        if not len(rows_p):
            continue
        dp = np.diff(np.r_[0.0, ep.a[:, rows_p[0]]])
        dq = np.diff(np.r_[0.0, ep.a[:, rows_q[0]]])
        assert (np.abs(dp) <= bound_p[:, b] + 1e-5).all()
        assert (np.abs(dq) <= bound_p[:, b] * np.abs(q_true[:, b]) / np.abs(p_true[:, b]) + 1e-5).all()
        checked += np.abs(ep.a[:, rows_p[0]]).max() > 1e-3
    assert checked >= 1  # the bound was exercised on a load the attack moves
    # every generator bus's reported output moves by at most its units' ramp from scan to scan, or by its
    # own true step where the dispatch moves faster (the true step plus the pretended one; into the
    # first scan the pretended step alone)
    for b in np.flatnonzero(att.gen_bus & np.isfinite(steps[0])):
        rows_p = np.flatnonzero((plan.kind == P) & (plan.bus == b))
        if len(rows_p):
            reported = np.diff(np.r_[0.0, ep.a[:, rows_p[0]]])
            reported[1:] += np.diff(p_true[:, b])
            allowed = np.r_[steps[0, b], np.maximum(steps[0, b], np.abs(np.diff(p_true[:, b])))]
            assert (np.abs(reported) <= allowed + 1e-5).all()
    # the reported flow of the target reaches its rating at the last scan, the true flow stays below
    b = ep.target_branch
    pf, qf = _flow_rows(plan, b)
    z_last = meas.h(vm[-1], va[-1])
    assert np.hypot(z_last[pf] + ep.a[-1, pf], z_last[qf] + ep.a[-1, qf]) >= ratings[b] - 1e-3
    assert np.hypot(*att.net.flows(vm[-1], va[-1])[:2])[b] < ratings[b]
    # global stealth at every scan: the WLS residual stays at the benign level, below the threshold
    thr = _threshold(meas, free_cols)
    for k in range(LENGTH):
        z_true = meas.h(vm[k], va[k])
        j_att = _wls_j(
            meas,
            noise,
            free_cols,
            n,
            noise.sample(z_true, rng(Settings(), f"ap-n{k}")) + ep.a[k],
            vm[k],
            va[k],
        )
        j_benign = _wls_j(
            meas, noise, free_cols, n, noise.sample(z_true, rng(Settings(), f"ap-b{k}")), vm[k], va[k]
        )
        assert j_att < thr and abs(j_att - j_benign) < 0.4 * j_benign


def test_am_unchanged_by_the_step_option(setup, pattern):
    att = setup[0]
    ratings, steps, hour, vm, va = pattern
    key = f"Am{AP_ONSET}"
    am = MsFdia(att, ratings, target_order="random").design((AP_ONSET, vm, va), rng(Settings(), key))
    loose = MsFdia(
        att,
        ratings,
        target_order="random",
        step_bound=np.full_like(steps, np.inf),
        frame_hour=hour,
        family="Ap",
    )
    again = loose.design((AP_ONSET, vm, va), rng(Settings(), key))
    assert am is not None and am.family == "Am" and again is not None
    # an unbinding step bound solves scan by scan through the same active set and lands on Am exactly
    assert np.array_equal(am.a, again.a) and np.array_equal(am.buses, again.buses)
    # Am itself overshoots the natural step somewhere: the bound Ap adds is not slack
    rows = {int(b): int(np.flatnonzero((setup[2].kind == P) & (setup[2].bus == b))[0]) for b in am.buses}
    over = [
        np.max(
            np.abs(np.diff(np.r_[0.0, am.a[:, r]]))
            / np.maximum(steps[hour[AP_ONSET : AP_ONSET + LENGTH], b], 1e-9)
        )
        for b, r in rows.items()
        if not att.gen_bus[b]
    ]
    assert max(over) > 1.0


def _pretended_loads(att, plan, vm, va, a):
    """Each New York load bus's pretended active load change over the window [L] (MW), with its true load."""
    load = np.stack([att.bus_load(vm[k], va[k]) for k in range(len(vm))])
    out = {}
    for b in np.flatnonzero(~att.gen_bus & (att.owner >= 0)):
        row = np.flatnonzero((plan.kind == P) & (plan.bus == b))
        if len(row):
            out[int(b)] = (-a[:, row[0]], load[:, b])
    return out


def test_zero_injection_buses_hold_and_loads_keep_their_sign(setup, pattern):
    """A bus with no load and no generation keeps reading zero injection change (the operator meters P
    and Q at every New York bus), and on this window no pretended load of Am or Ap changes sign inside
    the window unless its true load does (the warm-started scans keep each injection continuous)."""
    att, own, plan, meas, noise, _, _, n, free_cols = setup
    ratings, steps, hour, vm, va = pattern
    p_true = att.net.injections(vm, va)[0]
    zero = np.flatnonzero(~att.gen_bus & (att.owner >= 0) & (np.abs(p_true) < 1e-3).all(axis=0))
    assert len(zero)
    for fam in (
        MsFdia(att, ratings, target_order="random"),
        MsFdia(att, ratings, target_order="random", step_bound=steps, frame_hour=hour, family="Ap"),
    ):
        ep = fam.design((AP_ONSET, vm, va), rng(Settings(), f"Am{AP_ONSET}"))
        assert ep is not None
        rows = np.flatnonzero(np.isin(plan.kind, (P, Q)) & np.isin(plan.bus, zero))
        assert np.abs(ep.a[:, rows]).max() < 1e-4  # exactly held, not merely under the noise
        for b, (x, load) in _pretended_loads(att, plan, vm, va, ep.a).items():
            big = np.abs(x) > 0.1
            if big.any() and not ((load > 0).any() and (load < 0).any()):
                assert (x[big] > 0).all() or (x[big] < 0).all(), (ep.family, b, x)


def test_angle_changes_wrap(setup):
    att, own, plan, meas, noise, (_, vm, va), *_ = setup
    angle = np.flatnonzero(np.isin(plan.kind, (VA, IA)))
    d = np.zeros(len(plan.kind))
    d[angle] = 359.99  # a reading that crossed +-180 by 0.01 degree
    w = wrap_angles(d, plan.kind)
    assert np.allclose(w[angle], -0.01) and (w[np.setdiff1d(np.arange(len(d)), angle)] == 0).all()
    assert wrap_angles(np.full(len(d), -180.0), plan.kind)[angle[0]] == 180.0  # (-180, 180]
    # a crossing that moves the angle by less than its noise is not tampering
    assert not att.tampered(vm[:1], va[:1], d[None, :]).any()


def test_ap_p_only_bounds_the_active_step_alone(setup, pattern):
    """Ap as decided (Ben, 2026-10-27): only each pretended load's active step is bounded by its natural
    step; its reactive change is limited by the plausibility cap alone. A generator's reactive output
    moves within its ramp (N27), so its reactive bound is finite in both variants."""
    att, own, plan, meas, noise, _, _, n, free_cols = setup
    ratings, steps, hour, vm, va = pattern
    ap = MsFdia(att, ratings, target_order="random", step_bound=steps, frame_hour=hour, family="Ap")
    bound_p, bound_q, _ = ap._bounds(AP_ONSET, vm, va)
    assert (
        np.isinf(bound_q[:, ~att.gen_bus]).all()
        and np.isfinite(bound_p[:, ~att.gen_bus & (att.owner >= 0)]).any()
    )
    assert np.isfinite(bound_q[:, att.gen_bus & np.isfinite(steps[0])]).all()
    pq = MsFdia(
        att,
        ratings,
        target_order="random",
        step_bound=steps,
        frame_hour=hour,
        reactive_step=True,
        family="Ap",
    )
    assert np.isfinite(pq._bounds(AP_ONSET, vm, va)[1]).any()  # the P+Q option still bounds Q
    ep = ap.design((AP_ONSET, vm, va), rng(Settings(), f"Am{AP_ONSET}"))
    assert ep is not None and ep.family == "Ap"
    load = np.stack([att.bus_load(vm[k], va[k]) for k in range(LENGTH)])
    for b in np.flatnonzero(~att.gen_bus & (load > 1e-6).any(axis=0)):
        rows_p = np.flatnonzero((plan.kind == P) & (plan.bus == b))
        if len(rows_p):
            assert (np.abs(np.diff(np.r_[0.0, ep.a[:, rows_p[0]]])) <= bound_p[:, b] + 1e-5).all()
    b = ep.target_branch
    pf, qf = _flow_rows(plan, b)
    z_last = meas.h(vm[-1], va[-1])
    assert np.hypot(z_last[pf] + ep.a[-1, pf], z_last[qf] + ep.a[-1, qf]) >= ratings[b] - 1e-3


def test_zero_injection_edge_bus_is_absorbed_and_reads_zero(setup):
    """A zero-injection New York bus on an area's edge joins the interior (N25): its voltage is free
    while its injection stays at zero, and the grown interior's edge holds only buses the attack may
    change, the slack, or buses outside New York."""
    att, own, plan, meas, noise, (_, vm, va), ratings, n, _ = setup
    p, q = att.net.injections(vm[0], va[0])
    zero = ~att.gen_bus & (att.owner >= 0) & (np.abs(p) < 1e-3) & (np.abs(q) < 1e-3)
    free = np.zeros(n, bool)
    free[att.free_injection_buses(vm[0], va[0], np.arange(n))] = True
    load = att.bus_load(vm[0], va[0])
    checked = 0
    for c in att.load_buses(vm[0], va[0]):
        area = att.area(int(c), 1)
        edge_zero = [int(b) for b in att.boundary(area) if zero[b]]
        if not edge_zero:
            continue
        grown = att._absorb(vm[0], va[0], area)
        assert set(edge_zero) <= set(grown.tolist())  # absorbed into the interior
        e = att.boundary(grown)
        assert (free[e] | (att.owner[e] < 0) | (e == att.slack)).all()  # the edge moved out
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


def test_load_reactive_change_within_floored_cap(setup, pattern):
    """Every pretended load's reactive change stays within 0.5 max(|Q_true|, 0.2 P_true) (N24 with
    its floor), and its active change within 0.5 P_true, at every scan."""
    att, own, plan, meas, noise, *_ = setup
    ratings, steps, hour, vm, va = pattern
    ep = MsFdia(att, ratings, target_order="random").design(
        (AP_ONSET, vm, va), rng(Settings(), f"Am{AP_ONSET}")
    )
    assert ep is not None
    p, q = att.net.injections(vm, va)
    checked = 0
    for b in np.flatnonzero(~att.gen_bus & (att.owner >= 0)):
        load = -p[:, b]
        if (load <= 1.0).any():  # loads above the floor only (free injections)
            continue
        rp = np.flatnonzero((plan.kind == P) & (plan.bus == b))[0]
        rq = np.flatnonzero((plan.kind == Q) & (plan.bus == b))[0]
        assert (np.abs(ep.a[:, rp]) <= 0.5 * load + 1e-5).all()
        assert (np.abs(ep.a[:, rq]) <= 0.5 * np.maximum(np.abs(q[:, b]), 0.2 * load) + 1e-5).all()
        checked += np.abs(ep.a[:, rq]).max() > 1e-3
    assert checked >= 1


def test_the_interchange_consistent_attacker_holds_every_bus_outside_new_york(setup):
    """N62: with `hold_external` every bus outside New York keeps its true injection and no unit there takes
    up the change, and an external bus stays on the edge of the attack's interior instead of joining it,
    so the solve holds every tie at its true flow. Without it only dead external buses are held (N44)."""
    att, meas, (_, vms, vas) = setup[0], setup[3], setup[5]
    net, vm, va = att.net, vms[0], vas[0]
    case = net.case
    held = Attacker(net, meas, hold_external=True)
    ext = held.owner < 0
    assert held._held_bus(vm, va)[ext].all() and not att._held_bus(vm, va)[ext].all()
    every = np.arange(len(held.owner))
    assert (held.owner[held.free_injection_buses(vm, va, every)] >= 0).all()
    for b in np.flatnonzero(case.is_ny)[:40]:
        inside = held._absorb(vm, va, np.array([b]))
        assert (held.owner[inside] >= 0).all()  # never grown across the border
