from pathlib import Path

import numpy as np
import pytest
from nyfed_inputs import DATA, ROOT, TERRITORIES, needs_data

from fdia_graph.generation.nygrid.dispatch import hourly_dispatch
from fdia_graph.generation.nygrid.fixes import restore_765kv, split_zone_j
from fdia_graph.generation.nygrid.inputs import read_nygrid
from fdia_graph.generation.nygrid.npcc import modify_mpc, nygrid_units, read_npcc

REF = (
    Path(__file__).resolve().parent / "data" / "matlab_ref"
)  # operating points dumped from NYgrid's MATLAB code with the AC recipe
pytestmark = needs_data
TAGS = {"2019010100": "2019-01-01T00", "2019041514": "2019-04-15T14", "2019072017": "2019-07-20T17"}


@pytest.fixture(scope="module")
def case():
    return restore_765kv(split_zone_j(nygrid_units(modify_mpc(read_npcc(DATA / "npcc.mat")), DATA)))


@pytest.fixture(scope="module")
def hours():
    return read_nygrid(DATA, 2019)


def test_read_nygrid(hours):
    assert len(hours) == 8760
    h = hours[0]
    print(f"2019-01-01 00h NY load {h.ny_pd.sum():.1f} MW, renewables {h.renew_mw.sum():.1f} MW")
    # nyload column of the MATLAB year run: NY load net of wind and other renewables
    assert np.isclose(h.ny_pd.sum() - h.renew_mw.sum(), 13445.4, atol=0.1)
    assert len(h.thermal_mw) == 227 and len(h.nuclear_mw) == 6


@pytest.mark.skipif(not REF.exists(), reason="MATLAB reference dumps missing")
def test_dispatch_matches_matlab(case, hours):
    picked = [next(h for h in hours if str(h.time).startswith(t)) for t in TAGS.values()]
    d = hourly_dispatch(case, picked)
    for k, tag in enumerate(TAGS):
        gen = np.loadtxt(REF / f"gen_{tag}.csv", delimiter=",")
        bus = np.loadtxt(REF / f"bus_{tag}.csv", delimiter=",")
        assert len(gen) == d.unit_p.shape[1]
        assert np.array_equal(case.bus_id[case.unit_bus], gen[:, 0].astype(int))
        dp = np.abs(d.unit_p[k] - gen[:, 1]).max()
        dl = np.abs(d.pd[k] - bus[:, 1]).max()
        print(f"{tag}: max unit P diff {dp:.2e} MW, max bus load diff {dl:.2e} MW")
        assert dp < 1e-3 and dl < 1e-3


@pytest.mark.skipif(not REF.exists(), reason="MATLAB reference dumps missing")
def test_projection_matches_matlab(case, hours):
    from fdia_graph.generation.nygrid.solver import AcSolver

    picked = [next(h for h in hours if str(h.time).startswith(t)) for t in TAGS.values()]
    d = hourly_dispatch(case, picked)
    solver = AcSolver(case)
    for k, tag in enumerate(TAGS):
        r = solver.project(d.pd[k], d.qd[k], d.unit_p[k], d.unit_on[k], d.hvdc_p[k])
        ref = np.loadtxt(REF / f"acgen_{tag}.csv", delimiter=",")
        v = np.loadtxt(REF / f"vm_{tag}.csv", delimiter=",")
        dp, dv = np.abs(r["unit_p"] - ref[:, 0]).max(), np.abs(r["vm"] - v[:, 0]).max()
        print(f"{tag}: max unit P diff {dp:.2f} MW, max |V| diff {dv:.1e} p.u.")
        # MATLAB's MIPS stops at its own tolerance, so the optimum agrees to a few MW on a stiff hour.
        assert dp < 5.0 and dv < 1e-3


YEAR = ROOT / "results" / "nygrid.ac_year.csv"  # the MATLAB recipe over 2019, one row per hour


@pytest.mark.skipif(not YEAR.exists(), reason="MATLAB year results missing")
def test_projection_matches_matlab_year(case, hours):
    import pandas as pd

    from fdia_graph.generation.nygrid.solver import AcSolver

    year = pd.read_csv(YEAR).set_index("ts")
    tags = [
        "2019-01-07 00",
        "2019-04-12 00",
        "2019-07-17 13",
        "2019-11-29 02",
    ]  # one per season, all solved in MATLAB
    picked = [next(h for h in hours if str(h.time).replace("T", " ").startswith(t)) for t in tags]
    d = hourly_dispatch(case, picked)
    solver = AcSolver(case)
    ny_unit = case.is_ny[case.unit_bus]
    for k, tag in enumerate(tags):
        r = solver.project(d.pd[k], d.qd[k], d.unit_p[k], d.unit_on[k], d.hvdc_p[k])
        ref = year.loc[tag]
        redisp = np.abs(r["unit_p"] - d.unit_p[k])[ny_unit].sum() / 2
        vmin, vmax = r["vm"][case.is_ny].min(), r["vm"][case.is_ny].max()
        print(
            f"{tag}: NY re-dispatch {redisp:.1f} vs {ref.redisp_ny:.1f} MW, NY V {vmin:.4f}-{vmax:.4f} "
            f"vs {ref.ny_vmin:.4f}-{ref.ny_vmax:.4f}"
        )
        assert abs(redisp - ref.redisp_ny) < 2.0
        assert abs(vmin - ref.ny_vmin) < 3e-4 and abs(vmax - ref.ny_vmax) < 3e-4


def test_pool(case, hours, tmp_path):
    from fdia_graph.generation.nygrid.io import read_pool, write_pool
    from fdia_graph.generation.nygrid.loads import LoadSeries
    from fdia_graph.generation.nygrid.pool import NO_LOAD, SOLVED, PoolBuilder
    from fdia_graph.generation.nygrid.solver import AcSolver

    d = hourly_dispatch(case, hours[:2])
    renew = np.array([h.renew_mw for h in hours[:2]])
    # Frames at the two hourly points (half past each hour), one between them and one without load. The
    # loads at the points are the dispatch's own (gross of renewables, as the 5-minute loads are).
    half = np.timedelta64(30, "m")
    time = np.array([d.hour[0] + half, d.hour[0] + 2 * half, d.hour[1] + half, d.hour[1] + 2 * half])
    gross = d.pd + renew
    p = np.stack([gross[0], 0.5 * (gross[0] + gross[1]), gross[1], np.full(len(case.bus_id), np.nan)])
    q = np.stack([d.qd[0], 0.5 * (d.qd[0] + d.qd[1]), d.qd[1], d.qd[1]])
    solver = AcSolver(case)
    ramp = np.where(case.fuel == "Nuclear", 0.0, np.nan_to_num(case.ramp, nan=np.inf)) * 5  # MW per frame
    b = PoolBuilder(solver, LoadSeries(time=time, p=p, q=q), d, renew, ramp)
    pool = b.build()
    print(b.report())
    assert pool.solve_kind.tolist() == [SOLVED, SOLVED, SOLVED, NO_LOAD]
    assert np.isnan(pool.vm[3]).all() and np.isnan(pool.unit_p[3]).all()

    # N32, N39: each thermal unit's basepoint moves within its ramp of the previous frame's (the first from the
    # hour's AC point); nuclear holds the hourly AC points' output, hydro and external units stay near it; the
    # power flow at the basepoints leaves the reference bus only a residual.
    nuc = case.fuel == "Nuclear"
    thermal = ~np.isin(case.fuel, ("Nuclear", "Hydro", "Import"))
    prev = b.hour_points[0]["unit_p"]
    for f in range(3):
        step = np.abs(pool.unit_p[f] - prev)[thermal]
        print(
            f"frame {f}: cost {b.cost[f]:.0f} $/h, largest thermal step over ramp {np.max(step - ramp[thermal]):.2e} MW, "
            f"hydro and externals off schedule {b.anchor_dev[f]:.2f} MW, reference bus {b.slack_mw[f]:.1e} MW"
        )
        assert (step <= ramp[thermal] + 1e-3).all()
        prev = pool.unit_p[f]
        assert abs(b.slack_mw[f]) < 1e-3
    assert np.allclose(pool.unit_p[0, nuc], b.hour_points[0]["unit_p"][nuc])
    assert np.allclose(pool.unit_p[2, nuc], b.hour_points[1]["unit_p"][nuc])
    assert np.nanmax(b.anchor_dev[:3]) < 50.0

    # A build from the last state continues the dispatch: its first frame stays within the ramps.
    b2 = PoolBuilder(solver, LoadSeries(time=time[2:3], p=p[2:3], q=q[2:3]), d, renew, ramp)
    more = b2.build(start=b.final)
    assert more.solve_kind[0] == SOLVED
    assert np.allclose(b.final[0], pool.unit_p[2])
    assert (np.abs(more.unit_p[0] - pool.unit_p[2])[thermal] <= ramp[thermal] + 1e-3).all()
    pi, qi = solver.network.injections(pool.vm[:3], pool.va[:3])
    assert np.allclose(pool.p[:3], pi) and np.allclose(pool.q[:3], qi)

    write_pool(tmp_path / "pool.h5", pool)
    back = read_pool(tmp_path / "pool.h5")
    assert (back.time == pool.time).all() and np.array_equal(back.solve_kind, pool.solve_kind)
    assert np.array_equal(back.vm, pool.vm, equal_nan=True) and np.array_equal(back.p, pool.p, equal_nan=True)
    assert np.array_equal(back.unit_p, pool.unit_p, equal_nan=True)

    # N8: public ratings stay, every other branch gets its peak flow over the solved frames (N47).
    from fdia_graph.generation.nygrid.pool import peak_ratings
    from fdia_graph.generation.nygrid.ratings import line_ratings

    rated = line_ratings(case)
    filled = peak_ratings(rated, pool)
    pf, qf, pt, qt = solver.network.flows(pool.vm[:3], pool.va[:3])
    peak = np.maximum(np.hypot(pf, qf), np.hypot(pt, qt)).max(axis=0)
    public = rated.rating > 0
    assert np.array_equal(filled.rating[public], rated.rating[public])
    assert np.allclose(filled.rating[~public], peak[~public])


def test_pool_compensation(case, hours):
    """The shunt and reactive-limit sensitivity: units within 0.85 lagging and 0.95 leading power factor, New York's
    generator buses regulated to 1.035 p.u., New York held in 0.95 to 1.05 p.u. by switched banks. The release's
    solver has no bank at all."""
    from fdia_graph.generation.nygrid.loads import LoadSeries
    from fdia_graph.generation.nygrid.pool import SOLVED, PoolBuilder
    from fdia_graph.generation.nygrid.solver import AcSolver, Compensation, unit_q_limits

    q_lo, q_hi = unit_q_limits(case)
    loose = np.abs(case.qmax) >= 9999
    assert loose.any() and (~loose).any()
    assert np.allclose(q_hi[loose], np.tan(np.arccos(0.85)) * np.abs(case.pmax[loose]))
    assert np.allclose(q_lo[loose], -np.tan(np.arccos(0.95)) * np.abs(case.pmax[loose]))
    assert np.array_equal(q_hi[~loose], case.qmax[~loose]) and np.array_equal(q_lo[~loose], case.qmin[~loose])

    n = len(case.bus_id)
    ny = case.is_ny
    gen = np.zeros(n, bool)
    gen[case.unit_bus] = True
    comp = Compensation(
        shunt_lo=np.where(ny, -3000.0, 0.0),
        shunt_hi=np.where(ny, 3000.0, 0.0),
        v_schedule=np.where(ny & gen, 1.035, np.nan),
        q_lo=q_lo,
        q_hi=q_hi,
        v_lo=np.where(ny, 0.95, np.nan),
        v_hi=np.where(ny, 1.05, np.nan),
    )
    assert len(AcSolver(case)._opf.S) == 0  # the release's model
    d = hourly_dispatch(case, hours[:2])
    renew = np.array([h.renew_mw for h in hours[:2]])
    half = np.timedelta64(30, "m")
    time = np.array([d.hour[0] + half, d.hour[0] + 2 * half, d.hour[1] + half])
    gross = d.pd + renew
    p = np.stack([gross[0], 0.5 * (gross[0] + gross[1]), gross[1]])
    q = np.stack([d.qd[0], 0.5 * (d.qd[0] + d.qd[1]), d.qd[1]])
    ramp = np.where(case.fuel == "Nuclear", 0.0, np.nan_to_num(case.ramp, nan=np.inf)) * 5
    solver = AcSolver(case, comp)
    b = PoolBuilder(solver, LoadSeries(time=time, p=p, q=q), d, renew, ramp)
    pool = b.build()
    print(b.report(), np.round(b.shunt_q[:, np.abs(b.shunt_q).max(axis=0) > 1], 1))
    assert (pool.solve_kind == SOLVED).all() and not b.q_fallback.any()
    vm = pool.vm[:, ny]
    assert vm.min() >= 0.95 - 1e-4 and vm.max() <= 1.05 + 1e-4  # the band, where the release sits at 1.10
    sched = pool.vm[:, ny & gen]
    assert np.median(np.abs(sched - 1.035)) < 5e-3
    assert (b.shunt_q[:, ~ny] == 0).all() and np.abs(b.shunt_q).max() < 3000.0
    # the power flow at the dispatch's basepoints, the banks as injections, leaves the reference bus a residual
    assert np.abs(b.slack_mw).max() < 1e-3
    pi, qi = solver.network.injections(pool.vm, pool.va)
    assert np.allclose(pool.p, pi) and np.allclose(pool.q, qi)


LOADS = ROOT / "data" / "loads"


@pytest.mark.skipif(
    not (LOADS / "building_stock_ny_15min.npz").exists(), reason="15-minute building stock cache missing"
)
def test_load_profiles(case):
    import json

    from fdia_graph.federated.utilities import assign_clients
    from fdia_graph.generation.nygrid.config import Settings, rng
    from fdia_graph.generation.nygrid.inputs import ZONES, ZoneLoad, fetch_nyiso
    from fdia_graph.generation.nygrid.loads import LoadModel

    own = assign_clients(
        case,
        json.load(open(TERRITORIES / "nys_electric_territories.geojson", encoding="utf-8")),
        TERRITORIES / "ny_bus_locations.csv",
    )
    model = LoadModel(case, own, DATA, TERRITORIES / "ny_bus_locations.csv", LOADS)
    year = fetch_nyiso(2019, ROOT / "data" / "nyiso")
    week = slice(0, 7 * 288)
    zone = ZoneLoad(time=year.time[week], load=year.load[week])
    s = model.profiles(zone, rng(Settings(), "loads"))

    # The NY buses of each zone sum to its real load at every frame, external buses stay at base.
    for k, z in enumerate(ZONES):
        assert np.allclose(s.p[:, case.zone == z].sum(axis=1), zone.load[:, k], rtol=1e-9)
    assert np.allclose(s.p[:, ~case.is_ny], case.pd[~case.is_ny])
    assert np.allclose(s.q[:, model.load_bus], s.p[:, model.load_bus] * np.tan(np.arccos(0.98)))

    # Every bus carries its own shape: normalised profiles that differ from all others.
    x = s.p[:, model.load_bus]
    corr = np.corrcoef((x / x.mean(axis=0)).T)
    np.fill_diagonal(corr, 0)
    print(f"{len(model.load_bus)} load buses, max pairwise correlation {corr.max():.4f}")
    assert corr.max() < 1 - 1e-6

    # N31: within the hour a bus no longer moves in lockstep with its zone (pool_2019 had 0.95 to 0.99).
    steps = np.diff(x, axis=0)
    zone_of = model.zone_of
    within = [
        np.corrcoef(steps[:, i], steps[:, j])[0, 1]
        for i in range(len(zone_of))
        for j in range(i + 1, len(zone_of))
        if zone_of[i] == zone_of[j]
    ]
    print(
        f"within-zone five-minute step correlation: median {np.median(within):.3f}, max {np.max(within):.3f}"
    )
    assert np.median(within) < 0.9


def test_within_hour_calibration():
    """The N31 constants are what the fits give on the real data: the fluctuation on NYISO 2018 (zone H
    left out), the switching on zone D of 2019 with the model's industrial load there."""
    from fdia_graph.generation.nygrid import load_fit, loads
    from fdia_graph.generation.nygrid.inputs import fetch_nyiso

    fit = load_fit.fluctuation_fit(fetch_nyiso(2018, ROOT / "data" / "nyiso"))
    print({k: fit[k] for k in ("c", "beta", "phi", "c_residual", "r2")})
    assert np.isclose(fit["c"], loads.FLUCT_C, rtol=1e-3) and np.isclose(
        fit["beta"], loads.FLUCT_BETA, rtol=1e-3
    )
    assert np.isclose(fit["phi"], loads.FLUCT_PHI, atol=1e-3)
    sw = load_fit.switching_fit(fetch_nyiso(2019, ROOT / "data" / "nyiso"), loads.SWITCH_INDUSTRIAL_MW)
    print(
        {
            k: sw[k]
            for k in ("rate", "u", "xi", "scale", "max", "dur_mu", "dur_sigma", "off", "steps", "max_mw")
        }
    )
    assert sw["steps"] == 2799 and round(sw["max_mw"]) == 314
    assert np.isclose(sw["rate"], loads.SWITCH_RATE, rtol=1e-3) and np.isclose(
        sw["xi"], loads.SWITCH_XI, atol=1e-3
    )
