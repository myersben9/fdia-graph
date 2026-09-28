"""The hybrid meter model (the plan's D10): angles at the PMU buses only, the PMU branch-current
phasors [WU26, eqs. 19-20] emitted, stored, loaded and attacked, and the eq. (3) pseudo-measurements
an estimator can derive from them. The v0.8.3 meter model stays the default and writes none of it."""

import h5py
import numpy as np
import pytest

from fdia_graph.formulas.attacks import tampered_devices
from fdia_graph.formulas.estimation import pmu_pseudo_links, pmu_pseudo_voltages
from fdia_graph.formulas.network import branch_currents
from fdia_graph.formulas.noise import PMU_CURRENT_CLASS, current_sigma
from fdia_graph.models.grid import CURRENT, NODE

pytest.importorskip("pandapower")


@pytest.fixture(scope="module")
def gens():
    from fdia_graph.engine.core import FdiaGenerator

    return FdiaGenerator(14, seed=1), FdiaGenerator(14, seed=1, meter_model="hybrid")


@pytest.fixture(scope="module")
def pool():
    from fdia_graph.generation import _load_states

    return _load_states(14, None)[:200]


@pytest.fixture(scope="module")
def files(tmp_path_factory, pool):
    from fdia_graph.timeline import generate_timeline

    d = tmp_path_factory.mktemp("pmu")
    kw = dict(states=pool, seed=1, families=("At",), ramp_len=10, min_budget=32)
    return (
        generate_timeline(14, out=str(d / "v083.h5"), redundancy={"meter_model": "v083"}, **kw),
        generate_timeline(14, out=str(d / "hybrid.h5"), **kw),
    )


def test_a_scada_voltmeter_reads_no_angle_under_the_hybrid_model(gens):
    old, new = gens
    nm_old, _ = old.meter_masks()
    nm_new, _ = new.meter_masks()
    pmu = np.zeros(new.C, bool)
    pmu[sorted(new.meters.pmu)] = True
    assert (nm_old[:, NODE.theta] == nm_old[:, NODE.v]).all()  # v0.8.3: an angle at every voltmeter bus
    assert (nm_new[:, NODE.theta].astype(bool) == pmu).all()
    assert (nm_new[:, NODE.v] == nm_old[:, NODE.v]).all()  # the plan's draws are the same
    assert old.current_mask() is None and new.current_mask() is not None


def test_the_current_mask_follows_the_pmus(gens):
    _, g = gens
    cm = g.current_mask()
    pmu = np.zeros(g.C, bool)
    pmu[sorted(g.meters.pmu)] = True
    assert (cm[:, CURRENT.re_from].astype(bool) == pmu[g.ei[0]]).all()
    assert (cm[:, CURRENT.im_to].astype(bool) == pmu[g.ei[1]]).all()


def test_emitted_currents_sit_within_their_accuracy_class(gens, pool):
    _, g = gens
    scan = g.emit_from_state(pool[0])
    true = g.currents_from_states(pool[:1])[0]
    assert scan.i_x is not None and scan.i_m is not None
    sig = current_sigma(true, PMU_CURRENT_CLASS)
    on = scan.i_m > 0
    assert (np.abs(scan.i_x[on] - true[on]) < 6 * sig[on]).all()
    assert (scan.i_x[~on] == 0).all()


def test_the_exact_currents_carry_the_exact_flows(gens, pool):
    """S_from = V_from conj(I_from): the current channels agree with the flows of the same state."""
    _, g = gens
    X = pool[:3]
    cur = g.currents_from_states(X)
    flows = g.clean_flows_from_states(X)  # [T, E, 2] MW/MVAr, unmetered zeroed
    both = (g.current_mask()[:, CURRENT.re_from] > 0) & (flows[0, :, 0] != 0)
    th = np.deg2rad(X[:, :, NODE.theta])
    Vf = (X[:, :, NODE.v] * np.exp(1j * th))[:, g.ei[0]]
    S = Vf * np.conj(cur[..., CURRENT.re_from] + 1j * cur[..., CURRENT.im_from]) * g._base_mva
    assert both.any()
    np.testing.assert_allclose(S.real[:, both], flows[:, both, 0], atol=1e-4)
    np.testing.assert_allclose(S.imag[:, both], flows[:, both, 1], atol=1e-4)


def test_a_false_state_writes_its_currents(gens, pool):
    """a = h(x^a) - h(x) on the current channels, on exactly the PMU-read ends."""
    _, g = gens
    Xt = np.asarray(pool[5], float)
    Xa = Xt.copy()
    Xa[3, NODE.theta] += 0.05
    scan = g.emit_from_state(Xt)
    a = g._current_attack(Xa, Xt)
    ix, tamper = g._currents_with_attack(scan, Xa, Xt)
    assert a is not None and ix is not None and tamper is not None and scan.i_x is not None
    moved = (np.abs(a) > 1e-9) & (scan.i_m > 0)
    np.testing.assert_allclose(ix[moved] - scan.i_x[moved], a[moved], atol=1e-6)
    assert (ix[~moved] == scan.i_x[~moved]).all()


def test_a_tampered_current_counts_in_its_ends_pmu():
    N = 4
    ei = np.array([[0, 1], [1, 2]]).T  # branches 0-1 and 1-2
    node = np.zeros((N, 4), bool)
    edge = np.zeros((2, 2), bool)
    cur = np.zeros((2, 4), bool)
    cur[0, CURRENT.re_to] = True  # the to end of branch 0 is bus 1
    cur[1, CURRENT.im_from] = True  # the from end of branch 1 is bus 1 as well
    pmu = np.array([False, True, False, False])
    assert tampered_devices(node, edge, pmu, ei[0], cur, ei[1]).tolist() == [N + 1]


def test_the_search_counts_the_current_channels(gens, pool):
    from fdia_graph.engine.attacks.minimize import _Window
    from fdia_graph.generation import NOISE_FLOOR
    from fdia_graph.models.frames import AttackDesign, FrameKnobs, LoadGoal

    _, g = gens
    k = FrameKnobs(0.2, NOISE_FLOOR, 6, None, False, True, 2, g.operating_limits(pool), True, 256)
    pmu = sorted(g.meters.pmu)
    target = next(
        i for i, b in enumerate(g.load_bus) if b in pmu or any(n in pmu for n in g.local_region([b], 1))
    )
    states = [pool[0], pool[1]]
    goal = LoadGoal(tuple(AttackDesign(np.array([target]), 1.03) for _ in states))
    window = _Window(g, states, goal, k, stealth_bound=False)
    assert window.i_m is not None and window.i_sigma is not None and len(window.i_sigma) == 2
    area = np.asarray(g.local_region(np.array([g.load_bus[target]]), k.hops))
    cost = window.cost(area, None)
    moved = window._snapshot(0, area, window.prev)
    assert cost is not None and moved is not None and moved[2] is not None and moved[2].any()
    window.i_m = window.i_sigma = None  # the same window, blind to the currents
    blind = window.cost(area, None)
    assert blind is not None and blind[1] < cost[1]  # the current channels add to the tamper count


def test_the_hybrid_file_carries_the_current_layers_and_the_old_one_does_not(files):
    from fdia_graph import schema

    old, new = files
    with h5py.File(old) as f:
        assert not any(p in f for p in schema.CURRENT_LAYERS)
        assert "meter_model" not in f.attrs
    with h5py.File(new) as f:
        assert all(p in f for p in schema.CURRENT_LAYERS)
        assert f.attrs["meter_model"] == "hybrid"
        fam = f[schema.FAMILY][:]
        i, ib, tam = f[schema.PMU_I][:], f[schema.PMU_I_BENIGN][:], f[schema.PMU_I_TAMPER][:]
        benign = fam == 0
        assert (i[benign] == ib[benign]).all() and not tam[benign].any()
        assert ((i != ib) == tam.astype(bool)).all()
        assert (~benign).any() and tam[~benign].any()  # the ramps write currents


def test_the_loader_and_export_serve_the_currents(files):
    from fdia_graph.dataset import FdiaGraph

    old, new = files
    ds = FdiaGraph(new, split="train")
    rec = ds[0]
    assert rec["pmu_i"].shape == (ds.E, 4) and "pmu_i_m" in rec and "pmu_i_benign" in rec
    a = ds.export()
    assert a["pmu_i"].shape == (len(ds), ds.E, 4)
    batch = ds.collate([ds[0], ds[1]])
    assert batch["pmu_i"].shape == (2, ds.E, 4)
    legacy = FdiaGraph(old, split="train")
    assert not legacy.has_currents and "pmu_i" not in legacy[0] and "pmu_i" not in legacy.export()


def _case14():
    import pandapower as pp
    import pandapower.networks as pn
    from pandapower.pypower.makeYbus import makeYbus

    net = pn.case14()
    pp.runpp(net)
    ppc = net._ppc
    _, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    V = (
        ppc["internal"]["V"]
        if "internal" in ppc
        else ppc["bus"][:, 7] * np.exp(1j * np.deg2rad(ppc["bus"][:, 8]))
    )
    f, t = ppc["branch"][:, 0].real.astype(int), ppc["branch"][:, 1].real.astype(int)
    Yf, Yt = np.asarray(Yf.todense()), np.asarray(Yt.todense())
    pmu = np.zeros(len(V), bool)
    pmu[[1, 5, 8]] = True  # a transformer end (bus 5, IEEE-14's 4-6 tap) among them
    cm = np.zeros((len(f), 4))
    cm[pmu[f], 0:2] = 1
    cm[pmu[t], 2:4] = 1
    return V, Yf, Yt, f, t, pmu, cm


def test_eq3_is_exact_without_noise():
    V, Yf, Yt, f, t, pmu, cm = _case14()
    links = pmu_pseudo_links(pmu, cm, f, t, Yf, Yt)
    pv = pmu_pseudo_voltages(
        links, len(V), np.abs(V), np.angle(V), branch_currents(V, Yf, Yt) * cm, 0.0, 0.0, 0.0
    )
    r = pv.reached
    assert r.sum() >= 5 and not r[pmu].any()
    np.testing.assert_allclose(pv.v[r], np.abs(V)[r], atol=1e-12)
    np.testing.assert_allclose(pv.theta[r], np.angle(V)[r], atol=1e-12)
    assert np.isnan(pv.v[~r]).all()


def test_eq3_noise_matches_its_propagated_variance():
    V, Yf, Yt, f, t, pmu, cm = _case14()
    links = pmu_pseudo_links(pmu, cm, f, t, Yf, Yt)
    cur = branch_currents(V, Yf, Yt) * cm
    sv, st, si = 2e-3, 3e-3, 4e-3
    rng = np.random.default_rng(0)
    n, N = 20000, len(V)
    vv = np.abs(V) + rng.normal(0, sv, (n, N))
    tt = np.angle(V) + rng.normal(0, st, (n, N))
    cc = cur + rng.normal(0, si, (n, *cur.shape)) * cm
    mc = pmu_pseudo_voltages(links, N, vv, tt, cc, sv, st, si)
    an = pmu_pseudo_voltages(links, N, np.abs(V), np.angle(V), cur, sv, st, si)
    r = an.reached
    # 20000 draws: the sample variance is within about 3% of the true one (2 sigma of its own spread)
    np.testing.assert_allclose(mc.v[:, r].var(0), an.var_v[r], rtol=0.05)
    np.testing.assert_allclose(mc.theta[:, r].var(0), an.var_theta[r], rtol=0.05)


def test_pmu_pseudo_fills_only_unmetered_slots_and_needs_currents(files):
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.errors import MissingCapability
    from fdia_graph.se import WLS

    old, new = files
    with pytest.raises(MissingCapability, match="PMU currents"):
        WLS(pmu_pseudo=True).fit(FdiaGraph(old, split="train"))
    plain = WLS().fit(FdiaGraph(new, split="train"))
    est = WLS(pmu_pseudo=True).fit(FdiaGraph(new, split="train"))
    assert est.m == plain.m + len(est._pseudo_v) + len(est._pseudo_th) and len(est._pseudo_th) > 0
    x = est.estimate(FdiaGraph(new, split="test"))
    assert np.isfinite(x).all()


def test_new_generation_is_hybrid_and_the_legacy_recipe_pins_v083():
    import inspect

    from frozen_spec import TIMELINE_KW

    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.models.config import MeterSettings
    from fdia_graph.timeline import generate_timeline

    assert MeterSettings().meter_model == "hybrid"  # its own knob (D12), in the meter plan
    assert "meter_model" not in inspect.signature(generate_timeline).parameters  # no new top-level knob
    assert inspect.signature(FdiaGenerator).parameters["meter_model"].default == "v083"
    assert TIMELINE_KW["redundancy"] == {"meter_model": "v083"}  # the v0.8.3 recipe pins it
    with pytest.raises(ValueError):
        MeterSettings(meter_model="scada")


def test_the_previous_frame_carries_its_currents(files):
    from fdia_graph.dataset import FdiaGraph

    old, new = files
    ds = FdiaGraph(new, split="train")
    a = ds.export(["pmu_i", "prev_pmu_i"])
    rows = ds.idx
    with h5py.File(new) as f:
        whole = f["data/pmu_i"][:]
    np.testing.assert_array_equal(a["prev_pmu_i"], whole[np.maximum(rows - 1, 0)])
    with pytest.raises(ValueError, match="prev_pmu_i"):
        FdiaGraph(old, split="train").export(["prev_pmu_i"])


def test_jacobian_weighting_takes_the_pseudo_measurements(files):
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.se import JacobianWeighting

    _, new = files
    est = JacobianWeighting(pmu_pseudo=True).fit(FdiaGraph(new, split="train"))
    assert len(est._pseudo_th) > 0
    assert np.isfinite(est.estimate(FdiaGraph(new, split="test"))).all()


def test_a_current_bias_scales_with_the_end_phasor():
    """Review fix: the systematic error of a current channel is its relative bias times |I_end| (the
    C37.118 scale of `current_sigma`), so the Re channel of I = 1j still carries it."""
    from fdia_graph.formulas.noise import CURRENT_FLOOR_PU, biased_current

    true = np.array([[0.0, 1.0, 0.0, 0.0]])  # I_from = 1j pu, no PMU current at the to end
    bias = np.array([[0.004, -0.002, 0.003, 0.001]])
    got = biased_current(true, bias)
    assert got[0, 0] == pytest.approx(0.004) and got[0, 1] == pytest.approx(1.0 - 0.002)
    assert got[0, 2] == 0.0 and got[0, 3] == 0.0  # a zero end phasor carries no bias
    sig = current_sigma(true, PMU_CURRENT_CLASS)
    assert sig[0, 0] == pytest.approx(PMU_CURRENT_CLASS + CURRENT_FLOOR_PU) == sig[0, 1]
