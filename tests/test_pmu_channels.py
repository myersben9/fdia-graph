"""The hybrid meter model: angles at the PMU buses only, the PMU branch-current
phasors [WU26 eqs. 19-20] emitted, stored, loaded and attacked, and the eq. (3) pseudo-measurements
an estimator can derive from them. The v0.8.3 meter model stays the default and writes none of it."""

import h5py
import numpy as np
import pytest

from fdia_graph.constants.channels import CurrentChannel, NodeChannel
from fdia_graph.constants.meters import PMU_CURRENT_CLASS
from fdia_graph.formulas.attacks import tampered_devices
from fdia_graph.formulas.noise import current_sigma

pytest.importorskip("pandapower")


@pytest.fixture(scope="module")
def gens():
    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.systems.network import Network

    return FdiaGenerator(
        Network.from_ieee(14), seed=1
    )  # the hybrid meters, the only model the generator makes


@pytest.fixture(scope="module")
def pool():
    from fdia_graph.generation import _load_states

    return _load_states(14, None)[:200]


def test_a_scada_voltmeter_reads_no_angle_under_the_hybrid_model(gens):
    g = gens
    nm = g.masks.node
    pmu = np.zeros(g.n_bus, bool)
    pmu[g.pmu_buses] = True
    assert (nm[:, NodeChannel.THETA].astype(bool) == pmu).all()  # the angle is a PMU channel
    assert (nm[:, NodeChannel.V].astype(bool) >= pmu).all() and nm[
        :, NodeChannel.V
    ].sum() > pmu.sum()  # SCADA reads |V|


def test_the_current_mask_follows_the_pmus(gens):
    g = gens
    cm = g.masks.current
    pmu = np.zeros(g.n_bus, bool)
    pmu[g.pmu_buses] = True
    assert (cm[:, CurrentChannel.RE_FROM].astype(bool) == pmu[g.edge_index[0]]).all()
    assert (cm[:, CurrentChannel.IM_TO].astype(bool) == pmu[g.edge_index[1]]).all()


def test_emitted_currents_sit_within_their_accuracy_class(gens, pool):
    g = gens
    scan = g.emitter.emit_from_state(pool[0])
    true = g.emitter.currents_from_states(pool[:1])[0]
    assert scan.i_x is not None and scan.i_m is not None
    sig = current_sigma(true, PMU_CURRENT_CLASS)
    on = scan.i_m > 0
    assert (np.abs(scan.i_x[on] - true[on]) < 6 * sig[on]).all()
    assert (scan.i_x[~on] == 0).all()


def test_the_exact_currents_carry_the_exact_flows(gens, pool):
    """S_from = V_from conj(I_from): the current channels agree with the flows of the same state."""
    g = gens
    X = pool[:3]
    cur = g.emitter.currents_from_states(X)
    flows = g.emitter.clean_flows_from_states(X)  # [T, E, 2] MW/MVAr, unmetered zeroed
    both = (g.masks.current[:, CurrentChannel.RE_FROM] > 0) & (flows[0, :, 0] != 0)
    th = np.deg2rad(X[:, :, NodeChannel.THETA])
    v_from = (X[:, :, NodeChannel.V] * np.exp(1j * th))[:, g.edge_index[0]]
    S = (
        v_from
        * np.conj(cur[..., CurrentChannel.RE_FROM] + 1j * cur[..., CurrentChannel.IM_FROM])
        * g.case.base_mva
    )
    assert both.any()
    np.testing.assert_allclose(S.real[:, both], flows[:, both, 0], atol=1e-4)
    np.testing.assert_allclose(S.imag[:, both], flows[:, both, 1], atol=1e-4)


def test_a_false_state_writes_its_currents(gens, pool):
    # [WU26 eqs. 19-20]
    """a = h(x^a) - h(x) on the current channels, on exactly the PMU-read ends."""
    g = gens
    x_true = np.asarray(pool[5], float)
    x_false = x_true.copy()
    x_false[3, NodeChannel.THETA] += 0.05
    scan = g.emitter.emit_from_state(x_true)
    a = g.attack_frames.current_attack(x_false, x_true)
    ix, tamper = g.attack_frames.currents_with_attack(scan, x_false, x_true)
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
    cur[0, CurrentChannel.RE_TO] = True  # the to end of branch 0 is bus 1
    cur[1, CurrentChannel.IM_FROM] = True  # the from end of branch 1 is bus 1 as well
    pmu = np.array([False, True, False, False])
    assert tampered_devices(node, edge, pmu, ei[0], cur, ei[1]).tolist() == [N + 1]


def test_the_search_counts_the_current_channels(gens, pool):
    from fdia_graph.engine.attacks.window import AttackWindow
    from fdia_graph.models.frames import AttackDesign, FrameKnobs, LoadGoal

    g = gens
    k = FrameKnobs(2, g.false_state.operating_limits(pool), True, 256)
    pmu = g.pmu_buses.tolist()
    target = next(
        i for i, b in enumerate(g.load_bus) if b in pmu or any(n in pmu for n in g.area.local_region([b], 1))
    )
    states = [pool[0], pool[1]]
    goal = LoadGoal(tuple(AttackDesign(np.array([target]), 1.03) for _ in states))
    window = AttackWindow(g, states, goal, k, stealth_bound=False)
    assert window.i_m is not None and window.i_sigma is not None and len(window.i_sigma) == 2
    area = np.asarray(g.area.local_region(np.array([g.load_bus[target]]), k.hops))
    cost = window.cost(area, None)
    moved = window._snapshot(0, area, window.prev)
    assert cost is not None and moved is not None and moved[2] is not None and moved[2].any()
    window.i_m = np.zeros_like(window.i_m)  # the same window, blind to the currents
    blind = window.cost(area, None)
    assert blind is not None and blind[1] < cost[1]  # the current channels add to the tamper count


def test_the_hybrid_file_carries_the_current_layers_and_the_old_one_does_not(pmu_files):
    from fdia_graph import schema
    from tests.helpers import CURRENT_LAYERS

    old, new = pmu_files
    with h5py.File(old) as f:  # the v0.8.3 release: no current layers
        assert not any(p in f for p in CURRENT_LAYERS)
        assert "meter_model" not in f.attrs
    with h5py.File(new) as f:
        assert all(p in f for p in CURRENT_LAYERS)
        assert f.attrs["meter_model"] == "hybrid"
        fam = f[schema.FAMILY][:]
        i, ib, tam = f[schema.PMU_I][:], f[schema.PMU_I_BENIGN][:], f[schema.PMU_I_TAMPER][:]
        benign = fam == 0
        assert (i[benign] == ib[benign]).all() and not tam[benign].any()
        assert ((i != ib) == tam.astype(bool)).all()
        assert (~benign).any() and tam[~benign].any()  # the ramps write currents


def test_the_loader_and_export_serve_the_currents(pmu_files):
    from fdia_graph.dataset import FdiaGraph

    old, new = pmu_files
    ds = FdiaGraph(new, split="train")
    rec = ds[0]
    assert rec["pmu_i"].shape == (ds.E, 4) and "pmu_i_m" in rec and "pmu_i_benign" in rec
    a = ds.export()
    assert a["pmu_i"].shape == (len(ds), ds.E, 4)
    batch = ds.collate([ds[0], ds[1]])
    assert batch["pmu_i"].shape == (2, ds.E, 4)
    legacy = FdiaGraph(old, split="train")
    assert not legacy.has_currents and "pmu_i" not in legacy[0] and "pmu_i" not in legacy.export()


def test_new_generation_is_hybrid_only():
    import inspect

    from fdia_graph.engine.core import FdiaGenerator
    from fdia_graph.models.config import MeterSettings
    from fdia_graph.timeline import generate_timeline

    assert MeterSettings().meter_model == "hybrid"  # its own knob, in the meter plan
    assert "meter_model" not in inspect.signature(generate_timeline).parameters  # no new top-level knob
    assert inspect.signature(FdiaGenerator).parameters["meters"].default == MeterSettings()
    for old in ("v083", "scada"):  # the v0.8.3 meters stay readable in old releases, not generated
        with pytest.raises(ValueError):
            MeterSettings(meter_model=old)


def test_the_previous_frame_carries_its_currents(pmu_files):
    from fdia_graph.dataset import FdiaGraph

    old, new = pmu_files
    ds = FdiaGraph(new, split="train")
    a = ds.export(["pmu_i", "prev_pmu_i"])
    rows = ds.idx
    with h5py.File(new) as f:
        whole = f["data/pmu_i"][:]
    np.testing.assert_array_equal(a["prev_pmu_i"], whole[np.maximum(rows - 1, 0)])
    with pytest.raises(ValueError, match="prev_pmu_i"):
        FdiaGraph(old, split="train").export(["prev_pmu_i"])


def test_a_current_bias_scales_with_the_end_phasor():
    """Review fix: the systematic error of a current channel is its relative bias times |I_end| (the
    C37.118 scale of `current_sigma`), so the Re channel of I = 1j still carries it."""
    from fdia_graph.constants.meters import CURRENT_FLOOR_PU, KIND_ACCURACY, MeterKind
    from fdia_graph.sensing.meters import MeterPlan
    from fdia_graph.sensing.noise import NoiseModel
    from tests.helpers import FixedDraws

    # one branch with a PMU at each end, its four current channels in `CurrentChannel` order
    kind = np.array([MeterKind.IR, MeterKind.II, MeterKind.IR, MeterKind.II])
    ends = np.array([0, 0, 1, 1])
    rel, floor = KIND_ACCURACY[MeterKind.IR]
    plan = MeterPlan(
        kind=kind,
        bus=ends,
        branch=np.zeros(4, int),
        end=ends,
        owner=np.zeros(4, int),
        sigma_rel=np.full(4, rel),
        sigma_abs=np.full(4, floor),
        bias=np.array([0.004, -0.002, 0.003, 0.001]),
    )
    noise = NoiseModel(plan)
    true = np.array([0.0, 1.0, 0.0, 0.0])  # I_from = 1j pu, no current at the to end
    got = noise.sample(true, FixedDraws(0.0))  # type: ignore[arg-type]  # the readings before jitter
    assert got[0] == pytest.approx(0.004) and got[1] == pytest.approx(1.0 - 0.002)
    assert got[2] == 0.0 and got[3] == 0.0  # a zero end phasor carries no bias
    sig = noise.sigmas(true)
    assert sig[0] == pytest.approx(PMU_CURRENT_CLASS + CURRENT_FLOOR_PU) == sig[1]
    assert np.allclose(sig, current_sigma(true[None], PMU_CURRENT_CLASS)[0])
