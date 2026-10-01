"""The one-file timeline writer: its layout, its episodes and the At and Am frames.

One tiny IEEE-14 timeline is generated once per session (1000 frames, At and the overload Am,
20-frame episodes, the search capped as in conftest.TIMELINE_KW), and read here with h5py against
the layout the module docstring promises."""

import os

import h5py
import numpy as np
import pytest

pytest.importorskip("pandapower")

from fdia_graph.dataset.base import STEALTHY_FAMILIES  # noqa: E402
from fdia_graph.generation import _load_states  # noqa: E402
from fdia_graph.timeline import KIND, generate_timeline  # noqa: E402

SEED = 2  # puts At and Am in every split of the 1000-frame fixture


@pytest.fixture(scope="session")
def pool():
    return _load_states(14, None)[:1000]


@pytest.fixture(scope="session")
def timeline(tmp_path_factory, pool):
    """At and the overload Am, 1000 frames, 20-frame episodes, the search capped at 16 supports."""
    out = tmp_path_factory.mktemp("timeline") / "all.h5"
    return generate_timeline(
        14, states=pool, seed=SEED, ramp_len=20, min_tamper=False, min_budget=16, out=str(out)
    )


def _read(path):
    with h5py.File(path, "r") as f:
        arrays = {}
        f.visititems(
            lambda name, obj: arrays.__setitem__(name, obj[()]) if isinstance(obj, h5py.Dataset) else None
        )
        return arrays, dict(f.attrs)


def test_layout_is_the_documented_one(timeline):
    a, attrs = _read(timeline)
    T, N, E = attrs["T"], attrs["N"], attrs["E"]
    assert attrs["kind"] == KIND and attrs["system"] == 14 and T == 1000
    shapes = {
        "data/node_x": (T, N, 4),
        "data/node_m": (T, N, 4),
        "data/edge_x": (T, E, 2),
        "data/edge_m": (T, E, 2),
        "data/y": (T, N),
        "data/temporal_delta": (T, N, 2),
        "data/swing": (T, N, 2),
        "benign/node_benign": (T, N, 4),
        "benign/edge_benign": (T, E, 2),
        "clean/node_clean": (T, N, 4),
        "clean/edge_clean": (T, E, 2),
        "attack/node_tamper": (T, N, 4),
        "attack/edge_tamper": (T, E, 2),
        "graph/edge_index": (2, E),
    }
    for name, shape in shapes.items():
        assert a[name].shape == shape, name
    for name in ("family", "stealthy", "seq_id", "timestep", "split"):
        assert a[f"data/{name}"].shape == (T,), name
    assert a["data/timestep"].tolist() == list(range(T))
    K = len(a["episodes/onset"])
    assert a["episodes/bus_ptr"].shape == (K + 1,) and a["episodes/bus_ptr"][-1] == len(a["episodes/bus_idx"])
    assert a["attack/mag_ptr"].shape == (T + 1,) and a["attack/mag_ptr"][-1] == len(a["attack/mag"])
    assert len(a["attack/mag_bus"]) == len(a["attack/mag"])
    assert attrs["families"] == "0benign,1Aq,2Ad,3As,4Ar,5At,6Al,7Am"
    assert attrs["fallback_benign"] >= 0 and attrs["n_episodes"] == K


def test_every_family_appears_and_frames_are_balanced(timeline):
    a, _ = _read(timeline)
    fam = a["data/family"]
    present = set(np.unique(fam).tolist())
    assert present == {0, 5, 7}
    counts = np.bincount(fam[fam > 0], minlength=8)[[5, 7]]
    # the schedule weights families by inverse episode length; neither under a fifth of the mean
    assert counts.min() >= counts.mean() / 5, counts
    assert abs((fam > 0).mean() - 0.5) < 0.15


def test_episodes_index_the_frames(timeline):
    a, _ = _read(timeline)
    fam, seq, y = a["data/family"], a["data/seq_id"], a["data/y"]
    onset, length, efam = a["episodes/onset"], a["episodes/length"], a["episodes/family"]
    ptr, idx = a["episodes/bus_ptr"], a["episodes/bus_idx"]
    assert (seq[fam == 0] == -1).all() and (seq[fam > 0] >= 0).all()
    for k in range(len(onset)):
        rows = slice(onset[k], onset[k] + length[k])
        assert set(np.unique(fam[rows]).tolist()) <= {0, int(efam[k])}
        assert (seq[rows][fam[rows] > 0] == k).all()
        buses = set(idx[ptr[k] : ptr[k + 1]].tolist())
        assert set(np.where(y[rows].any(0))[0].tolist()) == buses
    assert sorted(np.unique(seq[seq >= 0]).tolist()) == list(range(len(onset)))
    assert set(np.unique(efam).tolist()) <= {5, 7}


def test_episodes_are_placed_at_random_without_overlap(timeline):
    """No episode overlaps another or crosses a split, the attacked fraction lands on the target, and
    the onsets spread over the whole timeline (no scheduler decides where an episode goes)."""
    a, attrs = _read(timeline)
    onset, length = a["episodes/onset"], a["episodes/length"]
    order = np.argsort(onset)
    assert (onset[order][1:] >= (onset + length)[order][:-1]).all()
    # each split's whole episodes come to half its frames (600/200/200 frames, 20-frame episodes:
    # 15/5/5 episodes), an infeasible episode moved rather than left benign; no frame of a built
    # episode falls back on this fixture
    assert int(length.sum()) == round(0.5 * attrs["T"]) and attrs["fallback_benign"] == 0
    assert round(attrs["attacked_frac"] * attrs["T"]) == int(length.sum())
    assert (attrs["episode_shortfall"] == 0).all() and np.allclose(attrs["split_attacked_frac"], 0.5)
    split = a["data/split"]
    assert all(split[o] == split[o + L - 1] for o, L in zip(onset, length))
    thirds = np.bincount(np.minimum(onset * 3 // attrs["T"], 2), minlength=3) / len(onset)
    assert thirds.min() > 0.2, thirds


def test_split_is_chronological_and_never_cuts_an_episode(timeline):
    a, _ = _read(timeline)
    split, onset, length = a["data/split"], a["episodes/onset"], a["episodes/length"]
    assert (np.diff(split) >= 0).all() and set(np.unique(split).tolist()) == {0, 1, 2}
    for o, n in zip(onset, length):
        assert len(np.unique(split[o : o + n])) == 1
    assert abs((split == 0).mean() - 0.6) < 0.1


def test_layers_and_tamper_masks_agree(timeline):
    a, _ = _read(timeline)
    fam = a["data/family"]
    nx, bn, cl, nm = a["data/node_x"], a["benign/node_benign"], a["clean/node_clean"], a["data/node_m"]
    nt, et = a["attack/node_tamper"], a["attack/edge_tamper"]
    ex, be, em = a["data/edge_x"], a["benign/edge_benign"], a["data/edge_m"]
    ben = fam == 0
    assert (nx[ben] == bn[ben]).all() and (ex[ben] == be[ben]).all()
    assert nt[ben].sum() == 0 and et[ben].sum() == 0
    assert (nt[nm == 0] == 0).all() and (et[em == 0] == 0).all()
    # untouched meters read the un-attacked value; a tampered meter is where the attacker wrote
    assert (nx[nt == 0] == bn[nt == 0]).all() and (ex[et == 0] == be[et == 0]).all()
    # At and Am write the meters of a local region: some, never none, never all
    stealthy = np.isin(fam, [5, 7])
    assert (nt[stealthy] <= nm[stealthy]).all()
    share = nt[stealthy].sum(axis=(1, 2)) / nm[stealthy].sum(axis=(1, 2))
    assert share.max() < 1 and (share[fam[stealthy] == 5] > 0).all()
    # an overload Am's first snapshot may tamper nothing: its drift-free goal there is the true flow (D9)
    assert (share[fam[stealthy] == 7] > 0).mean() > 0.9
    # the benign layer is the clean truth plus a small meter error on metered voltages
    v = nm[:, :, 0] > 0
    assert np.abs(bn[:, :, 0] - cl[:, :, 0])[v].max() < 0.02
    assert (a["data/stealthy"] == np.isin(fam, sorted(STEALTHY_FAMILIES))).all()


def test_generate_stream_is_the_timeline_as_a_dict(tmp_path, pool):
    import fdia_graph as fg

    with pytest.warns(DeprecationWarning, match="generate_stream is deprecated"):
        s = fg.generate_stream(
            14, pool[:60], 0.5, ("At",), ramp_len=10, seed=SEED, out=str(tmp_path / "s.h5"), min_tamper=False
        )
    with pytest.raises(TypeError):  # a 0.20 positional call: attack_intensity sat where ramp_rate is now
        fg.generate_stream(14, pool[:60], 0.5, ("At",), 0.2, 0.002)
    assert s.system == 14 and s.node_x.shape == (60, 14, 4) and s.node_m.shape == (14, 4)
    assert set(np.unique(s.family).tolist()) <= {0, 5} and len(s.episodes) > 0
    assert os.path.exists(tmp_path / "s.h5")


def test_stealthy_families_pass_the_residual_test(timeline):
    """Every stealthy frame is an exact local AC state plus the true scan's own meter noise: a WLS
    residual test at the 1% benign alarm level flags a stealthy frame no more often than it flags
    that frame's own benign twin (a noisy stretch of true states raises both alike)."""
    pytest.importorskip("torch")
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.se import WLS

    train, test = FdiaGraph(timeline, split="train"), FdiaGraph(timeline, split="test", order="time")
    est = WLS().fit(train)
    d = test.export(["node_x", "edge_x", "family", "benign", "edge_benign"])

    def alarm(nx, ex):
        z = est._z_of(nx, ex)
        return np.abs(est._nres(est._solve(z), z)).max(axis=1)

    r, twin = alarm(d["node_x"], d["edge_x"]), alarm(d["benign"], d["edge_benign"])
    level = np.quantile(r[d["family"] == 0], 0.99)
    for fid in (5, 7):
        rows = d["family"] == fid
        if rows.sum() >= 10:
            assert (r[rows] > level).mean() <= (twin[rows] > level).mean() + 0.05, fid


def test_a_stealthy_frame_is_the_benign_scan_plus_its_attack_vector(timeline):
    """observed - benign on an At frame equals h(x_false) - h(x_true) of the false state rebuilt
    from the frame's own labels (targets, magnitudes, the clean state), on every tampered channel,
    and is zero elsewhere: no second noise draw enters."""
    from fdia_graph.engine import FdiaGenerator

    a, attrs = _read(timeline)
    red = {k: float(attrs[k]) for k in ("vbus_frac", "pmu_frac", "flow_frac")}  # the file's meter plan
    g = FdiaGenerator(int(attrs["system"]), seed=int(attrs["seed"]), **red)
    pos = {int(b): i for i, b in enumerate(g.load_bus)}
    ptr, mag, mag_bus = a["attack/mag_ptr"], a["attack/mag"], a["attack/mag_bus"]
    frames = np.flatnonzero(a["data/family"] == 5)[:20]
    assert len(frames) >= 5
    for t in frames:
        Xt = a["clean/node_clean"][t].astype(np.float64)  # the pool state the frame was emitted from
        targets = np.array([pos[int(b)] for b in mag_bus[ptr[t] : ptr[t + 1]]])
        dev = mag[ptr[t] : ptr[t + 1]].astype(np.float64)  # unsigned: a rise or a drop of that size
        nt, et = a["attack/node_tamper"][t] > 0, a["attack/edge_tamper"][t] > 0
        dn = a["data/node_x"][t].astype(np.float64) - a["benign/node_benign"][t]
        de = a["data/edge_x"][t].astype(np.float64) - a["benign/edge_benign"][t]
        region = g.local_region(g.load_bus[targets], int(attrs["hops"]))
        rebuilt = []
        for sign in (1.0, -1.0):
            Lp = g.true_load(Xt)
            Lp[targets] *= 1.0 + sign * dev
            Xa = g.solve_local(Xt, region, Lp, g.true_reactive_load(Xt))
            if Xa is not None:
                rebuilt.append(g._attack_vector(Xa, Xt))
        # exactly one direction reproduces the frame; the clean layer is the pool state in float32, so
        # the rebuilt vector agrees to ~1e-4 MW, where a second noise draw would differ by about 1%
        match = [(n, e) for n, e in rebuilt if np.allclose(dn[nt], n[nt], rtol=1e-3, atol=1e-2)]
        assert len(match) == 1
        a_node, a_edge = match[0]
        assert np.abs(a_node[nt]).max() > 1e-2
        assert np.allclose(de[et], a_edge[et], rtol=1e-3, atol=1e-2)
        assert not dn[~nt].any() and not de[~et].any()


def test_a_slack_bus_load_is_never_a_target():
    """IEEE-57 carries a load on its slack bus; the slack never enters a region, so that load is
    not attackable (a frame scaling it would be labelled attacked with no attack in the state)."""
    from fdia_graph.engine import FdiaGenerator

    g = FdiaGenerator(57, seed=1)
    assert g.slack_bus in set(g.load_bus.tolist())
    assert g.slack_bus not in set(g.load_bus[g.attackable_pos].tolist())


def test_area_equivalent_loads_are_never_targets():
    """IEEE-145 lumps regions into gigawatt loads; the cap keeps them out of the target set and
    leaves every other ladder system's set as it was."""
    from fdia_graph.engine import FdiaGenerator

    g = FdiaGenerator(145, seed=1)
    p = np.abs(g.base.load.p_mw.values)
    assert (p > 2000).sum() >= 10 and p[g.attackable_pos].max() <= 2000
    assert set(g.attackable_pos) == set(np.flatnonzero((p > 0) & (p <= 2000)))
    g14 = FdiaGenerator(14, seed=1)
    assert set(g14.attackable_pos) == set(FdiaGenerator(14, seed=1, max_load_mw=None).attackable_pos)
    with pytest.raises(ValueError):
        FdiaGenerator(14, seed=1, max_load_mw=0)


def test_a_step_without_a_local_solution_is_halved(monkeypatch):
    """A ramp frame is built at the largest halving of its step that solves (STEP_HALVINGS at most),
    past the noise floor: its design step is sub-floor anyway."""
    from fdia_graph.engine import FdiaGenerator
    from fdia_graph.models.frames import AttackDesign, FrameKnobs

    g = FdiaGenerator(14, seed=1)
    steps: list[float] = []

    def fake(Xt, design, k):
        steps.append(float(np.max(np.abs(np.asarray(design.mult) - 1.0))))
        return "frame" if steps[-1] <= 0.01 else None

    monkeypatch.setattr(g, "_stealthy_frame", fake)
    k = FrameKnobs(hops=2)
    Xt = np.zeros((g.C, 4))
    two = g.stealthy_pos[:2]
    assert g._ramp_frame(Xt, AttackDesign(two, 1.2), k) == "frame"
    assert np.allclose(steps, [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])
    steps.clear()
    assert g._ramp_frame(Xt, AttackDesign(two, 1.9), k) is None  # six halvings, none small enough
    assert len(steps) == 7


def test_operating_limits_are_the_case_limits_widened_to_the_pool():
    """Bus and generator limits are the case's; a true state already outside one is its own bound
    and may not be made worse; the generator output behind a pool state is recovered exactly, and
    a false state that pushes a generator past its cap or a bus past its limit is refused."""
    from fdia_graph.engine import FdiaGenerator
    from fdia_graph.formulas.attacks import generator_output, within_limits

    g = FdiaGenerator(14, seed=1)
    net = g.base
    X0 = net.res_bus.reindex(sorted(net.bus.index))[["vm_pu", "p_mw", "q_mvar", "va_degree"]].to_numpy()
    for b, ps, qs in zip(net.shunt.bus, net.res_shunt.p_mw, net.res_shunt.q_mvar):
        X0[int(b), 1:3] -= (ps, qs)  # the pool stores injections without the shunt draw
    X = np.stack([X0, X0 * [[1.0, 1.3, 1.3, 1.0]]])  # the pool scales load and generation together
    lim = g.operating_limits(X)
    assert (lim.v_lo == g.v_case[:, 0]).all() and (lim.v_hi == g.v_case[:, 1]).all()  # voltages verbatim
    assert (lim.p_lo <= g.p_lim[:, 0]).all() and (lim.p_hi >= g.p_lim[:, 1]).all()  # generators widened
    assert X0[:, 0].max() > lim.v_hi.max()  # the base case runs above 1.06 pu at some bus ...
    gen = generator_output(X0, g.load_base, g.gen_base)
    on = np.flatnonzero(g.gen_base[:, 0] > 0)
    assert np.allclose(gen[on, 0], g.gen_base[on, 0])  # the base state: base generation exactly
    none = np.zeros(g.C)
    everywhere = np.arange(g.C)  # every bus inside the attacked subnetwork
    assert within_limits(X0, X0, gen, none, lim, everywhere)  # ... and is acceptable as its own bound
    worse = X0.copy()
    worse[int(np.argmax(X0[:, 0])), 0] += 0.01  # further above the limit than the true state
    assert not within_limits(worse, X0, gen, none, lim, everywhere)
    bad = X0.copy()
    bad[5, 0] = 0.8
    assert not within_limits(bad, X0, gen, none, lim, everywhere)
    bad = X0.copy()
    bad[on[0], 1] -= 1e4  # a 10 GW injection change at a generator bus inside the region: past any cap ...
    assert not within_limits(bad, X0, gen, none, lim, everywhere)
    pretended = none.copy()
    pretended[on[0]] = -1e4  # ... unless it is the load change the attacker pretends there
    assert within_limits(bad, X0, gen, pretended, lim, everywhere)
    outside = np.array([b for b in range(g.C) if b != on[0]])  # the generator on the boundary: unconstrained
    assert within_limits(bad, X0, gen, none, lim, outside)


def test_the_fixture_records_its_limits(timeline):
    _, attrs = _read(timeline)
    # an episode with no feasible design moves to another onset instead of falling back (the
    # split-first generation), and no frame of a built episode falls back here
    assert attrs["fallback_benign"] == 0 and attrs["max_load_mw"] == 2000.0
    assert attrs["v_lo"] <= 0.94 and attrs["v_hi"] >= 1.06


def test_local_region_keeps_a_boundary_and_the_slack_fixed():
    from fdia_graph.engine import FdiaGenerator
    from fdia_graph.formulas.network import subnetwork

    g = FdiaGenerator(14, seed=1)
    for hops in (0, 1, 2, 3):
        region = g.local_region(g.load_bus[:5], hops)
        assert region is not None and g.slack_bus not in region
        _, boundary = subnetwork(g.ei, region, 0, g.C)
        assert len(boundary) and not set(boundary.tolist()) & (set(g.zero_inj) - {g.slack_bus})
    whole = g.local_region(
        np.arange(g.C), 0
    )  # every bus as a seed: the slack alone is left to balance against
    assert whole is not None and sorted(whole.tolist()) == sorted(set(range(g.C)) - {g.slack_bus})


def test_attacked_frac_zero_is_all_benign(tmp_path, pool):
    out = generate_timeline(14, states=pool[:40], attacked_frac=0.0, out=str(tmp_path / "b.h5"))
    a, attrs = _read(out)
    assert (a["data/family"] == 0).all() and attrs["n_episodes"] == 0 and attrs["attacked_frac"] == 0.0
    with pytest.raises(ValueError, match="attacked_frac"):
        generate_timeline(14, states=pool[:20], attacked_frac=1.5, out=str(tmp_path / "x.h5"))


def test_empty_episode_lengths_are_refused(tmp_path, pool):
    for bad in (dict(ramp_len=0), dict(am_len=0)):
        with pytest.raises(ValueError, match="TimelineKnobs.(ramp_len|am_len) must be >= 1"):
            generate_timeline(14, states=pool[:20], out=str(tmp_path / "x.h5"), **bad)
    with pytest.raises(ValueError, match="hops"):
        generate_timeline(14, states=pool[:20], out=str(tmp_path / "x.h5"), hops=0)


def test_score_bundles_accept_the_seventh_family():
    """`SEBase.score` and `LocalizerBase.score` build their bundles from `FAMILIES`, so both carry
    an `Am` slot once a scored view holds Am frames."""
    from fdia_graph.models import ErrorPair, EstimatorScores, FamilyMetrics, LocalizerScores, OverallMetrics

    se = EstimatorScores(geo=ErrorPair(1.0, 0.1), Am=ErrorPair(2.0, 0.2))
    assert list(se) == ["Am", "geo"] and se["Am"].angle_mae_deg == 2.0
    fam = FamilyMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    loc = LocalizerScores(all=OverallMetrics(1.0, 1.0, 0.0, 1.0), Am=fam)
    assert list(loc) == ["all", "Am"] and loc.Am is fam


def test_a_producing_static_generator_bus_injects():
    """IEEE-89 buses 1, 65, 69, 79 are fed by a static generator alone: injection buses, metered
    as such, never zero-injection junctions; IEEE-200's static generators all produce 0 MW, so their
    buses stay zero-injection unless another element sits there."""
    from fdia_graph.engine import FdiaGenerator

    g = FdiaGenerator(89, seed=1)
    for b in (1, 65, 69, 79):
        assert b not in g.zero_inj and b in g.meters.inj
    g = FdiaGenerator(200, seed=1)
    base = g.base
    others = set(base.gen.bus) | set(base.load.bus) | set(base.ext_grid.bus) | set(base.shunt.bus)
    idle = [int(b) for b in base.sgen.bus if int(b) not in others]
    assert idle and all(b in g.zero_inj for b in idle)


def test_reactive_load_adds_back_the_generators_output():
    """At a load bus with a generator the stored Q injection is load minus generator output, so the
    reactive load adds it back; the local solve with unchanged loads reproduces the scan."""
    from fdia_graph.engine import FdiaGenerator
    from fdia_graph.profiles import _case_buses, _solve_states_chunk

    g = FdiaGenerator(14, seed=5)
    X = _solve_states_chunk(14, np.full((1, len(_case_buses(14))), 1.1))[0]
    Lq = g.true_reactive_load(X)
    for pos, b in enumerate(g.load_bus):
        want = 1.1 * g.load_base[b, 1] if g.has_gen[b] else X[b, 2]
        assert Lq[pos] == pytest.approx(want, abs=1e-6)
    interior = g.local_region(g.load_bus[[2]], 1)
    Xa = g.solve_local(X, interior, g.true_load(X), Lq)
    assert np.allclose(Xa, X, atol=1e-5)


def test_stealthy_families_never_target_a_generator_bus(timeline):
    """The ramp skips every load on a generator bus, zero-MW condensers included [BOY22] (an
    overload Am's labels are its support's free-injection buses, generators among them)."""
    import pandapower.networks as pn

    a, _ = _read(timeline)
    gen = np.zeros(a["data/y"].shape[1], bool)
    gen[pn.case14().gen.bus.values] = True
    fam, y = a["data/family"], a["data/y"].astype(bool)
    ramp = fam == 5
    assert ramp.any() and not (y[ramp] & gen).any()


def test_stealthy_targets_avoid_every_generator_and_live_static_generator_bus():
    """On the two systems with static generators, no stealthy target sits on a bus holding a
    generator or a static generator that produces P or Q; the in-place set is not narrowed by it."""
    from fdia_graph.engine import FdiaGenerator

    for C in (89, 300):
        g = FdiaGenerator(C, seed=1)
        sg = g.base.sgen[(g.base.sgen.p_mw.abs() > 0) | (g.base.sgen.q_mvar.abs() > 0)]
        banned = set(g.base.gen.bus) | set(sg.bus)
        assert not banned & set(g.load_bus[g.stealthy_pos].tolist())
        assert set(g.stealthy_pos) <= set(g.attackable_pos)


def test_a_family_with_nothing_to_attack_is_refused():
    from types import SimpleNamespace

    from fdia_graph.engine.attacks.episodes import EpisodeDesignMixin
    from fdia_graph.errors import NoAdmissibleTarget
    from fdia_graph.models.inputs import AdmissibleTargets

    none = EpisodeDesignMixin.target_counts(SimpleNamespace(stealthy_pos=np.array([], int)))
    AdmissibleTargets(["benign"], none)
    with pytest.raises(NoAdmissibleTarget, match="At"):
        AdmissibleTargets(["At"], none)
    with pytest.raises(NoAdmissibleTarget, match="At"):
        AdmissibleTargets(["ramp"], none)  # an alias resolves first


def test_the_single_snapshot_families_are_read_only(tmp_path, pool):
    """Generation makes At and Am; asking for an older family is refused with the families it makes,
    before any frame is walked (the families stay readable in old releases)."""
    from fdia_graph.models.inputs import GeneratedFamilies

    assert GeneratedFamilies(["At", "Am", "ramp"]).codes == (5, 7, 5)
    for old in (["Aq"], ["At", "Ad"], ["LRA"]):
        with pytest.raises(ValueError, match="makes At, Am only"):
            generate_timeline(14, states=pool[:20], families=old, out=str(tmp_path / "x.h5"))


def test_temporal_features_read_only_the_observed_frames(timeline, tmp_path):
    """delta and swing are functions of the observed node_x alone: rewriting the clean and benign
    layers and recomputing leaves both unchanged, and the stored layers equal a recomputation."""
    import shutil

    import h5py

    from fdia_graph import schema
    from fdia_graph.timeline import write_temporal_layers

    copy = str(tmp_path / "copy.h5")
    shutil.copyfile(timeline, copy)
    with h5py.File(copy, "r+") as f:
        stored = f[schema.SWING][:], f[schema.TEMPORAL_DELTA][:]
        for name in (schema.NODE_CLEAN, schema.NODE_BENIGN):
            f[name][...] = f[name][:] * 1.5 + 0.1
        write_temporal_layers(f)
        assert np.array_equal(f[schema.SWING][:], stored[0]) and np.array_equal(
            f[schema.TEMPORAL_DELTA][:], stored[1]
        )


def test_block_scale_matches_the_whole_series_kernel(timeline):
    """The bounded-memory scale equals the kernel over the whole observed series, for blocks shorter
    than the window and longer than it."""
    import h5py

    from fdia_graph import schema
    from fdia_graph.formulas.temporal import SWING_WINDOW, recent_change_scale
    from fdia_graph.timeline import _block_scale

    with h5py.File(timeline, "r") as f:
        nx = f[schema.NODE_X]
        T, N = nx.shape[0], nx.shape[1]
        pq = np.zeros((T, N, 4))
        pq[:, :, 1:3] = nx[:, :, 1:3]
        whole = recent_change_scale(pq, SWING_WINDOW, N)
        for block in (7, 250):
            parts = np.concatenate([_block_scale(nx, a, min(a + block, T)) for a in range(0, T, block)])
            assert np.allclose(parts, whole, rtol=1e-6, atol=1e-9)
