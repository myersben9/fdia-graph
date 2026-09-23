"""The one-file timeline writer (docs/plans/ONE_DATASET_PLAN.md step 1) and the Am family.

Two tiny IEEE-14 timelines are generated once per session: every family over 400 frames, and Am
alone over 150 frames. The loader for these files lands in step 2; here the file is read with
h5py against the layout the module docstring promises."""

import os

import h5py
import numpy as np
import pytest

pytest.importorskip("pandapower")

from fdia_graph.dataset.base import STEALTHY_FAMILIES  # noqa: E402
from fdia_graph.engine.records import AM_FAMILY, CORRUPT_KIND  # noqa: E402
from fdia_graph.generation import NOISE_FLOOR, _load_states  # noqa: E402
from fdia_graph.timeline import KIND, generate_timeline  # noqa: E402

SEED = 4  # covers every family in the 1000-frame fixture (the long families are few per 1000 frames)


@pytest.fixture(scope="session")
def pool():
    return _load_states(14, None)[:1000]


@pytest.fixture(scope="session")
def timeline(tmp_path_factory, pool):
    """Every family, 1000 frames, short episodes so several of each family fit."""
    out = tmp_path_factory.mktemp("timeline") / "all.h5"
    return generate_timeline(14, states=pool, seed=SEED, ramp_len=20, out=str(out))


@pytest.fixture(scope="session")
def am_timeline(tmp_path_factory, pool):
    out = tmp_path_factory.mktemp("timeline") / "am.h5"
    return generate_timeline(14, states=pool[:150], seed=SEED, families=("Am",), ramp_len=30, out=str(out))


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
    assert present == {0, 1, 2, 3, 4, 5, 6, 7}
    counts = np.bincount(fam[fam > 0], minlength=8)[1:]
    # the schedule weights families by inverse expected length; no family under a fifth of the mean
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
    # corrupt-in-place families are independent one-frame draws by default (corrupt_len=1)
    assert (length[np.isin(efam, list(CORRUPT_KIND))] == 1).all()


def test_episodes_are_placed_at_random_without_overlap(timeline):
    """No episode overlaps another, the attacked fraction lands near the target, and the onsets
    spread over the whole timeline (no scheduler decides where an episode goes)."""
    a, attrs = _read(timeline)
    onset, length = a["episodes/onset"], a["episodes/length"]
    order = np.argsort(onset)
    assert (onset[order][1:] >= (onset + length)[order][:-1]).all()
    # the episodes' frames are exactly the set fraction; only a non-converging frame falls back to benign
    assert int(length.sum()) == round(0.5 * attrs["T"])
    assert round(attrs["attacked_frac"] * attrs["T"]) + attrs["fallback_benign"] == int(length.sum())
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
    corrupt = np.isin(fam, list(CORRUPT_KIND))
    assert ((nx != bn) == (nt > 0))[corrupt].all()
    # the stealthy families write the meters of a local region: some, never none, never all
    stealthy = np.isin(fam, [1, 5, 6, 7])
    assert (nt[stealthy] <= nm[stealthy]).all()
    share = nt[stealthy].sum(axis=(1, 2)) / nm[stealthy].sum(axis=(1, 2))
    assert 0 < share.min() and share.max() < 1
    # the benign layer is the clean truth plus a small meter error on metered voltages
    v = nm[:, :, 0] > 0
    assert np.abs(bn[:, :, 0] - cl[:, :, 0])[v].max() < 0.02
    assert (a["data/stealthy"] == np.isin(fam, sorted(STEALTHY_FAMILIES))).all()


def test_am_is_a_sparse_sub_floor_ramp_of_a_held_redistribution(am_timeline):
    a, attrs = _read(am_timeline)
    fam, y = a["data/family"], a["data/y"]
    am = np.where(fam == AM_FAMILY)[0]
    assert len(am) > 40 and (a["data/stealthy"][am] == 1).all()
    nt, nm = a["attack/node_tamper"], a["data/node_m"]
    share = nt[am].sum(axis=(1, 2)) / nm[am].sum(axis=(1, 2))
    assert 0 < share.min() and share.max() < 0.8  # the region's meters, never the whole grid
    ptr, bus, mag = a["attack/mag_ptr"], a["attack/mag_bus"], a["attack/mag"]
    per = {t: (bus[ptr[t] : ptr[t + 1]], mag[ptr[t] : ptr[t + 1]]) for t in am}
    onset, length = a["episodes/onset"], a["episodes/length"]
    cap = attrs["am_rate"] * NOISE_FLOOR
    for o, n in zip(onset, length):
        frames = [t for t in range(o, o + n) if t in per]
        buses = per[frames[0]][0]
        for t0, t1 in zip(frames, frames[1:]):
            assert (per[t1][0] == buses).all()  # the bus set is held for the whole episode
            # the designed per-bus fraction moves under the noise-floor cap per frame (the load
            # itself drifts a few percent between frames, hence the slack)
            assert np.abs(per[t1][1] - per[t0][1]).max() < 1.5 * cap
        peak = max(per[t][1].max() for t in frames)
        assert cap < peak <= attrs["attack_intensity"] * 1.1
        assert (y[frames][:, buses] == 1).all()


def test_generate_stream_is_the_timeline_as_a_dict(tmp_path, pool):
    import fdia_graph as fg

    with pytest.warns(DeprecationWarning, match="generate_stream is deprecated"):  # the pre-0.18 positions
        s = fg.generate_stream(
            14, pool[:60], 0.5, ("Am", "Ad"), 0.2, 0.002, 10, None, None, SEED, str(tmp_path / "s.h5")
        )
    assert s.system == 14 and s.node_x.shape == (60, 14, 4) and s.node_m.shape == (14, 4)
    assert set(np.unique(s.family).tolist()) <= {0, 2, 7} and len(s.episodes) > 0
    assert os.path.exists(tmp_path / "s.h5")


def test_am_direction_sign_follows_the_engine_convention():
    """`lra_delta` lowers the target line's |flow| in the false state, so mask keeps its sign, and
    a given "both" draw maps to the same sign as before the fix (+1 below 0.5)."""
    from fdia_graph.timeline import _am_sign

    rng = np.random.default_rng(0)
    assert _am_sign("mask", rng) == 1.0 and _am_sign("induce", rng) == -1.0
    draws = np.random.default_rng(7).random(50)
    rng = np.random.default_rng(7)
    assert [_am_sign("both", rng) for _ in range(50)] == [1.0 if r < 0.5 else -1.0 for r in draws]


def test_stealthy_families_pass_the_residual_test(timeline):
    """Every stealthy frame is an exact local AC state plus the true scan's own meter noise: a WLS
    residual test at the 1% benign alarm level flags a stealthy frame no more often than it flags
    that frame's own benign twin (a noisy stretch of true states raises both alike), and flags
    the in-place corruption of Ad."""
    pytest.importorskip("torch")
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.se import WLS

    train, test = FdiaGraph(timeline, split="train"), FdiaGraph(timeline, split="test", order="time")
    est = WLS().fit(train)
    d = test.export(["node_x", "edge_x", "clean", "family", "benign", "edge_benign"])
    thsl = est._truth_of(d["clean"])["thsl"]

    def alarm(nx, ex):
        z = est._z_of(nx, ex)
        return np.abs(est._nres(est._solve(z, thsl), z, thsl)).max(axis=1)

    r, twin = alarm(d["node_x"], d["edge_x"]), alarm(d["benign"], d["edge_benign"])
    level = np.quantile(r[d["family"] == 0], 0.99)
    for fid in (1, 5, 6, 7):
        rows = d["family"] == fid
        if rows.sum() >= 10:
            assert (r[rows] > level).mean() <= (twin[rows] > level).mean() + 0.05, fid
    ad = d["family"] == 2
    if ad.sum() >= 5:
        assert (r[ad] > level).mean() >= 0.8


def test_a_stealthy_frame_is_the_benign_scan_plus_its_attack_vector(timeline):
    """observed - benign on an Aq frame equals h(x_false) - h(x_true) of the false state rebuilt
    from the frame's own labels (targets, magnitudes, the clean state), on every tampered channel,
    and is zero elsewhere: no second noise draw enters."""
    from fdia_graph.engine import FdiaGenerator
    from fdia_graph.engine.records import _attack_vector

    a, attrs = _read(timeline)
    red = {k: float(attrs[k]) for k in ("vbus_frac", "pmu_frac", "flow_frac")}  # the file's meter plan
    g = FdiaGenerator(int(attrs["system"]), seed=int(attrs["seed"]), **red)
    pos = {int(b): i for i, b in enumerate(g.load_bus)}
    ptr, mag, mag_bus = a["attack/mag_ptr"], a["attack/mag"], a["attack/mag_bus"]
    frames = np.flatnonzero(a["data/family"] == 1)[:20]
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
                rebuilt.append(_attack_vector(g, Xa, Xt))
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


def test_an_open_branch_is_not_a_hop():
    """On an N-1 generator the opened line is not walked: its far bus is neither interior nor boundary
    unless another live path reaches it."""
    from fdia_graph.engine import FdiaGenerator
    from fdia_graph.formulas.network import subnetwork

    g = FdiaGenerator(14, seed=1, outage=0)  # line 0 joins buses 0 and 1
    assert g.branch.status[0] == 0
    live = g._live_edges()
    assert live.shape[1] == g.E - 1
    interior = g.local_region(np.array([1]), 0)  # bus 1 alone as the seed
    assert interior is not None
    assert 0 not in set(subnetwork(live, interior, 0, g.C)[1].tolist())  # the far bus is not a boundary
    reach_live = set(subnetwork(live, np.array([1]), 1, g.C)[0].tolist())
    reach_all = set(subnetwork(g.ei, np.array([1]), 1, g.C)[0].tolist())
    assert reach_all - reach_live == {0}


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
    """The frame is built at the largest halving of the step that solves; an Aq step stops at the
    noise floor, a ramp frame does not."""
    from fdia_graph.engine import FdiaGenerator, records
    from fdia_graph.models.frames import FrameKnobs

    g = FdiaGenerator(14, seed=1)
    steps: list[float] = []

    def fake(g_, Xt, targets, mult, interior, k):
        steps.append(float(np.max(np.abs(np.asarray(mult) - 1.0))))
        return "frame" if steps[-1] <= 0.01 else None

    monkeypatch.setattr(records, "_stealthy_frame", fake)
    k = FrameKnobs(0.2, 0.02, 6, None, False, True, hops=2)
    Xt = np.zeros((g.C, 4))
    two = g.attackable_pos[:2]
    assert records._resolve_frame(g, Xt, 1, two, np.array([1.2, 1.1]), k) is None
    assert np.allclose(steps, [0.2, 0.1, 0.05, 0.025])  # three halvings above the floor, none solved
    steps.clear()
    assert records._resolve_frame(g, Xt, 1, two, np.array([1.05, 1.05]), k) is None
    assert np.allclose(steps, [0.05, 0.025])  # the next halving would fall under the floor
    steps.clear()
    assert records._resolve_frame(g, Xt, 5, two, 1.2, k) == "frame"
    assert np.allclose(steps, [0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625])  # the ramp halves past the floor


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


def test_the_fixture_has_no_fallback_frame(timeline):
    _, attrs = _read(timeline)
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


def test_a_short_am_episode_keeps_the_capped_rate():
    from fdia_graph.timeline import _AmShape

    full = _AmShape.under_floor(rel=0.2, length=60, am_rate=0.9, floor=0.02)
    short = _AmShape.under_floor(rel=0.2, length=4, am_rate=0.9, floor=0.02)
    assert full.rate == short.rate == pytest.approx(0.09) and full.rise == 12 and short.rise == 2
    assert max(short.at(i) for i in range(4)) == pytest.approx(0.18)  # never reaches the full delta
    assert max(full.at(i) for i in range(60)) == pytest.approx(1.0)


def test_split_boundaries_settle_in_order():
    from fdia_graph.timeline import _frame_split

    ep = [dict(onset=50, length=40), dict(onset=90, length=5)]  # 50..89 crosses both 60 and 80 of 100
    split = _frame_split(100, ep, (0.6, 0.2, 0.2))
    assert (split[:90] == 0).all() and (split[90:] == 2).all()  # the middle split is empty, nothing is cut
    split = _frame_split(100, [dict(onset=55, length=10)], (0.6, 0.2, 0.2))
    assert (split[:65] == 0).all() and (split[65:80] == 1).all() and (split[80:] == 2).all()


def test_attacked_frac_zero_is_all_benign(tmp_path, pool):
    out = generate_timeline(14, states=pool[:40], attacked_frac=0.0, out=str(tmp_path / "b.h5"))
    a, attrs = _read(out)
    assert (a["data/family"] == 0).all() and attrs["n_episodes"] == 0 and attrs["attacked_frac"] == 0.0
    with pytest.raises(ValueError, match="attacked_frac"):
        generate_timeline(14, states=pool[:20], attacked_frac=1.5, out=str(tmp_path / "x.h5"))


def test_empty_episode_lengths_are_refused(tmp_path, pool):
    for bad in (dict(ramp_len=0), dict(am_len=0), dict(corrupt_len=0)):
        with pytest.raises(ValueError, match="at least 1 frame"):
            generate_timeline(14, states=pool[:20], out=str(tmp_path / "x.h5"), **bad)
    with pytest.raises(ValueError, match="am_rate"):
        generate_timeline(14, states=pool[:20], out=str(tmp_path / "x.h5"), am_rate=0.0)
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


def test_redistribution_lightens_the_target_line_and_am_names_follow():
    """The engine's redistribution lowers the target line's |flow| in the false state (Al masks an
    overload), so Am's "mask" keeps its sign and "induce" flips it."""
    from fdia_graph.engine import FdiaGenerator
    from fdia_graph.formulas.network import branch_flows, complex_voltages
    from fdia_graph.profiles import _case_buses, _solve_states_chunk
    from fdia_graph.timeline import _am_sign

    g = FdiaGenerator(14, seed=5)
    g._pick_lra_target(0.2, 3)  # the target-line pool generation builds before any Al frame
    X = _solve_states_chunk(14, np.ones((1, len(_case_buses(14)))))[0]

    def flow(Xs, line):
        Vc = np.zeros(g._n_ppc_buses, complex)
        Vc[g._ppc_row[np.arange(g.C)]] = complex_voltages(Xs[:, 0], Xs[:, 3])
        return branch_flows(Vc, g._Yf, g._from_bus_ppc, g._base_mva).real[line]

    Lp, Lq = g.true_load(X), g.true_reactive_load(X)
    moved = 0
    for _ in range(10):
        red = g.lra_delta(Lp, 0.2, 3, floor=0.02, hops=2)
        if len(red.buses) == 0:
            continue
        for sign, lighter in ((_am_sign("mask", None), True), (_am_sign("induce", None), False)):
            Xa = g.solve_local(X, red.interior, Lp + sign * red.delta, Lq)
            if Xa is not None:
                assert (abs(flow(Xa, red.line)) < abs(flow(X, red.line))) == lighter
                moved += 1
    assert moved >= 4


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
    """The stealthy families skip every load on a generator bus, zero-MW condensers included
    [BOY22]; the in-place families may still tamper there."""
    import pandapower.networks as pn

    a, _ = _read(timeline)
    gen = np.zeros(a["data/y"].shape[1], bool)
    gen[pn.case14().gen.bus.values] = True
    fam, y = a["data/family"], a["data/y"].astype(bool)
    stealthy = np.isin(fam, sorted(STEALTHY_FAMILIES))
    assert stealthy.any() and not (y[stealthy] & gen).any()
    assert (y[np.isin(fam, [2, 3, 4])] & gen).any()  # Ad/As/Ar keep the full target set
