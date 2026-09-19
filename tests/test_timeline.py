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

SEED = 3  # covers every family in the 1000-frame fixture (the long families are rare per episode by design)


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
    resolved = np.isin(fam, [1, 5, 6])
    assert (nt[resolved] == nm[resolved]).all()
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
    assert share.max() < 0.6 and share.min() < 0.1  # sparse at the plateau, nearly empty on the rise
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


def test_am_is_refused_by_the_shard_and_stream_generators(tmp_path, pool):
    import fdia_graph as fg

    with pytest.raises(ValueError, match="timeline family"):
        fg.generate(
            "ieee14", "never", families=("Aq", "Am"), per_family=2, n_benign=2, out=str(tmp_path / "s.h5")
        )
    with pytest.raises(ValueError, match="timeline family"):
        fg.generate_stream(14, states=pool[:20], families=("Am",))


def test_am_direction_sign_follows_the_engine_convention():
    """`lra_delta` raises the target line's loading in the false state, so induce keeps its sign."""
    from fdia_graph.timeline import _am_sign

    rng = np.random.default_rng(0)
    assert _am_sign("induce", rng) == 1.0 and _am_sign("mask", rng) == -1.0
    assert {_am_sign("both", rng) for _ in range(50)} == {1.0, -1.0}


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


def test_am_direction_and_the_pool_as_hdf5(tmp_path, pool):
    with pytest.raises(ValueError, match="am_direction"):
        generate_timeline(
            14, states=pool[:20], families=("Am",), am_direction="up", out=str(tmp_path / "x.h5")
        )
    h5 = tmp_path / "pool.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("X", data=pool[:60])
    assert np.array_equal(_load_states(14, str(h5)), pool[:60])
    out = generate_timeline(
        14, states=str(h5), seed=SEED, families=("Am", "Ad"), ramp_len=10, out=str(tmp_path / "t.h5")
    )
    assert os.path.exists(out)
    a, attrs = _read(out)
    assert attrs["T"] == 60 and set(np.unique(a["data/family"]).tolist()) <= {0, 2, 7}


def test_score_bundles_accept_the_seventh_family():
    """`SEBase.score` and `LocalizerBase.score` build their bundles from `FAMILIES`, so both carry
    an `Am` slot once a scored view holds Am frames."""
    from fdia_graph.models import ErrorPair, EstimatorScores, FamilyMetrics, LocalizerScores, OverallMetrics

    se = EstimatorScores(geo=ErrorPair(1.0, 0.1), Am=ErrorPair(2.0, 0.2))
    assert list(se) == ["Am", "geo"] and se["Am"].angle_mae_deg == 2.0
    fam = FamilyMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    loc = LocalizerScores(all=OverallMetrics(1.0, 1.0, 0.0, 1.0), Am=fam)
    assert list(loc) == ["all", "Am"] and loc.Am is fam
