"""Split-first generation: the timeline is cut into train, val and test before any episode is placed;
each split gets the whole number of episodes closest to its attacked fraction, shared between the
families by largest remainder and placed uniformly without overlap or cut; an episode with no
feasible design moves to a later onset of its split; and every frame's jitter comes from its own
(seed, timestep) stream, so timelines that place different attacks carry the same benign frames."""

import warnings

import h5py
import numpy as np
import pytest

from fdia_graph import schema
from fdia_graph import timeline as tl
from fdia_graph.models.choices import FAMILY_CODE
from fdia_graph.schema import Attr

AT, AM = FAMILY_CODE["At"], FAMILY_CODE["Am"]


def _plan(fams=(AT, AM), length=60, frac=0.5):
    return tl._Schedule.build(list(fams), length, 0.002, length, frac)


# ---- 1. the splits ------------------------------------------------------------------------------
@pytest.mark.parametrize("T", [7, 100, 999, 1000, 72000])
def test_the_splits_are_cut_to_their_fractions_and_cover_the_timeline(T):
    bounds = tl._split_bounds(T, (0.6, 0.2, 0.2))
    assert bounds[0][0] == 0 and bounds[-1][1] == T
    assert all(b0[1] == b1[0] for b0, b1 in zip(bounds, bounds[1:]))
    assert bounds[0][1] == round(0.6 * T) and bounds[1][1] - bounds[1][0] == round(0.2 * T)
    split = tl._split_column(bounds, T)
    assert [int((split == c).sum()) for c in range(3)] == [b - a for a, b in bounds]


# ---- 2 and 3. the counts ------------------------------------------------------------------------
def test_largest_remainder_shares_whole_units():
    assert tl._largest_remainder(5, np.array([1.0, 1.0])).tolist() == [3, 2]  # a tie goes to the first
    assert tl._largest_remainder(7, np.array([0.5, 0.3, 0.2])).tolist() == [4, 2, 1]
    assert tl._largest_remainder(1, np.array([0.2, 0.8])).tolist() == [0, 1]
    assert tl._largest_remainder(0, np.array([0.5, 0.5])).tolist() == [0, 0]
    for total in range(50):
        assert tl._largest_remainder(total, np.array([0.25, 0.5, 0.25])).sum() == total


@pytest.mark.parametrize(
    "n, want", [(43200, [180, 180]), (14400, [60, 60]), (600, [3, 2]), (200, [1, 1]), (59, [0, 0])]
)
def test_each_split_gets_the_whole_episodes_closest_to_half(n, want):
    """At 72,000 frames every split is exactly 50%; a short split rounds to the nearest whole episode."""
    counts = tl._episode_counts(_plan(), n)
    assert counts.tolist() == want
    frames = 60 * counts.sum()
    assert all(abs(frames - 0.5 * n) <= abs(60 * k - 0.5 * n) for k in range(n // 60 + 1))


# ---- 5, 6, 7. the placement ---------------------------------------------------------------------
def _occupancy(placed, a, b):
    occ = np.zeros(b - a, int)
    for onset, _, length in placed:
        assert a <= onset and onset + length <= b  # inside its split: never cut
        occ[onset - a : onset - a + length] += 1
    return occ


def test_episodes_lie_whole_inside_their_split_and_never_overlap():
    rng = np.random.default_rng(0)
    plan = _plan()
    for bounds in tl._split_bounds(2000, (0.6, 0.2, 0.2)):
        rec = tl._Placement.empty(plan.families)
        placed = tl._place_split(rng, plan, 0, bounds, rec)
        assert len(placed) == rec.requested[0].sum()
        assert all(L == 60 for _, _, L in placed)  # full length, none clipped
        assert _occupancy(placed, *bounds).max() <= 1


def test_a_jammed_split_retries_then_drops_one_episode_and_records_it(monkeypatch):
    """A placement that jams is retried whole PLACE_TRIES times, then one episode of the family with
    the most is dropped, recorded and warned about, and the rest are placed."""
    from fdia_graph.errors import NoRoomForEpisode

    calls = []
    real = tl._place_once

    def jam_while_four(rng, episodes, n):
        calls.append(len(episodes))
        if len(episodes) >= 4:
            raise NoRoomForEpisode("jammed")
        return real(rng, episodes, n)

    monkeypatch.setattr(tl, "_place_once", jam_while_four)
    plan = _plan()
    rec = tl._Placement.empty(plan.families)
    with pytest.warns(RuntimeWarning, match="dropped"):
        placed = tl._place_split(np.random.default_rng(1), plan, 0, (0, 480), rec)
    assert rec.requested[0].tolist() == [2, 2] and rec.dropped[0].tolist() == [1, 0]  # the tie: At first
    assert calls == [4] * tl.PLACE_TRIES + [3] and len(placed) == 3
    assert _occupancy(placed, 0, 480).max() <= 1


def test_the_full_sized_splits_never_jam():
    plan = _plan()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for seed in range(100):
            rng = np.random.default_rng(seed)
            for split, bounds in enumerate(tl._split_bounds(72000, (0.6, 0.2, 0.2))):
                rec = tl._Placement.empty(plan.families)
                placed = tl._place_split(rng, plan, split, bounds, rec)
                assert len(placed) == rec.requested[split].sum() and not rec.dropped.any()


def test_onsets_are_uniform_over_the_split():
    """One episode in a 600-frame split: its onset is uniform over the 541 valid onsets (chi-square)."""
    plan = tl._Schedule.build([AT], 60, 0.002, 60, 0.1)
    rng = np.random.default_rng(7)
    onsets = []
    for _ in range(4000):
        rec = tl._Placement.empty(plan.families)
        onsets += [o for o, _, _ in tl._place_split(rng, plan, 0, (0, 600), rec)]
    hist, _ = np.histogram(onsets, bins=10, range=(0, 541))
    expected = len(onsets) / 10
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    assert chi2 < 27.9  # the 99.9% point of chi-square with 9 degrees of freedom


# ---- 8. an infeasible episode moves -------------------------------------------------------------
class _FakeWalk:
    def __init__(self):
        self.rng = np.random.default_rng(3)
        self.emitted: list[tuple[int, int]] = []


def _walk_with(monkeypatch, fails: int, bounds=(0, 600)):
    """Walk one split whose episode design fails at its first `fails` onsets."""
    w, tries = _FakeWalk(), []

    def run(w_, at, plan):
        tries.append(at[0])
        return None if len(tries) <= fails else at[0] + at[2]

    monkeypatch.setattr(tl, "_run_episode", run)
    monkeypatch.setattr(tl, "_benign_run", lambda w_, t, until: until)
    plan = tl._Schedule.build([AM], 10, 0.002, 10, 10 / 600)  # one 10-frame episode
    rec = tl._Placement.empty(plan.families)
    tl._walk_split(w, plan, 0, bounds, rec)
    return rec, tries


def test_an_infeasible_episode_moves_to_a_later_onset(monkeypatch):
    rec, tries = _walk_with(monkeypatch, fails=3)
    assert rec.redraws[0].tolist() == [3] and rec.built[0].tolist() == [1]
    assert tries == sorted(tries) and len(set(tries)) == 4  # each move is to a later onset


def test_an_episode_infeasible_everywhere_is_given_up_after_the_cap(monkeypatch):
    """Moved REDRAWS times with room always left, the episode is then given up, recorded and warned."""
    monkeypatch.setattr(tl, "_relocate", lambda rng, at, pending, end: (at[0] + 1, at[1], at[2]))
    with pytest.warns(RuntimeWarning, match="no feasible design"):
        rec, tries = _walk_with(monkeypatch, fails=10**6)
    assert rec.built[0].tolist() == [0] and rec.redraws[0].tolist() == [tl.REDRAWS]
    assert len(tries) == tl.REDRAWS + 1


def test_an_episode_with_no_later_room_is_given_up(monkeypatch):
    """When the split's end leaves no later onset, the episode is given up before the cap."""
    with pytest.warns(RuntimeWarning, match="no feasible design"):
        rec, tries = _walk_with(monkeypatch, fails=10**6)
    assert rec.built[0].tolist() == [0] and rec.redraws[0].sum() == len(tries) - 1 <= tl.REDRAWS


# ---- the generated files ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pool():
    pytest.importorskip("pandapower")
    from fdia_graph.generation import _load_states

    return _load_states(14, None)[:300]


@pytest.fixture(scope="module")
def variants(tmp_path_factory, pool):
    """The three release variants on one seed and pool (short episodes, a capped search)."""
    out = {}
    for name, fams in (("Am", ("Am",)), ("AtAm", ("At", "Am")), ("At", ("At",))):
        path = tmp_path_factory.mktemp("variants") / f"{name}.h5"
        tl.generate_timeline(
            14,
            states=pool,
            families=fams,
            ramp_len=10,
            seed=5,
            min_tamper=False,
            min_budget=8,
            out=str(path),
        )
        out[name] = str(path)
    return out


def test_the_file_records_what_the_placement_asked_for_and_got(variants):
    with h5py.File(variants["AtAm"], "r") as f:
        a = dict(f.attrs)
        split, seq, fam = f[schema.SPLIT][:], f[schema.SEQ_ID][:], f[schema.FAMILY][:]
        onset = f["episodes/onset"][:]
        length = f["episodes/length"][:]
        efam = f["episodes/family"][:]
    assert a[Attr.SPLIT_FRAC].tolist() == [0.6, 0.2, 0.2] and a[Attr.SPLIT_SIZES].tolist() == [180, 60, 60]
    assert a[Attr.PLACED_FAMILIES] == "At,Am" and a[Attr.JITTER_KEYED] == 1
    built, req = a[Attr.EPISODES_BUILT], a[Attr.EPISODES_REQUESTED]
    assert (a[Attr.EPISODE_SHORTFALL] == req - built).all()
    for s in range(3):
        on = split[onset] == s
        assert [int((on & (efam == f_)).sum()) for f_ in (AT, AM)] == built[s].tolist()
        assert a[Attr.SPLIT_ATTACKED_FRAC][s] == pytest.approx(float((seq[split == s] >= 0).mean()))
    # 4 and 5: every episode has its full length and lies inside one split
    assert all(int(L) == 10 for L in length)
    assert all(split[o] == split[o + L - 1] for o, L in zip(onset, length))
    assert (fam[seq < 0] == 0).all()


def test_every_frame_labelled_attacked_carries_an_attack(variants):
    """No labelled frame is a no-op: every attacked frame tampers at least one meter (the overload's
    first snapshot is kappa+1, with 1/T of the way to the rating, [WU26, eq. 25])."""
    for path in variants.values():
        with h5py.File(path, "r") as f:
            fam = f[schema.FAMILY][:]
            touched = f[schema.NODE_TAMPER][:].reshape(len(fam), -1).any(1) | f[schema.EDGE_TAMPER][
                :
            ].reshape(len(fam), -1).any(1)
            if schema.PMU_I_TAMPER in f:
                touched |= f[schema.PMU_I_TAMPER][:].reshape(len(fam), -1).any(1)
        assert touched[fam > 0].all(), path


def test_the_three_variants_carry_the_same_benign_frames(variants):
    """The benign twin of every frame (the attack removed, the noise kept), the split column and the
    meter plan are byte-identical across the Am, At+Am and At timelines on one seed and pool."""
    layers = (schema.NODE_BENIGN, schema.EDGE_BENIGN, schema.SPLIT, schema.NODE_M)
    ref = {}
    with h5py.File(variants["AtAm"], "r") as f:
        for name in layers:
            ref[name] = f[name][:]
    for key in ("Am", "At"):
        with h5py.File(variants[key], "r") as f:
            for name in layers:
                assert np.array_equal(f[name][:], ref[name]), (key, name)


def test_an_alias_of_a_named_family_adds_no_column(pool, tmp_path):
    """families=("At", "ramp") names At twice: one column, its requests and builds counted once."""
    out = tl.generate_timeline(
        14,
        states=pool,
        families=("At", "ramp"),
        ramp_len=10,
        seed=5,
        min_tamper=False,
        out=str(tmp_path / "a.h5"),
    )
    with h5py.File(out, "r") as f:
        a = dict(f.attrs)
    assert a[Attr.PLACED_FAMILIES] == "At" and (a[Attr.EPISODE_SHORTFALL] >= 0).all()
    assert (a[Attr.EPISODES_BUILT] <= a[Attr.EPISODES_REQUESTED]).all()


def test_the_same_seed_gives_the_same_file(variants, pool, tmp_path):
    again = tl.generate_timeline(
        14,
        states=pool,
        families=("At", "Am"),
        ramp_len=10,
        seed=5,
        min_tamper=False,
        min_budget=8,
        out=str(tmp_path / "again.h5"),
    )
    with h5py.File(variants["AtAm"], "r") as a, h5py.File(again, "r") as b:
        names: list[str] = []
        a.visit(lambda n: names.append(n) if isinstance(a[n], h5py.Dataset) else None)
        for n in names:
            assert np.array_equal(a[n][:], b[n][:]), n
        for k in a.attrs:
            x, y = np.asarray(a.attrs[k]), np.asarray(b.attrs[k])
            assert np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), k


@pytest.mark.parametrize(
    "bad", [(0.6, 0.4), (0.5, 0.3, 0.3), (0.0, 0.5, 0.5), (0.6, 0.2, float("nan")), "train"]
)
def test_a_malformed_split_is_refused_before_any_work(bad):
    from fdia_graph.models.validation import ConfigError

    with pytest.raises(ConfigError, match="SplitSettings"):
        tl.generate_timeline(14, states=np.zeros((5, 14, 4)), split=bad, out="never_written.h5")


def test_each_episode_counts_its_own_moves(monkeypatch):
    """Two episodes in one split: the one that fails is moved and given up on its own count, however
    the other fares."""
    w, tries = _FakeWalk(), []

    def run(w_, at, plan):
        tries.append(at)
        return None if at[1] == AM else at[0] + at[2]

    monkeypatch.setattr(tl, "_run_episode", run)
    monkeypatch.setattr(tl, "_benign_run", lambda w_, t, until: until)
    monkeypatch.setattr(
        tl,
        "_relocate",
        lambda rng, at, pending, end: (at[0] + 1, at[1], at[2]) if at[0] + at[2] < end else None,
    )
    plan = tl._Schedule.build([AT, AM], 10, 0.002, 10, 40 / 2000)  # two 10-frame episodes
    rec = tl._Placement.empty(plan.families)
    with pytest.warns(RuntimeWarning, match="no feasible design"):
        tl._walk_split(w, plan, 0, (0, 2000), rec)
    assert rec.requested[0].tolist() == [2, 2] and rec.built[0].tolist() == [2, 0]
    assert rec.redraws[0].tolist() == [0, 2 * tl.REDRAWS]  # each Am moved REDRAWS times, on its own count
