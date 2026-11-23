"""The rules a ramp is drawn and kept by: its targets are one neighbourhood; no frame of an emitted
ramp steps past the stealth bound, into the frame or out of the ramp's last one, on the fewest-tamper
support and on the whole area alike; and a ramp no support holds is given up after a fixed number of
tries, which the walk answers by moving the episode."""

import numpy as np
import pytest

pytest.importorskip("pandapower")

from fdia_graph.engine.core import FdiaGenerator  # noqa: E402
from fdia_graph.generation import _load_states  # noqa: E402
from fdia_graph.generation.ieee import IeeeSource  # noqa: E402
from fdia_graph.models.config import TimelineSettings  # noqa: E402
from fdia_graph.models.frames import FrameKnobs  # noqa: E402
from fdia_graph.systems.network import Network  # noqa: E402
from tests.helpers import ramp_frames_past_the_bound  # noqa: E402


def test_a_ramps_targets_are_one_neighbourhood():
    """One eligible load is drawn, and the others from the eligible loads within `hops` branches of
    it: at most `ramp_targets` in all, no load twice, and one alone when the count is one."""
    g = FdiaGenerator(Network.from_ieee(118), seed=1)
    rng, sizes = np.random.default_rng(0), set()
    for hops in (1, 2):
        knobs = FrameKnobs(hops=hops, ramp_targets=5)
        for _ in range(200):
            targets = g.episode_designs.draw_ramp(rng, 60, knobs).targets
            buses = g.load_bus[targets]
            assert np.isin(targets, g.stealthy_pos).all() and len(set(targets.tolist())) == len(targets)
            assert np.isin(buses, g.net.neighbors(buses[0], hops)).all()
            assert 1 <= len(targets) <= 5
            sizes.add(len(targets))
    assert len(sizes) > 1  # fewer than five where fewer are in reach
    alone = g.episode_designs.draw_ramp(rng, 60, FrameKnobs(ramp_targets=1)).targets
    assert len(alone) == 1


@pytest.mark.parametrize("min_tamper", [True, False])
def test_no_emitted_ramp_frame_steps_past_its_stealth_bound(tmp_path, min_tamper):
    """Every attacked frame of every ramp in a generated IEEE-14 timeline keeps the bound the search
    tests: no metered channel steps from the frame before by more than its meter's sigma. With the
    search off the ramp runs on its whole area, and is kept only when the area holds the bound."""
    import h5py

    from fdia_graph import schema

    X = _load_states("ieee14", None)[:1500]
    settings = TimelineSettings.of(families=("At",), min_tamper=min_tamper)
    source = IeeeSource("ieee14", X, settings, seed=11)
    path = source.generate(tmp_path / "ramps.h5")
    with h5py.File(path, "r") as f:
        ramps = len(f[schema.Group.EPISODES][schema.EPISODE_ONSET])
        if min_tamper:  # every ramp is recorded on the support the search found
            assert (f[schema.Group.EPISODES][schema.EPISODE_MIN_DEVICES][:] >= 0).all()
    assert ramps >= 5
    assert ramp_frames_past_the_bound(source.g, source.X, path) == 0


def test_a_ramp_that_ends_before_its_return_is_refused_when_its_last_step_shows():
    """A ramp whose rise and hold leave no room for the whole return ends with several rate-steps
    still on, and the frame after it is benign: that one step out is held to the stealth bound like
    every step in. On this window the area keeps every step in, and the step out breaks the bound, so
    the window does not hold; with the last step taken out of the check it does."""
    from fdia_graph.engine.attacks.episodes import probe_frames
    from fdia_graph.engine.attacks.window import AttackWindow
    from fdia_graph.models.frames import LoadGoal, RampDesign

    X = _load_states("ieee14", None)[:1500]
    source = IeeeSource("ieee14", X, TimelineSettings.of(families=("At",)), seed=11)
    g, knobs, rate = source.g, source.knobs, source.settings.ramp.rate
    onset, length = 140, 60
    design = RampDesign(g.stealthy_pos[:2], 1.0, 25, 13)  # 2 * 25 + 13 > 60: four rate-steps left at the end
    steps = [g.episode_designs.ramp_step(design, i, rate) for i in range(length)]
    assert steps[-1].mult == pytest.approx(1 + 4 * rate)
    frames = probe_frames(len(X), onset, length)
    support = g.false_state.with_support(steps[0], knobs).support
    window = AttackWindow(g, [X[u] for u in frames], LoadGoal(tuple(steps)), knobs)
    entered = all(
        not window._steps_past(t, vector, prev) for t, (vector, prev) in enumerate(_vectors(window, support))
    )
    assert entered  # every step into a frame is inside the bound
    last = _vectors(window, support)[-1][0]
    assert window._steps_past(length - 1, window.after, last)  # the step out is not
    assert not window.holds(support)


def _vectors(window, support):
    """(the attack vector of each snapshot on the support, the one before it), bound unchecked."""
    out, prev = [], window.prev
    held, window.stealth_bound = window.stealth_bound, False
    for t in range(len(window.states)):
        _, vector = window._snapshot(t, support, prev)
        out.append((vector, prev))
        prev = vector
    window.stealth_bound = held
    return out


def test_a_ramp_no_support_holds_is_given_up_after_a_fixed_number_of_tries(monkeypatch):
    """`ramp_design` tries the supports of at most `RAMP_SUPPORT_TRIES` designs, counting only designs
    that have a false state on every frame, and returns None when none keeps the bound. Here every
    other draw has no false state and no window holds: eight windows are tried, on the fifteenth draw."""
    from fdia_graph.constants.numerics import RAMP_SUPPORT_TRIES
    from fdia_graph.engine.attacks.window import AttackWindow

    g = FdiaGenerator(Network.from_ieee(14), seed=1)
    X = _load_states("ieee14", None)[:200]
    drawn, tried = [], []
    draw = g.episode_designs.draw_ramp

    def counted_draw(rng, ramp_len, knobs):
        drawn.append(1)
        return draw(rng, ramp_len, knobs)

    def never_holds(self, support):
        tried.append(1)
        return False

    monkeypatch.setattr(g.episode_designs, "draw_ramp", counted_draw)
    monkeypatch.setattr(g.false_state, "is_feasible", lambda x_true, design, knobs: len(drawn) % 2 == 1)
    monkeypatch.setattr(AttackWindow, "holds", never_holds)
    knobs = FrameKnobs(hops=2, min_tamper=False)
    assert g.episode_designs.ramp_design(X, 10, (12, 0.002), knobs) is None
    assert len(tried) == RAMP_SUPPORT_TRIES
    assert len(drawn) == 2 * RAMP_SUPPORT_TRIES - 1  # a draw with no false state is not a try


def test_the_walk_moves_a_ramp_that_cannot_begin_and_counts_the_move(tmp_path, monkeypatch):
    """A ramp whose design is refused at its onset is moved to a later onset of its split, and the file
    counts the move in `episode_redraws`: with the first three designs refused, the file holds three
    more moves than the same run without, and still places its ramps."""
    import h5py

    from fdia_graph import schema
    from fdia_graph.engine.attacks.episodes import EpisodeDesigns

    X = _load_states("ieee14", None)[:900]
    settings = TimelineSettings.of(families=("At",), ramp_len=20, min_tamper=False)

    def moves(path) -> tuple[int, int]:
        with h5py.File(path, "r") as f:
            redraws = int(np.asarray(f.attrs[schema.Attr.EPISODE_REDRAWS]).sum())
            return redraws, len(f[schema.Group.EPISODES][schema.EPISODE_ONSET])

    plain = moves(IeeeSource("ieee14", X, settings, seed=11).generate(tmp_path / "plain.h5"))
    design, calls = EpisodeDesigns.ramp_design, []

    def refused_three_times(self, *args, **kwargs):
        calls.append(1)
        return None if len(calls) <= 3 else design(self, *args, **kwargs)

    monkeypatch.setattr(EpisodeDesigns, "ramp_design", refused_three_times)
    moved = moves(IeeeSource("ieee14", X, settings, seed=11).generate(tmp_path / "moved.h5"))
    assert moved[0] >= plain[0] + 3 and moved[1] >= 1


def test_every_frame_of_an_accepted_ramp_is_built_at_its_designed_step(timeline, tmp_path):
    """A ramp is accepted only with a false state on every frame of its window, and a frame is that
    same solve on the same state: no frame inside a ramp of the two golden timelines is stored benign.
    (A step with no solution was once halved at
    frame time; an accepted ramp never needs it.)"""
    import h5py

    import fdia_graph as fg
    from fdia_graph import schema
    from fdia_graph.constants.families import AT_CODE
    from tests.test_timeline_digests import SEARCHED_KW

    searched = tmp_path / "searched.h5"
    fg.generate("ieee14", "ramp_rules_searched", out=str(searched), **SEARCHED_KW)
    for path in (fg.load(timeline).path, searched):
        with h5py.File(path, "r") as f:
            episodes, seq = f[schema.Group.EPISODES], f[schema.SEQ_ID][:]
            ramps = np.flatnonzero(episodes[schema.EPISODE_FAMILY][:] == AT_CODE)
            assert len(ramps)
            for k in ramps:
                onset, length = (
                    int(episodes[schema.EPISODE_ONSET][k]),
                    int(episodes[schema.EPISODE_LENGTH][k]),
                )
                assert (seq[onset : onset + length] == k).all()
