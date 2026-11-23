"""The rules a ramp is drawn and kept by: its targets are one neighbourhood, and no frame of an
emitted ramp steps past the stealth bound, on the fewest-tamper support and on the whole area alike."""

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
