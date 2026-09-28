"""Names that moved to engine.attacks stay importable from their old modules for one minor release,
with a DeprecationWarning that says where they live now."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

OLD_MODULES = ("fdia_graph.engine.records", "fdia_graph.timeline")


def _moved_names():
    return [(m, name) for m in OLD_MODULES for name in importlib.import_module(m)._MOVED]


@pytest.mark.parametrize("module, name", _moved_names())
def test_an_old_path_still_resolves_and_warns(module, name):
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=rf"{name} moved to engine\.attacks\..* retires in 0\.22"):
        value = getattr(mod, name)
    assert value is mod._MOVED[name][1]


def test_an_unknown_name_is_still_an_attribute_error():
    import fdia_graph.engine.records as records

    with pytest.raises(AttributeError, match="no attribute '_never_here'"):
        records._never_here  # noqa: B018


def test_an_old_path_keeps_its_old_signature():
    """Each alias takes the arguments the old private function took, and gives what the new home gives."""
    import fdia_graph.engine.records as records
    import fdia_graph.timeline as timeline
    from fdia_graph.engine.attacks.episodes import EpisodeDesignMixin, am_sign, draw_ramp, ramp_dev

    with pytest.warns(DeprecationWarning):
        ramp = timeline._ramp_dev
        sign = timeline._am_sign
        counts = timeline._target_counts
        draw = timeline._draw_ramp
        changed = records._changed_meters
    assert ramp(3, 5, 2, 0.01) == ramp_dev(3, 5, 2, 0.01)
    assert sign("mask", None) == am_sign("mask", None)
    case = SimpleNamespace(stealthy_pos=np.arange(4), _target_lines=[1], attackable_pos=np.arange(6))
    assert counts(case) == EpisodeDesignMixin.target_counts(case)
    ctx = SimpleNamespace(g=SimpleNamespace(stealthy_pos=np.arange(8)))
    old = draw(ctx, np.random.default_rng(3), 60)
    new = draw_ramp(np.random.default_rng(3), np.arange(8), 60)
    assert np.array_equal(old[0], new.targets) and old[1:] == (new.direction, new.rise, new.hold)
    scan = SimpleNamespace(node_m=np.ones((2, 4)), edge_m=np.ones((1, 2)))
    node, edge = changed(np.eye(2, 4), np.zeros((1, 2)), scan)
    assert node.sum() == 2 and not edge.any()


def test_the_single_shot_alias_draws_from_the_generator_it_is_given():
    """`timeline._draw_single_shot(ctx, rng, ...)` drew from `rng`; the alias still does, and leaves
    the generator's own stream where it was."""
    import fdia_graph.timeline as timeline
    from fdia_graph.engine.attacks.episodes import EpisodeDesignMixin
    from fdia_graph.models.choices import FAMILY_CODE

    fid = FAMILY_CODE["Ad"]  # a corrupt-in-place family: no feasibility probe, only the draws
    own = np.random.default_rng(1)
    g = SimpleNamespace(rng=own, stealthy_pos=np.arange(8), attackable_pos=np.arange(8))
    g.single_shot_design = lambda *a, **kw: EpisodeDesignMixin.single_shot_design(g, *a, **kw)
    ctx = SimpleNamespace(g=g, X=np.zeros((10, 14, 4)), knobs=SimpleNamespace(intensity=0.2))
    before = own.bit_generator.state
    with pytest.warns(DeprecationWarning):
        draw = timeline._draw_single_shot
    got = draw(ctx, np.random.default_rng(5), 0, fid, 1)
    want = EpisodeDesignMixin.single_shot_design(g, ctx.X, 0, fid, 1, ctx.knobs, rng=np.random.default_rng(5))
    assert own.bit_generator.state == before
    assert np.array_equal(got[0], want.targets) and np.allclose(got[1], want.mult)
