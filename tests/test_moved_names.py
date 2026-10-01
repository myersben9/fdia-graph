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
    from fdia_graph.engine.attacks.episodes import EpisodeDesignMixin, draw_ramp, ramp_dev

    with pytest.warns(DeprecationWarning):
        ramp = timeline._ramp_dev
        counts = timeline._target_counts
        draw = timeline._draw_ramp
        changed = records._changed_meters
    assert ramp(3, 5, 2, 0.01) == ramp_dev(3, 5, 2, 0.01)
    case = SimpleNamespace(stealthy_pos=np.arange(4))
    assert counts(case) == EpisodeDesignMixin.target_counts(case)
    ctx = SimpleNamespace(g=SimpleNamespace(stealthy_pos=np.arange(8)))
    old = draw(ctx, np.random.default_rng(3), 60)
    new = draw_ramp(np.random.default_rng(3), np.arange(8), 60)
    assert np.array_equal(old[0], new.targets) and old[1:] == (new.direction, new.rise, new.hold)
    scan = SimpleNamespace(node_m=np.ones((2, 4)), edge_m=np.ones((1, 2)))
    node, edge = changed(np.eye(2, 4), np.zeros((1, 2)), scan)
    assert node.sum() == 2 and not edge.any()
