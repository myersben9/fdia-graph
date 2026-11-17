"""The analysis toolkit's part of tests/test_models.py:
the tests that run an estimator, a localizer, a trust selector or an analysis formula. Of that module:
Bundle, the dual-access base of the typed records: every dict use the package's callers rely
on must keep working, and the attribute view must agree with it."""

import numpy as np
import pytest


def test_group_built_bundles_are_keyword_only():
    """Positional construction would bind to the inherited field order, not the documented one, so
    it is refused outright rather than allowed to mis-bind silently."""
    from fdia_analysis.models.data import TrueState
    from fdia_graph.models.data import RecordBundle

    with pytest.raises(TypeError, match="by keyword"):
        RecordBundle(np.zeros((2, 1)), np.ones((3, 4)))
    assert TrueState(np.zeros((2, 5)), np.zeros(2)).thsl.shape == (
        2,
    )  # explicit bundles still take positionals
