"""The safety net for the readability series (docs/plans/READABILITY_PLAN.md, section 5).

Every array the generated tiny timeline contains, every file attribute, and the estimator and
localizer scores on it are compared with references written by
tools/freeze_reference.py. A refactor that changes what a user gets, by a single byte in strict
mode, fails here.

Two modes. Integer, boolean and string arrays and every attribute must always be equal in value:
they carry the meter plan, the labels, the family draws and the split, which depend on the RNG
call order and never on floating-point details (only numpy's default integer width may differ,
int32 on Windows against int64 on Linux, so integer arrays are compared by kind and value).
Floating arrays and scores are compared exactly when FDIA_FROZEN_STRICT=1, the mode to run
locally, on the machine the references were written on, before every PR of the series.
Otherwise, and in CI on Linux, floating arrays must agree to 1e-7 relative, because pandapower's
power flow and BLAS differ in the last bits between platforms, and scores to 1e-4 relative,
because the iterative estimators (Huber passes with a settling test) amplify those last-bit
differences and the tiny timeline averages few records per family: the first CI run measured the
Huber Ad angle error at 3.12094e-2 on Linux against 3.12089e-2 frozen on Windows.
"""

import json
import os

import numpy as np
import pytest
from conftest import TINY
from frozen_spec import file_arrays, loc_scores, se_scores

FROZEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frozen")
STRICT = os.environ.get("FDIA_FROZEN_STRICT") == "1"


ARRAY_RTOL = 1e-7  # floating arrays, cross-platform mode
SCORE_RTOL = 1e-4  # estimator and localizer scores, cross-platform mode (see the module docstring)


def _same_array(name, got, ref):
    assert got.shape == ref.shape, f"{name}: shape {got.shape} vs frozen {ref.shape}"
    if got.dtype.kind in "iu" and not STRICT:  # numpy's default integer width is platform-dependent
        assert got.dtype.kind == ref.dtype.kind, f"{name}: dtype {got.dtype} vs frozen {ref.dtype}"
    else:
        assert got.dtype == ref.dtype, f"{name}: dtype {got.dtype} vs frozen {ref.dtype}"
    if got.dtype.kind in "iub" or got.dtype.kind in "SU" or STRICT:
        assert np.array_equal(got, ref), f"{name}: values differ from the frozen reference"
    else:
        assert np.allclose(got, ref, rtol=ARRAY_RTOL, atol=1e-9, equal_nan=True), _differ(name, got, ref)


def _differ(name, got, ref) -> str:
    """What differs, for the CI log: how many elements, the largest relative gap, and the rows."""
    g, r = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    bad = ~np.isclose(g, r, rtol=ARRAY_RTOL, atol=1e-9, equal_nan=True)
    rel = np.abs(g - r) / np.maximum(np.abs(r), 1e-12)
    rows = np.unique(np.nonzero(bad)[0])[:20] if bad.ndim else []
    return (
        f"{name}: {int(bad.sum())} of {bad.size} elements differ beyond {ARRAY_RTOL} "
        f"(max relative gap {rel[bad].max():.3e}), first rows {rows.tolist()}"
    )


def _same_scores(got, ref, path=""):
    if isinstance(ref, dict):
        assert set(got) == set(ref), f"{path}: keys {sorted(got)} vs frozen {sorted(ref)}"
        for k in ref:
            _same_scores(got[k], ref[k], f"{path}/{k}")
    elif isinstance(ref, float):
        if STRICT:
            assert got == ref, f"{path}: {got!r} vs frozen {ref!r}"
        else:
            assert got == pytest.approx(ref, rel=SCORE_RTOL, abs=1e-12), f"{path}: {got!r} vs frozen {ref!r}"
    else:
        assert got == ref, f"{path}: {got!r} vs frozen {ref!r}"


def test_tiny_timeline_matches_frozen_reference(timeline):
    import fdia_graph as fg

    ds = fg.load(timeline)
    got, attrs = file_arrays(ds.path)
    ref = np.load(os.path.join(FROZEN, "tiny_ieee14_timeline.npz"), allow_pickle=False)
    ref_attrs = json.load(open(os.path.join(FROZEN, "tiny_ieee14_timeline_attrs.json")))
    assert set(got) == set(ref.files), "the set of datasets in the timeline changed"
    for k in ref.files:
        _same_array(k, np.asarray(got[k]), ref[k])
    assert attrs == ref_attrs, "a timeline attribute changed"


def test_estimator_scores_match_frozen_reference(timeline):
    pytest.importorskip("torch")
    _same_scores(se_scores(TINY), json.load(open(os.path.join(FROZEN, "tiny_ieee14_se.json"))))


def test_localizer_scores_match_frozen_reference(timeline):
    pytest.importorskip("torch")
    _same_scores(loc_scores(TINY), json.load(open(os.path.join(FROZEN, "tiny_ieee14_localization.json"))))
