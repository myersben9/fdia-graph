"""The safety net for the readability series (docs/plans/READABILITY_PLAN.md, section 5).

Every array a generated shard or stream contains, every file attribute, and the estimator and
localizer scores on the tiny shard are compared with references written by
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
differences and the tiny shard averages few records per family: the first CI run measured the
Huber Ad angle error at 3.12094e-2 on Linux against 3.12089e-2 frozen on Windows.
"""

import json
import os

import numpy as np
import pytest

from conftest import TINY
from frozen_spec import STREAM_T, loc_scores, se_scores, shard_arrays, stream_arrays

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
        assert np.allclose(got, ref, rtol=ARRAY_RTOL, atol=1e-9, equal_nan=True), (
            f"{name}: values differ beyond {ARRAY_RTOL}"
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


def test_tiny_shard_matches_frozen_reference(shard):
    import fdia_graph as fg

    ds = fg.load(shard)
    got, attrs = shard_arrays(ds.path)
    ref = np.load(os.path.join(FROZEN, "tiny_ieee14_shard.npz"), allow_pickle=False)
    ref_attrs = json.load(open(os.path.join(FROZEN, "tiny_ieee14_shard_attrs.json")))
    assert set(got) == set(ref.files), "the set of datasets in the shard changed"
    for k in ref.files:
        _same_array(k, np.asarray(got[k]), ref[k])
    # The full-pool hashes are bitwise by nature; outside strict mode the sliced rows carry the check.
    hashes = [k for k in ref_attrs if k.endswith("::sha256")]
    if not STRICT:
        attrs = {k: v for k, v in attrs.items() if k not in hashes}
        ref_attrs = {k: v for k, v in ref_attrs.items() if k not in hashes}
    assert attrs == ref_attrs, "a shard attribute changed"


def test_stream_matches_frozen_reference():
    pytest.importorskip("pandapower")
    import fdia_graph as fg
    from fdia_graph.generation import _load_states

    s = fg.generate_stream(14, states=_load_states(14, None)[:STREAM_T], seed=1)
    got = stream_arrays(s)
    ref = np.load(os.path.join(FROZEN, "ieee14_stream.npz"), allow_pickle=False)
    assert set(got) == set(ref.files), "the set of stream arrays changed"
    for k in ref.files:
        _same_array(k, got[k], ref[k])


def test_estimator_scores_match_frozen_reference(shard):
    pytest.importorskip("torch")
    _same_scores(se_scores(TINY), json.load(open(os.path.join(FROZEN, "tiny_ieee14_se.json"))))


def test_localizer_scores_match_frozen_reference(shard):
    pytest.importorskip("torch")
    _same_scores(loc_scores(TINY), json.load(open(os.path.join(FROZEN, "tiny_ieee14_localization.json"))))
