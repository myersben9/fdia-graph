"""Golden snapshots of nygrid's outputs on fixed inputs, the safety net of the readability refactor.

Each test runs one public entry point (the interface attack's design; the persistence rule's snapshot is taken by
the analysis toolkit's analysis/tests/test_detect.py) on fixed inputs and
compares a digest of what it returns with the JSON stored under `golden/`. A refactor that moves code without
changing behavior reproduces every digest; one that changes a number fails here and has to say why. Set
`NYGRID_UPDATE_GOLDEN=1` to rewrite the snapshots after an intended change.

The interface attack's design rates lines over the year, so it needs the whole 2019 pool (`inputs.py`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from .inputs import POOL, ROOT, needs_data, needs_pool, needs_territories

GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("NYGRID_UPDATE_GOLDEN") == "1"

pytestmark = needs_data


def _plain(x):
    """A JSON-ready copy: arrays and numpy scalars to lists and floats, NaN kept as None."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return (
            [_plain(v) for v in np.asarray(x).tolist()]
            if isinstance(x, np.ndarray)
            else [_plain(v) for v in x]
        )
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    return x


def _same(got, want, path: str = "", rtol: float = 1e-9, atol: float = 1e-9) -> None:
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path}: keys differ: {sorted(set(got) ^ set(want))}"
        for k in want:
            _same(got[k], want[k], f"{path}.{k}", rtol, atol)
    elif isinstance(want, list):
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _same(g, w, f"{path}[{i}]", rtol, atol)
    elif isinstance(want, float) or isinstance(got, float):
        assert got is not None and want is not None, f"{path}: {got} != {want}"
        assert np.isclose(got, want, rtol=rtol, atol=atol), f"{path}: {got!r} != {want!r}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def check(name: str, got: dict, rtol: float = 1e-9, atol: float = 1e-9) -> None:
    """Compare `got` with the snapshot `name` (floats within `rtol` and `atol`), or write it with
    NYGRID_UPDATE_GOLDEN=1."""
    got = _plain(got)
    path = GOLDEN / f"{name}.json"
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(json.dumps(got, indent=1, sort_keys=True))
        return
    assert path.exists(), f"no snapshot {path.name}; run once with NYGRID_UPDATE_GOLDEN=1"
    _same(got, json.loads(path.read_text()), name, rtol, atol)


def _digest(a: np.ndarray) -> dict:
    """The shape, the count of non-finite entries, and the sum, absolute sum and largest magnitude of the finite
    ones: a value that changes, or a result that goes missing (NaN), both show."""
    a = np.asarray(a, float)
    finite = np.isfinite(a)
    v = a[finite]
    return {
        "shape": list(a.shape),
        "nonfinite": int((~finite).sum()),
        "sum": float(v.sum()),
        "abs": float(np.abs(v).sum()),
        "max": float(np.abs(v).max()) if v.size else 0.0,
    }


@needs_territories
@needs_pool
@pytest.mark.parametrize("overshoot", (0.0, 175.0))
def test_interface_attack_design(overshoot):
    """The Am design on one onset (frame 48220, 2019-06-17), at the limit and 175 MW past it."""
    from fdia_graph.generation.nygrid.config import Settings, rng
    from fdia_graph.generation.nygrid.source import NewYorkSource
    from fdia_graph.models.config import NewYorkRelease

    release = NewYorkRelease(families=("Am",), interfaces=("SPR/DUN-SOUTH",), workers=1, survey=0, root=ROOT)
    b = NewYorkSource.production(str(POOL.relative_to(ROOT)), release)
    fam = b.families["Am"]
    onset, L = 48220, b.lengths()["Am"]
    w = slice(onset, onset + L)
    ep = fam.with_overshoot(overshoot).design(
        (onset, b.pool.vm[w], b.pool.va[w]), rng(Settings(), f"Am{onset}")
    )
    assert ep is not None
    tam = b.att.tampered(b.pool.vm[w], b.pool.va[w], ep.a)
    check(
        f"design_am_48220_{int(overshoot)}",
        {
            "buses": sorted(int(x) for x in ep.buses),
            "clients": sorted(int(x) for x in ep.clients),
            "target_branch": int(ep.target_branch),
            "length": int(ep.length),
            "a": _digest(ep.a),
            "a_per_scan": [_digest(row)["abs"] for row in ep.a],
            "meters_tampered": int(tam.any(axis=0).sum()),
        },
    )
