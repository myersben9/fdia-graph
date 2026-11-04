"""Golden snapshots of nyfed's outputs on fixed inputs, the safety net of the readability refactor.

Each test runs one public entry point (the operator's `respond`, the detection tools, the interface attack's
design) on one attack window of the damage study and compares a digest of what it returns with the JSON stored
under `golden/`. A refactor that moves code without changing behavior reproduces every digest; one that changes a
number fails here and has to say why. Set `NYFED_UPDATE_GOLDEN=1` to rewrite the snapshots after an intended change.

The inputs are the NYgrid checkout, the 2019 pool and the damage-study day files under `NYFED_ROOT` (the nyfed
working folder), so these tests run where that data exists and skip elsewhere.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get("NYFED_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data" / "NYgrid" / "Data"
POOL = ROOT / "out" / "pool_2019_v3.h5"
DAYS = ROOT / "out" / "damage_study" / "days"
WINDOWS = ("am_48220_0.h5", "am_48220_175.h5")  # one onset, at the limit and past it
CALIB = (28.41, 13.76)  # the interchange check's calibration, results/nygrid.tie_check.json
RULE = (428.7, 2, 3)  # the residual test's persistence rule
GOLDEN = Path(__file__).parent / "golden"
UPDATE = os.environ.get("NYFED_UPDATE_GOLDEN") == "1"
TOOLS = Path(__file__).resolve().parents[2] / "tools" / "nyfed"

pytestmark = pytest.mark.skipif(
    not (DATA.exists() and POOL.exists() and all((DAYS / w).exists() for w in WINDOWS)),
    reason="NYgrid checkout, the 2019 pool or the damage-study days missing under NYFED_ROOT",
)


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


def _same(got, want, path: str = "") -> None:
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path}: keys differ: {sorted(set(got) ^ set(want))}"
        for k in want:
            _same(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, list):
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for i, (g, w) in enumerate(zip(got, want)):
            _same(g, w, f"{path}[{i}]")
    elif isinstance(want, float) or isinstance(got, float):
        assert got is not None and want is not None, f"{path}: {got} != {want}"
        assert np.isclose(got, want, rtol=1e-9, atol=1e-9), f"{path}: {got!r} != {want!r}"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def check(name: str, got: dict) -> None:
    """Compare `got` with the snapshot `name`, or write it with NYFED_UPDATE_GOLDEN=1."""
    got = _plain(got)
    path = GOLDEN / f"{name}.json"
    if UPDATE:
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(json.dumps(got, indent=1, sort_keys=True))
        return
    assert path.exists(), f"no snapshot {path.name}; run once with NYFED_UPDATE_GOLDEN=1"
    _same(got, json.loads(path.read_text()), name)


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


def _response(r) -> dict:
    from fdia_graph.nyfed.operator.response import summarize

    twins = {
        side: {f.name: _digest(getattr(t, f.name)) for f in fields(t)}  # every array of the run
        for side, t in (("attacked", r.attacked), ("benign", r.benign))
    }
    return {"alarm": r.alarm, "summary": summarize(r), "twins": twins}


@pytest.fixture(scope="module")
def days():
    from fdia_graph.nyfed.operator.response import operator_day

    return {w: operator_day(DAYS / w, POOL, DATA) for w in WINDOWS}


@pytest.fixture(scope="module")
def tools():
    sys.path.insert(0, str(TOOLS))
    import adaptive_bdd
    import bdd_rule
    import closed_loop_bdd
    import naive_tie_bias

    return closed_loop_bdd, adaptive_bdd, naive_tie_bias, bdd_rule


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("mode", ("trusting", "accounting", "oracle"))
@pytest.mark.parametrize("closed_loop", (False, True))
def test_respond_modes(days, window, mode, closed_loop):
    from fdia_graph.nyfed.operator.response import respond

    r = respond(days[window], mode=mode, calib=CALIB, closed_loop=closed_loop)
    check(f"respond_{Path(window).stem}_{mode}_{'cl' if closed_loop else 'ol'}", _response(r))


@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize(
    "options",
    (
        {"after": "benign"},
        {"after": "freeze"},
        {"after": "freeze_cs"},
        {"entry": "peak"},
        {"entry": "west"},
        {"delay": 1},
        {"over": 50.0},
    ),
    ids=lambda o: "_".join(f"{k}-{v}" for k, v in o.items()),
)
def test_respond_options(days, window, options):
    from fdia_graph.nyfed.operator.response import respond

    day = days[window]
    kw = dict(options)
    if "after" in kw:
        kw["alarm_at"] = day.onset + 5
    r = respond(day, mode="trusting", calib=CALIB, closed_loop=True, **kw)
    tag = "_".join(f"{k}-{v}" for k, v in options.items())
    check(f"respond_{Path(window).stem}_{tag}", _response(r))


@pytest.mark.parametrize("window", WINDOWS)
def test_closed_loop_residual_test(tools, window):
    closed_loop_bdd = tools[0]
    row = closed_loop_bdd._job(str(DAYS / window), str(POOL), str(DATA), CALIB, "trusting", 420.1, 2)
    check(f"closed_loop_bdd_{Path(window).stem}", row)


@pytest.mark.parametrize("window", WINDOWS)
def test_adaptive_attacker(tools, window):
    adaptive_bdd = tools[1]
    row = adaptive_bdd._job(str(DAYS / window), str(POOL), str(DATA), CALIB, "trusting", RULE)
    check(f"adaptive_bdd_{Path(window).stem}", row)


@pytest.mark.parametrize("window", WINDOWS)
def test_naive_tie_bias(tools, window):
    naive_tie_bias = tools[2]
    row = naive_tie_bias._job(str(DAYS / window), str(POOL), str(DATA), RULE)
    check(f"naive_tie_bias_{Path(window).stem}", row)


def test_persistence_rule(tools):
    bdd_rule = tools[3]
    j = np.array([400.0, 430.0, 410.0, 431.0, np.nan, 440.0, 441.0, 300.0])
    check(
        "persistence_rule",
        {"first": bdd_rule.first_alarm(j, *RULE), "none": bdd_rule.first_alarm(j[:3], *RULE)},
    )


TERRITORIES = ROOT / "data" / "territories"


@pytest.mark.skipif(
    not all((TERRITORIES / n).exists() for n in ("nys_electric_territories.geojson", "ny_bus_locations.csv")),
    reason="utility territories missing under NYFED_ROOT/data/territories (the builder assigns clients)",
)
@pytest.mark.parametrize("overshoot", (0.0, 175.0))
def test_interface_attack_design(overshoot):
    """The Am design on the onset of `WINDOWS`, at the limit and at the damage study's overshoot."""
    from fdia_graph.nyfed.build import production_builder
    from fdia_graph.nyfed.config import Settings, rng

    b = production_builder(
        str(POOL.relative_to(ROOT)), ("Am",), interfaces=("SPR/DUN-SOUTH",), workers=1, survey=0, root=ROOT
    )
    fam = b.families["Am"]
    onset, L = 48220, b.episode_lengths()["Am"]
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
