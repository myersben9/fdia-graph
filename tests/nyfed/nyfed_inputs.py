"""The inputs of the nyfed tests in one place, and which tests each one lets run.

They come in three kinds:

- Public inputs under `NYFED_ROOT/data` (NYFED_ROOT defaults to the checkout): NYgrid's `Data/` folder at the
  pinned commit, the NYS utility territories and NYISO's five-minute load archives; `fetch_inputs(root)` from
  `fdia_graph.generation.nygrid.build` downloads them (a test fetches another year's archives itself). Most tests need
  only these, and CI downloads them (`.github/workflows/nyfed.yml`).
- Checked in under `tests/nyfed/data/days` (its README says how they were made): two days of the damage study (one
  attack onset, at the limit and past it), the single-episode day of 2019-07-05, and the 2019 pool's frames of
  those two days. The operator's tests and most golden snapshots run on them with the public inputs.
- Built locally under `NYFED_ROOT/out`: the whole 2019 pool (about 370 MB). The tests that rate lines or survey
  attack onsets over the year (the builder, the timeline and the interface attack's design) skip without it. The
  attack families' tests read the first pool, `out/pool_2019.h5`, on whose windows they were written.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(
    os.environ.get("NYFED_ROOT", Path(__file__).resolve().parents[2])
)  # the nyfed inputs, data/ and out/
DATA = ROOT / "data" / "NYgrid" / "Data"
TERRITORIES = ROOT / "data" / "territories"
POOL = ROOT / "out" / "pool_2019_v3.h5"  # the whole year

DAYS = Path(__file__).parent / "data" / "days"
DAYS_POOL = DAYS / "pool_2019_v3_days.h5"  # the whole pool's frames of the days below
DAMAGE_DAYS = ("am_48220_0.h5", "am_48220_175.h5")  # one onset, at the limit and past it
SINGLE_DAY = DAYS / "sd_south_2019-07-05.h5"

_FETCH = "fdia_graph.generation.nygrid.build.fetch_inputs(root) downloads it"
needs_data = pytest.mark.skipif(
    not DATA.exists(), reason=f"NYgrid's Data/ missing under {ROOT}/data ({_FETCH})"
)
needs_territories = pytest.mark.skipif(
    not all((TERRITORIES / n).exists() for n in ("nys_electric_territories.geojson", "ny_bus_locations.csv")),
    reason=f"the utility territories missing under {TERRITORIES} ({_FETCH})",
)
needs_pool = pytest.mark.skipif(
    not POOL.exists(), reason=f"the whole 2019 pool missing at {POOL} (built locally)"
)
# the first 2019 pool (before the real-time dispatch of N32): the attack families' tests chose their windows on it
POOL_V1 = ROOT / "out" / "pool_2019.h5"
needs_pool_v1 = pytest.mark.skipif(
    not POOL_V1.exists(), reason=f"the first 2019 pool missing at {POOL_V1} (built locally)"
)
