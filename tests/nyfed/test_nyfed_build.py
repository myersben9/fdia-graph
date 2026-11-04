"""The release builder's attacker sensitivities stay with the attacker: a timeline built with `v_margin` records the
case's true voltage limits, while the attacker designs against the widened ones (N71). Runs where the NYgrid
checkout, the territories and the 2019 pool are under NYFED_ROOT."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(os.environ.get("NYFED_ROOT", Path(__file__).resolve().parents[2]))
POOL = ROOT / "out" / "pool_2019_v3.h5"
pytestmark = pytest.mark.skipif(
    not (POOL.exists() and (ROOT / "data" / "territories").exists() and (ROOT / "data" / "NYgrid").exists()),
    reason="the 2019 pool, the territories or the NYgrid checkout missing under NYFED_ROOT",
)


def test_a_timeline_built_with_a_voltage_margin_records_the_true_limits(tmp_path):
    from fdia_graph.errors import DataConditionError
    from fdia_graph.nyfed.build import production_builder
    from fdia_graph.nyfed.io import open_timeline

    b = production_builder(
        str(POOL.relative_to(ROOT)), ("At",), workers=1, survey=0, root=ROOT, v_margin=0.02
    )
    true = b.meas.net.case
    assert np.allclose(b.att.net.case.vmax, true.vmax + 0.02)  # the attacker designs on the widened limits
    path = tmp_path / "one.h5"
    for onset in (48220, 50000, 52000, 56000):  # the first window where the ramp finds an attack
        try:
            b.build_episode(path, onset, "At")
            break
        except DataConditionError:
            continue
    else:
        pytest.skip("no ramp attack on the windows tried")
    h, case, _, _ = open_timeline(path)
    h.close()
    assert np.array_equal(case.vmax, true.vmax) and np.array_equal(case.vmin, true.vmin)
