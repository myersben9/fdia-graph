"""The release builder's attacker sensitivities stay with the attacker: a timeline built with `v_margin` records the
case's true voltage limits, while the attacker designs against the widened ones (N71). Needs the public inputs and
the whole 2019 pool (nyfed_inputs.py)."""

from __future__ import annotations

import numpy as np
import pytest
from nyfed_inputs import POOL, ROOT, needs_data, needs_pool, needs_territories

pytestmark = [needs_data, needs_territories, needs_pool]


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
