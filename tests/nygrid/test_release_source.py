"""The release source's attacker sensitivities stay with the attacker: a timeline built with `v_margin` records the
case's true voltage limits, while the attacker designs against the widened ones (N71). Needs the public inputs and
the whole 2019 pool (inputs.py)."""

from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pytest

from .inputs import POOL, ROOT, needs_data, needs_pool, needs_territories

pytestmark = [needs_data, needs_territories, needs_pool]


def test_a_timeline_built_with_a_voltage_margin_records_the_true_limits(tmp_path):
    from fdia_graph import schema
    from fdia_graph.errors import DataConditionError
    from fdia_graph.generation.nygrid.source import NewYorkSource, SingleEpisodeSource
    from fdia_graph.models.config import NewYorkRelease

    release = NewYorkRelease(families=("At",), workers=1, survey=0, root=ROOT, v_margin=0.02)
    b = NewYorkSource.production(str(POOL.relative_to(ROOT)), release)
    true = b.meas.net.case
    assert np.allclose(b.att.net.case.vmax, true.vmax + 0.02)  # the attacker designs on the widened limits
    path = tmp_path / "one.h5"
    for onset in (48220, 50000, 52000, 56000):  # the first window where the ramp finds an attack
        try:
            SingleEpisodeSource(b, onset, "At").generate(path)
            break
        except DataConditionError:
            continue
    else:
        pytest.skip("no ramp attack on the windows tried")
    with h5py.File(path, "r") as h:
        graph = h[schema.Group.GRAPH]
        vmax, vmin = graph[schema.Static.BUS_VMAX][:], graph[schema.Static.BUS_VMIN][:]
    assert np.array_equal(vmax, true.vmax) and np.array_equal(vmin, true.vmin)


def test_the_load_cap_and_band_sensitivities_reach_the_opf_attacker():
    from fdia_graph.engine.attacks.attacker import generator_band
    from fdia_graph.generation.nygrid.source import NewYorkSource
    from fdia_graph.models.config import NewYorkRelease

    pool = str(POOL.relative_to(ROOT))
    release = NewYorkRelease(families=("Am",), workers=1, survey=0, root=ROOT)
    base = NewYorkSource.production(pool, release)
    knob = NewYorkSource.production(pool, replace(release, load_cap=0.05, band=0.02))
    case = base.meas.net.case
    assert base.families["Am"].load_cap == 0.1 and knob.families["Am"].load_cap == 0.05
    assert np.array_equal(base.families["Am"].gen_band, generator_band(case))
    gen = np.isfinite(generator_band(case))
    assert np.allclose(knob.families["Am"].gen_band[gen], generator_band(case)[gen] * 2 / 3)


def test_a_lagged_attack_is_the_earlier_windows_design_added_to_the_true_scans(tmp_path):
    from fdia_graph import schema
    from fdia_graph.dataset.clients import Dataset
    from fdia_graph.errors import ConfigError, DataConditionError
    from fdia_graph.generation.nygrid.source import NewYorkSource, SingleEpisodeSource
    from fdia_graph.models.config import NewYorkRelease

    release = NewYorkRelease(families=("At",), workers=1, survey=0, root=ROOT)
    b = NewYorkSource.production(str(POOL.relative_to(ROOT)), release)
    lag = 12  # an hour
    for known in (48220, 50000, 52000, 56000):  # the first window where the ramp finds an attack
        try:
            SingleEpisodeSource(b, known, "At").generate(tmp_path / "known.h5")
            SingleEpisodeSource(b, known + lag, "At", lag=lag).generate(tmp_path / "lagged.h5")
            break
        except DataConditionError:
            continue
    else:
        pytest.skip("no ramp attack on the windows tried")

    def attack(name: str) -> tuple[int, np.ndarray, np.ndarray]:
        ds = Dataset(tmp_path / name)
        h, every = ds.h, np.arange(ds.h[schema.Y].shape[0])
        on = h[schema.SEQ_ID][:] >= 0
        out = (
            int(h[schema.Group.EPISODES][schema.EPISODE_ONSET][0]),
            (ds.z(every) - ds.z(every, benign=True))[on],
            h[schema.NODE_CLEAN][:, :, 0][on],  # the true voltage magnitudes
        )
        h.close()
        return out

    k_onset, k_a, k_vm = attack("known.h5")
    l_onset, l_a, l_vm = attack("lagged.h5")
    assert k_onset == l_onset == 0  # each file starts at its own onset
    assert np.allclose(k_a, l_a, atol=1e-3)  # the same vector (z is float32), designed on the earlier window
    assert not np.allclose(k_vm, l_vm)  # added to the scans of the later one
    with pytest.raises(ConfigError):
        SingleEpisodeSource(b, 5, "At", lag=10)
