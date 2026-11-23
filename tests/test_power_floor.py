"""The floor under the power meters' class as a setting: the plan carries it, the noise and every
count follow it, the file records it, and the readers of both toolkits weight by it."""

import numpy as np
import pytest

pytest.importorskip("pandapower")

from fdia_graph.constants.meters import (  # noqa: E402
    KIND_ACCURACY,
    POWER_KINDS,
    POWER_NOISE_FLOOR_MW,
    WU26_SCADA_NOISE_PU,
)
from fdia_graph.errors import ConfigError  # noqa: E402
from fdia_graph.generation import _load_states  # noqa: E402
from fdia_graph.generation.search import SearchSource  # noqa: E402
from fdia_graph.models.config import MeterSettings, TimelineSettings  # noqa: E402
from fdia_graph.systems.network import Network  # noqa: E402
from fdia_graph.systems.ownership import Ownership  # noqa: E402


def _source(floor):
    """A small ramp-only IEEE-14 source with three made-up clients, its power meters on `floor`."""
    net = Network.from_ieee(14)
    bus_client = np.arange(len(net.case.bus_id)) % 3
    own = Ownership(np.array(["a", "b", "c"]), bus_client, bus_client[net.case.f] != bus_client[net.case.t])
    meters = MeterSettings(power_floor_pu=floor)
    settings = TimelineSettings.of(families=("At",), ramp_len=10, attacked_frac=0.2, redundancy=meters)
    return SearchSource(net, _load_states("ieee14", None)[:150], settings, seed=3, own=own)


def test_the_default_floor_is_the_packages_own_whatever_the_base():
    assert MeterSettings().power_floor_pu is None
    assert (
        MeterSettings().power_floor_mw(100.0) == MeterSettings().power_floor_mw(250.0) == POWER_NOISE_FLOOR_MW
    )
    assert MeterSettings(power_floor_pu=WU26_SCADA_NOISE_PU).power_floor_mw(100.0) == pytest.approx(3.0)
    record = TimelineSettings().as_record()
    assert "power_floor_pu" not in record["meters"] and "targets" not in record["ramp"]  # their hash stands
    for refused in (0.0, -0.01, float("inf")):
        with pytest.raises(ConfigError):
            MeterSettings(power_floor_pu=refused)


def test_the_floor_is_under_every_power_meter_and_no_other():
    plain, floored = _source(None).g.plan, _source(WU26_SCADA_NOISE_PU).g.plan
    power = np.isin(floored.kind, POWER_KINDS)
    assert power.any() and np.allclose(floored.sigma_abs[power], 3.0)
    assert np.array_equal(floored.sigma_abs[~power], plain.sigma_abs[~power])
    assert np.array_equal(floored.sigma_rel, plain.sigma_rel) and np.array_equal(floored.bias, plain.bias)
    assert all(
        plain.sigma_abs[plain.kind == k].tolist() == [KIND_ACCURACY[k][1]] * int((plain.kind == k).sum())
        for k in POWER_KINDS
    )


def test_a_file_records_its_floor_and_both_readers_weight_by_it(tmp_path):
    """A file made with a floor of its own stores it; `fg.load`, the clients' reader and the state
    estimator read the stored value. A file made without one stores nothing and reads the package's own."""
    import h5py

    import fdia_graph as fg
    from fdia_analysis.se.methods import WLS
    from fdia_graph import schema
    from fdia_graph.dataset.clients import Dataset

    for floor, mw in ((WU26_SCADA_NOISE_PU, 3.0), (None, POWER_NOISE_FLOOR_MW)):
        source = _source(floor)
        path = source.generate(tmp_path / f"floor_{floor}.h5")
        with h5py.File(path, "r") as f:
            assert (schema.Attr.POWER_FLOOR_MW in f.attrs) == (floor is not None)
        ds = fg.FdiaGraph(str(path))
        assert ds.power_floor_mw == pytest.approx(mw)
        clients = Dataset(path)
        power = np.isin(clients.plan.kind, POWER_KINDS)
        assert np.allclose(clients.plan.sigma_abs[power], mw)
        assert np.array_equal(
            clients.plan.sigma_abs, source.g.plan.sigma_abs[_as_read(source.g.plan, clients.plan)]
        )
        clients.h.close()
        est = WLS()
        est._build_network(ds, need_clean=False)
        assert est.power_floor_mw == pytest.approx(mw)
        # the estimator's class sigma of every power meter reading nothing is the floor, per unit
        sigma = est._class_sigma(np.zeros(int(est.mask.sum())))
        n, e = est.N, est.E  # the slots in the estimator's order: |V|, P, Q, angle per bus, then the flows
        power = np.r_[np.zeros(n, bool), np.ones(2 * n, bool), np.zeros(n, bool), np.ones(2 * e, bool)]
        power = power[est.mask]
        assert power.any() and np.allclose(sigma[power], mw / est.baseMVA)


def _as_read(written, read):
    """For each meter of the plan `read` from a file, its row in the plan the file was `written` with."""
    key = {
        (int(k), int(b), int(e), int(d)): i
        for i, (k, b, e, d) in enumerate(zip(written.kind, written.bus, written.branch, written.end))
    }
    return np.array(
        [
            key[(int(k), int(b), int(e), int(d))]
            for k, b, e, d in zip(read.kind, read.bus, read.branch, read.end)
        ]
    )
