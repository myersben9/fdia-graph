"""The synthetic New York grid the timeline tests share: the packaged case file with made-up clients, eight
weeks of hourly made-up states and two families that write a bias on one meter, so a timeline is generated
with no NYgrid data and no solver."""

from __future__ import annotations

import numpy as np
import pytest

WEEK_HOURS = 7 * 24
T = 8 * WEEK_HOURS  # eight weeks of hourly frames: six train, one val, one test
LENGTHS = {"Am": 6, "At": 4}
UNSOLVED = slice(300, 303)


def _family(name, plan):
    from fdia_graph.engine.attacks.family import AttackFamily, Episode

    class Bias(AttackFamily):
        """A growing bias on one meter drawn from the design's stream; no attack at an onset divisible by 7."""

        def design(self, window, rng):
            onset, vm, _ = window
            if onset % 7 == 0:
                self.last_miss = "no_target"
                return None
            meter = int(rng.integers(len(plan.kind)))
            a = np.zeros((len(vm), len(plan.kind)))
            a[:, meter] = 5.0 * (1 + np.arange(len(vm)))
            owner, bus = np.array([plan.owner[meter]]), np.array([plan.bus[meter]])
            return Episode(name, owner, onset, len(vm), bus, -1, a)

    return Bias()


@pytest.fixture(scope="module")
def grid():
    """(pool, attacker, families, ownership): the packaged case, three made-up clients, a made-up pool."""
    pytest.importorskip("matpowercaseframes")  # the packaged case file's reader (the [ny] extra)
    pytest.importorskip("scipy")
    from fdia_graph.engine.attacks.attacker import Attacker
    from fdia_graph.engine.sensing.measure import Measurement
    from fdia_graph.engine.sensing.meters import plan_meters
    from fdia_graph.generation.nygrid.case import nygrid_case
    from fdia_graph.generation.nygrid.config import Settings, rng
    from fdia_graph.generation.nygrid.network import Network
    from fdia_graph.generation.nygrid.ownership import Ownership
    from fdia_graph.generation.nygrid.pool import FAILED, SOLVED, Pool

    case = nygrid_case()
    n = len(case.bus_id)
    bus_client = np.where(case.is_ny, np.arange(n) % 3, -1)
    own = Ownership(np.array(["a", "b", "c"]), bus_client, bus_client[case.f] != bus_client[case.t])
    net = Network(case)
    meas = Measurement(net, plan_meters(net, own, rng(Settings(), "meters")))
    draw = np.random.default_rng(7)
    vm, va = 1.0 + 0.01 * draw.standard_normal((T, n)), 2.0 * draw.standard_normal((T, n))
    kind = np.full(T, SOLVED)
    vm[UNSOLVED], va[UNSOLVED], kind[UNSOLVED] = np.nan, np.nan, FAILED
    time = (np.datetime64("2019-01-01T00:00:00") + np.arange(T) * np.timedelta64(1, "h")).astype(
        "datetime64[s]"
    )
    pool = Pool(time, vm, va, np.zeros((T, n)), np.zeros((T, n)), kind)
    families = {name: _family(name, meas.plan) for name in LENGTHS}
    return pool, Attacker(net, meas), families, own
