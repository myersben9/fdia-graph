"""The damage split by area adds up to the whole: the zones and the utilities each partition New York's units, the
neighbors' supply is the inadvertent interchange, and the attack surface counts the meters the day's labels mark."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from nyfed_inputs import DAMAGE_DAYS, DATA, DAYS, DAYS_POOL, needs_data

pytestmark = needs_data


@pytest.fixture(scope="module")
def run():
    from fdia_graph.nyfed.operator.areas import area_damage, neighbor_prices
    from fdia_graph.nyfed.operator.response import operator_day, respond

    day = operator_day(DAYS / DAMAGE_DAYS[1], DAYS_POOL, DATA)
    r = respond(day, mode="trusting", closed_loop=True)
    return day, r, area_damage(r, day, neighbor_prices(DATA, day.time))


def test_zones_and_utilities_each_add_up_to_new_york(run):
    from fdia_graph.nyfed.operator.response import summarize

    day, r, rows = run
    s = summarize(r)
    system = next(row for row in rows if row["kind"] == "system")
    for kind in ("zone", "utility"):
        part = [row for row in rows if row["kind"] == kind]
        assert np.isclose(sum(row["fuel_usd"] for row in part), s["fuel_usd"], rtol=1e-9, atol=1e-6)
        assert np.isclose(sum(row["sales_usd"] for row in part), s["settled_usd"], rtol=1e-9, atol=1e-6)
    assert np.isclose(system["fuel_usd"], s["fuel_usd"])
    # every New York load bus belongs to one zone and one utility, so both splits hold the same load
    load = {k: sum(row["load_mw"] for row in rows if row["kind"] == k) for k in ("zone", "utility")}
    assert np.isclose(load["zone"], load["utility"], rtol=1e-9)


def test_the_neighbors_supply_the_inadvertent_interchange_and_it_is_priced(run):
    day, r, rows = run
    supplied = [row for row in rows if row["kind"] == "neighbor"]
    inadv = (r.attacked.uncovered - r.benign.uncovered).sum() / 12  # MWh, five-minute scans
    assert np.isclose(sum(row["supplied_mwh"] for row in supplied), inadv, rtol=1e-9)
    system = next(row for row in rows if row["kind"] == "system")
    assert system["borrowed_mwh"] > 100 and system["borrowed_usd"] > 0  # the damaging attack borrows energy
    assert np.isclose(system["net_cost_usd"], system["fuel_usd"] + system["borrowed_usd"])


def test_the_attack_surface_counts_the_meters_the_labels_mark():
    from fdia_graph.nyfed.operator.areas import attack_surface
    from fdia_graph.nyfed.operator.response import operator_day

    path = DAYS / DAMAGE_DAYS[1]
    rows = attack_surface(operator_day(path, DAYS_POOL, DATA))
    with h5py.File(path, "r") as h:
        labelled = int(h["labels/meter_tampered"][:].any(axis=0).sum())
    for kind in ("zone", "utility"):
        assert sum(row["meters_tampered"] for row in rows if row["kind"] == kind) == labelled


def test_only_new_york_city_moves_the_flow_into_it_much():
    """Sprain Brook/Dunwoodie South feeds New York City alone, so zone J's loads and units are nearly the whole
    lever on it."""
    from fdia_graph.nyfed.operator.areas import zone_levers
    from fdia_graph.nyfed.operator.response import operator_day

    rows = [
        r
        for r in zone_levers(operator_day(DAYS / DAMAGE_DAYS[0], DAYS_POOL, DATA))
        if r["interface"] == "SPR/DUN-SOUTH"
    ]
    lever = {r["zone"]: r["lever_mw"] for r in rows}
    assert lever["J"] > 0.8 * sum(lever.values())
