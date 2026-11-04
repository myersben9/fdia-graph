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


def test_energy_without_a_price_is_unknown_not_free():
    """A neighbor's energy in a scan with no price makes its value NaN; a missing price where nothing flowed does not."""
    from fdia_graph.nyfed.operator.areas import _value

    price = np.array([20.0, np.nan, 30.0])
    assert np.isnan(_value(np.array([12.0, 12.0, 0.0]), price))
    assert np.isclose(_value(np.array([12.0, 0.0, 24.0]), price), (12 * 20 + 24 * 30) / 12)


def test_the_least_cost_dispatch_prices_the_false_congestion():
    """Under the least-cost dispatch the false congestion binds: the attacked interface takes a positive shadow
    price, every enforced price is at least zero, and the zones' and utilities' payment changes each sum to New
    York's. The least-movement dispatch reports no prices, so its payments are zero."""
    from fdia_graph.nyfed.operator.areas import area_damage, neighbor_prices
    from fdia_graph.nyfed.operator.response import operator_day, respond

    day = operator_day(DAYS / DAMAGE_DAYS[1], DAYS_POOL, DATA)
    prices = neighbor_prices(DATA, day.time)
    r = respond(day, mode="trusting", closed_loop=True, dispatch="cost")
    mu = r.attacked.congestion[np.isfinite(r.attacked.congestion)]
    assert (mu >= -1e-6).all() and r.attacked.congestion[:, 0].max() > 1.0
    assert np.isfinite(r.attacked.energy[day.onset : day.onset + day.length]).all()
    rows = area_damage(r, day, prices)
    total = next(row for row in rows if row["kind"] == "system")["payments_usd"]
    for kind in ("zone", "utility"):
        assert np.isclose(sum(row["payments_usd"] for row in rows if row["kind"] == kind), total, rtol=1e-6)
    still = area_damage(respond(day, mode="trusting", closed_loop=True), day, prices)
    assert next(row for row in still if row["kind"] == "system")["payments_usd"] == 0.0


def test_an_unknown_objective_is_a_config_error():
    from fdia_graph.errors import ConfigError
    from fdia_graph.nyfed.operator.response import OperatorPolicy

    with pytest.raises(ConfigError):
        OperatorPolicy(dispatch="cheapest")


def _cost_sced(day):
    from fdia_graph.nyfed.operator.sced import DcSced

    load = np.zeros_like(day.vm)  # no load to shed, so only redispatch and the slack can relieve
    return DcSced(day.case, day.unit_p, day.cost, day.interfaces, load, objective="cost")


def test_an_overload_no_redispatch_relieves_takes_the_shortage_cap():
    """An interface read far over its limit cannot be relieved: the flow left over it is reported, and its shadow
    price is the 4,000 $/MWh cap of the shortage curve, never the slack weight of the least-movement dispatch."""
    from fdia_graph.nyfed.operator.response import operator_day, seen_interfaces
    from fdia_graph.nyfed.operator.sced import SHORTAGE

    day = operator_day(DAYS / DAMAGE_DAYS[0], DAYS_POOL, DATA)
    t = day.onset
    seen = seen_interfaces(day.plan, day.interfaces, day.z_benign[t : t + 1])[0]
    limit = seen.copy()
    limit[0] = seen[0] - 1e5  # a hundred gigawatts over: no dispatch reaches it
    step = _cost_sced(day).step(t, seen, limit, 0.0, None)
    assert step.unrelieved[0] > 1e4
    assert np.isclose(step.congestion[0], SHORTAGE[-1][1])
    assert np.nanmax(step.congestion) <= SHORTAGE[-1][1] + 1e-6


def test_a_load_the_units_cannot_follow_takes_the_energy_cap():
    from fdia_graph.nyfed.operator.response import operator_day, seen_interfaces
    from fdia_graph.nyfed.operator.sced import ENERGY_CAP

    day = operator_day(DAYS / DAMAGE_DAYS[0], DAYS_POOL, DATA)
    t = day.onset
    seen = seen_interfaces(day.plan, day.interfaces, day.z_benign[t : t + 1])[0]
    step = _cost_sced(day).step(t, seen, np.full_like(seen, np.nan), 1e6, None)  # a terawatt more load
    assert step.unmatched > 1e5 and np.isclose(step.energy, ENERGY_CAP)
