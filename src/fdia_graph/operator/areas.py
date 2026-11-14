"""Who bears an attack's damage, and whose grid the attacker uses: the consequence split by New York zone, by
utility and by neighboring area, with what the attack tampered in each.

`summarize` reports an attack for the whole system. `area_damage` splits the same run, attacked less benign:

- each New York zone and utility: the energy its units produced less (dispatch and regulation), what that saved
  in fuel and lost in sales at the zone's real price, the load it shed, how far the attack cut the true transfer
  into it or ate the margin of the interfaces that feed it (`FEEDS`), and, under the least-cost dispatch, what its
  load pays more at the congestion prices the false reading moves;
- each neighboring area: the inadvertent interchange it supplied and that energy's value at its proxy bus price
  (`PROXY`), what New York borrowed and must repay in kind (NERC BAL-006);
- the system: the fuel New York saved, the borrowed energy's value, and their sum, the net cost of the attack
  once the borrowed energy is priced.

`attack_surface` reads, from the day itself, which zones' and utilities' meters the attack tampered beyond
their noise and how much injection it pretended there. `zone_levers` measures what each zone could give an
attacker: how far its plausible false load and generation could move each interface's flow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fdia_graph.engine.attacks.attacker import wrap_angles
from fdia_graph.engine.sensing.measure import Measurement
from fdia_graph.engine.sensing.meters import KINDS
from fdia_graph.generation.nygrid.case import REF
from fdia_graph.generation.nygrid.inputs import ZONES
from fdia_graph.generation.nygrid.network import Network
from fdia_graph.operator.acsced import ac_shift
from fdia_graph.operator.day import OperatorDay, node_lbmp
from fdia_graph.operator.policy import entry_share
from fdia_graph.operator.run import Response
from fdia_graph.operator.sced import FRAME_HOURS, shift_factors

__all__ = ["FEEDS", "NEIGHBORS", "PROXY", "area_damage", "attack_surface", "neighbor_prices", "zone_levers"]

# The zones on the receiving side of each interface the operator enforces: the New York network with the
# interface's branches removed splits into these zones and the rest. Central East does not split the reduced
# network (parallel paths through its equivalents), so it takes NYISO's definition, zones F to K.
FEEDS = {
    "SPR/DUN-SOUTH": "J",
    "UPNY CONED": "HIJK",
    "CENTRAL EAST - VC": "FGHIJK",
    "MOSES SOUTH": "ABCEFGHIJK",
}
# the external areas of the source model that carry inadvertent interchange (the Quebec import is HVDC)
NEIGHBORS = {1: "ISO-NE", 4: "Ontario", 5: "MISO", 6: "PJM"}
# the NYISO proxy bus whose price values each area's energy; MISO has no proxy bus of its own and reaches New
# York through Ontario and PJM, so its energy takes the mean of their two prices
PROXY = {1: ("NPX",), 4: ("O H",), 5: ("PJM", "O H"), 6: ("PJM",)}


def neighbor_prices(data_dir: str | Path, time: np.ndarray, year: int = 2019) -> dict[int, np.ndarray]:
    """`{area: [T]}` the real hourly price ($/MWh) at which each neighboring area's energy reaches New York."""
    nodes = sorted({n for names in PROXY.values() for n in names})
    table = node_lbmp(data_dir, time, nodes, year)
    return {a: table[:, [nodes.index(n) for n in names]].mean(axis=1) for a, names in PROXY.items()}


def _bus_load(day: OperatorDay) -> np.ndarray:
    """`[T, N]` the true load at each bus (MW): the units' basepoint at the bus less its net injection."""
    c = day.case
    p, _ = Network(c).injections(day.vm, day.va)
    units = np.zeros_like(p)
    np.add.at(units.T, c.unit_bus, np.nan_to_num(day.unit_p).T)
    return np.clip(units - p, 0.0, None)


def _window(day: OperatorDay) -> slice:
    return slice(day.onset, day.onset + day.length)


def _interface_moves(r: Response, day: OperatorDay) -> dict[str, tuple[float, float]]:
    """`{interface: (rise, drop)}` the largest rise and drop (MW, both positive) of each interface's true flow,
    attacked less benign: a rise eats its margin, a drop is transfer the operator gave up."""
    d = r.attacked.iface_change - r.benign.iface_change  # [T, I], NaN where a power flow failed
    out = {}
    for k, it in enumerate(day.interfaces):
        col = d[:, k][np.isfinite(d[:, k])]
        out[it.name] = (
            (max(float(col.max()), 0.0), max(-float(col.min()), 0.0)) if col.size else (np.nan, np.nan)
        )
    return out


def _exposure(zones: set[str], moves: dict[str, tuple[float, float]]) -> dict:
    """The worst margin loss and transfer cut over the interfaces that feed any of `zones`."""
    feeding = [moves[name] for name, fed in FEEDS.items() if name in moves and zones & set(fed)]
    return {
        "margin_lost_mw": max((m[0] for m in feeding), default=0.0),
        "transfer_lost_mw": max((m[1] for m in feeding), default=0.0),
    }


def _payments(r: Response, day: OperatorDay) -> np.ndarray:
    """`[T, N]` what each bus's true load pays more ($ per scan) at the prices the false reading moves, as a market
    prices them: the energy price at the reference bus plus, for each interface, minus its shadow price times the
    bus's shift factor on it, attacked less benign (each run's `price`, which around the AC point also weighs the
    energy price by the bus's delivery factor). Zero under the least-movement dispatch, which reports no prices."""
    a, b = r.attacked, r.benign
    if a.price is not None and b.price is not None:
        return np.nan_to_num(a.price - b.price) * _bus_load(day) * FRAME_HOURS
    if a.congestion is None or b.congestion is None or a.energy is None or b.energy is None:
        return np.zeros((len(day.time), len(day.case.bus_id)))
    ptdf = shift_factors(day.case)
    sf = np.array([it.signs @ ptdf[it.branches] for it in day.interfaces])  # [I, N]
    dmu = np.nan_to_num(a.congestion) - np.nan_to_num(b.congestion)  # [T, I] $/MWh per MW
    dlam = np.nan_to_num(a.energy) - np.nan_to_num(b.energy)  # [T] $/MWh
    return (dlam[:, None] - dmu @ sf) * _bus_load(day) * FRAME_HOURS


def _ny_rows(r: Response, day: OperatorDay) -> list[dict]:
    """One row per New York zone and per utility: generation, fuel, sales, shed, load and exposure."""
    c = day.case
    a, b = r.attacked, r.benign
    dgen = (a.dev + a.reg) - (b.dev + b.reg)  # [T, G] MW
    unit_zone = c.zone[c.unit_bus].astype(str)
    zi = np.array([ZONES.index(z) if z in ZONES else 0 for z in unit_zone])
    price = np.where(np.isin(unit_zone, ZONES), day.lbmp[:, zi], 0.0)  # [T, G]
    fuel, sales = day.cost * dgen * FRAME_HOURS, price * dgen * FRAME_HOURS
    shed = (a.shed - b.shed) * FRAME_HOURS  # [T, N] MWh
    paid = _payments(r, day)  # [T, N] $
    load = _bus_load(day)[_window(day)].max(axis=0)  # [N] each bus's peak load over the window
    bus_zone = c.zone.astype(str)
    moves = _interface_moves(r, day)
    areas = [("zone", z, unit_zone == z, bus_zone == z) for z in ZONES]
    areas += [
        ("utility", str(n), r.unit_client == k, r.bus_client == k) for k, n in enumerate(r.client_names)
    ]
    rows = []
    for kind, name, units, buses in areas:
        zones = {z for z in bus_zone[buses & (load > 0)]} if kind == "utility" else {name}
        rows.append(
            {
                "kind": kind,
                "area": name,
                "gen_mwh": float(dgen[:, units].sum() * FRAME_HOURS),
                "gen_down_mw": float(max(-(dgen[:, units].sum(axis=1)).min(), 0.0)),
                "fuel_usd": float(np.nansum(fuel[:, units])),
                "sales_usd": float(np.nansum(sales[:, units])),
                "shed_mwh": float(shed[:, buses].sum()),
                "load_mw": float(load[buses].sum()),
                "payments_usd": float(np.nansum(paid[:, buses])),
                **_exposure(zones, moves),
            }
        )
    return rows


def _value(mw: np.ndarray, price: np.ndarray) -> float:
    """The energy `mw [T]` at `price [T]` ($): NaN when energy flowed in a scan with no price, so missing prices
    never read as free energy."""
    if (np.isnan(price) & (mw != 0)).any():
        return float("nan")
    return float(np.nansum(mw * price) * FRAME_HOURS)


def _neighbor_rows(r: Response, day: OperatorDay, prices: dict[int, np.ndarray], entry: str) -> list[dict]:
    """One row per neighboring area: the inadvertent interchange it supplied (MW, MWh) and its value ($)."""
    c = day.case
    inadv = r.attacked.uncovered - r.benign.uncovered  # [T] MW, positive where the neighbors supply
    share = entry_share(c, entry)
    unit_area = c.area[c.unit_bus]
    rows = []
    for area, name in NEIGHBORS.items():
        mw = inadv * float(share[unit_area == area].sum())
        rows.append(
            {
                "kind": "neighbor",
                "area": name,
                "supplied_mw": float(max(mw.max(), 0.0)),
                "supplied_mwh": float(mw.sum() * FRAME_HOURS),
                "supplied_usd": _value(mw, prices[area]),
            }
        )
    return rows


def area_damage(
    r: Response, day: OperatorDay, prices: dict[int, np.ndarray], entry: str = "capacity"
) -> list[dict]:
    """The run `r` of the operator on `day`, attacked less benign, split by area: a row per New York zone, per
    utility and per neighboring area, then one for the system. `prices` is `neighbor_prices` of the day's
    frames and `entry` the `ENTRY` the run took the inadvertent interchange by (the run's policy)."""
    ny = _ny_rows(r, day)
    out = _neighbor_rows(r, day, prices, entry)
    zones = [row for row in ny if row["kind"] == "zone"]
    fuel = sum(row["fuel_usd"] for row in zones)
    borrowed = sum(row["supplied_usd"] for row in out)
    system = {
        "kind": "system",
        "area": "NY",
        "fuel_usd": fuel,
        "sales_usd": sum(row["sales_usd"] for row in zones),
        "borrowed_mwh": sum(row["supplied_mwh"] for row in out),
        "borrowed_usd": borrowed,
        "net_cost_usd": fuel + borrowed,  # the fuel saved now, plus the energy owed back at its value
        "relief_usd": float(r.attacked.relief.sum() - r.benign.relief.sum()),
        "shortage_usd": float(r.attacked.shortage.sum() - r.benign.shortage.sum()),
        "payments_usd": sum(row["payments_usd"] for row in zones),  # New York's load at the moved prices
    }
    return ny + out + [system]


def attack_surface(day: OperatorDay) -> list[dict]:
    """Per New York zone and utility, what the day's attack used: the meters it tampered beyond their noise at
    any scan, the buses whose injection meter it moved, and the largest pretended net injection change of the
    area in one scan (MW, negative where it pretends more load)."""
    c, plan = day.case, day.plan
    d = wrap_angles(
        day.z - day.z_benign, plan.kind
    )  # the attack vector of every scan (the noise is the same in both)
    if day.meter_tampered is not None:
        # the labels the design stored, exact: the readings are stored in float32, and the search leaves many
        # changes right at their meter's noise, where that rounding would flip the count
        tampered = day.meter_tampered.any(axis=0)  # [M]
    else:  # the labels' rule (`Attacker.tampered`): against the noise of the noiseless reading
        sigma = plan.sigma_rel * np.abs(Measurement(Network(c), plan).h(day.vm, day.va)) + plan.sigma_abs
        with np.errstate(invalid="ignore"):  # an unsolved scan's NaN readings count as untouched
            tampered = (np.abs(d) > sigma).any(axis=0)
    p_meter = plan.kind == KINDS.index("p")
    meter_zone = c.zone[plan.bus].astype(str)
    areas = [("zone", z, meter_zone == z) for z in ZONES]
    areas += [("utility", str(n), plan.owner == k) for k, n in enumerate(day.own.names)]
    rows = []
    for kind, name, meters in areas:
        net = d[:, meters & p_meter].sum(axis=1)
        net = net[np.isfinite(net)]  # unsolved scans carry NaN readings
        rows.append(
            {
                "kind": kind,
                "area": name,
                "meters_tampered": int((tampered & meters).sum()),
                "buses_moved": int(np.unique(plan.bus[tampered & meters & p_meter]).size),
                "pretended_mw": float(net[np.argmax(np.abs(net))]) if net.size else 0.0,
            }
        )
    return rows


def zone_levers(day: OperatorDay, tau: float = 0.1, band: float = 0.03, ac: bool = False) -> list[dict]:
    """What each New York zone could give an attacker at the last rise scan: for each interface, the most its
    buses alone could move the interface's flow in the DC model (MW), or with `ac` by the AC sensitivities at that
    scan's true state (`ac_shift`), every load within `tau` of itself and every unit within `band` of its upper limit,
    as the attack's bounds allow. A zone's lever ignores the balance the rest of the grid must keep, so it is an upper
    bound on its share of a push, not a push."""
    c = day.case
    t = min(day.onset + day.length // 2 - 1, len(day.time) - 1)  # the last rise scan
    load = _bus_load(day)[t]
    pmax = np.zeros(len(c.bus_id))
    np.add.at(pmax, c.unit_bus, np.where(np.isfinite(c.pmax), c.pmax, 0.0))
    room = tau * load + band * pmax  # MW each bus may move
    room[(c.bus_type == REF) | ~c.is_ny] = 0.0
    bus_zone = c.zone.astype(str)
    if ac:
        shift = ac_shift(c, day.vm[t], day.va[t], day.interfaces)
    else:
        ptdf = shift_factors(c)
        shift = np.array([it.signs @ ptdf[it.branches] for it in day.interfaces])
    rows = []
    for i, it in enumerate(day.interfaces):
        sf = np.abs(shift[i])  # [N] interface MW per MW at each bus
        for z in ZONES:
            inside = bus_zone == z
            rows.append(
                {
                    "interface": it.name,
                    "zone": z,
                    "lever_mw": float((sf * room)[inside].sum()),
                    "load_room_mw": float((tau * load)[inside].sum()),
                    "unit_room_mw": float((band * pmax)[inside].sum()),
                }
            )
    return rows
