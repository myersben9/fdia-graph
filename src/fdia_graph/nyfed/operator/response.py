"""What the operator does under an attack, and what it costs on the true grid (D1 to D6).

The operator dispatches on what it sees each scan (`DcSced`): the attacked scans, and separately the
benign twin of the same scans, each in closed loop on its own dispatch. Its dispatch is then applied to
the true grid, whose load the attack never changed: the imbalance the dispatch leaves is picked up by
regulation, the online free units in proportion to their five-minute ramp (AGC), and an AC power flow at
the true loads gives the true interface flows. The attack's consequence is the attacked twin less the
benign twin, so whatever the model does alike in both (DC against AC, the cost model) cancels.

Reported per scan: the dispatch moved, the regulation it calls, the interface flow the operator could not
relieve, the true interface change, the worst single-outage interface flow over the New York 345 kV
branches (the N-1 consequence, O2), the production cost at the units' costs, and the energy settled at
the hour's real NYISO zonal LBMP, with the regulation and the dispatch split by the utility whose
territory holds each unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fdia_graph.nyfed.grid.case import Case
from fdia_graph.nyfed.grid.clients import Ownership
from fdia_graph.nyfed.grid.interfaces import Interface, hourly_interfaces
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.io import open_timeline, read_pool
from fdia_graph.nyfed.operator.sced import FRAME_HOURS, DcSced, outage_factors, shift_factors
from fdia_graph.nyfed.scenario.dispatch import hourly_dispatch
from fdia_graph.nyfed.scenario.inputs import ZONES, read_nygrid
from fdia_graph.nyfed.scenario.solver import AcSolver
from fdia_graph.nyfed.sensing.meters import KINDS, MeterPlan

# NYISO's zone names in priceHourly, by zone letter
LBMP_ZONE = {
    "A": "WEST",
    "B": "GENESE",
    "C": "CENTRL",
    "D": "NORTH",
    "E": "MHK VL",
    "F": "CAPITL",
    "G": "HUD VL",
    "H": "MILLWD",
    "I": "DUNWOD",
    "J": "N.Y.C.",
    "K": "LONGIL",
}


@dataclass(frozen=True)
class OperatorDay:
    """One timeline file's frames with what the operator needs: the true state `vm`, `va [T, N]`, the
    attacked and benign scans `z`, `z_benign [T, M]`, the pool's basepoint `unit_p [T, G]`, the units'
    cost `cost [T, G]` ($/MWh, NYgrid's hourly costs interpolated on the frame clock as the pool build
    does), the real zonal LBMP `lbmp [T, 11]` ($/MWh, zones A to K), the `interfaces` the operator
    enforces, `solved [T]`, and the file's first episode (`onset`, `length`)."""

    case: Case
    own: Ownership
    plan: MeterPlan
    time: np.ndarray
    vm: np.ndarray
    va: np.ndarray
    z: np.ndarray
    z_benign: np.ndarray
    unit_p: np.ndarray
    cost: np.ndarray
    lbmp: np.ndarray
    interfaces: list[Interface]
    solved: np.ndarray
    onset: int
    length: int


def _hour_weights(time: np.ndarray, hours: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each frame the hour before, the hour after and the weight of the hour after, on the hours'
    midpoints (as `PoolBuilder._brackets`)."""
    mid = (hours.astype("datetime64[s]") + np.timedelta64(1800, "s")).astype(np.int64).astype(float)
    pos = np.interp(time.astype("datetime64[s]").astype(np.int64).astype(float), mid, np.arange(len(mid)))
    lo = np.floor(pos).astype(int)
    return lo, np.minimum(lo + 1, len(mid) - 1), pos - lo


def _zonal_lbmp(data_dir: Path, time: np.ndarray, year: int = 2019) -> np.ndarray:
    """`[T, 11]` the real hourly zonal LBMP ($/MWh) of each frame's hour, zones A to K."""
    price = pd.read_csv(data_dir / f"priceHourly_{year}.csv")
    price["t"] = pd.to_datetime(price.TimeStamp, format="%m/%d/%Y %H:%M:%S")
    table = price.pivot_table(index="t", columns="ZoneName", values="LBMP")
    hour = pd.DatetimeIndex(time.astype("datetime64[s]")).floor("h")
    return table.reindex(index=hour, columns=[LBMP_ZONE[z] for z in ZONES]).to_numpy(float)


def operator_day(
    timeline: str | Path,
    pool: str | Path,
    data_dir: str | Path,
    interfaces: tuple[str, ...] = ("SPR/DUN-SOUTH",),
) -> OperatorDay:
    """An `OperatorDay` from a timeline file (a single-episode file, or any timeline), the pool it was
    built from and NYgrid's `Data/` folder."""
    data_dir = Path(data_dir)
    h, case, own, plan = open_timeline(timeline)
    time = h["frames/time"][:]
    stamps = time.astype("datetime64[s]")
    p = read_pool(pool)
    k = np.searchsorted(p.time.astype("datetime64[s]"), stamps)
    if not (p.time.astype("datetime64[s]")[k] == stamps).all() or p.unit_p is None:
        raise ValueError("the pool does not hold every frame of the timeline with its unit dispatch")
    hours = read_nygrid(data_dir, 2019)
    when = np.array([x.time for x in hours]).astype("datetime64[s]")
    near = np.flatnonzero(
        (when >= stamps[0] - np.timedelta64(3600, "s")) & (when <= stamps[-1] + np.timedelta64(3600, "s"))
    )
    disp = hourly_dispatch(case, [hours[i] for i in near])
    lo, hi, w = _hour_weights(stamps, disp.hour)
    cost = (1 - w)[:, None] * disp.unit_cost[lo] + w[:, None] * disp.unit_cost[hi]
    vm, va = h["frames/vm"][:], h["frames/va"][:]
    eps = h["episodes"]
    day = OperatorDay(
        case=case,
        own=own,
        plan=plan,
        time=time,
        vm=vm,
        va=va,
        z=h["frames/z"][:].astype(float),
        z_benign=h["frames/z_benign"][:].astype(float),
        unit_p=p.unit_p[k],
        cost=cost,
        lbmp=_zonal_lbmp(data_dir, time),
        interfaces=hourly_interfaces(case, time, data_dir, list(interfaces)),
        solved=np.isfinite(vm).all(axis=1) & np.isfinite(p.unit_p[k]).all(axis=1),
        onset=int(eps["onset"][0]) if len(eps["onset"]) else -1,
        length=int(eps["length"][0]) if len(eps["length"]) else 0,
    )
    h.close()
    return day


def seen_interfaces(plan: MeterPlan, interfaces: list[Interface], z: np.ndarray) -> np.ndarray:
    """`[T, I]` each interface's flow as the operator reads it: the signed sum of its branches' P-flow
    meters, at the from end, or the to end with its sign turned where the from end has none."""
    pf = KINDS.index("pf")
    out = np.zeros((len(z), len(interfaces)))
    for i, it in enumerate(interfaces):
        for b, s in zip(it.branches, it.signs):
            mine = np.flatnonzero((plan.kind == pf) & (plan.branch == b))
            rows = {int(plan.end[r]): int(r) for r in mine}
            if 0 in rows:
                out[:, i] += s * z[:, rows[0]]
            else:
                out[:, i] -= s * z[:, rows[1]]
    return out


@dataclass(frozen=True)
class Twin:
    """One run of the operator over the frames [T]: `dev`, `reg [T, G]` the dispatch's change from the
    basepoint and the regulation (MW), `unrelieved [T, I]` and `unmatched [T]` what the dispatch could not
    cover as the operator sees it (MW), `iface_change [T, I]` the true interface flow change from the
    basepoint's (AC, MW), `n1_change [T]` the change in the worst single-outage interface flow (MW),
    `fuel [T]` the production cost change and `settled [T]` the energy change at the zonal LBMP ($ per
    scan)."""

    dev: np.ndarray
    reg: np.ndarray
    unrelieved: np.ndarray
    unmatched: np.ndarray
    iface_change: np.ndarray
    n1_change: np.ndarray
    fuel: np.ndarray
    settled: np.ndarray


class _TrueGrid:
    """The true grid of a day: AC power flows at the true loads for a given unit dispatch, the true
    interface flows and the worst single outage of a New York 345 kV branch."""

    def __init__(self, day: OperatorDay):
        c = day.case
        self.day, self.net = day, Network(c)
        self.solver = AcSolver(c)
        pt, qt = self.net.injections(day.vm, day.va)
        gen = np.zeros_like(pt)
        np.add.at(gen.T, c.unit_bus, np.nan_to_num(day.unit_p).T)
        has_unit = np.zeros(len(c.bus_id), bool)
        has_unit[c.unit_bus] = True
        self.pd, self.qd = gen - pt, np.where(has_unit, 0.0, -qt)  # loads net of the HVDC links
        ny345 = c.is_ny[c.f] & c.is_ny[c.t] & (c.kv[c.f] >= 345) & (c.kv[c.t] >= 345)
        self.outages = np.flatnonzero(ny345)
        self.lodf = outage_factors(c, shift_factors(c), self.outages)
        self._base: dict[int, tuple[np.ndarray, float]] = {}

    def _flows(self, t: int, units: np.ndarray) -> np.ndarray | None:
        d = self.day
        r = self.solver.power_flow(
            self.pd[t],
            self.qd[t],
            units,
            units > 0.01,
            d.vm[t, d.case.unit_bus],
            np.zeros(len(d.case.dc_f)),
            start=(d.vm[t], d.va[t]),
        )
        return None if r is None else self.net.flows(r["vm"], r["va"])[0]

    def _measure(self, pf: np.ndarray) -> tuple[np.ndarray, float]:
        iface = np.array([it.model_flow(pf) for it in self.day.interfaces])
        it = self.day.interfaces[0]
        after = iface[0] + it.signs @ (self.lodf[it.branches] * pf[self.outages])
        return iface, float(after.max())

    def change(self, t: int, units: np.ndarray) -> tuple[np.ndarray, float]:
        """The true interface flows' and the worst N-1 flow's change from the basepoint at scan `t`."""
        if t not in self._base:
            pf = self._flows(t, self.day.unit_p[t])
            self._base[t] = (
                self._measure(pf) if pf is not None else (np.full(len(self.day.interfaces), np.nan), np.nan)
            )
        pf = self._flows(t, units)
        if pf is None:
            return np.full(len(self.day.interfaces), np.nan), np.nan
        iface, worst = self._measure(pf)
        return iface - self._base[t][0], worst - self._base[t][1]


def _regulation(day: OperatorDay, t: int, imbalance: float) -> np.ndarray:
    """The imbalance (MW, positive when the units produce too little) spread over the online free units
    in proportion to their five-minute ramp, as automatic generation control picks it up."""
    c = day.case
    on = (
        ~np.isin(c.fuel.astype(str), ("Nuclear", "Hydro", "Import"))
        & (day.unit_p[t] > 0.5)
        & np.isfinite(c.ramp)
    )
    share = np.where(on, c.ramp, 0.0)
    return imbalance * share / share.sum()


def _run(day: OperatorDay, seen: np.ndarray, load_change: np.ndarray, delay: int, grid: _TrueGrid) -> Twin:
    c, T, G, I = day.case, len(day.time), len(day.case.unit_bus), len(day.interfaces)
    sced = DcSced(c, day.unit_p, day.cost, day.interfaces)
    pf_true = Network(c).flows(day.vm, day.va)[0]
    true_iface = np.array([it.model_flow(pf_true) for it in day.interfaces]).T
    limit = true_iface + np.array([it.limit - it.flow for it in day.interfaces]).T  # D3
    unit_zone = np.array([ZONES.index(z) if z else -1 for z in c.zone.astype(str)[c.unit_bus]])
    price = np.where(unit_zone >= 0, day.lbmp[:, np.maximum(unit_zone, 0)], np.nan)  # [T, G]
    out = {
        "dev": np.zeros((T, G)),
        "reg": np.zeros((T, G)),
        "unrelieved": np.full((T, I), np.nan),
        "unmatched": np.full(T, np.nan),
        "iface_change": np.zeros((T, I)),
        "n1_change": np.zeros(T),
        "fuel": np.zeros(T),
        "settled": np.zeros(T),
    }
    prev = None
    for t in range(T):
        if not (day.solved[t] and (t == 0 or day.solved[t - 1])):
            out["iface_change"][t], out["n1_change"][t] = np.nan, np.nan
            prev = None if not day.solved[t] else prev
            continue
        s = max(t - delay, 0)  # the operator acts on the scan `delay` scans back
        # a scan the operator has no reading of (an unsolved frame) shows it no change
        step = sced.step(t, seen[s], limit[t], float(np.nan_to_num(load_change[s])), prev)
        reg = _regulation(day, t, -float(step.dev.sum()))
        prev = step.dev
        out["dev"][t], out["reg"][t] = step.dev, reg
        out["unrelieved"][t], out["unmatched"][t] = step.unrelieved, step.unmatched
        moved = step.dev + reg
        if np.abs(moved).max() > 1e-3:
            out["iface_change"][t], out["n1_change"][t] = grid.change(t, day.unit_p[t] + moved)
        out["fuel"][t] = float(day.cost[t] @ moved) * FRAME_HOURS
        out["settled"][t] = float(np.nansum(price[t] * moved)) * FRAME_HOURS
    return Twin(**out)


@dataclass(frozen=True)
class Response:
    """The operator's run on the attacked scans and on their benign twin, with `unit_client [G]` the
    client whose territory holds each unit (-1 outside New York)."""

    attacked: Twin
    benign: Twin
    unit_client: np.ndarray
    client_names: list[str]
    onset: int
    length: int
    delay: int
    over: float


def respond(day: OperatorDay, delay: int = 0, over: float = 0.0) -> Response:
    """Run the operator (O1) on `day`, acting `delay` scans late (O5). `over` (MW) is a what-if: the
    attacked interface reading pushed that far past what the attack reached, in proportion to the push
    over the episode, to show how the consequence grows if an attacker overshoots the limit."""
    seen_b = seen_interfaces(day.plan, day.interfaces, day.z_benign)
    seen_a = seen_interfaces(day.plan, day.interfaces, day.z)
    if over and day.onset >= 0:
        push = seen_a - seen_b
        w = slice(day.onset, day.onset + day.length)
        seen_a[w] += over * push[w] / max(float(np.abs(push[w]).max()), 1e-9)
    p = KINDS.index("p")
    rows = np.flatnonzero(day.plan.kind == p)
    load_change = -(day.z[:, rows] - day.z_benign[:, rows]).sum(axis=1)  # the seen load's change, MW
    grid = _TrueGrid(day)
    return Response(
        attacked=_run(day, seen_a, load_change, delay, grid),
        benign=_run(day, seen_b, np.zeros(len(day.time)), delay, grid),
        unit_client=day.own.bus_client[day.case.unit_bus],
        client_names=[str(n) for n in day.own.names],
        onset=day.onset,
        length=day.length,
        delay=delay,
        over=over,
    )


def summarize(r: Response) -> dict:
    """The attack's consequence over the day, attacked less benign: the most dispatch moved and regulation
    called (MW), the regulation energy (MWh), the most interface flow left unrelieved (MW), the largest true
    interface and N-1 changes (MW, both signs), the production cost and LBMP settlement changes ($), and
    the regulation energy and its settlement by utility."""
    a, b = r.attacked, r.benign
    reg_a = np.clip(a.reg, 0, None) * FRAME_HOURS  # MWh each unit produced for regulation, upward
    out = {
        "delay": r.delay,
        "over_mw": r.over,
        "dispatch_moved_mw": float(np.abs(a.dev).sum(axis=1).max() - np.abs(b.dev).sum(axis=1).max()),
        "regulation_mw": float(a.reg.sum(axis=1).max()),
        "regulation_mwh": float(reg_a.sum() - np.clip(b.reg, 0, None).sum() * FRAME_HOURS),
        "unrelieved_mw": float(np.nanmax(a.unrelieved)) if np.isfinite(a.unrelieved).any() else 0.0,
        "unmatched_mw": float(np.nanmax(np.abs(a.unmatched))),
        "iface_down_mw": float(np.nanmin(a.iface_change - b.iface_change)),
        "iface_up_mw": float(np.nanmax(a.iface_change - b.iface_change)),
        "n1_up_mw": float(np.nanmax(a.n1_change - b.n1_change)),
        "fuel_usd": float(a.fuel.sum() - b.fuel.sum()),
        "settled_usd": float(a.settled.sum() - b.settled.sum()),
    }
    for c, name in enumerate(r.client_names):
        mine = r.unit_client == c
        out[f"regulation_mwh_{name}"] = float(reg_a[:, mine].sum())
    return out
