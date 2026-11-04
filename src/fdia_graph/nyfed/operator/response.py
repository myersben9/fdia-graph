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

from fdia_graph.errors import ConfigError, DataConditionError
from fdia_graph.nyfed.grid.case import Case
from fdia_graph.nyfed.grid.clients import Ownership
from fdia_graph.nyfed.grid.interfaces import Interface, hourly_interfaces
from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.grid.nygrid import HQ_BUS
from fdia_graph.nyfed.io import open_timeline, read_pool
from fdia_graph.nyfed.operator.sced import (
    FRAME_HOURS,
    SHED_AFTER,
    DcSced,
    outage_factors,
    shift_factors,
    shortage_usd,
)
from fdia_graph.nyfed.scenario.dispatch import hourly_dispatch
from fdia_graph.nyfed.scenario.inputs import ZONES, read_nygrid
from fdia_graph.nyfed.scenario.pool import Pool
from fdia_graph.nyfed.scenario.solver import AcSolver
from fdia_graph.nyfed.sensing.meters import JITTER_FRAC, KINDS, MeterPlan

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


# the interfaces the operator enforces, the attacked one first (N64)
INTERFACES = ("SPR/DUN-SOUTH", "UPNY CONED", "CENTRAL EAST - VC", "MOSES SOUTH")
REG_CAP = 275.0  # MW of regulation the operator holds, the summer requirement at its highest hour (N63)
# N64: small gas, oil and kerosene units in New York City and Long Island, the quick-start fleet
QUICK_FUELS, QUICK_ZONES, QUICK_MAX_MW = ("Natural Gas", "Kerosene", "Fuel Oil 2"), ("J", "K"), 200.0
_POOLS: dict[str, Pool] = {}  # pools and NYgrid's hourly tables read once per process: a study runs many days
_HOURS: dict[str, list] = {}


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


def _target_first(interfaces: list[Interface], branch: int) -> list[Interface]:
    """The interfaces with the one holding the attack's target branch first (the summary reads it as the
    attacked interface), every one still enforced."""
    k = next((i for i, it in enumerate(interfaces) if branch in set(it.branches.tolist())), 0)
    return [interfaces[k]] + interfaces[:k] + interfaces[k + 1 :]


def operator_day(
    timeline: str | Path,
    pool: str | Path,
    data_dir: str | Path,
    interfaces: tuple[str, ...] = INTERFACES,
) -> OperatorDay:
    """An `OperatorDay` from a timeline file (a single-episode file, or any timeline), the pool it was
    built from and NYgrid's `Data/` folder."""
    data_dir = Path(data_dir)
    h, case, own, plan = open_timeline(timeline)
    time = h["frames/time"][:]
    stamps = time.astype("datetime64[s]")
    if str(pool) not in _POOLS:
        _POOLS[str(pool)] = read_pool(pool)
    p = _POOLS[str(pool)]
    k = np.searchsorted(p.time.astype("datetime64[s]"), stamps)
    if not (p.time.astype("datetime64[s]")[k] == stamps).all() or p.unit_p is None:
        raise DataConditionError("the pool does not hold every frame of the timeline with its unit dispatch")
    if str(data_dir) not in _HOURS:
        _HOURS[str(data_dir)] = read_nygrid(data_dir, 2019)
    hours = _HOURS[str(data_dir)]
    when = np.array([x.time for x in hours]).astype("datetime64[s]")
    near = np.flatnonzero(
        (when >= stamps[0] - np.timedelta64(3600, "s")) & (when <= stamps[-1] + np.timedelta64(3600, "s"))
    )
    disp = hourly_dispatch(case, [hours[i] for i in near])
    lo, hi, w = _hour_weights(stamps, disp.hour)
    cost = (1 - w)[:, None] * disp.unit_cost[lo] + w[:, None] * disp.unit_cost[hi]
    vm, va = h["frames/vm"][:], h["frames/va"][:]
    target = int(h["episodes/target_branch"][0]) if len(h["episodes/target_branch"]) else -1
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
        interfaces=_target_first(hourly_interfaces(case, time, data_dir, list(interfaces)), target),
        solved=np.isfinite(vm).all(axis=1) & np.isfinite(p.unit_p[k]).all(axis=1),
        onset=int(eps["onset"][0]) if len(eps["onset"]) else -1,
        length=int(eps["length"][0]) if len(eps["length"]) else 0,
    )
    h.close()
    return day


NEIGHBOUR_SEED = 20190101  # the neighbours' tie meters: one bias each, the same every day


def tie_meters(case: Case, plan: MeterPlan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ties to a neighbour with a P-flow meter at their New York end: the branches, which end is New
    York's (0 from, 1 to) and the row of that meter in the plan."""
    pf_k = KINDS.index("pf")
    ny_f, ny_t = case.is_ny[case.f], case.is_ny[case.t]
    out = []
    for b in np.flatnonzero(ny_f != ny_t):
        end = 0 if ny_f[b] else 1
        rows = np.flatnonzero((plan.kind == pf_k) & (plan.branch == b) & (plan.end == end))
        if len(rows):
            out.append((int(b), end, int(rows[0])))
    a = np.array(out, int).reshape(-1, 3)
    return a[:, 0], a[:, 1], a[:, 2]


def neighbour_reads(
    plan: MeterPlan, rows: np.ndarray, far: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """The neighbours' readings `[T, K]` of the flows `far` at their ends of the ties, through meters of the
    same accuracy class as the New York end (`rows` in the plan): a constant bias drawn from a fixed
    stream, so the same every day, and a fresh jitter per scan from `rng`, split as in M2."""
    sr, sa = plan.sigma_rel[rows], plan.sigma_abs[rows]
    rel = sr > 0
    bias = np.random.default_rng(NEIGHBOUR_SEED).normal(
        0.0, np.sqrt(1.0 - JITTER_FRAC**2) * np.where(rel, sr, sa)
    )
    jitter = JITTER_FRAC * sr * np.abs(far) + np.where(rel, sa, JITTER_FRAC * sa)
    biased = np.where(rel, far * (1.0 + bias), far + bias)
    return biased + rng.normal(0.0, 1.0, far.shape) * jitter


def tie_gap(day: OperatorDay, z: np.ndarray) -> np.ndarray:
    """`[T]` the net interchange the operator reads (the P-flow meters at the New York end of every tie
    to a neighbour, as flow leaving New York) less what the neighbours meter at their own end, turned to
    the same direction (N62). Ties are metered at both ends and the readings exchanged, so a gap beyond
    the two meters' noise shows that the reported interchange is wrong. The neighbours' jitter is drawn
    from a stream keyed by the day, so the attacked and the benign call see the same neighbour readings."""
    c, plan = day.case, day.plan
    br, end, rows = tie_meters(c, plan)
    pf, _, pt, _ = Network(c).flows(day.vm, day.va)
    far = np.where(end == 0, -pt[:, br], -pf[:, br])  # the neighbour's end, in the New York direction
    day_key = int(np.asarray(day.time)[0]) if len(day.time) else 0
    neighbour = neighbour_reads(plan, rows, far, np.random.default_rng([NEIGHBOUR_SEED, day_key]))
    return np.nan_to_num(z[:, rows]).sum(axis=1) - neighbour.sum(axis=1)


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
    cover as the operator sees it (MW), `uncovered [T]` the true imbalance regulation had no room for
    (MW, positive short), `iface_change [T, I]` the true interface flow change from the
    basepoint's (AC, MW), `n1_change [T]` the change in the worst single-outage interface flow (MW), `margin [T]` the tightest
    true margin of any interface to its limit (MW, negative over it) and `margin_at [T]` that interface (-1 unknown),
    `shed [T, N]` the load the operator shed (MW), `relief [T]` what relieving the interfaces cost in
    production ($, `Step.relief_usd`), `shortage [T]` the flow left over the limit at the shortage curve ($), `fuel [T]` the production cost change and `settled [T]` the energy
    change at the zonal LBMP ($ per scan). `pf_failed [T]` marks the scans where the AC power flow of the
    operator's dispatch on the true grid failed: their interface changes and margin are NaN, unknown rather than
    safe, and `summarize` counts them."""

    dev: np.ndarray
    reg: np.ndarray
    unrelieved: np.ndarray
    unmatched: np.ndarray
    uncovered: np.ndarray
    iface_change: np.ndarray
    n1_change: np.ndarray
    margin: np.ndarray
    margin_at: np.ndarray
    shed: np.ndarray
    relief: np.ndarray
    shortage: np.ndarray
    fuel: np.ndarray
    settled: np.ndarray
    pf_failed: np.ndarray


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

    def _flows(self, t: int, units: np.ndarray, shed: np.ndarray | None = None) -> np.ndarray | None:
        d = self.day
        q = self.qd[t]
        if shed is not None:  # a shed load takes its reactive demand with it, at its own power factor
            pf = np.divide(self.qd[t], self.pd[t], out=np.zeros_like(self.qd[t]), where=self.pd[t] > 1e-6)
            q = self.qd[t] - shed * pf
        r = self.solver.power_flow(
            self.pd[t] if shed is None else self.pd[t] - shed,
            q,
            units,
            np.ones(len(units), bool),  # every unit in service for Q, as the pool solves it
            d.vm[t, d.case.unit_bus],
            np.zeros(len(d.case.dc_f)),
            start=(d.vm[t], d.va[t]),
        )
        return None if r is None else self.net.flows(r["vm"], r["va"])[0]

    def _measure(self, pf: np.ndarray) -> tuple[np.ndarray, float]:
        """Every interface's flow, and the largest flow any of them carries after the worst single outage."""
        iface = np.array([it.model_flow(pf) for it in self.day.interfaces])
        worst = max(
            float((iface[i] + it.signs @ (self.lodf[it.branches] * pf[self.outages])).max())
            for i, it in enumerate(self.day.interfaces)
        )
        return iface, worst

    def change(self, t: int, units: np.ndarray, shed: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """The true interface flows' and the worst N-1 flow's change from the basepoint at scan `t`, with
        `shed [N]` (MW) taken off the true loads."""
        if t not in self._base:
            pf = self._flows(t, self.day.unit_p[t])
            self._base[t] = (
                self._measure(pf) if pf is not None else (np.full(len(self.day.interfaces), np.nan), np.nan)
            )
        pf = self._flows(t, units, shed)
        if pf is None:
            return np.full(len(self.day.interfaces), np.nan), np.nan
        iface, worst = self._measure(pf)
        return iface - self._base[t][0], worst - self._base[t][1]


def _regulation(
    day: OperatorDay, t: int, imbalance: float, dispatched: np.ndarray, before: np.ndarray | None
) -> tuple[np.ndarray, float]:
    """The imbalance (MW, positive when the units produce too little) spread over the online free units
    in proportion to their five-minute ramp, as automatic generation control picks it up, each capped by
    its room after the dispatch: its limits, and its ramp from its true output `before` (the scan
    before, None at the first). A unit at its cap passes its share on to the others. Returns the
    regulation [G] and what no unit had room for (MW, positive short)."""
    c = day.case
    on = (
        ~np.isin(c.fuel.astype(str), ("Nuclear", "Hydro", "Import"))
        & (day.unit_p[t] > 0.5)
        & np.isfinite(c.ramp)
    )
    up = imbalance > 0
    if up:
        room = np.maximum(c.pmax, day.unit_p[t]) - dispatched
        if before is not None:
            room = np.minimum(room, before + c.ramp * 5.0 - dispatched)
    else:
        room = dispatched - np.minimum(c.pmin, day.unit_p[t])
        if before is not None:
            room = np.minimum(room, dispatched - (before - c.ramp * 5.0))
    room = np.where(on, np.clip(np.nan_to_num(room), 0.0, None), 0.0)
    # regulation stops at what the operator holds (N63); beyond it the imbalance is an ACE excursion
    reg, need = np.zeros(len(room)), min(abs(imbalance), REG_CAP)
    excess = abs(imbalance) - need
    while need > 1e-9:
        open_ = room - reg > 1e-9
        if not open_.any():
            break
        share = np.where(open_, c.ramp, 0.0)
        add = np.minimum(need * share / share.sum(), room - reg)
        reg += add
        need -= float(add.sum())
    sign = 1.0 if up else -1.0
    return sign * reg, sign * (need + excess)


MODES = ("trusting", "accounting", "oracle")
# N78: how an operator answers an alarm: it reads the scans without the attack, or holds its dispatch with its
# AGC on the tie meters it reads or on the common-source tie values
AFTER = ("benign", "freeze", "freeze_cs")


# N72: where the inadvertent interchange enters New York, as weights per external area of the source model (1
# ISO-NE, 4 Ontario, 5 MISO, 6 PJM), split within an area by the units' capacity. "capacity" weights every
# synchronous external unit by its capacity alone (N69); "peak" weights each area by its approximate 2019 peak
# load in GW, a proxy for its frequency bias; the others put all of it in one region.
ENTRY: dict[str, dict[int, float] | None] = {
    "capacity": None,
    "peak": {1: 24.4, 4: 21.8, 5: 121.6, 6: 151.3},
    "pjm": {6: 1.0},
    "isone": {1: 1.0},
    "west": {4: 1.0, 5: 1.0, 6: 1.0},
}


def _entry_share(c: Case, entry: str) -> np.ndarray:
    """[U] the share of the inadvertent interchange each unit supplies (N69, N72): synchronous units outside
    New York (the HVDC import from Quebec cannot carry it), by capacity within the weights of `ENTRY[entry]`."""
    if entry not in ENTRY:
        raise ConfigError(f"entry must be one of {tuple(ENTRY)}, not {entry!r}")
    cap = np.nan_to_num(c.pmax)
    ext = ~c.is_ny[c.unit_bus] & (c.bus_id[c.unit_bus] != HQ_BUS) & (cap > 0)
    share = np.where(ext, cap, 0.0)
    weights = ENTRY[entry]
    if weights is not None:
        area = c.area[c.unit_bus]
        out = np.zeros_like(share)
        for a, w in weights.items():
            inside = share * (area == a)
            if inside.sum() > 0:
                out += w * inside / inside.sum()
        share = out
    return share / share.sum() if share.sum() > 0 else share


def _run(
    day: OperatorDay,
    seen: np.ndarray,
    load_change: np.ndarray,
    delay: int,
    grid: _TrueGrid,
    gap: np.ndarray,
    mode: str,
    alarm: int | None,
    seen_benign: np.ndarray,
    entry: str = "capacity",
    closed_loop: bool = False,
    after: str | None = None,
) -> Twin:
    """One run of the operator over the day (N66). `gap [T]` is the interchange it reads at the New York
    ends of the ties less the true one (zero in the benign twin). A trusting operator dispatches on the
    metered injections and its AGC reads the area control error through those same New York-end tie
    meters, so the false interchange hides the imbalance it causes. An accounting operator takes its load
    as generation plus the interchange agreed with the neighbours and runs its AGC on the agreed values,
    so neither carries the false interchange, while the interface it reads is still the attacked one. An
    oracle operator trusts until the interchange alarm and from then on dispatches on the scans as they
    would be without the attack, the most any defense that finds the attack at that scan can save."""
    c, T, G, I = day.case, len(day.time), len(day.case.unit_bus), len(day.interfaces)
    fuel_kind, zone = c.fuel.astype(str), c.zone.astype(str)[c.unit_bus]
    quick = np.isin(fuel_kind, QUICK_FUELS) & np.isin(zone, QUICK_ZONES) & (c.pmax <= QUICK_MAX_MW)
    load = np.clip(grid.pd, 0.0, None) * day.case.is_ny
    sced = DcSced(c, day.unit_p, day.cost, day.interfaces, load, quick_start=quick)
    sf_units = sced.sf[:, c.unit_bus]
    # N69: the inadvertent interchange enters over the AC ties, from the synchronous units outside New York in
    # proportion to their capacity (the HVDC import from Quebec cannot carry it), not at the reference bus
    ext_share = _entry_share(c, entry)
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
        "uncovered": np.zeros(T),
        "shed": np.zeros((T, len(c.bus_id))),
        "relief": np.zeros(T),
        "shortage": np.zeros(T),
        "iface_change": np.zeros((T, I)),
        "n1_change": np.zeros(T),
        "margin": np.full(T, np.nan),
        "margin_at": np.full(T, -1),
        "fuel": np.zeros(T),
        "settled": np.zeros(T),
        "pf_failed": np.zeros(T, bool),
    }
    prev, before = None, None
    over = np.zeros(I, int)  # scans in a row each interface has been over its limit
    # N76: with `closed_loop` the operator reads each interface as it truly flows after its last dispatch, so
    # the gap between that true change and what its shift factors predicted enters its next reading
    mismatch = np.zeros(I)
    last_step = None
    for t in range(T):
        if not (day.solved[t] and (t == 0 or day.solved[t - 1])):
            out["iface_change"][t], out["n1_change"][t] = np.nan, np.nan
            prev, before = (None, None) if not day.solved[t] else (prev, before)
            continue
        s = max(t - delay, 0)  # the operator acts on the scan `delay` scans back
        # N78: past the alarm the oracle reads the scans without the attack; with `after` any operator responds
        # to the alarm: "benign" as the oracle, "freeze" holding its dispatch while its AGC keeps reading the tie
        # meters it reads, "freeze_cs" holding its dispatch with the AGC on the common-source tie values
        alarmed = alarm is not None and t >= alarm and (mode == "oracle" or after is not None)
        style = "benign" if mode == "oracle" else after
        fallback = alarmed and style == "benign"  # decided at scan t, whatever s is
        frozen = alarmed and style in ("freeze", "freeze_cs") and last_step is not None
        view = (seen_benign[s] if fallback else seen[s]) + (mismatch if closed_loop else 0.0)
        dload = 0.0 if fallback else float(np.nan_to_num(load_change[s]))
        if mode == "accounting" and not fallback:  # load from generation plus the agreed interchange
            dload += float(np.nan_to_num(gap[s]))  # the read export the attack added hides as much load
        step = (
            last_step
            if frozen and last_step is not None
            else sced.step(t, view, limit[t], dload, prev, over >= SHED_AFTER)
        )
        last_step = step
        short = np.nan_to_num(step.unrelieved)
        # an interface's clock runs while it is over its limit, or would be without the load it sheds
        bare = np.nan_to_num(view + sf_units @ step.dev - limit[t]) > 1e-6
        over = np.where((short > 1e-6) | ((over >= SHED_AFTER) & bare), over + 1, 0)
        out["shortage"][t] = sum(shortage_usd(u) for u in short)  # the curve is per constraint
        dispatched = day.unit_p[t] + step.dev
        # the true load lost what was shed and nothing else, so the units' shortfall is the dispatch's
        # change plus the shed (the attack's pretended load never existed)
        short_mw = -float(step.dev.sum() + step.shed.sum())
        # the AGC sees the shortfall through the tie meters it reads: the trusting operator's New
        # York-end meters carry the false interchange, the agreed ones do not
        reads_ties = mode == "trusting" or (mode == "oracle" and not fallback)
        if alarmed and style in (
            "benign",
            "freeze_cs",
        ):  # the alarm moved the AGC off the New York-end meters
            reads_ties = False
        seen_err = float(np.nan_to_num(gap[t])) if reads_ties else 0.0
        reg, _ = _regulation(day, t, short_mw - seen_err, dispatched, before)
        prev, before = step.dev, dispatched + reg
        out["uncovered"][t] = short_mw - float(
            reg.sum()
        )  # what the neighbours supply: inadvertent interchange
        out["dev"][t], out["reg"][t] = step.dev, reg
        out["unrelieved"][t], out["unmatched"][t] = step.unrelieved, step.unmatched
        out["shed"][t], out["relief"][t] = step.shed, step.relief_usd
        moved = step.dev + reg
        lent = out["uncovered"][t] * ext_share  # what the neighbours supply, at their units
        if np.abs(moved).max() > 1e-3 or step.shed.any() or np.abs(lent).max() > 1e-3:
            out["iface_change"][t], out["n1_change"][t] = grid.change(
                t, day.unit_p[t] + moved + lent, step.shed
            )
            out["pf_failed"][t] = bool(np.isnan(out["iface_change"][t]).any())
        if closed_loop:  # what the next scan reads beyond the operator's own model of its dispatch
            mismatch = np.nan_to_num(out["iface_change"][t] - sf_units @ step.dev)
        # the tightest true margin of any interface to its limit (MW, negative over it)
        # unknown (NaN) where the AC power flow of either run failed, never the benign margin
        change = out["iface_change"][t]
        if not np.isnan(change).any():
            room = limit[t] - (true_iface[t] + change)
            out["margin"][t] = float(np.nanmin(room))
            out["margin_at"][t] = int(np.nanargmin(room))
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
    bus_client: np.ndarray
    client_names: list[str]
    onset: int
    length: int
    delay: int
    over: float
    tie_attacked: np.ndarray  # [T] the interchange read less the neighbours' meters (MW, `tie_gap`)
    tie_benign: np.ndarray
    alarm: float  # scans from the onset to the interchange alarm, NaN if it never fires
    mode: str


def _alarm_scan(
    day: OperatorDay, tie_a: np.ndarray, tie_b: np.ndarray, calib: tuple[float, float] | None = None
) -> int | None:
    """The first scan at or after the onset whose read interchange is off the neighbours' meters by
    more than four standard deviations of the gap's benign noise (N62). `calib` is that noise's mean and
    standard deviation (MW), calibrated on other days (`tools/nyfed/tie_check.py`); by default the benign
    day's own, which is in sample."""
    if day.onset < 0:
        return None
    b = tie_b[np.isfinite(tie_b)]
    mean, sd = (float(np.mean(b)), float(np.std(b))) if calib is None else calib
    off = np.abs(tie_a - mean) > 4.0 * sd
    hits = np.flatnonzero(off[day.onset :])
    return day.onset + int(hits[0]) if len(hits) else None


def respond(
    day: OperatorDay,
    delay: int = 0,
    over: float = 0.0,
    mode: str = "trusting",
    calib: tuple[float, float] | None = None,
    entry: str = "capacity",
    closed_loop: bool = False,
    alarm_at: int | None = None,
    after: str | None = None,
) -> Response:
    """Run the operator on `day` in one of `MODES` (N66), acting `delay` scans late (O5). `over` (MW) is a
    what-if: the attacked interface reading pushed that far past what the attack reached, in proportion
    to the push over the episode. `entry` is where the inadvertent interchange enters (`ENTRY`, N72). With `closed_loop` the operator reads each interface as it truly flows after its own last dispatch (N76). `calib` (MW) is the mean and standard deviation of the tie gap's benign noise the alarm is calibrated on, from other days."""
    if mode not in MODES:
        raise ConfigError(f"mode must be one of {MODES}, not {mode!r}")
    if after is not None and after not in AFTER:
        raise ConfigError(f"after must be one of {AFTER} or None, not {after!r}")
    if entry not in ENTRY:
        raise ConfigError(f"entry must be one of {tuple(ENTRY)}, not {entry!r}")
    tie_a, tie_b = tie_gap(day, day.z), tie_gap(day, day.z_benign)
    # the oracle falls back at the interchange alarm, or at `alarm_at` (a scan of the day) when another alarm is studied
    alarm = _alarm_scan(day, tie_a, tie_b, calib) if alarm_at is None else alarm_at
    seen_b = seen_interfaces(day.plan, day.interfaces, day.z_benign)
    seen_a = seen_interfaces(day.plan, day.interfaces, day.z)
    if over and day.onset >= 0:
        push = seen_a - seen_b
        w = slice(day.onset, day.onset + day.length)
        seen_a[w] += over * push[w] / max(float(np.abs(push[w]).max()), 1e-9)
    p = KINDS.index("p")
    rows = np.flatnonzero(day.plan.kind == p)
    load_change = -(day.z[:, rows] - day.z_benign[:, rows]).sum(axis=1)  # the seen load's change, MW
    gap = tie_a - tie_b  # the false interchange, the meters' own noise cancelling against the twin
    grid = _TrueGrid(day)
    zero = np.zeros(len(day.time))
    return Response(
        attacked=_run(
            day, seen_a, load_change, delay, grid, gap, mode, alarm, seen_b, entry, closed_loop, after
        ),
        benign=_run(day, seen_b, zero, delay, grid, zero, mode, None, seen_b, entry, closed_loop),
        unit_client=day.own.bus_client[day.case.unit_bus],
        bus_client=day.own.bus_client,
        client_names=[str(n) for n in day.own.names],
        onset=day.onset,
        length=day.length,
        delay=delay,
        over=over,
        tie_attacked=tie_a,
        tie_benign=tie_b,
        alarm=float("nan") if alarm is None else float(alarm - day.onset),
        mode=mode,
    )


def _over(r: Response) -> dict:
    """Where the attacked run takes an interface past its true limit (N72): the scans over, the longest run of
    them, the energy over (MWh) and the interface most often over (-1 none). A scan whose power flow failed has no
    margin and is not counted over; `pf_failed_scans` says how many such scans there are."""
    m = r.attacked.margin
    over = np.nan_to_num(m, nan=1.0) < 0
    run = best = 0
    for o in over:
        run = run + 1 if o else 0
        best = max(best, run)
    at = r.attacked.margin_at[over]
    return {
        "over_scans": int(over.sum()),
        "over_run_scans": int(best),
        "over_mwh": float(-m[over].sum() * FRAME_HOURS),
        "over_iface": int(np.bincount(at).argmax()) if len(at) else -1,
    }


def summarize(r: Response, voll: float | None = None) -> dict:
    """The attack's consequence over the day, attacked less benign scan by scan, so what the benign twin
    already shows (a lagged reading over its limit, load the units cannot follow) is not counted: the
    most dispatch moved and regulation called (MW), the regulation energy (MWh, upward, in total and by the
    utility whose territory holds each unit), the most interface flow left unrelieved, load unmatched and
    imbalance uncovered (MW), the largest true interface drop and rise and N-1 rise (MW), and the
    production cost and LBMP settlement changes ($). Then the escalation (N59): the most load shed (MW),
    the energy shed (MWh, in total and by utility), what relieving the interfaces cost in production
    ($, the fake congestion's cost), the shortage cost of the flow left over the limit ($), and with
    `voll` ($/MWh) the shed energy at that value."""
    a, b = r.attacked, r.benign

    def peak(x: np.ndarray) -> float:
        return float(np.nanmax(x)) if np.isfinite(x).any() else 0.0

    reg_up = (np.clip(a.reg, 0, None) - np.clip(b.reg, 0, None)) * FRAME_HOURS  # [T, G] MWh, attack only
    out = {
        "delay": r.delay,
        "over_mw": r.over,
        "dispatch_moved_mw": peak(np.abs(a.dev).sum(axis=1) - np.abs(b.dev).sum(axis=1)),
        "regulation_mw": peak(np.abs(a.reg).sum(axis=1) - np.abs(b.reg).sum(axis=1)),
        "regulation_mwh": float(reg_up.sum()),
        "unrelieved_mw": peak(np.nan_to_num(a.unrelieved) - np.nan_to_num(b.unrelieved)),
        "unmatched_mw": peak(np.abs(np.nan_to_num(a.unmatched)) - np.abs(np.nan_to_num(b.unmatched))),
        "uncovered_mw": peak(np.abs(a.uncovered) - np.abs(b.uncovered)),
        "iface_down_mw": float(np.nanmin(a.iface_change[:, 0] - b.iface_change[:, 0])),
        "iface_up_mw": float(np.nanmax(a.iface_change[:, 0] - b.iface_change[:, 0])),
        "other_iface_up_mw": peak(a.iface_change[:, 1:] - b.iface_change[:, 1:])
        if a.iface_change.shape[1] > 1
        else 0.0,
        "inadvertent_mw": peak(np.abs(a.uncovered) - np.abs(b.uncovered)),
        "inadvertent_mwh": float(
            np.clip(np.abs(a.uncovered) - np.abs(b.uncovered), 0, None).sum() * FRAME_HOURS
        ),
        "tie_gap_mw": peak(np.abs(r.tie_attacked - r.tie_benign)),
        "alarm_scans": r.alarm,
        "mode": r.mode,
        "n1_up_mw": float(np.nanmax(a.n1_change - b.n1_change)),
        "margin_min_mw": float(np.nanmin(a.margin)) if np.isfinite(a.margin).any() else float("nan"),
        "margin_lost_mw": peak(b.margin - a.margin),
        **_over(r),
        "pf_failed_scans": int(a.pf_failed.sum()),
        "pf_failed_benign_scans": int(b.pf_failed.sum()),
        "fuel_usd": float(a.fuel.sum() - b.fuel.sum()),
        "settled_usd": float(a.settled.sum() - b.settled.sum()),
        "relief_usd": float(a.relief.sum() - b.relief.sum()),
        "shortage_usd": float(a.shortage.sum() - b.shortage.sum()),
        "shed_mw": peak(a.shed.sum(axis=1) - b.shed.sum(axis=1)),
        "shed_mwh": float((a.shed - b.shed).sum() * FRAME_HOURS),
    }
    if voll is not None:
        out["shed_usd"] = out["shed_mwh"] * voll
    for c, name in enumerate(r.client_names):
        out[f"regulation_mwh_{name}"] = float(reg_up[:, r.unit_client == c].sum())
        out[f"shed_mwh_{name}"] = float((a.shed - b.shed)[:, r.bus_client == c].sum() * FRAME_HOURS)
    return out
