"""The operator scan by scan (D1 to D6). It dispatches on what it reads (`DcSced`), the attacked scans and,
separately, their benign twin, each in closed loop on its own dispatch. The dispatch is applied to the true grid,
whose load the attack never changed: the imbalance it leaves is picked up by regulation, the online free units in
proportion to their five-minute ramp (AGC), up to `REG_CAP`, and the rest by the neighbours as inadvertent
interchange. The attack's consequence is the attacked run less the benign one, so whatever the model does alike in
both (DC against AC, the cost model) cancels. `respond` runs both and returns the `Response`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from fdia_graph.nyfed.grid.network import Network
from fdia_graph.nyfed.operator.acsced import AcSced
from fdia_graph.nyfed.operator.day import OperatorDay, pool_frames
from fdia_graph.nyfed.operator.policy import OperatorPolicy, entry_share
from fdia_graph.nyfed.operator.sced import (
    FRAME_HOURS,
    SHED_AFTER,
    DcSced,
    shortage_usd,
)
from fdia_graph.nyfed.operator.ties import interchange_alarm, seen_interfaces, tie_gap
from fdia_graph.nyfed.operator.truegrid import TrueGrid
from fdia_graph.nyfed.scenario.inputs import ZONES
from fdia_graph.nyfed.sensing.meters import KINDS

AC_DISPATCH = {"ac": "movement", "ac_cost": "cost"}  # the AC dispatches and their `AcSced` objective
REG_CAP = 275.0  # MW of regulation the operator holds, the summer requirement at its highest hour (N63)
# N64: small gas, oil and kerosene units in New York City and Long Island, the quick-start fleet
QUICK_FUELS, QUICK_ZONES, QUICK_MAX_MW = ("Natural Gas", "Kerosene", "Fuel Oil 2"), ("J", "K"), 200.0


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
    safe, and `summarize` counts them. `congestion [T, I]` is each interface's shadow price under the least-cost
    dispatch ($/MWh per MW, `Step.congestion`), zero under the least-movement one and NaN where not enforced (None in
    a run built without it), `energy [T]` the balance's shadow price under the least-cost dispatch ($/MWh at the
    reference bus, NaN otherwise, `Step.energy`), and `price [T, N]` each bus's price under a least-cost dispatch
    ($/MWh, `DcSced.prices`; NaN under the least-movement one, None in a run built without it)."""

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
    congestion: np.ndarray | None = None
    energy: np.ndarray | None = None
    price: np.ndarray | None = None


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


def _run(
    day: OperatorDay,
    seen: np.ndarray,
    load_change: np.ndarray,
    delay: int,
    grid: TrueGrid,
    gap: np.ndarray,
    mode: str,
    alarm: int | None,
    seen_benign: np.ndarray,
    entry: str = "capacity",
    closed_loop: bool = False,
    after: str | None = None,
    dispatch: str = "movement",
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
    sced = DcSced(
        c,
        day.unit_p,
        day.cost,
        day.interfaces,
        load,
        quick_start=quick,
        objective=dispatch,
        state=(day.vm, day.va),
    )
    sf_units = sced.sf[:, c.unit_bus]
    # N69: the inadvertent interchange enters over the AC ties, from the synchronous units outside New York in
    # proportion to their capacity (the HVDC import from Quebec cannot carry it), not at the reference bus
    ext_share = entry_share(c, entry)
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
        "congestion": np.full((T, I), np.nan),
        "energy": np.full(T, np.nan),
        "price": np.full((T, len(c.bus_id)), np.nan),
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
        # change plus the shed (the attack's pretended load never existed), as delivered at the reference bus
        short_mw = -sced.delivered(t, step.dev, step.shed)
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
        if step.congestion is not None:
            out["congestion"][t] = step.congestion
        out["energy"][t] = step.energy
        if step.congestion is not None and np.isfinite(step.energy):
            out["price"][t] = sced.prices(t, step.energy, step.congestion)
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


def reported_changes(day: OperatorDay, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`[T, N]` each bus's active and reactive injection change the operator reads in `z` against the benign scans
    (MW, MVAr): its New York injection meters' difference, so the meters' own noise cancels as in `load_change`."""
    plan, n = day.plan, len(day.case.bus_id)
    out = []
    for kind in ("p", "q"):
        rows = np.flatnonzero(plan.kind == KINDS.index(kind))
        d = np.zeros((len(day.time), n))
        np.add.at(d.T, plan.bus[rows], np.nan_to_num(z[:, rows] - day.z_benign[:, rows]).T)
        out.append(d)
    return out[0], out[1]


def _run_ac(
    day: OperatorDay,
    d_p: np.ndarray,
    d_q: np.ndarray,
    push: np.ndarray,
    flow_read: np.ndarray,
    delay: int,
    grid: TrueGrid,
    gap: np.ndarray,
    mode: str,
    alarm: int | None,
    entry: str = "capacity",
    closed_loop: bool = False,
    after: str | None = None,
    objective: str = "movement",
    force: set[int] | None = None,
) -> tuple[Twin, set[int]]:
    """One run of the AC operator over the day (`AcSced`), as `_run` runs the DC one. Each scan it reads the true
    loads less the injection changes `d_p`, `d_q [T, N]` its meters carry (zero in the benign twin; the accounting
    operator adds the false interchange `gap` back as load, spread over New York's loads by their size), and solves its
    AC OPF on them. `push [T, I]` is a what-if push of the read interface flows (`respond`'s `over`). With
    `closed_loop` each interface's reading carries what the operator read after its last dispatch beyond its own AC
    model: the true flow then, plus the attack's change of the reading `flow_read [T, I]`, less the model's flow. The
    regulation, the neighbours' supply and the true grid follow `_run`. `force` names scans solved even where quiet (the
    benign twin of a least-cost run solves where the attacked one did, so both carry prices there). Returns the run and
    the scans its dispatch solved."""
    c, T, G, I = day.case, len(day.time), len(day.case.unit_bus), len(day.interfaces)
    fuel_kind, zone = c.fuel.astype(str), c.zone.astype(str)[c.unit_bus]
    quick = np.isin(fuel_kind, QUICK_FUELS) & np.isin(zone, QUICK_ZONES) & (c.pmax <= QUICK_MAX_MW)
    load = np.clip(grid.pd, 0.0, None) * c.is_ny
    frames = pool_frames(day)
    sced = AcSced(
        c,
        day.unit_p,
        day.cost,
        day.interfaces,
        (day.vm, day.va),
        (grid.pd, frames.qd),
        frames,
        load,
        quick_start=quick,
        objective=objective,
    )
    true_q = frames.qd
    ext_share = entry_share(c, entry)
    true_iface = sced.flow0
    limit = true_iface + np.array([it.limit - it.flow for it in day.interfaces]).T  # D3
    unit_zone = np.array([ZONES.index(z) if z else -1 for z in c.zone.astype(str)[c.unit_bus]])
    price = np.where(unit_zone >= 0, day.lbmp[:, np.maximum(unit_zone, 0)], np.nan)
    ny_load = np.clip(grid.pd, 0.0, None) * c.is_ny
    spread = ny_load / np.maximum(ny_load.sum(axis=1, keepdims=True), 1e-9)  # [T, N] conformal shares
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
        "congestion": np.full((T, I), np.nan),
        "energy": np.full(T, np.nan),
        "price": np.full((T, len(c.bus_id)), np.nan),
    }
    prev, before = None, None
    over = np.zeros(I, int)
    corr = np.zeros(I)  # the closed loop's reading correction
    last_step = None
    for t in range(T):
        if not (day.solved[t] and (t == 0 or day.solved[t - 1])):
            out["iface_change"][t], out["n1_change"][t] = np.nan, np.nan
            prev, before = (None, None) if not day.solved[t] else (prev, before)
            continue
        s = max(t - delay, 0)
        alarmed = alarm is not None and t >= alarm and (mode == "oracle" or after is not None)
        style = "benign" if mode == "oracle" else after
        fallback = alarmed and style == "benign"
        frozen = alarmed and style in ("freeze", "freeze_cs") and last_step is not None
        if fallback:
            load_p, load_q, read = grid.pd[s].copy(), true_q[s].copy(), np.zeros(I)
        else:
            load_p, load_q = grid.pd[s] - d_p[s], true_q[s] - d_q[s]
            if mode == "accounting":  # load from generation plus the agreed interchange
                load_p = load_p + float(np.nan_to_num(gap[s])) * spread[s]
            read = push[s]
        view = read + (corr if closed_loop else 0.0)
        step = (
            last_step
            if frozen and last_step is not None
            else sced.step(
                t,
                load_p,
                load_q,
                view,
                limit[t],
                prev,
                over >= SHED_AFTER,
                force=force is not None and t in force,
            )
        )
        last_step = step
        short = np.nan_to_num(step.unrelieved)
        over = np.where((short > 1e-6) | ((over >= SHED_AFTER) & (step.shed.sum() > 1e-6)), over + 1, 0)
        out["shortage"][t] = sum(shortage_usd(u) for u in short)
        dispatched = day.unit_p[t] + step.dev
        short_mw = -sced.delivered(t, step.dev, step.shed)
        reads_ties = mode == "trusting" or (mode == "oracle" and not fallback)
        if alarmed and style in ("benign", "freeze_cs"):
            reads_ties = False
        seen_err = float(np.nan_to_num(gap[t])) if reads_ties else 0.0
        reg, _ = _regulation(day, t, short_mw - seen_err, dispatched, before)
        prev, before = step.dev, dispatched + reg
        out["uncovered"][t] = short_mw - float(reg.sum())
        out["dev"][t], out["reg"][t] = step.dev, reg
        out["unrelieved"][t], out["unmatched"][t] = step.unrelieved, step.unmatched
        out["shed"][t], out["relief"][t] = step.shed, step.relief_usd
        if step.congestion is not None:
            out["congestion"][t] = step.congestion
        out["energy"][t] = step.energy
        if step.price is not None:
            out["price"][t] = step.price
        moved = step.dev + reg
        lent = out["uncovered"][t] * ext_share
        if np.abs(moved).max() > 1e-3 or step.shed.any() or np.abs(lent).max() > 1e-3:
            out["iface_change"][t], out["n1_change"][t] = grid.change(
                t, day.unit_p[t] + moved + lent, step.shed
            )
            out["pf_failed"][t] = bool(np.isnan(out["iface_change"][t]).any())
        if closed_loop and step.flow is not None:  # what it reads next, beyond its own AC model of this scan
            truth = (
                true_iface[t] + np.nan_to_num(out["iface_change"][t]) + (0.0 if fallback else flow_read[t])
            )
            corr = np.nan_to_num(truth - step.flow)
        change = out["iface_change"][t]
        if not np.isnan(change).any():
            room = limit[t] - (true_iface[t] + change)
            out["margin"][t] = float(np.nanmin(room))
            out["margin_at"][t] = int(np.nanargmin(room))
        out["fuel"][t] = float(day.cost[t] @ moved) * FRAME_HOURS
        out["settled"][t] = float(np.nansum(price[t] * moved)) * FRAME_HOURS
    return Twin(**out), sced.solved_at


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
    dispatch: str = "movement",
    *,
    policy: OperatorPolicy | None = None,
) -> Response:
    """Run the operator on `day`, attacked and on its benign twin, under the choices of an `OperatorPolicy`: pass
    one as `policy`, or its fields as keywords (`mode`, `delay`, `over`, `entry`, `closed_loop`, `after`, `dispatch`), checked the
    same way. `calib` (MW) is the mean and standard deviation of the tie gap's benign noise the interchange alarm is
    calibrated on, from other days (by default the benign day's own, in sample). `alarm_at` (a scan of the day) puts
    the alarm there instead, for studying another detector's alarm; `after` says how the operator answers it."""
    p = policy or OperatorPolicy(  # the choices, checked by the record before anything runs
        mode=mode,
        delay=delay,
        over=over,
        entry=entry,
        closed_loop=closed_loop,
        after=after,
        dispatch=dispatch,
    )
    mode, delay, over, entry, closed_loop, after = p.mode, p.delay, p.over, p.entry, p.closed_loop, p.after
    dispatch = p.dispatch
    tie_a, tie_b = tie_gap(day, day.z), tie_gap(day, day.z_benign)
    # the oracle falls back at the interchange alarm, or at `alarm_at` (a scan of the day) when another alarm is studied
    alarm = interchange_alarm(day, tie_a, tie_b, calib) if alarm_at is None else alarm_at
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
    zero = np.zeros(len(day.time))
    if dispatch in AC_DISPATCH:
        grid = TrueGrid(day, ac=True)
        objective = AC_DISPATCH[dispatch]
        d_p, d_q = reported_changes(day, day.z)
        raw = seen_interfaces(day.plan, day.interfaces, day.z)
        flat = np.zeros_like(seen_a)
        attacked, solved = _run_ac(
            day,
            d_p,
            d_q,
            seen_a - raw,
            raw - seen_b,
            delay,
            grid,
            gap,
            mode,
            alarm,
            entry,
            closed_loop,
            after,
            objective,
        )
        benign, _ = _run_ac(
            day,
            np.zeros_like(d_p),
            np.zeros_like(d_q),
            flat,
            flat,
            delay,
            grid,
            zero,
            mode,
            None,
            entry,
            closed_loop,
            objective=objective,
            force=solved if objective == "cost" else None,  # its prices where the attacked run has them
        )
        return Response(
            attacked=attacked,
            benign=benign,
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
    grid = TrueGrid(day)
    return Response(
        attacked=_run(
            day,
            seen_a,
            load_change,
            delay,
            grid,
            gap,
            mode,
            alarm,
            seen_b,
            entry,
            closed_loop,
            after,
            dispatch,
        ),
        benign=_run(
            day, seen_b, zero, delay, grid, zero, mode, None, seen_b, entry, closed_loop, dispatch=dispatch
        ),
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


@runtime_checkable
class Operator(Protocol):
    """An operator: anything that runs on an attacked day and returns its `Response` (the attacked run and its
    benign twin), so `summarize` and the detection tools work on it unchanged. `NyisoOperator` is the built-in one."""

    def respond(
        self, day: OperatorDay, *, calib: tuple[float, float] | None = None, alarm_at: int | None = None
    ) -> Response: ...


@dataclass(frozen=True)
class NyisoOperator:
    """The New York operator of this package under one `OperatorPolicy`: `respond` with that policy."""

    policy: OperatorPolicy = OperatorPolicy()

    def respond(
        self, day: OperatorDay, *, calib: tuple[float, float] | None = None, alarm_at: int | None = None
    ) -> Response:
        return respond(day, calib=calib, alarm_at=alarm_at, policy=self.policy)
