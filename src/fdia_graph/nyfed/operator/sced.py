"""The operator's real-time dispatch: a DC security-constrained dispatch around the current basepoint,
as NYISO's real-time dispatch is a DC dispatch on shift factors (D2).

Each five-minute scan the operator compares what it sees with what its dispatch must hold: the total load
it reads (generation must follow it) and each interface's flow against its limit. It moves the free units
(not nuclear, hydro or imports, which follow their schedules as in the pool, N39) within their limits and
five-minute ramps from its own previous dispatch. A corrective operator moves as little as it must: the
dispatch solves for the least total movement first, then for the least cost at that movement, so a scan
that needs nothing moves nothing beyond what the ramps force, and the benign basepoint stays as the pool
dispatched it. What the units
cannot cover (an interface still over, load still unmatched) is left as a slack, priced far above any
unit so it is used only when nothing else can.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from fdia_graph.nyfed.grid.case import REF, Case
from fdia_graph.nyfed.grid.interfaces import Interface

HELD = ("Nuclear", "Hydro", "Import")  # units that follow their schedule (N39), never redispatched
SLACK_WEIGHT = 1e6  # per MW of unrelieved interface flow or unmatched load, above any movement, shed or cost
SHED_WEIGHT = 1e3  # per MW shed: only where no redispatch relieves (a MW moved weighs 1)
SHED_FACTOR = -0.1  # a bus sheds only where a MW shed takes at least 0.1 MW off the interface
SHED_AFTER = 3  # scans over the limit before load relief: 15 minutes (Emergency Operations Manual 4.5)
# the 2019 transmission shortage curve ($/MWh): 350 for the first 5 MW, 1,175 for the next 15 MW, the
# 4,000 shadow-price cap beyond (NYISO's constraint-specific shortage pricing, in force June 20, 2017)
SHORTAGE = ((5.0, 350.0), (15.0, 1175.0), (np.inf, 4000.0))
FRAME_HOURS = 5.0 / 60.0


def shortage_usd(mw: float) -> float:
    """The shortage cost of one scan with `mw` of interface flow over its limit, at the 2019 curve ($)."""
    cost, left = 0.0, max(float(mw), 0.0)
    for width, price in SHORTAGE:
        part = min(left, width)
        cost, left = cost + part * price, left - part
    return cost * FRAME_HOURS


def shift_factors(case: Case) -> np.ndarray:
    """`[L, N]` DC shift factors: the from-end active flow of each branch per MW injected at each bus and
    withdrawn at the reference bus (zero in the reference column)."""
    n, m = len(case.bus_id), len(case.f)
    ref = int(np.flatnonzero(case.bus_type == REF)[0])
    b = 1.0 / (case.x * np.where(case.tap == 0, 1.0, case.tap))
    inc = np.zeros((m, n))
    inc[np.arange(m), case.f] = 1.0
    inc[np.arange(m), case.t] = -1.0
    keep = np.arange(n) != ref
    bbus = inc.T @ (b[:, None] * inc)
    out = np.zeros((m, n))
    out[:, keep] = (b[:, None] * inc[:, keep]) @ np.linalg.inv(bbus[np.ix_(keep, keep)])
    return out


def outage_factors(case: Case, ptdf: np.ndarray, outages: np.ndarray) -> np.ndarray:
    """`[L, K]` line outage distribution factors: the change in each branch's flow per MW that branch
    `outages[k]` carried before it opened. A branch whose outage islands the grid gets a zero column."""
    move = ptdf[:, case.f[outages]] - ptdf[:, case.t[outages]]  # [L, K] flow per MW sent from f_k to t_k
    own = move[outages, np.arange(len(outages))]
    lodf = np.where(np.abs(1.0 - own) > 1e-6, move / np.where(np.abs(1.0 - own) > 1e-6, 1.0 - own, 1.0), 0.0)
    lodf[outages, np.arange(len(outages))] = -1.0
    return lodf


@dataclass(frozen=True)
class Step:
    """One scan's dispatch: `dev [G]` each unit's change from the pool basepoint (MW), `shed [N]` the load
    the operator sheds at each bus (MW), `unrelieved [I]` each interface's flow left over its limit as the
    operator sees it (MW), `unmatched` the seen load the units could not follow (MW, positive short), and
    `relief_usd` what relieving the interfaces costs this scan: the production cost of this dispatch less
    that of the same dispatch with no interface limit ($)."""

    dev: np.ndarray
    shed: np.ndarray
    unrelieved: np.ndarray
    unmatched: float
    relief_usd: float


class DcSced:
    """The dispatch over one day's frames: `unit_p [T, G]` the pool basepoint, `cost [T, G]` $/MWh, the
    `interfaces` it enforces, the case's unit limits and ramps, and `bus_load [T, N]` (MW) the load it may
    shed at each bus. `step` solves one scan.

    The operator escalates as NYISO's procedures do (N59): it redispatches first and sheds load only where
    redispatch cannot relieve an interface, at the buses where shedding relieves it (a shift factor of at
    most `SHED_FACTOR`), and any flow it still cannot relieve is left as the slack."""

    def __init__(
        self,
        case: Case,
        unit_p: np.ndarray,
        cost: np.ndarray,
        interfaces: list[Interface],
        bus_load: np.ndarray | None = None,
    ):
        self.case, self.unit_p, self.cost = case, unit_p, cost
        ptdf = shift_factors(case)
        self.sf = np.array([it.signs @ ptdf[it.branches] for it in interfaces])  # [I, N]
        self.free = ~np.isin(case.fuel.astype(str), HELD)
        self.ramp = np.where(np.isfinite(case.ramp), case.ramp * 5.0, np.inf)  # MW per scan
        self.bus_load = np.zeros((len(unit_p), len(case.bus_id))) if bus_load is None else bus_load

    def _bounds(self, t: int, prev: np.ndarray | None, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each unit's reachable change from its basepoint, signed: its limits, and the ramp from the
        operator's previous dispatch. After an attack the interval can exclude the basepoint (a unit moved
        far cannot return in one scan), so recovery is ramp limited. Where the ramp and the limits do not
        meet, the unit goes as far toward its limits as the ramp allows."""
        c, p0 = self.case, self.unit_p[t, g]
        lo = np.minimum(c.pmin[g], p0) - p0
        hi = np.maximum(c.pmax[g], p0) - p0
        if prev is not None:
            was = self.unit_p[t - 1, g] + prev[g]
            lo = np.maximum(lo, was - self.ramp[g] - p0)
            hi = np.minimum(hi, was + self.ramp[g] - p0)
        return np.minimum(lo, hi), hi

    def _solve(self, t, seen, limit, load_change, prev, enforce: bool, shed_ok: bool = True):
        """The two-stage dispatch: least weighted movement (redispatch at 1 per MW, shedding at
        `SHED_WEIGHT`, slack at `SLACK_WEIGHT`), then least cost at no more of each. Returns the units,
        the dispatch [G], the shed [N], the interface slack [I] and the unmatched load."""
        g = np.flatnonzero(self.free & (self.unit_p[t] > 0.5))
        n = len(g)
        lo, hi = self._bounds(t, prev, g)
        on = np.isfinite(seen) & np.isfinite(limit) if enforce else np.zeros(len(seen), bool)
        k = int(on.sum())
        sb = (
            np.flatnonzero((self.sf[on] <= SHED_FACTOR).any(axis=0) & (self.bus_load[t] > 0))
            if k and shed_ok
            else np.zeros(0, int)
        )
        s = len(sb)
        # variables: up [n], down [n], shed [s], interface slack [k], short, over; dev = up - down
        a_if = self.sf[on][:, self.case.unit_bus[g]]
        a_ub = np.hstack([a_if, -a_if, self.sf[on][:, sb], -np.eye(k), np.zeros((k, 2))])
        b_ub = (limit - seen)[on]
        a_eq = np.r_[np.ones(n), -np.ones(n), np.ones(s), np.zeros(k), 1.0, -1.0][None, :]
        # up and down carry the signed interval: a mandatory move is a lower bound on one of them
        bounds = (
            [(max(a, 0.0), max(b, 0.0)) for a, b in zip(lo, hi)]
            + [(max(-b, 0.0), max(-a, 0.0)) for a, b in zip(lo, hi)]
            + [(0.0, float(x)) for x in self.bus_load[t, sb]]
            + [(0.0, None)] * (k + 2)
        )
        weight = np.r_[np.ones(2 * n), np.full(s, SHED_WEIGHT), np.full(k + 2, SLACK_WEIGHT)]
        kw = dict(A_eq=a_eq, b_eq=[load_change], method="highs")
        first = linprog(weight, A_ub=a_ub if k else None, b_ub=b_ub if k else None, bounds=bounds, **kw)
        if first.status != 0:
            raise RuntimeError(f"scan {t}: {first.message}")
        x1 = first.x
        # the least cost among the dispatches that move, shed and leave no more than the first
        bounds2 = bounds[: 2 * n] + [(0.0, v * (1 + 1e-9) + 1e-9) for v in x1[2 * n :]]
        cap = np.r_[np.ones(2 * n), np.zeros(s + k + 2)][None, :]
        cost = np.r_[self.cost[t, g], -self.cost[t, g], np.zeros(s + k + 2)]
        second = linprog(
            cost + np.r_[np.zeros(2 * n), weight[2 * n :]],
            A_ub=np.vstack([a_ub, cap]) if k else cap,
            b_ub=np.r_[b_ub, x1[: 2 * n].sum() * (1 + 1e-9) + 1e-6] if k else [x1[: 2 * n].sum() + 1e-6],
            bounds=bounds2,
            **kw,
        )
        x = second.x if second.status == 0 else x1
        dev = np.zeros(len(self.free))
        dev[g] = x[:n] - x[n : 2 * n]
        shed = np.zeros(len(self.case.bus_id))
        shed[sb] = x[2 * n : 2 * n + s]
        slack = np.full(len(seen), np.nan)
        slack[on] = x[2 * n + s : 2 * n + s + k]
        return dev, shed, slack, float(x[-2] - x[-1])

    def step(
        self,
        t: int,
        seen: np.ndarray,
        limit: np.ndarray,
        load_change: float,
        prev: np.ndarray | None,
        shed_ok: bool = True,
    ) -> Step:
        """Scan `t`: interface flows `seen [I]` against `limit [I]` (a NaN in either leaves that interface
        unenforced), the seen load's change from the basepoint's `load_change` (MW), and `prev [G]` the
        operator's deviation at the scan before (None at the first scan). `shed_ok` allows load relief, which
        the operator's procedures reach only after `SHED_AFTER` scans over the limit."""
        dev, shed, slack, unmatched = self._solve(t, seen, limit, load_change, prev, True, shed_ok)
        relief = 0.0
        if np.abs(dev).sum() > 1e-6 and np.isfinite(seen).any():
            free_dev, _, _, _ = self._solve(t, seen, limit, load_change, prev, False)
            relief = float(self.cost[t] @ (dev - free_dev)) * FRAME_HOURS
        return Step(dev, shed, slack, unmatched, relief)
