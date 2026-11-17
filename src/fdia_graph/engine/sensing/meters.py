"""The meter plan: every meter of a grid as one row of a table, with its owner and accuracy class.

`plan_meters` places the meters of the New York buses, `MeterPlan.from_coverage` those of an IEEE
case (`from_placement` at given places), and `MeterPlan.from_masks` reads the plan back from a New York file."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Optional

import numpy as np

from fdia_graph.constants.channels import TO_END_OFFSET
from fdia_graph.constants.meters import (
    ACCURACY_CLASS,
    BUS_KINDS,
    FLOW_KINDS,
    JITTER_FRAC,
    KIND_ACCURACY,
    METER_SLOT,
    PHASOR_KINDS,
    PMU_CURRENT_CLASS,
    POLAR_FILE_KINDS,
    MeterKind,
)
from fdia_graph.generation.nygrid.case import Case
from fdia_graph.generation.nygrid.network import Network
from fdia_graph.generation.nygrid.ownership import Ownership
from fdia_graph.models.config import MeterSettings
from fdia_graph.models.frames import MeterLayers

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


@dataclass(frozen=True)
class MeterPlan:
    """One row per meter [M]. `branch` and `end` (0 from, 1 to) are -1 for bus meters; `owner` is the
    client of `bus`. The accuracy class is sigma = sigma_rel |z| + sigma_abs (on a relative meter
    `sigma_abs` is its floor). `bias` is constant: a gain error on a relative meter, an offset otherwise."""

    kind: np.ndarray
    bus: np.ndarray
    branch: np.ndarray
    end: np.ndarray
    owner: np.ndarray
    sigma_rel: np.ndarray
    sigma_abs: np.ndarray
    bias: np.ndarray
    # Where the readings sit in a file's layers, worked out once: for each layer of `MeterLayers` (node,
    # flow, current) its meters' rows in this table, then their row (bus or branch) and column there.
    slots: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        layer = np.array([METER_SLOT[k][0] for k in self.kind], int)
        column = np.array([METER_SLOT[k][1] for k in self.kind], int)
        column = column + TO_END_OFFSET * np.maximum(self.end, 0)  # a to end's column
        row = np.where(self.branch >= 0, self.branch, self.bus)
        meters = [np.flatnonzero(layer == k) for k in range(len(MeterLayers._fields))]
        object.__setattr__(self, "slots", tuple((m, row[m], column[m]) for m in meters))

    def select(self, meters: np.ndarray) -> MeterPlan:
        """The plan of the rows `meters` of this one (a bool mask [M] or positions), in their order."""
        columns = [f.name for f in fields(self) if f.init]  # `slots` is worked out from them
        return type(self)(**{name: getattr(self, name)[meters] for name in columns})

    def buses(self, kind: MeterKind) -> np.ndarray:
        """The buses with a meter of `kind`, sorted (`MeterKind.VA`: the PMU buses)."""
        return np.unique(self.bus[self.kind == kind])

    def phasor_partner(self) -> np.ndarray:
        """[M] for the real or imaginary part of a current phasor, the row of its other part; -1 on
        every other meter. Both parts of a phasor are metered, in the same order."""
        partner = np.full(len(self.kind), -1)
        real, imag = (np.flatnonzero(self.kind == k) for k in PHASOR_KINDS)
        partner[real], partner[imag] = imag, real
        return partner

    @classmethod
    def from_coverage(
        cls,
        case: Case,
        coverage: MeterSettings,
        rng: np.random.Generator,
        pmu_at: Optional[np.ndarray] = None,
        flow_at: Optional[np.ndarray] = None,
    ) -> MeterPlan:
        """The meters of an IEEE case at random places (`from_placement` says what each reads): a
        voltmeter at `vbus_frac` of the buses, a PMU at `pmu_frac` (at least one), a flow meter on
        `flow_frac` of the branches. Three draws from `rng`, in this order, then the biases. `pmu_at`
        [N] and `flow_at` [E] (bool) stand in for the drawn PMUs and flow meters (a paper's metering);
        the draws are made all the same, so the voltmeters and the biases stay those of the stream."""
        N, E = len(case.bus_id), len(case.f)
        volt, pmu = np.zeros(N, bool), np.zeros(N, bool)
        volt[rng.choice(N, int(coverage.vbus_frac * N), replace=False)] = True
        pmu[rng.choice(N, max(1, int(coverage.pmu_frac * N)), replace=False)] = True
        flow = rng.random(E) < coverage.flow_frac
        pmu = pmu if pmu_at is None else np.asarray(pmu_at, bool)
        flow = flow if flow_at is None else np.asarray(flow_at, bool)
        return cls.from_placement(case, volt, pmu, flow, rng)

    @classmethod
    def from_placement(
        cls, case: Case, volt: np.ndarray, pmu: np.ndarray, flow: np.ndarray, rng: np.random.Generator
    ) -> MeterPlan:
        """The meters of an IEEE case, owned by one client, with a voltmeter at the buses `volt` [N], a
        PMU at `pmu` [N] and a flow meter on the branches `flow` [E] (all bool): P and Q injection at every
        bus, P and Q flow at a metered branch's from end, |V| at voltmeters and PMUs, and at a PMU also the
        angle and the current phasor of every branch end at its bus, as real and imaginary part [WU26 eqs.
        17-20]."""
        # The rows are in the order a scan's noise is drawn: bus by bus (|V|, angle, P, Q), then the
        # flows and then the currents, branch by branch. The biases are drawn from `rng` for every bus
        # and branch, metered or not, in the order every released file fixes: P, Q, |V|, angle, P flow,
        # Q flow, the currents.
        N, E = len(case.bus_id), len(case.f)
        constant = np.sqrt(1.0 - JITTER_FRAC**2)  # the share of a class that is a constant bias
        p = rng.normal(0.0, constant * ACCURACY_CLASS["pi"], N)
        q = rng.normal(0.0, constant * ACCURACY_CLASS["qi"], N)
        v = rng.normal(0.0, constant * ACCURACY_CLASS["v"], N)
        angle = np.degrees(rng.normal(0.0, constant * ACCURACY_CLASS["va"], N))  # the class is in radians
        pf = rng.normal(0.0, constant * ACCURACY_CLASS["pf"], E)
        qf = rng.normal(0.0, constant * ACCURACY_CLASS["qf"], E)
        current_bias = rng.normal(0.0, constant * PMU_CURRENT_CLASS, (E, 4))

        # one column per kind a bus, a flow meter and a PMU's branch ends can hold; row-major order
        b, k = np.nonzero(np.c_[volt | pmu, pmu, np.ones((N, 2), bool)])  # columns: BUS_KINDS
        e, c = np.nonzero(np.c_[flow, flow])  # columns: FLOW_KINDS, at the from end
        at_from, at_to = pmu[case.f], pmu[case.t]
        i, part = np.nonzero(np.c_[at_from, at_from, at_to, at_to])  # columns: CurrentChannel
        end = part // TO_END_OFFSET
        kind = np.r_[np.array(BUS_KINDS)[k], np.array(FLOW_KINDS)[c], np.array(PHASOR_KINDS)[part % 2]]
        none = np.full(len(b), -1)
        return cls(
            kind=kind,
            bus=np.r_[b, case.f[e], np.where(end == 0, case.f[i], case.t[i])],
            branch=np.r_[none, e, i],
            end=np.r_[none, np.zeros(len(e), int), end],
            owner=np.zeros(len(kind), int),
            sigma_rel=np.array([KIND_ACCURACY[m][0] for m in kind]),
            sigma_abs=np.array([KIND_ACCURACY[m][1] for m in kind]),
            bias=np.r_[np.c_[v, angle, p, q][b, k], np.c_[pf, qf][e, c], current_bias[i, part]],
        )

    def scatter(self, z: np.ndarray, buses: int, branches: int) -> MeterLayers:
        """One value per meter `z [..., M]` (readings, or a mask) laid out as a file's layers over `buses`
        buses and `branches` branches, zero where no meter sits. `gather` is its inverse."""
        z = np.asarray(z)
        lead = z.shape[:-1]
        out = MeterLayers(*(np.zeros(lead + (rows, 4), z.dtype) for rows in (buses, branches, branches)))
        for values, (meters, row, column) in zip(out, self.slots):
            values[..., row, column] = z[..., meters]
        return out

    def gather(self, node: np.ndarray, edge: np.ndarray, current: np.ndarray) -> np.ndarray:
        """The flat vector `[..., M]` of this plan read back from a file's layers, in the plan's row order."""
        z = np.empty(node.shape[:-2] + (len(self.kind),), node.dtype)
        for values, (meters, row, column) in zip((node, edge, current), self.slots):
            z[..., meters] = values[..., row, column]
        return z

    @classmethod
    def from_masks(cls, masks: MeterLayers, edges: np.ndarray, bus_client: np.ndarray) -> MeterPlan:
        """The plan a file's meter masks describe, in the row order of `plan_meters` (kind by kind, bus
        meters by bus, branch meters by branch then end), so `gather` on the file's layers gives the vector
        it was written from. `edges` is [2, E]. A file does not store `bias`: it reads NaN."""
        kind, bus, branch, end = [], [], [], []
        for k in POLAR_FILE_KINDS:
            layer, column = METER_SLOT[k]
            at_bus = layer == 0
            # row-major, so the meters come by bus, or by branch and then end
            columns = slice(column, column + 1) if at_bus else slice(column, None, TO_END_OFFSET)
            row, e = np.nonzero(masks[layer][:, columns])
            none = np.full(len(row), -1)
            kind.append(np.full(len(row), k))
            bus.append(row if at_bus else edges[e, row])
            branch.append(none if at_bus else row)
            end.append(none if at_bus else e)
        kinds, buses = np.concatenate(kind), np.concatenate(bus)
        return cls(
            kind=kinds,
            bus=buses,
            branch=np.concatenate(branch),
            end=np.concatenate(end),
            owner=bus_client[buses],
            sigma_rel=np.array([KIND_ACCURACY[k][0] for k in kinds]),
            sigma_abs=np.array([KIND_ACCURACY[k][1] for k in kinds]),
            bias=np.full(len(kinds), np.nan),
        )


class _PmuSearch:
    """The branch and bound of `_min_pmus` over bitsets: `near` [N, N] bool says which buses a PMU at
    each bus sees (itself and its neighbours), `best` is the smallest covering set found so far."""

    def __init__(self, near: np.ndarray) -> None:
        self.near, self.n = near, len(near)
        self.cover = [sum(1 << j for j in np.flatnonzero(row)) for row in near]
        self.full = (1 << self.n) - 1
        self.best = list(range(self.n))

    def search(self, chosen: list[int], covered: int) -> None:
        """Extend `chosen` (which sees the buses of the bitset `covered`) to every cover that can beat `best`."""
        if covered == self.full:
            if len(chosen) < len(self.best):
                self.best = chosen
            return
        gains = [(self.cover[c] & ~covered).bit_count() for c in range(self.n)]
        left = (self.full & ~covered).bit_count()
        if len(chosen) + -(-left // max(gains)) >= len(self.best):
            return
        uncovered = [b for b in range(self.n) if not covered >> b & 1]
        b = min(uncovered, key=lambda u: self.near[u].sum())
        for c in sorted(np.flatnonzero(self.near[b]).tolist(), key=lambda c: -gains[c]):
            self.search(chosen + [c], covered | self.cover[c])


def _min_pmus(adjacency: spmatrix) -> np.ndarray:
    """The smallest bus set whose PMUs see every bus, sorted: each bus has a PMU or a neighbour with one,
    since a PMU's branch current fixes the voltage at the far end of the branch [WU26]. Exact, by branch
    and bound over bitsets. Ties go to the lower bus position, so the set is deterministic."""
    found = _PmuSearch(adjacency.toarray().astype(bool) | np.eye(adjacency.shape[0], dtype=bool))
    found.search([], 0)
    return np.array(sorted(found.best), dtype=int)


def plan_meters(net: Network, own: Ownership, rng: np.random.Generator, vbus_frac: float = 0.6) -> MeterPlan:
    """The meters of the New York buses (external buses carry none), each owned by the client of its bus.
    SCADA reads P and Q injection at every NY bus, P and Q flow at every branch end on a NY bus (each side of
    a tie meters its own end), and |V| at a random `vbus_frac` of the NY buses. PMUs sit at the smallest set
    that sees every NY bus and read voltage and current phasors [WU26] eqs. 19, 20."""
    # draws from `rng`, in this order: the voltmeter buses, then the biases
    case = net.case
    ny = np.flatnonzero(own.bus_client >= 0)
    pmu = ny[_min_pmus(net.adjacency[ny][:, ny])]
    volt = np.union1d(rng.choice(ny, int(vbus_frac * len(ny)), replace=False), pmu)

    # branch ends on NY buses, in branch order, from end first
    ends = np.r_[
        np.c_[np.arange(len(case.f)), np.zeros(len(case.f), int), case.f],
        np.c_[np.arange(len(case.t)), np.ones(len(case.t), int), case.t],
    ]
    ends = ends[own.bus_client[ends[:, 2]] >= 0]
    ends = ends[np.lexsort((ends[:, 1], ends[:, 0]))]
    pmu_ends = ends[np.isin(ends[:, 2], pmu)]

    # (kind, bus, branch, end) of each group of meters; a bus meter has no branch and no end (-1)
    at_buses = ((MeterKind.V, volt), (MeterKind.VA, pmu), (MeterKind.P, ny), (MeterKind.Q, ny))
    groups = [(k, b, np.full(len(b), -1), np.full(len(b), -1)) for k, b in at_buses]
    groups += [(k, ends[:, 2], ends[:, 0], ends[:, 1]) for k in (MeterKind.PF, MeterKind.QF)]
    groups += [(k, pmu_ends[:, 2], pmu_ends[:, 0], pmu_ends[:, 1]) for k in (MeterKind.I, MeterKind.IA)]
    kind = np.concatenate([np.full(len(b), k) for k, b, _, _ in groups])
    bus = np.concatenate([b for _, b, _, _ in groups])
    branch = np.concatenate([br for _, _, br, _ in groups])
    end = np.concatenate([e for _, _, _, e in groups])

    sigma_rel = np.array([KIND_ACCURACY[k][0] for k in kind])
    sigma_abs = np.array([KIND_ACCURACY[k][1] for k in kind])
    scale = np.sqrt(1.0 - JITTER_FRAC**2) * np.where(sigma_rel > 0, sigma_rel, sigma_abs)
    return MeterPlan(
        kind=kind,
        bus=bus,
        branch=branch,
        end=end,
        owner=own.bus_client[bus],
        sigma_rel=sigma_rel,
        sigma_abs=sigma_abs,
        bias=rng.normal(0.0, scale),
    )
