"""The meter plan: every meter on the New York buses as one row of a table, with its owner and accuracy class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdia_graph.constants.channels import TO_END_OFFSET
from fdia_graph.constants.meters import JITTER_FRAC, KIND_ACCURACY, METER_SLOT, MeterKind
from fdia_graph.generation.nygrid.network import Network
from fdia_graph.generation.nygrid.ownership import Ownership
from fdia_graph.models.frames import MeterLayers


@dataclass(frozen=True)
class MeterPlan:
    """One row per meter [M]: `kind` (a `MeterKind`), `bus` (the bus it sits at), `branch` and `end`
    (0 from, 1 to; -1 for bus meters), `owner` (the client of `bus`), the accuracy class `sigma_rel`,
    `sigma_abs` (sigma = sigma_rel |z| + sigma_abs; on a relative meter `sigma_abs` is its floor) and the
    constant `bias` (a gain error on a relative meter, an offset on an absolute one)."""

    kind: np.ndarray
    bus: np.ndarray
    branch: np.ndarray
    end: np.ndarray
    owner: np.ndarray
    sigma_rel: np.ndarray
    sigma_abs: np.ndarray
    bias: np.ndarray

    def _slots(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Where each meter's reading sits in a file's layers: the layer (of `METER_SLOT`), the row (its bus,
        or its branch) and the column."""
        layer = np.array([METER_SLOT[k][0] for k in self.kind])
        column = np.array([METER_SLOT[k][1] for k in self.kind]) + TO_END_OFFSET * np.maximum(self.end, 0)
        return layer, np.where(self.branch >= 0, self.branch, self.bus), column

    def scatter(self, z: np.ndarray, buses: int, branches: int) -> MeterLayers:
        """One value per meter `z [..., M]` (readings, or a mask) laid out as a file's layers over `buses`
        buses and `branches` branches, zero where no meter sits. `gather` is its inverse."""
        z = np.asarray(z)
        layer, row, column = self._slots()
        out = MeterLayers(
            *(np.zeros(z.shape[:-1] + (rows, 4), z.dtype) for rows in (buses, branches, branches))
        )
        for k, values in enumerate(out):
            here = layer == k
            values[..., row[here], column[here]] = z[..., here]
        return out

    def gather(self, node: np.ndarray, edge: np.ndarray, current: np.ndarray) -> np.ndarray:
        """The flat vector `[..., M]` of this plan read back from a file's layers, in the plan's row order."""
        layer, row, column = self._slots()
        z = np.empty(node.shape[:-2] + (len(self.kind),), node.dtype)
        for k, values in enumerate((node, edge, current)):
            here = layer == k
            z[..., here] = values[..., row[here], column[here]]
        return z

    @classmethod
    def from_masks(cls, masks: MeterLayers, edges: np.ndarray, bus_client: np.ndarray) -> MeterPlan:
        """The plan a file's meter masks describe (`masks`, each layer of one frame), in the row order of
        `plan_meters`: kind by kind, the bus meters by bus, the branch meters by branch and then end. So
        `gather` on that file's layers gives the flat vector the file was written from. `edges [2, E]` are
        the branch ends and `bus_client [N]` the owner of each bus. The accuracy class is the table's
        (`KIND_ACCURACY`); the constant `bias` of each meter is not in a file (the generator draws it from its
        seed) and reads NaN."""
        kind, bus, branch, end = [], [], [], []
        for k in MeterKind:
            layer, column = METER_SLOT[k]
            at_bus = layer == 0
            # row-major, so the meters come by bus, or by branch and then end
            row, e = np.nonzero(
                masks[layer][:, column : column + 1] if at_bus else masks[layer][:, column::TO_END_OFFSET]
            )
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


def _min_pmus(adjacency) -> np.ndarray:
    """The smallest bus set whose PMUs see every bus, sorted: each bus has a PMU or a neighbour with one,
    since a PMU's branch current fixes the voltage at the far end of the branch [1]. Exact, by branch and
    bound over bitsets: branch on the uncovered bus with the fewest ways to cover it, bound by the most
    buses one more PMU can newly cover. Ties go to the lower bus position, so the set is deterministic."""
    a = adjacency.toarray().astype(bool) | np.eye(adjacency.shape[0], dtype=bool)
    n = len(a)
    cover = [sum(1 << j for j in np.flatnonzero(row)) for row in a]
    full = (1 << n) - 1
    best = list(range(n))

    def search(chosen: list[int], covered: int) -> None:
        nonlocal best
        if covered == full:
            if len(chosen) < len(best):
                best = chosen
            return
        gains = [(cover[c] & ~covered).bit_count() for c in range(n)]
        left = (full & ~covered).bit_count()
        if len(chosen) + -(-left // max(gains)) >= len(best):
            return
        uncovered = [b for b in range(n) if not covered >> b & 1]
        b = min(uncovered, key=lambda u: a[u].sum())
        for c in sorted(np.flatnonzero(a[b]).tolist(), key=lambda c: -gains[c]):
            search(chosen + [c], covered | cover[c])

    search([], 0)
    return np.array(sorted(best), dtype=int)


def plan_meters(net: Network, own: Ownership, rng: np.random.Generator, vbus_frac: float = 0.6) -> MeterPlan:
    """The meters of the New York buses; external buses carry none (N4). SCADA reads P and Q at every NY
    bus, P and Q flow at every branch end on a NY bus (so each side of a tie line meters its own end), and
    |V| at a random `vbus_frac` of the NY buses. PMUs sit at the smallest set that sees every NY bus (N7)
    and read the voltage phasor and the current phasor at their end of each branch [1] eqs. 19 and 20.
    Each meter is owned by the client of its bus. Draws from `rng`: the voltmeter buses, then the biases."""
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

    none = np.empty(0, int)
    groups = [
        (MeterKind.V, volt, none, none),
        (MeterKind.VA, pmu, none, none),
        (MeterKind.P, ny, none, none),
        (MeterKind.Q, ny, none, none),
    ]
    groups += [(k, ends[:, 2], ends[:, 0], ends[:, 1]) for k in (MeterKind.PF, MeterKind.QF)]
    groups += [(k, pmu_ends[:, 2], pmu_ends[:, 0], pmu_ends[:, 1]) for k in (MeterKind.I, MeterKind.IA)]
    kind = np.concatenate([np.full(len(b), k) for k, b, _, _ in groups])
    bus = np.concatenate([b for _, b, _, _ in groups])
    branch = np.concatenate([br if len(br) else np.full(len(b), -1) for _, b, br, _ in groups])
    end = np.concatenate([e if len(e) else np.full(len(b), -1) for _, b, _, e in groups])

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
