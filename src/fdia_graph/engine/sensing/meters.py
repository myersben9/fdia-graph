"""The meter plan: every meter on the New York buses as one row of a table, with its owner and accuracy class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdia_graph.federated.utilities import Ownership
from fdia_graph.generation.nygrid.network import Network

V, VA, P, Q, PF, QF, I, IA = range(8)  # meter kinds, the `kind` codes
KINDS = ("v", "va", "p", "q", "pf", "qf", "i", "ia")

# Accuracy class of each kind as (relative, absolute) standard deviations, sigma = rel |z| + abs.
# |V|, angle, injections and flows are the instrument-transformer classes of Asprou et al. (IEEE Trans.
# Instrum. Meas. 2014): absolute for |V| (p.u.) and the angle, 1.7% of the reading for powers with a
# 1e-3 MW floor so a near-zero reading still has noise. The PMU branch currents take IEEE C37.118.1's
# 1% total vector error as three standard deviations of the phasor: 1/300 of |I| on the magnitude and
# 1/300 rad on the angle, with a 1e-5 p.u. floor. Angles are in degrees, like the state.
ACCURACY = {
    V: (0.0, 0.0012),
    VA: (0.0, float(np.degrees(0.00168))),
    P: (0.017, 1e-3),
    Q: (0.017, 1e-3),
    PF: (0.017, 1e-3),
    QF: (0.017, 1e-3),
    I: (0.01 / 3, 1e-5),
    IA: (0.0, float(np.degrees(0.01 / 3))),
}
# Share of each class that is per-scan jitter; the rest is a constant per-meter bias, with
# bias^2 + jitter^2 = class^2. Meter error is mostly a fixed calibration offset.
JITTER_FRAC = 0.25


@dataclass(frozen=True)
class MeterPlan:
    """One row per meter [M]: `kind` (a `KINDS` code), `bus` (the bus it sits at), `branch` and `end`
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
    groups = [(V, volt, none, none), (VA, pmu, none, none), (P, ny, none, none), (Q, ny, none, none)]
    groups += [(k, ends[:, 2], ends[:, 0], ends[:, 1]) for k in (PF, QF)]
    groups += [(k, pmu_ends[:, 2], pmu_ends[:, 0], pmu_ends[:, 1]) for k in (I, IA)]
    kind = np.concatenate([np.full(len(b), k) for k, b, _, _ in groups])
    bus = np.concatenate([b for _, b, _, _ in groups])
    branch = np.concatenate([br if len(br) else np.full(len(b), -1) for _, b, br, _ in groups])
    end = np.concatenate([e if len(e) else np.full(len(b), -1) for _, b, _, e in groups])

    sigma_rel = np.array([ACCURACY[k][0] for k in kind])
    sigma_abs = np.array([ACCURACY[k][1] for k in kind])
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
