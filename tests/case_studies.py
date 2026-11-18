"""[WU26]'s case studies as test inputs: the lines each scenario overloads, its PMU buses and the attacker's
area, in MATPOWER bus numbers, and their positions in a generator's grid.

The tests of the fewest-tamper search, the trust schedule, the row reduction and the attack area run on these
scenarios because the paper states what an attack on them looks like.
"""

from __future__ import annotations

import numpy as np

from fdia_graph.engine.core import FdiaGenerator
from fdia_graph.models.config import MeterSettings
from fdia_graph.sensing.meters import MeterPlan
from fdia_graph.systems.network import Network

# the two lines each scenario overloads at once
SCENARIOS: dict[int, tuple[tuple[tuple[int, int], tuple[int, int]], ...]] = {
    14: (((3, 4), (6, 11)), ((1, 2), (4, 5))),  # Section V-A
    118: (((84, 85), (99, 100)),),  # Section V-B, inside the attacker's local network of Fig. 9
}
PMUS: dict[int, tuple[int, ...]] = {
    14: (1, 4, 6, 13),  # Section V-A, the trusted-PMU sequence 1 -> 4 -> 6 -> 13
    # Fig. 9 ("IEEE 118-bus system topology and PMU distribution"), the buses drawn in red: 11 bars, each read
    # against its bus label at 1,200 dpi; bus 94's label sits beside bus 95's bar, but the red bar is 94's
    118: (76, 78, 80, 83, 89, 92, 94, 100, 105, 106, 110),
}
# Fig. 9's red dashed region, the attacker's local network on IEEE-118: every bus drawn inside it
ATTACK_AREA: dict[int, tuple[int, ...]] = {118: (*range(74, 113), 118)}


def _numbers(g: FdiaGenerator) -> np.ndarray:
    """The MATPOWER number of every bus of the generator's grid, by position."""
    return g.case.bus_id


def buses(g: FdiaGenerator, numbers: tuple[int, ...] | list[int]) -> np.ndarray:
    """The generator's bus positions of the MATPOWER bus numbers `numbers`."""
    number = _numbers(g)
    return np.array([int(np.flatnonzero(number == n)[0]) for n in numbers], np.int64)


def branch(g: FdiaGenerator, a: int, b: int) -> int:
    """The generator's branch between the MATPOWER buses a and b (either order): the first match."""
    number = _numbers(g)
    ends = [{int(number[f]), int(number[t])} for f, t in g.ei.T]
    return next(e for e, pair in enumerate(ends) if pair == {a, b})


def area(g: FdiaGenerator) -> tuple[int, ...]:
    """The attacker's area the paper states, as bus positions: IEEE-14's whole system but the slack ("knowledge
    of the system's global generation and load forecasts", p. 659), IEEE-118's local network of Fig. 9."""
    if g.C == 14:
        return tuple(int(b) for b in range(g.C) if b != g.slack_bus)
    return tuple(int(b) for b in buses(g, ATTACK_AREA[g.C]))


def paper_metering(g: FdiaGenerator, pmu_numbers: tuple[int, ...], flow: np.ndarray) -> FdiaGenerator:
    """The generator of `g`'s case and seed (`g` at the default coverage) metered as the paper does: PMUs at
    the MATPOWER buses `pmu_numbers`, flow meters on the branches `flow` [E] bool, and `g`'s own voltmeters
    and biases (the plan is drawn again from `g`'s seed, with the paper's places standing in)."""
    pmu = np.zeros(g.C, bool)
    pmu[buses(g, pmu_numbers)] = True
    stream = np.random.default_rng(g.seed)
    plan = MeterPlan.from_coverage(g.case, MeterSettings(), stream, pmu_at=pmu, flow_at=flow)
    return FdiaGenerator(Network.from_ieee(g.C), g.seed, plan=plan)
