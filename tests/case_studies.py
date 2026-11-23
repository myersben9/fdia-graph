"""[WU26]'s case studies as test inputs: the lines each scenario overloads, its PMU buses and the attacker's
area, in MATPOWER bus numbers, their positions in a generator's grid, and the paper's own measurement noise.

The tests of the fewest-tamper search, the trust schedule, the row reduction and the attack area run on these
scenarios because the paper states what an attack on them looks like.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fdia_graph.constants.meters import (
    ANGLE_KINDS,
    BUS_KINDS,
    FLOW_KINDS,
    WU26_SCADA_NOISE_PU,
    WU26_WAMS_NOISE_PU,
    MeterKind,
)
from fdia_graph.engine.core import FdiaGenerator
from fdia_graph.models.choices import LimitSlack
from fdia_graph.models.config import MeterSettings
from fdia_graph.models.frames import OperatingLimits
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
    ends = [{int(number[f]), int(number[t])} for f, t in g.edge_index.T]
    return next(e for e, pair in enumerate(ends) if pair == {a, b})


def area(g: FdiaGenerator) -> tuple[int, ...]:
    """The attacker's area the paper states, as bus positions: IEEE-14's whole system but the slack ("knowledge
    of the system's global generation and load forecasts", p. 659), IEEE-118's local network of Fig. 9."""
    if g.n_bus == 14:
        return tuple(int(b) for b in range(g.n_bus) if b != g.slack_bus)
    return tuple(int(b) for b in buses(g, ATTACK_AREA[g.n_bus]))


def paper_noise(plan: MeterPlan, pmu: np.ndarray, base_mva: float) -> MeterPlan:
    """`plan` with the flat noise the paper's case studies state (Section V-A) in place of the package's
    accuracy classes, in the stored units: 0.03 pu on a SCADA channel (a power injection or flow, on the case
    base, and the voltage at a bus with no PMU) and 0.01 pu on a PMU channel (its voltage, its angle in degrees
    of 0.01 rad, its branch currents). The package counts tampering against the noise of the meters, so a
    scenario of the paper is counted on the paper's noise by giving the meters that noise. The paper's
    limits stay hard (`paper_limits`): its noise enters the count only."""
    flat = np.full(len(plan.kind), WU26_WAMS_NOISE_PU)  # the PMU channels: branch currents and what follows
    power = np.isin(plan.kind, (*BUS_KINDS[2:], *FLOW_KINDS))
    flat[power] = WU26_SCADA_NOISE_PU * base_mva
    flat[(plan.kind == MeterKind.V) & ~pmu[plan.bus]] = WU26_SCADA_NOISE_PU
    flat[np.isin(plan.kind, ANGLE_KINDS)] = np.degrees(WU26_WAMS_NOISE_PU)
    return replace(plan, sigma_rel=np.zeros(len(plan.kind)), sigma_abs=flat)


def paper_limits(g: FdiaGenerator, X: np.ndarray) -> OperatingLimits:
    """The operating limits of a paper scenario: the generator's, held hard [WU26 eqs. 21-23]. The
    package's default lets a generator sit past a limit by what its bus's injection meter resolves,
    which on the paper's flat noise would be 3 MW and 3 MVAr at every bus; the paper has no such
    slack."""
    return g.false_state.operating_limits(X)._replace(slack=LimitSlack.HARD.value)


def paper_metering(g: FdiaGenerator, pmu_numbers: tuple[int, ...], flow: np.ndarray) -> FdiaGenerator:
    """The generator of `g`'s case and seed (`g` at the default coverage) metered as the paper does: PMUs at
    the MATPOWER buses `pmu_numbers`, flow meters on the branches `flow` [E] bool, `g`'s own voltmeters and
    biases (the plan is drawn again from `g`'s seed, with the paper's places standing in), and the paper's
    noise on every meter (`paper_noise`)."""
    pmu = np.zeros(g.n_bus, bool)
    pmu[buses(g, pmu_numbers)] = True
    stream = np.random.default_rng(g.seed)
    plan = MeterPlan.from_coverage(g.case, MeterSettings(), stream, pmu_at=pmu, flow_at=flow)
    return FdiaGenerator(Network.from_ieee(g.n_bus), g.seed, plan=paper_noise(plan, pmu, g.case.base_mva))
