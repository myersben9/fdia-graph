"""The records the attack code passes around: a scan, a frame, an attack vector, a goal, a design, the
operating limits, a search result.

This module imports no other module of the package. Also here: the attack settings of a run
(`FrameKnobs`), the layers of a file (`MeterLayers`), the result of a flow solve (`FlowSolve`), the cost
of a support (`SupportCost`) and the score of a candidate area (`AreaScore`)."""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import numpy as np

from ..constants.numerics import RAMP_TARGETS


class MeterLayers(NamedTuple):
    """One value per meter of a meter table, laid out as a file's layers
    (`sensing.meters.MeterPlan.scatter`), zero where no meter sits. A flow is metered at the from
    end of its branch; a current is the real and imaginary part of the phasor at each end."""

    node: np.ndarray  # [..., N, 4] |V|, P_inj, Q_inj, theta
    edge: np.ndarray  # [..., E, 2] P_from, Q_from (`scatter` lays out four columns, the last two empty)
    current: np.ndarray  # [..., E, 4] Re and Im of the current at the from end, then at the to end


class Scan(NamedTuple):
    """One emitted measurement scan in the shard's physical units: readings and their meter masks."""

    node_x: np.ndarray  # [N, 4] |V|, P_inj, Q_inj, theta (zero where unmetered)
    node_m: np.ndarray  # [N, 4] meter mask
    edge_x: np.ndarray  # [E, 2] P_from, Q_from
    edge_m: np.ndarray  # [E, 2] meter mask
    i_x: np.ndarray  # [E, 4] the PMU branch-current phasors (`CurrentChannel` columns, per unit)
    i_m: np.ndarray  # [E, 4] their meter mask


class AttackDesign(NamedTuple):
    """A stealthy attack's design on one scan: the loads it moves, by how much, and over which
    support. The local false state is solved from it (engine.attacks, `stealthy_state`)."""

    targets: np.ndarray  # positions in the generator's load table (not bus numbers)
    mult: Union[float, np.ndarray]  # load multiplier: a scalar for the ramp, one per target otherwise
    support: Optional[np.ndarray] = (
        None  # the buses the false state re-solves; None: the area within `hops` of the targets
    )


class RampDesign(NamedTuple):
    """One At episode, drawn at onset: a fixed load set scaled along a rise, a hold and a return
    (engine.attacks, `ramp_design`); frame i applies `g.episode_designs.ramp_step(design, i, rate)`."""

    targets: np.ndarray  # positions in the generator's load table
    direction: float  # +1 a load rise, -1 a load drop
    rise: int  # frames of the rise
    hold: int  # frames held at the peak
    # with the fewest-tamper knob: the support the minimizer chose, held for every frame of the
    # episode, and its result; None: every frame is solved on the area around its targets
    support: Optional[np.ndarray] = None
    tamper: Optional[MinimizerResult] = None


class Frame(NamedTuple):
    """One emitted scan. Measurement arrays are in the shard's physical units and column order."""

    node_x: np.ndarray  # [N, 4] observed |V|, P_inj, Q_inj, theta (zero where unmetered)
    node_m: np.ndarray  # [N, 4] meter mask
    edge_x: np.ndarray  # [E, 2] observed P_from, Q_from
    edge_m: np.ndarray  # [E, 2] meter mask
    y: np.ndarray  # [N] uint8 per-bus attack label
    stealthy: int  # 1 when the scan is a re-solved state (evades bad-data detection), else 0
    mag_bus: np.ndarray  # buses with a designed magnitude (int), empty on benign scans
    mag: np.ndarray  # designed |change| / |base| per entry of mag_bus, the plausibility-band record
    benign_node_x: Optional[np.ndarray]  # un-attacked node measurement of the same scan (with_benign)
    benign_edge_x: Optional[np.ndarray]  # un-attacked branch flows of the same scan (with_benign)
    i_x: np.ndarray  # [E, 4] observed PMU branch currents (per unit; an IEEE file: `CurrentChannel` columns)
    i_m: np.ndarray  # [E, 4] their meter mask
    # The meters the attacker wrote: ([N, 4], [E, 2]) boolean masks, the meters whose true value
    # the local false state moves.
    tamper: Optional[tuple[np.ndarray, np.ndarray]] = None
    benign_i_x: Optional[np.ndarray] = None  # [E, 4] un-attacked PMU branch currents of the same scan
    i_tamper: Optional[np.ndarray] = None  # [E, 4] the current channels the attacker wrote

    @classmethod
    def from_scan(
        cls,
        scan: Scan,
        seen: MeterLayers,
        y: np.ndarray,
        mag: Optional[tuple[np.ndarray, np.ndarray]] = None,
        tamper: Optional[MeterLayers] = None,
    ) -> Frame:
        """The frame of a scan: `seen` its observed layers (the scan's own on a benign frame), `y` its
        labels [N], `mag` the designed (buses, magnitudes), and `tamper` the meters the attacker
        wrote, laid out like the readings (None: a benign frame)."""
        mag_bus, size = (np.zeros(0, int), np.zeros(0, float)) if mag is None else mag
        masks, currents = (None, None) if tamper is None else ((tamper.node, tamper.edge), tamper.current)
        attacked = int(tamper is not None)
        return cls(
            seen.node, scan.node_m, seen.edge, scan.edge_m, y, attacked, mag_bus, size,
            scan.node_x, scan.edge_x, seen.current, scan.i_m, masks, scan.i_x, currents,
        )  # fmt: skip


class OperatingLimits(NamedTuple):
    """The security and operational constraints a false state must satisfy [WU26 eqs. 21-23]:
    every bus voltage magnitude within the case's limits (a bus the true state already holds
    outside a limit may not be made worse) and every generator's implied output within its P and
    Q limits widened to the range the benign pool ran it over (the slack, whose output is the
    balance, and buses without a generator are unbounded)."""

    v_lo: np.ndarray  # [N] lowest voltage magnitude a false state may show at each bus, pu
    v_hi: np.ndarray  # [N] highest, pu
    p_lo: np.ndarray  # [N] lowest generator active output per bus, MW (-inf without a generator)
    p_hi: np.ndarray  # [N] highest, MW (+inf without a generator)
    q_lo: np.ndarray  # [N] lowest generator reactive output per bus, MVAr
    q_hi: np.ndarray  # [N] highest, MVAr
    # how far past a generator limit counts as past it (`LimitSlack`): what the bus's injection
    # meter resolves, or "hard", the numerical slack alone
    slack: str = "meter"


class FrameKnobs(NamedTuple):
    """The attack settings of one generation run, fixed for every scan."""

    # the attacks are local false states [WU26]: the attacker solves the area within `hops`
    # branches of the attacked buses (or the target lines) with the boundary voltages held true
    hops: int = 2
    limits: Optional[OperatingLimits] = None  # a false state outside the box is rejected
    # [WU26 eq. 12]: an episode's support is the one that tampers the fewest devices (off: the area
    # within `hops`), searched over at most `min_budget` candidate supports
    min_tamper: bool = False
    min_budget: int = 256
    # a multiplier on At's stealth bound, whose unit is the rated accuracy Am has none
    stealth_scale: float = 1.0
    # the overload attack's load-plausibility cap tau: None leaves the load changes unbounded
    load_cap: Optional[float] = None
    n_lines: int = 1  # the lines an overload episode drives at once (a generated timeline: 2)
    # the overload attack's support method (`OverloadSettings.support_method`): "search" or "rref"
    support_method: str = "search"
    # what the search minimizes first (`CostUnit`): the tampered devices, or the tampered measurements
    # (the l0 of [WU26]'s eqs. 12, 28 and 33 taken literally); the other count breaks ties
    objective: str = "devices"
    # the attacker's area [WU26] Sec. III-A: its buses when the caller gives them bus by bus,
    # else chosen by `area_rule` (`AreaRule`): "hops", the buses within `hops`
    # of the goal, or "rules", the area the paper's four principles pick (`AttackArea.rule_area`, ours)
    area: Optional[tuple[int, ...]] = None
    area_rule: str = "hops"
    ramp_targets: int = RAMP_TARGETS  # the most loads one ramp moves (`RampSettings.targets`)


class AttackVector(NamedTuple):
    """An attack vector h(x_false) - h(x_true) of one scan, per channel group in the scan's units."""

    node: np.ndarray  # [N, 4] |V|, P_inj, Q_inj, theta
    edge: np.ndarray  # [E, 2] P_from, Q_from
    current: np.ndarray  # [E, 4] the PMU branch-current channels


class LoadGoal(NamedTuple):
    """What a load-changing attack must realize at each snapshot of its window: the attack design of
    each snapshot (the loads it moves and their multipliers). The goal of At; the fewest-tamper
    search holds one support for all of them. Am's goal is `FlowGoal`."""

    designs: tuple[AttackDesign, ...]  # one per snapshot, in window order
    kind: str = "load"  # the solve the fewest-tamper search applies per snapshot (FalseState.goal_state)


class FlowGoal(NamedTuple):
    """What the overload attack of [WU26 eqs. 24-25] must realize at each snapshot of its window: the
    apparent flow (MVA) that the tampered measurements carry before noise on each target branch,
    `S_{l,t} = S_true_{l,t} + (t - kappa)/T (S_max - S_true_{l,kappa+T})` (drift-free),
    reaching the branch's rating at the window's end. The free injections of the support (attackable
    loads and generators) move; one support is held for all snapshots. A goal drives one or more
    lines on that support: `line` is the first target and `more` the others."""

    line: int  # the first target branch (position in the edge index)
    targets: tuple[float, ...]  # the first target's MVA per snapshot, in window order
    kind: str = "flow"  # the solve the fewest-tamper search applies per snapshot (FalseState.goal_state)
    more: tuple[tuple[int, tuple[float, ...]], ...] = ()  # further (branch, MVA per snapshot) pairs

    @property
    def lines(self) -> tuple[int, ...]:
        """Every target branch, the first one first."""
        return (self.line, *(line for line, _ in self.more))

    def targets_at(self, t: int) -> tuple[float, ...]:
        """Every target branch's flow at snapshot t (MVA), in `lines` order."""
        return (self.targets[t], *(targets[t] for _, targets in self.more))


Goal = Union[LoadGoal, FlowGoal]  # what a window must realize: the loads of At, the flow of Am


class AmOverloadDesign(NamedTuple):
    """One `Am` episode as the overload attack of [WU26]: the target branches (`goal.lines`), their
    ratings, the flow each frame must reach, and the fewest-tamper support held for the window with
    the search's result."""

    goal: FlowGoal
    ratings: tuple[float, ...]  # each target branch's rating S_max, MVA, in `goal.lines` order
    support: np.ndarray  # the buses whose voltages the false state moves, held for every frame
    tamper: MinimizerResult  # the search's result for the window


class FlowSolve(NamedTuple):
    """What the flow solve of a goal on a support returns [WU26 eqs. 24-25]
    (`engine.attacks.false_state.FalseState.solve_flow_local`)."""

    x_false: Optional[np.ndarray]  # the false state [N, 4]; None when the solve fails, a bound cannot be
    # met or the support has no free bus
    converged: bool  # False only when the local solve itself failed
    load_change: np.ndarray  # the pretended load change per bus [N], MW; zero at a generator bus


class SupportCost(NamedTuple):
    """What a support costs over an attack window [WU26 eq. 12]: the devices and the channels its
    false states move beyond the meters' noise at some snapshot, and its size in buses. Both counts are
    zero for a support whose changes all stay under the noise."""

    devices: int  # devices with a channel moved beyond noise at some snapshot
    channels: int  # channels moved beyond noise at some snapshot
    size: int  # buses in the support

    def key(self, channels_first: bool) -> tuple[int, int, int]:
        """The cost in the order the search compares it: the objective first (devices, or channels
        with `FrameKnobs.objective` "channels"), the other count next, the size last."""
        if channels_first:
            return self.channels, self.devices, self.size
        return self.devices, self.channels, self.size


class MinimizerResult(NamedTuple):
    """The cheapest support found for one attack window [WU26 eq. 12] (the search holds no claim of
    optimality: [WU26] does not prove its minimum either)."""

    support: np.ndarray  # the buses whose voltages the false state moves (sorted)
    devices: int  # devices with a channel moved beyond its noise at some snapshot: 0 or more (a ramp
    # may stay under the noise of every meter for its whole window; an overload moves at least one)
    channels: int  # channels moved beyond their noise at some snapshot: 0 or more
    evaluated: int  # candidate supports solved


class AreaScore(NamedTuple):
    """How a candidate area meets [WU26]'s Section III-A principles: rules 1 and 2 as checks, rules 3
    and 4 as terms in [0, 1] (ours), and the size the score is charged for."""

    buses: np.ndarray
    observable: bool  # rule 1: a metered voltage or injection channel in the area
    connected: bool  # rule 2: one connected region over live branches
    load_share: float  # rule 3: the area's share of the system's active load over the window
    # rule 3: mean |AC flow sensitivity| of the goal lines to the area's injections (0 for a load goal)
    spread: float
    stability: float  # rule 4: 1 / (1 + the mean coefficient of variation of the area's loads)
    observability: float  # rule 4: metered channels per area bus against the system's, capped at 1
    size: float  # the area's buses as a share of the system's
