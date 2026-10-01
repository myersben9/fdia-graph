"""What the generator passes around per scan: a measurement scan, the emitted frame with its labels,
the run's attack knobs, the design of an At ramp and of an Am overload episode, and what the
fewest-tamper search returns."""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import numpy as np


class Scan(NamedTuple):
    """One emitted measurement scan in the shard's physical units: readings and their meter masks."""

    node_x: np.ndarray  # [N, 4] |V|, P_inj, Q_inj, theta (zero where unmetered)
    node_m: np.ndarray  # [N, 4] meter mask
    edge_x: np.ndarray  # [E, 2] P_from, Q_from
    edge_m: np.ndarray  # [E, 2] meter mask
    # the PMU branch-current phasors [E, 4] (CURRENT columns, per unit) and their mask; None when the
    # meter plan has no currents (the v0.8.3 meter model)
    i_x: Optional[np.ndarray] = None
    i_m: Optional[np.ndarray] = None


class AttackDesign(NamedTuple):
    """A stealthy attack's design on one scan: the loads it moves, by how much, and over which
    subnetwork. The local false state is solved from it (engine.attacks, `stealthy_state`)."""

    targets: np.ndarray  # positions in the generator's load table (not bus numbers)
    mult: Union[float, np.ndarray]  # load multiplier: a scalar for the ramp, one per target otherwise
    interior: Optional[np.ndarray] = None  # the attacker's subnetwork; None: the region around targets


class RampDesign(NamedTuple):
    """One At episode, drawn at onset: a fixed load set scaled along a rise, a hold and a return
    (engine.attacks, `ramp_design`); frame i applies `ramp_step(design, i, rate)`."""

    targets: np.ndarray  # positions in the generator's load table
    direction: float  # +1 a load rise, -1 a load drop
    rise: int  # frames of the rise
    hold: int  # frames held at the peak
    # with the fewest-tamper knob: the support the minimizer chose, held for every frame of the
    # episode, and its result; None: every frame is solved on the region around its targets
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
    # The meters the attacker wrote: ([N, 4], [E, 2]) boolean masks, the meters whose true value
    # the local false state moves.
    tamper: Optional[tuple[np.ndarray, np.ndarray]] = None
    # the PMU branch-current channels, observed and un-attacked [E, 4] (per unit), their mask and the
    # current channels the attacker wrote; None without currents in the meter plan
    i_x: Optional[np.ndarray] = None
    i_m: Optional[np.ndarray] = None
    benign_i_x: Optional[np.ndarray] = None
    i_tamper: Optional[np.ndarray] = None


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


class FrameKnobs(NamedTuple):
    """The attack settings of one generation run, fixed for every scan."""

    # the attacks are local false states [WU26]: the attacker solves the subnetwork within `hops`
    # branches of the attacked buses (or the target lines) with the boundary voltages held true
    hops: int = 2
    limits: Optional[OperatingLimits] = None  # a false state outside the box is rejected (then halved)
    # [WU26 eq. 12]: an episode's support is the one that tampers the fewest devices (off: the region
    # within `hops`), searched over at most `min_budget` candidate supports
    min_tamper: bool = False
    min_budget: int = 256
    # a multiplier on At's stealth bound, whose unit is the rated accuracy [D7]; Am has none [D11]
    stealth_scale: float = 1.0
    # the overload attack's load-plausibility cap tau [D16]: None leaves the load changes unbounded
    load_cap: Optional[float] = None
    n_lines: int = 1  # the lines an overload episode drives at once (D17; new generation: 2)
    # the overload attack's support method (`OverloadSettings.support_method`): "search" or "rref"
    support_method: str = "search"
    # what the search minimizes first (`CostUnit`): the tampered devices, or the tampered measurements
    # (the l0 of [WU26]'s eqs. 12, 28 and 33 taken literally); the other count breaks ties
    objective: str = "devices"
    # the attacker's area [WU26] Sec. III-A: its buses when the caller knows them (the paper's own area,
    # `OverloadMixin.wu26_area`), else chosen by `area_rule` (`AreaRule`): "hops", the buses within `hops`
    # of the goal, or "rules", the region the paper's four principles pick (`AreaMixin.rule_area`, ours)
    area: Optional[tuple[int, ...]] = None
    area_rule: str = "hops"


class AttackVector(NamedTuple):
    """An attack vector h(x_false) - h(x_true) of one scan, per channel group in the scan's units."""

    node: np.ndarray  # [N, 4] |V|, P_inj, Q_inj, theta
    edge: np.ndarray  # [E, 2] P_from, Q_from
    current: Optional[np.ndarray] = None  # [E, 4] the PMU branch-current channels, when the plan has them


class LoadGoal(NamedTuple):
    """What a load-changing attack must realize at each snapshot of its window: the attack design of
    each snapshot (the loads it moves and their multipliers). The goal of At; the fewest-tamper
    search holds one support for all of them. The Am overload goal of [WU26 eqs. 24-25], a target
    line's reported flow reaching its rating, is its own goal type."""

    designs: tuple[AttackDesign, ...]  # one per snapshot, in window order
    kind: str = "load"  # the solve the fewest-tamper search applies per snapshot (MinimizeMixin.goal_state)


class FlowGoal(NamedTuple):
    """What the overload attack of [WU26 eqs. 24-25] must realize at each snapshot of its window: the
    apparent flow (MVA) that the tampered measurements carry before noise on each target branch,
    `S_{l,t} = S_true_{l,t} + (t - kappa)/T (S_max - S_true_{l,kappa+T})` (drift-free, [D9]),
    reaching the branch's rating at the window's end. The free injections of the support (attackable
    loads and generators) move; the fewest-tamper search holds one support for all snapshots. A goal
    drives one or more lines at once on the same support: `line` is the first target and `more` the
    others. Generated episodes drive `OverloadSettings.n_lines` lines, two by default as the paper's
    case studies do, or one [D17]."""

    line: int  # the first target branch (position in the edge index)
    targets: tuple[float, ...]  # the first target's MVA per snapshot, in window order
    kind: str = "flow"  # the solve the fewest-tamper search applies per snapshot (MinimizeMixin.goal_state)
    more: tuple[tuple[int, tuple[float, ...]], ...] = ()  # further (branch, MVA per snapshot) pairs

    @property
    def lines(self) -> tuple[int, ...]:
        """Every target branch, the first one first."""
        return (self.line, *(line for line, _ in self.more))

    def targets_at(self, t: int) -> tuple[float, ...]:
        """Every target branch's flow at snapshot t (MVA), in `lines` order."""
        return (self.targets[t], *(targets[t] for _, targets in self.more))


class AmOverloadDesign(NamedTuple):
    """One `Am` episode as the overload attack of [WU26]: the target branches (`goal.lines`), their
    ratings, the flow each frame must reach, and the fewest-tamper support held for the window with
    the search's result."""

    goal: FlowGoal
    ratings: tuple[float, ...]  # each target branch's rating S_max, MVA, in `goal.lines` order
    support: np.ndarray  # the buses whose voltages the false state moves, held for every frame
    tamper: MinimizerResult  # the search's result for the window


class MinimizerResult(NamedTuple):
    """The cheapest support found for one attack window [WU26 eq. 12] (the search holds no claim of
    optimality: [WU26] does not prove its minimum either)."""

    support: np.ndarray  # the buses whose voltages the false state moves (sorted)
    devices: int  # devices with a channel moved beyond its noise at some snapshot; -1 when no candidate
    # support meets every constraint and moves at least one device beyond its noise (the episode then
    # runs on the region)
    channels: int  # channels moved beyond their noise at some snapshot
    evaluated: int  # candidate supports solved


class AreaScore(NamedTuple):
    """How a candidate area meets [WU26]'s Section III-A principles: rules 1 and 2 as checks, rules 3
    and 4 as terms in [0, 1] (ours), and the size the score is charged for."""

    buses: np.ndarray
    observable: bool  # rule 1: a metered voltage or injection channel in the area
    connected: bool  # rule 2: one connected region over live branches
    load_share: float  # rule 3: the area's share of the system's active load over the window
    spread: float  # rule 3: mean |DC PTDF| of the goal lines to the area's injections (0 for a load goal)
    stability: float  # rule 4: 1 / (1 + the mean coefficient of variation of the area's loads)
    observability: float  # rule 4: metered channels per area bus against the system's, capped at 1
    size: float  # the area's buses as a share of the system's
