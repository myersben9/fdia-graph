"""What the generator passes around per scan: a measurement scan, the emitted frame with its labels,
the run's attack knobs, the design and target of one attack (and of a ramp or Am episode), and the
two intermediate results of the physics (a load redistribution, a re-solved pool)."""

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


class Band(NamedTuple):
    """The plausibility band of an in-place tamper: each realized |change| / |reading| lies in
    [floor, cap], above the meter noise floor (a smaller change resolves to noise) and below the
    largest change the literature treats as credible."""

    floor: float  # lower edge, the noise floor (generation.NOISE_FLOOR)
    cap: float  # upper edge, the attack intensity


class TamperTarget(NamedTuple):
    """Where an in-place attack (Ad, As, Ar) writes: the attacked buses and every branch incident to
    one of them, whose flows the attacker corrupts to match."""

    buses: np.ndarray  # attacked bus indices
    branches: list[int]  # branch positions with an attacked bus at either end


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


class AmDesign(NamedTuple):
    """One Am episode, drawn at onset: a load redistribution held over the episode and the ramp that
    reaches it (engine.attacks, `am_design`); frame i applies `am_step(design, Xt, i)`."""

    targets: np.ndarray  # positions in the generator's load table
    delta: np.ndarray  # MW per target at full strength (load-conserving)
    interior: Optional[np.ndarray]  # the attacker's subnetwork around the target line
    rate: float  # fraction of the full redistribution added per frame on the rise and the fall
    rise: int
    hold: int


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
    # The meters the attacker wrote: ([N, 4], [E, 2]) boolean masks. For the stealthy families the
    # meters whose true value the local false state moves; None for the in-place families, whose
    # tamper set is the meters that differ from the benign twin.
    tamper: Optional[tuple[np.ndarray, np.ndarray]] = None
    # the PMU branch-current channels, observed and un-attacked [E, 4] (per unit), their mask and the
    # current channels the attacker wrote; None without currents in the meter plan
    i_x: Optional[np.ndarray] = None
    i_m: Optional[np.ndarray] = None
    benign_i_x: Optional[np.ndarray] = None
    i_tamper: Optional[np.ndarray] = None


class OperatingLimits(NamedTuple):
    """The security and operational constraints a false state must satisfy [WU26, eqs. 21-23]:
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

    intensity: float  # attack_intensity: load-shift bound of Aq/Al and the plausibility cap of Ad/As/Ar
    floor: float  # lower edge of the plausibility band (NOISE_FLOOR)
    lra_k: int  # most buses an LRA redistribution may touch
    replay_tau: Optional[int]  # Ar replay depth in scans, None = random lag of at least REPLAY_MIN_LAG
    reject_below_floor: bool  # shards: reject a within-noise scan so the draw loop redraws
    with_benign: bool  # streams: also emit the un-attacked twin of the scan
    # the stealthy families are local false states [WU26]: the attacker solves the subnetwork within
    # `hops` branches of the attacked buses (or the target line) with the boundary voltages held true
    hops: int = 2
    limits: Optional[OperatingLimits] = None  # a false state outside the box is rejected (then halved)
    # [WU26, eq. 12]: an episode's support is the one that tampers the fewest devices (off: the region
    # within `hops`), searched over at most `min_budget` candidate supports
    min_tamper: bool = False
    min_budget: int = 256
    # a multiplier on At's stealth bound, whose unit is the rated accuracy (D7); Am has none (D11)
    stealth_scale: float = 1.0
    # the overload attack's load-plausibility cap tau (D16): None leaves the load changes unbounded
    load_cap: Optional[float] = None
    n_lines: int = 1  # the lines an overload episode drives at once (D17; new generation: 2)

    @property
    def band(self) -> Band:
        """The plausibility band of the in-place families: [floor, intensity]."""
        return Band(self.floor, self.intensity)


class Redistribution(NamedTuple):
    """A load-redistribution attack: the per-load-bus delta (MW), the attacked load-table positions,
    the flow change it induces on the target line (MW), the target line, and the attacker's
    interior buses (the subnetwork the false state is solved on)."""

    delta: np.ndarray
    buses: np.ndarray
    line_flow_change: float
    line: int = -1
    interior: Optional[np.ndarray] = None


class ResolvedPool(NamedTuple):
    """A pool re-solved under this generator's topology: the states [T', N, 4] and the boolean mask
    of the pool timesteps that converged (T' = mask.sum())."""

    states: np.ndarray
    converged: np.ndarray


class AttackVector(NamedTuple):
    """An attack vector h(x_false) - h(x_true) of one scan, per channel group in the scan's units."""

    node: np.ndarray  # [N, 4] |V|, P_inj, Q_inj, theta
    edge: np.ndarray  # [E, 2] P_from, Q_from
    current: Optional[np.ndarray] = None  # [E, 4] the PMU branch-current channels, when the plan has them


class LoadGoal(NamedTuple):
    """What a load-changing attack must realize at each snapshot of its window: the attack design of
    each snapshot (the loads it moves and their multipliers). The goal of At; the fewest-tamper
    search holds one support for all of them. The Am overload goal of [WU26, eqs. 24-25], a target
    line's reported flow reaching its rating, is its own goal type."""

    designs: tuple[AttackDesign, ...]  # one per snapshot, in window order
    kind: str = "load"  # the solve the fewest-tamper search applies per snapshot (MinimizeMixin.goal_state)


class FlowGoal(NamedTuple):
    """What the overload attack of [WU26, eqs. 24-25] must realize at each snapshot of its window: the
    apparent flow (MVA) that the tampered measurements carry before noise on each target branch,
    `S_{l,t} = S_true_{l,t} + (t - kappa)/T (S_max - S_true_{l,kappa+T})` (drift-free, the plan's D9),
    reaching the branch's rating at the window's end. The free injections of the support (attackable
    loads and generators) move; the fewest-tamper search holds one support for all snapshots. A goal
    drives one or more lines at once on the same support: `line` is the first target and `more` the
    others. Generated episodes drive `OverloadSettings.n_lines` lines, two by default as the paper's
    case studies do, or one (the plan's D14, D17)."""

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
    """The fewest-tamper support of one attack window [WU26, eq. 12] and how it was found."""

    support: np.ndarray  # the buses whose voltages the false state moves (sorted)
    devices: int  # devices with a channel moved beyond its noise at some snapshot (the objective); -1 when
    # no support held for the window meets every constraint and moves at least one device beyond its
    # accuracy sigma (the episode then runs on the region)
    channels: int  # channels moved beyond their noise at some snapshot, for analysis
    proven: (
        bool  # True: no other support in the area tampers fewer devices (search exhausted or at the bound)
    )
    evaluated: int  # candidate supports solved
    lower_bound: int  # devices every support must tamper (the target buses' own changed meters)
    unsolved: int = 0  # candidates whose local solve did not converge (their feasibility unknown)


class Certificate(NamedTuple):
    """How close the fewest-tamper search's attack is to the global optimum of [WU26, eq. 12] over the
    attacker's area: the search's device count (an upper bound) against the optimum of a convex
    relaxation of the same problem (a lower bound; docs/plans/RELAX_CERTIFIER_PLAN.md)."""

    upper: int  # the search's devices; -1 when it found no attack
    lower: int  # devices every attack in the area must tamper (the relaxation's bound, rounded with a margin)
    certified: bool  # verdict "certified": the bounds meet, clear of SCIP's tolerances
    status: str  # the solver's status of the mixed-integer relaxation ("optimal", or the limit it hit)
    seconds: float  # the relaxation's solve time
    cone_gap: float  # largest relative slack of the relaxed point's cones (0: every cone tight)
    mismatch: float  # largest injection mismatch, MW, between the relaxed W and its voltages (0: an AC state)
    area: np.ndarray  # the attacker's area both bounds range over
    support: np.ndarray  # the search's support (empty without an attack)
    devices: np.ndarray  # the devices the relaxation's optimum tampers
    verdict: str  # `CertifyVerdict`: "certified", "gap" or "uncertain"
    reason: str  # why the verdict is "uncertain" (empty otherwise)
    cone_lower: int  # the bound of the cone relaxation alone, which the cut families may only raise


class BoundClaim(NamedTuple):
    """What one relaxation solve proves for `certify`: a lower bound on the device count, SCIP's
    status, the relaxed point and device binaries (None and zeros when infeasible), and a doubt,
    empty when the claim is clear of SCIP's tolerances."""

    lower: int  # devices every attack must tamper, per this solve
    status: str  # SCIP's status ("optimal", "infeasible", or the limit it hit)
    x: Optional[np.ndarray]  # the relaxed variables per snapshot [T, n], None without a point
    binaries: np.ndarray  # the device binaries of the relaxed point
    doubt: str  # why the claim is not clear of the tolerances (empty when it is)
