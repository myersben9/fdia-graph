"""What the generator passes around per scan: a measurement scan, the emitted frame with its labels,
the run's attack knobs, the design and target of one attack, and the two intermediate results of the
physics (a load redistribution, a re-solved pool)."""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import numpy as np


class Scan(NamedTuple):
    """One emitted measurement scan in the shard's physical units: readings and their meter masks."""

    node_x: np.ndarray  # [N, 4] |V|, P_inj, Q_inj, theta (zero where unmetered)
    node_m: np.ndarray  # [N, 4] meter mask
    edge_x: np.ndarray  # [E, 2] P_from, Q_from
    edge_m: np.ndarray  # [E, 2] meter mask


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
    subnetwork. The local false state is solved from it (engine.records.stealthy_state)."""

    targets: np.ndarray  # positions in the generator's load table (not bus numbers)
    mult: Union[float, np.ndarray]  # load multiplier: a scalar for the ramp, one per target otherwise
    interior: Optional[np.ndarray] = None  # the attacker's subnetwork; None: the region around targets


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
