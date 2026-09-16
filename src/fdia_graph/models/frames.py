"""What the generators pass around per scan: a measurement scan, the emitted frame with its labels,
the run's attack knobs, a finished shard record, and the two intermediate results of the physics
(a load redistribution, a re-solved pool)."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class Scan(NamedTuple):
    """One emitted measurement scan in the shard's physical units: readings and their meter masks."""

    node_x: np.ndarray  # [N, 4] |V|, P_inj, Q_inj, theta (zero where unmetered)
    node_m: np.ndarray  # [N, 4] meter mask
    edge_x: np.ndarray  # [E, 2] P_from, Q_from
    edge_m: np.ndarray  # [E, 2] meter mask


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


class FrameKnobs(NamedTuple):
    """The attack settings of one generation run, fixed for every scan."""

    intensity: float  # attack_intensity: load-shift bound of Aq/Al and the plausibility cap of Ad/As/Ar
    floor: float  # lower edge of the plausibility band (NOISE_FLOOR)
    lra_k: int  # most buses an LRA redistribution may touch
    replay_tau: Optional[int]  # Ar/As replay depth in scans, None = random lag of at least REPLAY_MIN_LAG
    reject_below_floor: bool  # shards: reject a within-noise scan so the draw loop redraws
    with_benign: bool  # streams: also emit the un-attacked twin of the scan


class Record(NamedTuple):
    """One finished shard record: the emitted scan, its labels and ids, and its two temporal features."""

    node_x: np.ndarray  # [N, 4] |V|, P_inj, Q_inj, theta (physical units), zero where unmetered
    node_m: np.ndarray  # [N, 4] meter mask
    edge_x: np.ndarray  # [E, 2] P_from, Q_from
    edge_m: np.ndarray  # [E, 2] meter mask
    y: np.ndarray  # [N] per-bus attack label
    family: int  # attack family id (0 benign)
    seq_id: int  # ramp sequence id, -1 otherwise
    timestep: int  # pool timestep the record was built on
    gap: int  # 1 for a gap (skipped scan) record, else 0
    stealthy: int  # 1 when the scan is a re-solved state
    temporal_delta: np.ndarray  # [N, 2] injection change vs the previous pool scan
    swing: np.ndarray  # [N, 2] that change as a z-score of the bus's typical recent change


class Redistribution(NamedTuple):
    """A load-redistribution attack: the per-load-bus delta (MW), the attacked load-table positions,
    and the flow change it induces on the target line (MW)."""

    delta: np.ndarray
    buses: np.ndarray
    line_flow_change: float


class ResolvedPool(NamedTuple):
    """A pool re-solved under this generator's topology: the states [T', N, 4] and the boolean mask
    of the pool timesteps that converged (T' = mask.sum())."""

    states: np.ndarray
    converged: np.ndarray
