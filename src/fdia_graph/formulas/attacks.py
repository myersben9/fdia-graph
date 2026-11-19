"""Attack shapes defined by the dataset [DAT26] that are pure functions of their parameters."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..constants.channels import CurrentChannel, NodeChannel
from ..constants.grid import UNRATED_MVA
from ..constants.numerics import LIMIT_TOL_PQ, LIMIT_TOL_V, ZERO_TOL
from ..models.frames import OperatingLimits


def operating_limits(
    v_case: np.ndarray,
    p_lim: np.ndarray,
    q_lim: np.ndarray,
    gen: np.ndarray,
) -> OperatingLimits:
    """The limits of [WU26 eqs. 21-23]: the case's voltage limits `v_case` [N, 2] as given, and the
    generator limits widened to the range the benign pool's generator output `gen` [T, N, 2] spans (the
    pool never enforces nameplate): p_lo_i = min(P_i^min, min_t P_gen,i,t), p_hi_i = max(P_i^max, max_t P_gen,i,t), likewise Q."""
    return OperatingLimits(
        v_case[:, 0],
        v_case[:, 1],
        np.minimum(p_lim[:, 0], gen[:, :, 0].min(axis=0)),
        np.maximum(p_lim[:, 1], gen[:, :, 0].max(axis=0)),
        np.minimum(q_lim[:, 0], gen[:, :, 1].min(axis=0)),
        np.maximum(q_lim[:, 1], gen[:, :, 1].max(axis=0)),
    )


def generator_output(X: np.ndarray, load_base: np.ndarray, gen_base: np.ndarray) -> np.ndarray:
    """The generator output [..., N, 2] (P MW, Q MVAr) behind pool states X [..., N, 4] [DAT26]. The pool
    scales a bus's load and generation by one factor s (P_inj, Q_inj are load-positive; s = 1 where the
    base net is 0):
    s = P_inj / (P_load_base - P_gen_base),  P_gen = s P_gen_base,  Q_gen = s Q_load_base - Q_inj."""
    net = load_base[:, 0] - gen_base[:, 0]
    safe = np.where(np.abs(net) > ZERO_TOL, net, 1.0)
    s = np.where(np.abs(net) > ZERO_TOL, X[..., NodeChannel.P_INJ] / safe, 1.0)
    return np.stack([s * gen_base[:, 0], s * load_base[:, 1] - X[..., NodeChannel.Q_INJ]], axis=-1)


def element_loads(bus_p: np.ndarray, load_bus: np.ndarray, p0: np.ndarray) -> np.ndarray:
    """A bus's scan load (MW, from `engine.physics.LoadSplit.load`) split over its load elements by their base shares
    [DAT26]: P_e = P_bus(e) · P0_e / Σ_{e' at bus(e)} P0_e'. Returns [n_loads] MW; an element on a
    bus whose base loads sum to zero gets zero."""
    tot = np.zeros(len(bus_p))
    np.add.at(tot, load_bus, p0)
    t = tot[load_bus]
    return bus_p[load_bus] * np.where(t != 0, p0 / np.where(t != 0, t, 1.0), 0.0)


def within_limits(
    x_false: np.ndarray,
    x_true: np.ndarray,
    gen_true: np.ndarray,
    load_delta: np.ndarray,
    limits: OperatingLimits,
    interior: np.ndarray,
) -> bool:
    """Whether the false state x_false [N, 4] satisfies [WU26 eqs. 21-23]: every |V| inside v_lo_i =
    min(V_i^min, |V_i^true|), v_hi_i = max(V_i^max, |V_i^true|), and, at the interior's generators, the
    implied output P_gen - (ΔP_inj - ΔP_load) and Q_gen - ΔQ_inj inside its limits. `load_delta` is MW."""
    V, Vt = x_false[:, NodeChannel.V], x_true[:, NodeChannel.V]
    if np.any(V < np.minimum(limits.v_lo, Vt) - LIMIT_TOL_V):
        return False
    if np.any(V > np.maximum(limits.v_hi, Vt) + LIMIT_TOL_V):
        return False
    I_ = np.asarray(interior, int)
    p = gen_true[I_, 0] - (x_false[I_, NodeChannel.P_INJ] - x_true[I_, NodeChannel.P_INJ] - load_delta[I_])
    q = gen_true[I_, 1] - (x_false[I_, NodeChannel.Q_INJ] - x_true[I_, NodeChannel.Q_INJ])
    return bool(
        np.all(p >= limits.p_lo[I_] - LIMIT_TOL_PQ)
        and np.all(p <= limits.p_hi[I_] + LIMIT_TOL_PQ)
        and np.all(q >= limits.q_lo[I_] - LIMIT_TOL_PQ)
        and np.all(q <= limits.q_hi[I_] + LIMIT_TOL_PQ)
    )


def ramp_profile(i: int, rise: int, hold: int, rate_up: float, rate_down: float) -> float:
    """Deviation of the slow ramp At at step i [DAT26]: rise, hold, then return, never below zero.
    The caller applies the direction (surge or dip) as 1 +/- dev.

        dev(i) = rate_up * i                                  for i < rise
               = rate_up * rise                               for rise <= i < rise + hold
               = max(0, rate_up * rise - rate_down * (i - rise - hold))   after
    """
    peak = rate_up * rise
    if i < rise:
        return rate_up * i
    if i < rise + hold:
        return peak
    return max(0.0, peak - rate_down * (i - rise - hold))


def tampered_channels(
    a_node: np.ndarray,
    a_edge: np.ndarray,
    sig_node: np.ndarray,
    sig_edge: np.ndarray,
    node_m: np.ndarray,
    edge_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks [N, 4] and [E, 2] of the metered channels an attack vector a = h(x_false) -
    h(x_true) moves by more than their noise, the l0 count of [WU26 eq. 12]: tampered_m = metered_m
    and |a_m| > sigma_m. Sigma is the rated accuracy (`NoiseModel.sigmas`), not the jitter."""
    node = (np.abs(a_node) > sig_node) & (np.asarray(node_m) > 0)
    edge = (np.abs(a_edge) > sig_edge) & (np.asarray(edge_m) > 0)
    return node, edge


def tampered_devices(
    node_mask: np.ndarray,
    edge_mask: np.ndarray,
    pmu_bus: np.ndarray,
    from_bus: np.ndarray,
    current_mask: Optional[np.ndarray] = None,
    to_bus: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Sorted unique ids of the devices holding the tampered channels, the unit [WU26] counts: the
    SCADA terminal of bus b is b, the PMU of bus b is N + b.

        SCADA terminal of bus b : P and Q injection at b, the flows metered at b (the from end), and
                                  |V| and angle at b when b has no PMU
        PMU of bus b            : |V| and angle at b, and the branch currents at b's end of each
                                  branch [WU26 eqs. 19-20] (`current_mask` [E, 4], with `to_bus`)
    """
    node_mask = np.asarray(node_mask, bool)
    N = node_mask.shape[0]
    pmu = np.asarray(pmu_bus, bool)
    power = node_mask[:, NodeChannel.P_INJ] | node_mask[:, NodeChannel.Q_INJ]
    voltage = node_mask[:, NodeChannel.V] | node_mask[:, NodeChannel.THETA]
    scada = power | (voltage & ~pmu)
    flow_buses = np.asarray(from_bus)[np.asarray(edge_mask, bool).any(axis=1)]
    ids = [np.where(scada)[0], flow_buses, N + np.where(voltage & pmu)[0]]
    if current_mask is not None and to_bus is not None:
        cm = np.asarray(current_mask, bool)
        from_end = cm[:, CurrentChannel.RE_FROM] | cm[:, CurrentChannel.IM_FROM]
        to_end = cm[:, CurrentChannel.RE_TO] | cm[:, CurrentChannel.IM_TO]
        ids += [N + np.asarray(from_bus)[from_end], N + np.asarray(to_bus)[to_end]]
    return np.unique(np.concatenate(ids).astype(np.int64))


def branch_ratings(ends: list[tuple[int, int]], rows: list[tuple[int, int, float]]) -> np.ndarray:
    """Each branch's thermal rating [E] (MVA), the S_max of [WU26 eq. 25], from (from bus, to bus,
    rate_a) rows matched by end buses in either order; `ends` are in the table's bus numbering.
    Parallel rows give the smallest rating; no row, or MATPOWER's 9,900 MVA placeholder, gives NaN."""
    table: dict[frozenset[int], float] = {}
    for f, t, r in rows:
        key = frozenset((int(f), int(t)))
        table[key] = min(r, table.get(key, np.inf))
    rating = np.array([table.get(frozenset((int(a), int(b))), np.nan) for a, b in ends], np.float64)
    rating[rating >= UNRATED_MVA] = np.nan
    return rating
