"""Attack shapes defined by the dataset [DAT26] that are pure functions of their parameters."""

from __future__ import annotations

import numpy as np

from ..models.frames import OperatingLimits
from ..models.grid import NODE

# Slack under a limit, below what a meter resolves (0.0012 pu on |V|, 1e-3 MW on P and Q): a true
# state outside a limit is itself the bound (a regulated bus at its setpoint), and a false state
# that moves it by less than the meter floor violates nothing an operator could see.
LIMIT_TOL_V = 1e-3  # pu
LIMIT_TOL_PQ = 1e-3  # MW, MVAr


def operating_limits(
    v_case: np.ndarray,
    p_lim: np.ndarray,
    q_lim: np.ndarray,
    X: np.ndarray,
    base: tuple[np.ndarray, np.ndarray],
) -> OperatingLimits:
    """The constraints of [WU26, eqs. 21-23] for one system: the case's per-bus voltage limits
    `v_case` [N, 2] verbatim, and the generator limits `p_lim`, `q_lim` [N, 2] widened per bus to
    the range the benign pool X [T, N, 4] spans. The pools were built with generation scaled by the
    load factor and nameplate never enforced, so a generator's benign output is the range the grid
    actually ran it over, and the nameplate alone would refuse the state the pool already holds.
    `base` = (load_base, gen_base) for `generator_output`.

        p_lo_i = min(P_i^min, min_t P_gen,i,t),   p_hi_i = max(P_i^max, max_t P_gen,i,t),   likewise Q
    """
    gen = generator_output(X, *base)  # [T, N, 2]
    return OperatingLimits(
        v_case[:, 0],
        v_case[:, 1],
        np.minimum(p_lim[:, 0], gen[:, :, 0].min(axis=0)),
        np.maximum(p_lim[:, 1], gen[:, :, 0].max(axis=0)),
        np.minimum(q_lim[:, 0], gen[:, :, 1].min(axis=0)),
        np.maximum(q_lim[:, 1], gen[:, :, 1].max(axis=0)),
    )


def generator_output(X: np.ndarray, load_base: np.ndarray, gen_base: np.ndarray) -> np.ndarray:
    """The generator output [..., N, 2] (P MW, Q MVAr) behind pool states X [..., N, 4], exact for
    the pool construction (profiles.generate_states): load and generation at a bus scale by one
    factor s, so

        s = P_inj / (P_load_base - P_gen_base),   P_gen = s P_gen_base,   Q_gen = s Q_load_base - Q_inj

    with P_inj, Q_inj load-positive; a bus whose base net injection is zero keeps s = 1."""
    net = load_base[:, 0] - gen_base[:, 0]
    safe = np.where(np.abs(net) > 1e-9, net, 1.0)
    s = np.where(np.abs(net) > 1e-9, X[..., NODE.p_inj] / safe, 1.0)
    return np.stack([s * gen_base[:, 0], s * load_base[:, 1] - X[..., NODE.q_inj]], axis=-1)


def within_limits(
    Xa: np.ndarray,
    Xt: np.ndarray,
    gen_true: np.ndarray,
    load_delta: np.ndarray,
    limits: OperatingLimits,
    interior: np.ndarray,
) -> bool:
    """Whether the false state Xa [N, 4] satisfies [WU26, eqs. 21-23] given the true state Xt, the
    true generator output [N, 2], the load change the attacker pretends per bus `load_delta` [N]
    (MW) and the attacker's `interior`: every |V| inside its bus limits, where a bus the true state
    already holds outside a limit may not be made worse (the bound there is the true value), and,
    for the generators of the attacked subnetwork (the paper's constraint ranges over the
    generators of the attack area; a boundary bus is outside it, its voltage held true and its
    injection whatever balances the region), the implied output P_gen - (ΔP_inj - ΔP_load) and
    Q_gen - ΔQ_inj inside its limits.

        v_lo_i = min(V_i^min, |V_i^true|),  v_hi_i = max(V_i^max, |V_i^true|)
    """
    V, Vt = Xa[:, NODE.v], Xt[:, NODE.v]
    if np.any(V < np.minimum(limits.v_lo, Vt) - LIMIT_TOL_V) or np.any(
        V > np.maximum(limits.v_hi, Vt) + LIMIT_TOL_V
    ):
        return False
    I_ = np.asarray(interior, int)
    p = gen_true[I_, 0] - (Xa[I_, NODE.p_inj] - Xt[I_, NODE.p_inj] - load_delta[I_])
    q = gen_true[I_, 1] - (Xa[I_, NODE.q_inj] - Xt[I_, NODE.q_inj])
    return bool(
        np.all(p >= limits.p_lo[I_] - LIMIT_TOL_PQ)
        and np.all(p <= limits.p_hi[I_] + LIMIT_TOL_PQ)
        and np.all(q >= limits.q_lo[I_] - LIMIT_TOL_PQ)
        and np.all(q <= limits.q_hi[I_] + LIMIT_TOL_PQ)
    )


def ramp_profile(i: int, rise: int, hold: int, rate_up: float, rate_down: float) -> float:
    """Deviation of the slow ramp At at step i of a sequence [DAT26]: rise at rate_up for `rise`
    steps to the peak, hold there for `hold` steps, then return at rate_down and never below zero.
    The direction (surge or dip) is applied by the caller as 1 +/- dev.

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
