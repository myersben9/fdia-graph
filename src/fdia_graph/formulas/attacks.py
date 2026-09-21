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


def operating_limits(v_case: np.ndarray, p_lim: np.ndarray, q_lim: np.ndarray) -> OperatingLimits:
    """The constraints of [WU26, eqs. 21-23] for one system, verbatim from the case data: the
    per-bus voltage limits `v_case` [N, 2] and the per-bus generator limits `p_lim`, `q_lim`
    [N, 2] (±inf without a generator)."""
    return OperatingLimits(v_case[:, 0], v_case[:, 1], p_lim[:, 0], p_lim[:, 1], q_lim[:, 0], q_lim[:, 1])


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
    Xa: np.ndarray, Xt: np.ndarray, gen_true: np.ndarray, load_delta: np.ndarray, limits: OperatingLimits
) -> bool:
    """Whether the false state Xa [N, 4] satisfies [WU26, eqs. 21-23] given the true state Xt, the
    true generator output [N, 2] and the load change the attacker pretends per bus `load_delta`
    [N] (MW): every |V| inside its bus limits, and every generator's implied output, the injection
    change not explained by that load change, P_gen - (ΔP_inj - ΔP_load) and Q_gen - ΔQ_inj,
    inside its limits. Where the true state itself is outside a limit (a case that runs below its
    own minimum, a pool whose generators exceed nameplate) the false state may not make it worse:
    the bound at that bus is the true value.

        v_lo_i = min(V_i^min, |V_i^true|),  v_hi_i = max(V_i^max, |V_i^true|),  likewise P_gen, Q_gen
    """
    V, Vt = Xa[:, NODE.v], Xt[:, NODE.v]
    if np.any(V < np.minimum(limits.v_lo, Vt) - LIMIT_TOL_V) or np.any(
        V > np.maximum(limits.v_hi, Vt) + LIMIT_TOL_V
    ):
        return False
    p = gen_true[:, 0] - (Xa[:, NODE.p_inj] - Xt[:, NODE.p_inj] - load_delta)
    q = gen_true[:, 1] - (Xa[:, NODE.q_inj] - Xt[:, NODE.q_inj])
    return bool(
        np.all(p >= np.minimum(limits.p_lo, gen_true[:, 0]) - LIMIT_TOL_PQ)
        and np.all(p <= np.maximum(limits.p_hi, gen_true[:, 0]) + LIMIT_TOL_PQ)
        and np.all(q >= np.minimum(limits.q_lo, gen_true[:, 1]) - LIMIT_TOL_PQ)
        and np.all(q <= np.maximum(limits.q_hi, gen_true[:, 1]) + LIMIT_TOL_PQ)
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
