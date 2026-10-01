"""The pieces of the convex relaxation that certifies the fewest-tamper attack
(docs/plans/RELAX_CERTIFIER_PLAN.md): the voltage box, the big-M constants derived from it, the
sectors that outer-approximate the exterior of a circle, the cone gap that says how tight a
relaxed point's pairwise cones are, and the roundoff slack that keeps the search's float32 noise
thresholds reproduced conservatively. numpy only; the model itself is built in
`engine/attacks/certify.py`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

SECTORS = (
    24  # sectors of the flow goal's |S| >= target disjunction: the cut sits at cos(pi / 24) = 0.991 of it
)
FLOAT32_EPS = float(np.finfo(np.float32).eps)  # 2^-23, the spacing of float32 numbers at 1
ROUNDOFF_FLOOR = 1e-6  # absolute slack for the relaxation's own float64 arithmetic


def roundoff_slack(sigma: npt.ArrayLike, reading: npt.ArrayLike) -> np.ndarray:
    """How far above its noise threshold `sigma` a channel's exact attack value may lie while the
    fewest-tamper search still counts it within noise, per channel (a rule of ours, so the relaxation
    never cuts the search's own attack). The search rounds readings to float32 (`_Window`: a flow as
    fl32(h(x_false)) - fl32(h(x_true)), a node or current channel as the float64 difference rounded
    once), so the value it compares with sigma is off by at most eps32/2 (|h_true| + |h_false| + |a|);
    within noise |a| <= sigma and |h_false| <= |h_true| + |a|, which bounds the error by
    eps32 (|h_true| + sigma) to first order. The slack doubles that and adds a floor:

        slack = 2 eps32 (|reading| + sigma) + 1e-6

    `reading` is the true reading the search rounds (0 where only the difference is rounded)."""
    return (
        2.0 * FLOAT32_EPS * (np.abs(np.asarray(reading, float)) + np.asarray(sigma, float)) + ROUNDOFF_FLOOR
    )


def step_roundoff_slack(
    now: tuple[npt.ArrayLike, npt.ArrayLike],
    before: tuple[npt.ArrayLike, npt.ArrayLike],
    step_sigma: npt.ArrayLike,
) -> np.ndarray:
    """How far above its step threshold `step_sigma` a channel's exact step between two snapshots may
    lie while the search, which compares float32 attack values (`_Window`: a_t - a_(t-1), each rounded
    as `roundoff_slack` says, then subtracted in float32), still finds it within the stealth bound (a
    rule of ours). Unlike a count, a step is taken between attack values of any size, so each endpoint's
    error scales with the most |a| can be at its snapshot, not with sigma: `now` and `before` are
    (that bound, the true reading the search rounds) per channel, and an endpoint the search and the
    relaxation share exactly (the frame before the window, given as data) passes zeros.

        slack = roundoff_slack(bound_now, reading_now) + roundoff_slack(bound_before, reading_before)
                + roundoff_slack(step_sigma, 0)
    """
    return (
        roundoff_slack(now[0], now[1])
        + roundoff_slack(before[0], before[1])
        + roundoff_slack(step_sigma, 0.0)
    )


def voltage_box(
    v_true: np.ndarray, v_lo: np.ndarray, v_hi: np.ndarray, tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds (lo, hi) on W_ii = |V_i|^2 from [WU26 eq. 21] as the search applies it
    (`formulas.attacks.within_limits`): the case limits widened to the true value and by `tol`.

        W_ii in [(min(v_lo, |V_true|) - tol)^2, (max(v_hi, |V_true|) + tol)^2]
    """
    lo = np.maximum(np.minimum(v_lo, v_true) - tol, 0.0)
    hi = np.maximum(v_hi, v_true) + tol
    return lo**2, hi**2


def big_m(rows: np.ndarray, radius: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """The big-M of each linear channel a = A (x - x0): the most |a| can exceed its noise sigma
    anywhere in the variables' box, where `radius` is each variable's largest distance from its true
    value x0 inside the box. No feasible point is cut off, since |a| <= |A| r.

        M_m = max(sum_k |A_mk| r_k - sigma_m, 0)
    """
    return np.maximum(np.abs(rows) @ radius - sigma, 0.0)


def sector_cuts(k: int = SECTORS) -> tuple[np.ndarray, float]:
    """(unit directions [k, 2], cos(pi / k)) of the disjunction that replaces |S| >= target: every
    point S with |S| >= target lies in some sector k, where u_k . S >= target cos(pi / k)."""
    phi = 2.0 * np.pi * np.arange(k) / k
    return np.stack([np.cos(phi), np.sin(phi)], axis=1), float(np.cos(np.pi / k))


def cone_gap(w_i: np.ndarray, w_j: np.ndarray, re: np.ndarray, im: np.ndarray) -> float:
    """The largest relative slack of the cones |W_ij|^2 <= W_ii W_jj over a relaxed point: 0 when
    every pairwise cone is tight, up to 1. A tight cone makes each 2x2 block of W rank one, not W as a
    whole: on a meshed network the angles around a cycle need not add up, so a zero gap does not make
    the point an AC voltage (`_Relaxation.mismatch` says whether it is).

        gap = max (W_ii W_jj - |W_ij|^2) / (W_ii W_jj)
    """
    if len(w_i) == 0:
        return 0.0
    prod = np.maximum(np.asarray(w_i) * np.asarray(w_j), 1e-12)
    return float(np.max((prod - np.asarray(re) ** 2 - np.asarray(im) ** 2) / prod))


def angle_bound(rho: np.ndarray, v0: np.ndarray) -> np.ndarray:
    """The most a voltage's angle can move when the voltage stays within `rho` of its true value v0
    (|V - V0| <= rho): asin(rho / |V0|), and pi / 2 (no bound worth using) once rho reaches |V0|.

        |angle(V) - angle(V0)| <= asin(rho / |V0|)
    """
    ratio = np.asarray(rho, float) / np.maximum(np.abs(np.asarray(v0)), 1e-12)
    return np.where(ratio < 1.0, np.arcsin(np.clip(ratio, 0.0, 1.0)), np.pi / 2)


def polygon_radius(support: np.ndarray, k: int) -> float:
    """The largest |p| over the points p with u_i . p <= support_i for k unit directions spaced evenly
    around the circle: some direction lies within pi / k of p, so u_i . p >= |p| cos(pi / k).

        |p| <= max_i support_i / cos(pi / k)
    """
    return float(np.max(support) / np.cos(np.pi / k))


def cos_envelope(m: float) -> tuple[float, float]:
    """(c, lo) of the convex envelope of cos(phi) on |phi| <= m < pi / 2 [QC relaxation]: the concave
    cos lies under the concave parabola and over the chord at the bounds.

        cos(phi) <= 1 - c phi^2,  c = (1 - cos m) / m^2;   cos(phi) >= cos m
    """
    if m < 1e-9:
        return 0.5, 1.0
    return (1.0 - float(np.cos(m))) / m**2, float(np.cos(m))


def sin_envelope(m: float) -> tuple[float, float]:
    """(slope, offset) of the envelope of sin(phi) on |phi| <= m <= pi / 2 [QC relaxation]: sin is
    concave above zero and convex below, so the tangents at +-m/2 bound it.

        sin(phi) <= slope (phi - m/2) + offset,  sin(phi) >= slope (phi + m/2) - offset,
        slope = cos(m/2),  offset = sin(m/2)
    """
    return float(np.cos(m / 2)), float(np.sin(m / 2))


def mccormick(a: tuple[float, float], b: tuple[float, float]) -> list[tuple[str, float, float, float]]:
    """The McCormick envelope of p = a b for a in [a_lo, a_hi], b in [b_lo, b_hi]: four rows
    (sense, coefficient of a, coefficient of b, constant) meaning p >= or <= ca a + cb b + c0.

        p >= a_lo b + b_lo a - a_lo b_lo,   p >= a_hi b + b_hi a - a_hi b_hi
        p <= a_hi b + b_lo a - a_hi b_lo,   p <= a_lo b + b_hi a - a_lo b_hi
    """
    (al, au), (bl, bu) = a, b
    return [
        (">=", bl, al, -al * bl),
        (">=", bu, au, -au * bu),
        ("<=", bl, au, -au * bl),
        ("<=", bu, al, -al * bu),
    ]
