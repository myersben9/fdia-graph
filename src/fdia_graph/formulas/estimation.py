"""Weighted least squares state estimation and its robust variants, one function per equation
[SCH70], [HAN75], [HUB64], [EST26].

The estimator classes in `fdia_graph.se` own the measurement function h(x), the chord Jacobian,
the iteration loops and their divergence guard; every algebraic step inside those loops is one
of the functions here. Each keeps the exact floating-point expression the estimator used before
it existed, so cached results reproduce bit for bit.

Shapes: m measurements, k state (or basis) coordinates, n records. `w` is a weight vector
(1/σ²) per measurement, `H` the m×k Jacobian, `Ai` the inverse normal matrix (HᵀWH)⁻¹.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..models.grid import PseudoLinks, PseudoVoltages


def normal_matrix(H: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The gain (normal) matrix of weighted least squares, G = Hᵀ W H [SCH70, part II].

    H       : [m, k] Jacobian (or Jacobian times a basis)
    w       : [m] measurement weights 1/σ²
    returns : [k, k]
    """
    return H.T @ (w[:, None] * H)


def wls_step(residual: np.ndarray, w: np.ndarray, B: np.ndarray, Ai: np.ndarray) -> np.ndarray:
    """One Gauss-Newton step of weighted least squares with a shared Jacobian [SCH70, part II].

        Δc = (Bᵀ W B)⁻¹ Bᵀ W (z − h(x))

    residual : [n, m] z − h(x) per record
    w        : [m] shared measurement weights
    B        : [m, k] the (chord) Jacobian, or Jacobian times a basis
    Ai       : [k, k] inverse normal matrix (Bᵀ W B)⁻¹
    returns  : [n, k] the step for every record
    """
    return (residual * w) @ B @ Ai.T


def wls_step_batched(residual: np.ndarray, w: np.ndarray, B: np.ndarray, Ai: np.ndarray) -> np.ndarray:
    """The same step with per-record weights, so each record has its own normal matrix.

    residual : [n, m] z − h(x) per record
    w        : [n, m] per-record measurement weights
    B        : [m, k]
    Ai       : [n, k, k] per-record inverse normal matrices
    returns  : [n, k]
    """
    return np.einsum("bij,bj->bi", Ai, (residual * w) @ B)


def weighted_objective(residual: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The weighted least-squares objective J(x) = Σ_i w_i (z_i − h_i(x))² per record [SCH70].

    residual : [n, m]
    w        : [n, m] or [m]
    returns  : [n]
    """
    return (w * residual**2).sum(axis=1)


def residual_covariance_diag(H: np.ndarray, w: np.ndarray, Ai: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonal of the residual covariance Ω = R − H G⁻¹ Hᵀ [HAN75], with R = diag(1/w).

    H       : [m, k] Jacobian
    w       : [m] measurement weights 1/σ²
    Ai      : [k, k] inverse normal matrix G⁻¹
    returns : (Ω_ii [m], R_ii [m])
    """
    R = 1.0 / w
    om = R - np.einsum("ij,jk,ik->i", H, Ai, H)
    return om, R


def critical_measurements(om: np.ndarray, R: np.ndarray, rel: float = 1e-6) -> np.ndarray:
    """A measurement is critical when its residual variance is structurally zero, Ω_ii < rel R_ii
    [HAN75]; its residual carries no bad-data information and it must never be removed.

    returns : [m] bool
    """
    return om < rel * R


def floored_covariance(om: np.ndarray, R: np.ndarray, rel: float = 1e-12) -> np.ndarray:
    """Ω_ii floored at rel R_ii so a critical measurement's normalized residual stays finite."""
    return np.maximum(om, rel * R)


def normalized_residual(residual: np.ndarray, om: np.ndarray) -> np.ndarray:
    """The normalized residual r_N,i = |z_i − h_i(x̂)| / √Ω_ii [HAN75].

    residual : [n, m] z − h(x̂)
    om       : [m] residual covariance diagonal (floored)
    returns  : [n, m] non-negative
    """
    return np.abs(residual) / np.sqrt(om)[None, :]


def huber_weights(r_n: np.ndarray, c: float, eps: float = 1e-9) -> np.ndarray:
    """The Huber weight a_i = min(1, c / |r_N,i|) [HUB64]: full weight inside the band, decaying
    outside it, so a gross error is bounded in influence.

    r_n     : [n, m] normalized residuals (non-negative)
    c       : the band half-width in normalized units
    eps     : floor on |r_N,i| against division by zero
    returns : [n, m] in (0, 1]
    """
    return np.minimum(1.0, c / np.maximum(r_n, eps))


def whitened_svd_basis(X: np.ndarray, rank_frac: float) -> tuple[int, np.ndarray]:
    """The learned operating-point prior [EST26]: whiten the benign states per coordinate, take
    the SVD, keep the leading rank_frac fraction of directions, un-whiten and re-orthonormalize.

    X         : [n, d] benign training states
    rank_frac : fraction of d to keep, in (0, 1]
    returns   : (K, V_K [d, K]) with V_Kᵀ V_K = I; the estimate is restricted to x = mean + V_K c

    The whitened basis is otherwise catastrophically ill conditioned near full rank, which is
    why the QR re-orthonormalization is part of the formula.
    """
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-9)  # angle and voltage differ ~10x in scale
    _, _, Vt = np.linalg.svd((X - mean) / std, full_matrices=False)
    K = max(1, int(round(rank_frac * X.shape[1])))
    VK = np.linalg.qr(Vt[:K].T * std[:, None])[0]  # un-whiten, re-orthonormalize
    return K, VK


def gate_weights(w: np.ndarray, flags: np.ndarray, incidence: list[np.ndarray], factor: float) -> np.ndarray:
    """Localization-gated weights [EST26]: every meter incident to a flagged bus is down-weighted
    by `factor` before the solve, so the prior supplies the state there.

    w         : [m] shared measurement weights
    flags     : [n, N] bool, the flagged buses per record
    incidence : per bus, the masked-measurement indices touching it (see projection.bus_incidence)
    factor    : in (0, 1]
    returns   : [n, m] per-record weights
    """
    out = np.repeat(w[None, :], flags.shape[0], axis=0)
    for b, ix in enumerate(incidence):
        if len(ix):
            hit = flags[:, b]
            out[np.ix_(hit, ix)] *= factor
    return out


def accuracy_class_sigma(
    mean_abs: np.ndarray, cls: np.ndarray, relative: np.ndarray, floor: float
) -> np.ndarray:
    """Each meter's error std from its accuracy class [ASP14]: a relative class scales the meter's
    mean absolute reading, an absolute class is the std itself,

        sigma_i = c_i |z_i| + floor   (relative: power injections and flows)
        sigma_i = c_i                 (absolute: voltage magnitude and angle)

    Equipment data and measurements only; no state enters.

    mean_abs : [m] mean |z_i| over the calibration scans
    cls      : [m] accuracy class per meter
    relative : [m] bool, True where the class is relative
    floor    : absolute floor of a relative meter's std (its reading's units)
    returns  : [m]
    """
    mean_abs, cls = np.asarray(mean_abs, np.float64), np.asarray(cls, np.float64)
    return np.where(np.asarray(relative, bool), cls * mean_abs + floor, cls)


def pmu_pseudo_links(
    pmu: np.ndarray,
    current_m: np.ndarray,
    f_bus: np.ndarray,
    t_bus: np.ndarray,
    Yf: np.ndarray,
    Yt: np.ndarray,
) -> PseudoLinks:
    """The branch ends where [WU26, eq. (3)] places a pseudo voltage phasor: a PMU at the metered end
    reads the bus voltage and the branch current, which fix the voltage at the far end when the far
    bus has no PMU of its own.

    pmu        : [N] bool, the PMU buses
    current_m  : [E, 4] the PMU branch-current mask (`CURRENT` columns)
    f_bus, t_bus : [E] the branch terminals, in the same bus order as the columns of Yf and Yt
    Yf, Yt     : [E, N] complex branch admittances, I_f = Yf V and I_t = Yt V
    returns    : `PseudoLinks`, one row per usable metered end
    """
    pmu = np.asarray(pmu, bool)
    cm = np.asarray(current_m) > 0
    f_bus, t_bus = np.asarray(f_bus, np.int64), np.asarray(t_bus, np.int64)
    # a PMU at the from end (Yf row) or at the to end (Yt row), the far bus without a PMU of its own
    ef = np.where(cm[:, 0] & pmu[f_bus] & ~pmu[t_bus])[0]
    et = np.where(cm[:, 2] & pmu[t_bus] & ~pmu[f_bus])[0]
    near = np.concatenate([f_bus[ef], t_bus[et]])
    far = np.concatenate([t_bus[ef], f_bus[et]])
    y_nn = np.concatenate([Yf[ef, f_bus[ef]], Yt[et, t_bus[et]]]).astype(complex)
    y_nf = np.concatenate([Yf[ef, t_bus[ef]], Yt[et, f_bus[et]]]).astype(complex)
    edge = np.concatenate([ef, et])
    end = np.concatenate([np.zeros(len(ef), np.int64), np.ones(len(et), np.int64)])
    ok = y_nf != 0  # an open branch couples nothing
    return PseudoLinks(near[ok], far[ok], edge[ok], end[ok], y_nn[ok], y_nf[ok])


def _rotation(c: np.ndarray) -> np.ndarray:
    """Complex multiplication by c as a real 2x2 matrix on (Re, Im): [..., 2, 2]."""
    return np.stack([np.stack([c.real, -c.imag], -1), np.stack([c.imag, c.real], -1)], -2)


def pmu_pseudo_voltages(
    links: PseudoLinks,
    n_bus: int,
    v: np.ndarray,
    theta: np.ndarray,
    current: np.ndarray,
    sigma_v: Union[float, np.ndarray],
    sigma_theta: Union[float, np.ndarray],
    sigma_i: Union[float, np.ndarray],
) -> PseudoVoltages:
    """The pseudo voltage phasors of [WU26, eq. (3)] at the far end of every PMU-metered branch,
    averaged over the links that reach a bus, with their first-order propagated noise.

    From the metered end n of a branch to its far end f (pi model, taps and shifts included):

        I_n = y_nn V_n + y_nf V_f   =>   V_f = (I_n - y_nn V_n) / y_nf

    which is eq. (3) written for a general branch (y_nn = y_s + j b/2 and y_nf = -y_s on a line
    without a tap). A bus reached by k links takes the mean of its k estimates.

    Noise: with independent errors on the PMU readings (|V_n| and its angle with standard deviations
    sigma_v and sigma_theta, Re and Im of I_n each with its sigma_i), the first-order error of one
    estimate as a real 2-vector is

        dV_f = R(1/y_nf) dI_n - R(y_nn/y_nf) P_n (d|V_n|, dtheta_n)

    with R(c) the real 2x2 matrix of multiplying by c and P_n the polar-to-rectangular Jacobian at
    V_n. The mean over k links sums the current terms as independent and groups the voltage terms
    by PMU bus (two branches from one PMU share its voltage error), giving a 2x2 covariance C_f;
    the polar variances follow through the rectangular-to-polar gradients at the estimate,

        var|V_f| = g_r C_f g_r^T,   g_r = (x, y)/|V_f|
        var theta_f = g_t C_f g_t^T,   g_t = (-y, x)/|V_f|^2          (V_f = x + j y)

    so with a round error on V_f of complex variance s^2 = tr C_f, var|V_f| = s^2/2 and
    var theta_f = s^2/(2|V_f|^2).

    links      : `pmu_pseudo_links`
    n_bus      : N
    v, theta   : [n, N] or [N] bus |V| (pu) and angle (rad), read at the PMU buses
    current    : [n, E, 4] or [E, 4] the PMU branch-current readings (`CURRENT` columns, pu)
    sigma_v, sigma_theta : [N] or [n, N] the standard deviations of the PMU |V| and angle
    sigma_i    : [n, E, 4] or [E, 4] the standard deviation of every current channel
    returns    : `PseudoVoltages`, NaN where no link reaches
    """
    single = np.ndim(v) == 1
    v, theta = np.atleast_2d(v).astype(float), np.atleast_2d(theta).astype(float)
    cur = np.asarray(current, float).reshape(v.shape[0], -1, 4)
    s_i = np.broadcast_to(np.asarray(sigma_i, float), cur.shape)
    s_v = np.broadcast_to(np.asarray(sigma_v, float), v.shape)
    s_t = np.broadcast_to(np.asarray(sigma_theta, float), v.shape)
    n, N = v.shape[0], n_bus
    near, far, e, re = links.near, links.far, links.edge, 2 * links.end
    k = np.bincount(far, minlength=N).astype(float)
    w = 1.0 / np.maximum(k[far], 1.0)  # [J] each link's share of its bus's mean
    I_n = cur[:, e, re] + 1j * cur[:, e, re + 1]
    V_n = v[:, near] * np.exp(1j * theta[:, near])
    V_f = np.zeros((n, N), complex)
    np.add.at(V_f.T, far, (w * (I_n - links.y_nn * V_n) / links.y_nf).T)  # .T: sum over the bus axis
    # the current terms, one independent source per link
    Ri = _rotation(1.0 / links.y_nf)  # [J, 2, 2]
    CI = np.zeros((n, len(far), 2, 2))
    CI[..., 0, 0], CI[..., 1, 1] = s_i[:, e, re] ** 2, s_i[:, e, re + 1] ** 2
    C = np.zeros((n, N, 2, 2))
    Cb = C.transpose(1, 0, 2, 3)  # a view with the bus axis first, for np.add.at
    np.add.at(Cb, far, ((w**2)[:, None, None] * (Ri @ CI @ np.swapaxes(Ri, -1, -2))).transpose(1, 0, 2, 3))
    # the voltage terms, grouped by (far bus, PMU bus): one source per PMU voltage
    pair, g = np.unique(far * N + near, return_inverse=True)
    A = np.zeros((len(pair), 2, 2))
    np.add.at(A, g, -w[:, None, None] * _rotation(links.y_nn / links.y_nf))
    g_far, g_near = pair // N, pair % N
    c, s, r = np.cos(theta[:, g_near]), np.sin(theta[:, g_near]), v[:, g_near]
    P = np.stack([np.stack([c, -r * s], -1), np.stack([s, r * c], -1)], -2)  # [n, G, 2, 2]
    D = np.zeros((n, len(pair), 2, 2))
    D[..., 0, 0], D[..., 1, 1] = s_v[:, g_near] ** 2, s_t[:, g_near] ** 2
    AP = A @ P
    np.add.at(Cb, g_far, (AP @ D @ np.swapaxes(AP, -1, -2)).transpose(1, 0, 2, 3))
    reached = k > 0
    V_f[:, ~reached] = np.nan
    x, y, mag = np.real(V_f), np.imag(V_f), np.abs(V_f)
    g_r = np.stack([x, y], -1) / mag[..., None]
    g_t = np.stack([-y, x], -1) / (mag**2)[..., None]
    var_v = np.einsum("nba,nbac,nbc->nb", g_r, C, g_r)
    var_t = np.einsum("nba,nbac,nbc->nb", g_t, C, g_t)
    th = np.angle(V_f)
    if single:
        return PseudoVoltages(mag[0], th[0], var_v[0], var_t[0], reached)
    return PseudoVoltages(mag, th, var_v, var_t, reached)
