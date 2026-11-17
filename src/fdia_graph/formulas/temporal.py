"""The temporal features of a scan [FED26]: the one-step injection change and its z-score against
the bus's typical recent change, and the recent-change scale itself."""

from __future__ import annotations

import numpy as np


def wrap_degrees(d: np.ndarray) -> np.ndarray:
    """A difference of two angle readings, degrees, wrapped to (-180, 180]: readings either side of
    the +-180 wrap differ by almost 360 while the angles themselves barely moved.

        wrapped = -((-d + 180) mod 360 - 180)

    d       : differences of angles, degrees, any shape
    returns : the same shape and dtype, each value in (-180, 180]
    """
    return -((-d + 180.0) % 360.0 - 180.0)


def recent_change_scale(X: np.ndarray, window: int, n_bus: int) -> np.ndarray:
    """Per-timestep typical recent change of every bus's injections [FED26].

        scale[t] = std over the last `window` scans before t of the scan-to-scan |change| in [P_inj, Q_inj],
                   plus 1e-3 (and 1e-3 alone where fewer than three changes are available)

    X       : [T, N, 4] a series of scans in [|V|, P_inj, Q_inj, theta] order (a timeline's observed
              frames: the scale uses measurements only)
    returns : [T, N, 2] float32; scale[t] uses the changes strictly before t (prefix sums, one pass)
    """
    T = len(X)
    D = np.abs(np.diff(X[:, :, 1:3], axis=0))  # [T-1, N, 2] scan-to-scan |change| in [Pinj, Qinj]
    c1 = np.concatenate([np.zeros((1,) + D.shape[1:]), np.cumsum(D, 0)], 0)  # prefix sum
    c2 = np.concatenate([np.zeros((1,) + D.shape[1:]), np.cumsum(D**2, 0)], 0)  # prefix sum of squares
    scale = np.full((T, n_bus, 2), 1e-3, np.float32)
    # every t at once: the window covers D[max(0, t-W) .. t-2], used where it holds >= 3 changes
    t = np.arange(2, T)
    s, e = np.maximum(0, t - window), t - 1
    n = e - s
    ok = n >= 3
    s, e, t, n = s[ok], e[ok], t[ok], n[ok].astype(np.float64)[:, None, None]
    su, sq = c1[e] - c1[s], c2[e] - c2[s]
    scale[t] = np.sqrt(np.maximum(sq / n - (su / n) ** 2, 0.0)) + 1e-3
    return scale


def temporal_delta(nx: np.ndarray, prev: np.ndarray, metered: np.ndarray) -> np.ndarray:
    """One-step injection change at injection-metered buses, zero elsewhere [FED26].

        delta[b] = [P_inj, Q_inj](t) - [P_inj, Q_inj](t-1)

    nx, prev : [N, 4] current and previous scan in [|V|, P_inj, Q_inj, theta] order
    metered  : [N] bool, buses with an injection meter
    returns  : [N, 2] float32 (MW, MVAr)
    """
    td = np.zeros((nx.shape[0], 2), np.float32)
    td[metered, 0] = nx[metered, 1] - prev[metered, 1]
    td[metered, 1] = nx[metered, 2] - prev[metered, 2]
    return td


def swing_zscore(nx: np.ndarray, prev: np.ndarray, scale_t: np.ndarray, metered: np.ndarray) -> np.ndarray:
    """The one-step injection change as a z-score of the bus's typical recent change [FED26]:
    spikes (Aq, Al) read large, the slow ramp and benign scans stay near 1.

        swing[b] = ([P_inj, Q_inj](t) - [P_inj, Q_inj](t-1)) / scale_t[b]

    The difference is divided before it is rounded to float32 (the writer does the same over a block
    of frames at once, `generation.write`).
    nx, prev : [N, 4] current and previous scan; scale_t : [N, 2] from recent_change_scale
    metered  : [N] bool, buses with an injection meter
    returns  : [N, 2] float32, dimensionless
    """
    sw = np.zeros((nx.shape[0], 2), np.float32)
    sw[metered, 0] = (nx[metered, 1] - prev[metered, 1]) / scale_t[metered, 0]
    sw[metered, 1] = (nx[metered, 2] - prev[metered, 2]) / scale_t[metered, 1]
    return sw
