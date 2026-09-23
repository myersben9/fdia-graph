"""Statistics that combine across federated clients without moving their records [FED26].

Shapes: a client's feature block is [n_records, n_buses, C]; its moments are taken per channel over
records and buses, the way the papers standardize the per-bus vector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Moments = tuple[float, np.ndarray, np.ndarray]  # (count, mean [C], variance [C])


def channel_moments(X: np.ndarray) -> Moments:
    """Count, mean and population variance of every channel over records and buses [FED26].

    X       : [n, N, C]
    returns : (n * N, mean [C], var [C])
    """
    return float(X.shape[0] * X.shape[1]), X.mean(axis=(0, 1)), X.var(axis=(0, 1))


def pool_moments(parts: Sequence[Moments]) -> Moments:
    """The moments of the union of several parts from each part's moments alone [CGL79]:

        n = n_a + n_b,   mean = mean_a + (mean_b - mean_a) n_b / n,
        M2 = M2_a + M2_b + (mean_b - mean_a)^2 n_a n_b / n,   var = M2 / n

    folded left from the first part, so one part comes back unchanged (bit for bit).

    parts   : (count, mean [C], var [C]) per part
    returns : (count, mean [C], var [C]) of the union
    """
    n, mean, var = parts[0]
    m2 = var * n
    for nb, mb, vb in parts[1:]:
        tot = n + nb
        d = mb - mean
        mean = mean + d * (nb / tot)
        m2 = m2 + vb * nb + d * d * (n * nb / tot)
        n = tot
    return n, mean, (m2 / n if len(parts) > 1 else var)
