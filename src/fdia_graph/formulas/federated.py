"""Statistics that combine across federated clients without moving their records [FED26].

Shapes: a client's feature block is [n_records, n_buses, C]; its moments are taken per channel over
records and buses, the way the papers standardize the per-bus vector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Moments = tuple[float, np.ndarray, np.ndarray]  # (count, mean [C], variance [C])


def _check_graph(assignment: np.ndarray, A: np.ndarray) -> None:
    """One client per bus against a square adjacency of the same size."""
    if np.ndim(assignment) != 1 or np.shape(A) != (len(assignment), len(assignment)):
        raise ValueError(
            f"need an [N] assignment and an [N, N] adjacency, got {np.shape(assignment)} and {np.shape(A)}"
        )


def channel_moments(X: np.ndarray) -> Moments:
    """Count, mean and population variance of every channel over records and buses [FED26].

    X       : [n, N, C]
    returns : (n * N, mean [C], var [C])
    """
    if np.ndim(X) != 3 or X.shape[0] * X.shape[1] == 0:
        raise ValueError(f"need a non-empty [n, N, C] block, got shape {np.shape(X)}")
    return float(X.shape[0] * X.shape[1]), X.mean(axis=(0, 1)), X.var(axis=(0, 1))


def pool_moments(parts: Sequence[Moments]) -> Moments:
    """The moments of the union of several parts from each part's moments alone [CGL79]:

        n = n_a + n_b,   mean = mean_a + (mean_b - mean_a) n_b / n,
        M2 = M2_a + M2_b + (mean_b - mean_a)^2 n_a n_b / n,   var = M2 / n

    folded left from the first part, so one part comes back unchanged (bit for bit).

    parts   : (count, mean [C], var [C]) per part
    returns : (count, mean [C], var [C]) of the union
    """
    if not len(parts):
        raise ValueError("pool_moments needs at least one part")
    n, mean, var = parts[0]
    m2 = var * n
    for nb, mb, vb in parts[1:]:
        tot = n + nb
        d = mb - mean
        mean = mean + d * (nb / tot)
        m2 = m2 + vb * nb + d * d * (n * nb / tot)
        n = tot
    return n, mean, (m2 / n if len(parts) > 1 else var)


def fedavg(arrays: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """The federated average of one parameter tensor [MCM17]:  theta = sum_k (n_k / n) theta_k,
    accumulated in float64 and cast back to the parameters' dtype (one client returns its own
    tensor exactly).

    arrays  : one tensor per client, all the same shape
    weights : n_k per client (record counts), any positive scale
    returns : the averaged tensor in the dtype of arrays[0]
    """
    if not len(arrays) or len(arrays) != len(weights):
        raise ValueError(
            f"need one weight per client tensor, got {len(arrays)} tensors and {len(weights)} weights"
        )
    w = np.asarray(weights, np.float64)
    if not (np.isfinite(w) & (w > 0)).all():
        raise ValueError("client weights must be finite and positive")
    shape = np.shape(arrays[0])
    if any(np.shape(a) != shape for a in arrays):
        raise ValueError(f"every client tensor must have the shape {shape}")
    w = w / w.max()  # scale first: finite weights near the float limit cannot overflow the sum
    w = w / w.sum()
    acc = np.zeros(np.shape(arrays[0]), np.float64)
    for wk, a in zip(w, arrays):
        acc += wk * np.asarray(a, np.float64)
    return acc.astype(np.asarray(arrays[0]).dtype)


def attackable_affinity(A: np.ndarray, attackable: np.ndarray, heavy: float = 8.0) -> np.ndarray:
    """An adjacency re-weighted so a normalized cut prefers edges away from attackable buses
    [FED26]:  W = A * sqrt(m m^T),  m_v = heavy if bus v is attackable else 1.

    A          : [N, N] 0/1 adjacency
    attackable : [N] bool
    returns    : [N, N] float32 affinity
    """
    attackable = np.asarray(attackable, bool)
    if attackable.shape != (A.shape[0],):
        raise ValueError(f"need a [{A.shape[0]}] attackable mask")
    if not (np.isfinite(heavy) and 0 < heavy <= np.finfo(np.float32).max):
        raise ValueError(f"heavy must be a finite positive float32 weight, got {heavy}")
    m = np.where(attackable, heavy, 1.0)
    return (A * np.sqrt(np.outer(m, m))).astype(np.float32)


def interior_boundary(assignment: np.ndarray, A: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Each client's interior (every neighbour in the same client, or no neighbour) and boundary
    buses [VLX07], [FED26].

    assignment : [N] client of every bus
    A          : [N, N] 0/1 adjacency
    returns    : (interior [K, N] bool, boundary [K, N] bool)
    """
    _check_graph(assignment, A)
    N = len(assignment)
    foreign = (A > 0) & (assignment[None, :] != assignment[:, None])  # [N, N] edges to another client
    inner = ~foreign.any(axis=1)
    own = assignment[None, :] == np.arange(K)[:, None]  # [K, N]
    return own & inner[None, :N], own & ~inner[None, :N]


def cut_edge_count(assignment: np.ndarray, A: np.ndarray) -> int:
    """Edges whose ends lie in different clients, each unordered pair once [VLX07].

    assignment : [N]
    A          : [N, N] 0/1 symmetric adjacency
    """
    _check_graph(assignment, A)
    u, v = np.nonzero(np.triu(A, 1))
    return int((assignment[u] != assignment[v]).sum())


def halo_nodes(assignment: np.ndarray, A: np.ndarray, k: int, depth: int) -> tuple[np.ndarray, int]:
    """Client k's compute buses: its own buses, then the other clients' buses within `depth` hops,
    ring by ring in increasing bus order, held as read-only context [FED26].

    assignment : [N]
    A          : [N, N] 0/1 adjacency
    returns    : (bus indices, own buses first; the number of own buses)
    """
    _check_graph(assignment, A)
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")
    owned = np.flatnonzero(assignment == k)
    if not len(owned):
        raise ValueError(f"client {k} owns no bus")
    seen = np.zeros(len(assignment), bool)
    seen[owned] = True
    frontier, halo = owned, []
    for _ in range(depth):
        ring = np.flatnonzero(A[frontier].any(axis=0) & ~seen & (assignment != k))
        if not len(ring):
            break
        seen[ring] = True
        halo.append(ring)
        frontier = ring
    return (np.concatenate([owned, *halo]) if halo else owned), len(owned)
