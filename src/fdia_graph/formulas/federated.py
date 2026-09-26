"""Statistics that combine across federated clients without moving their records [FED26].

Shapes: a client's feature block is [n_records, n_buses, C]; its moments are taken per channel over
records and buses, the way the papers standardize the per-bus vector.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..models.validation import expect

Moments = tuple[float, np.ndarray, np.ndarray]  # (count, mean [C], variance [C])


def _check_graph(assignment: np.ndarray, A: np.ndarray) -> None:
    """One client per bus against a square adjacency of the same size."""
    expect(
        np.ndim(assignment) == 1 and np.shape(A) == (len(assignment), len(assignment)),
        f"need an [N] assignment and an [N, N] adjacency, got {np.shape(assignment)} and {np.shape(A)}",
    )


def channel_moments(X: np.ndarray) -> Moments:
    """Count, mean and population variance of every channel over records and buses [FED26].

    X       : [n, N, C]
    returns : (n * N, mean [C], var [C])
    """
    expect(
        np.ndim(X) == 3 and X.shape[0] * X.shape[1] != 0,
        f"need a non-empty [n, N, C] block, got shape {np.shape(X)}",
    )
    return float(X.shape[0] * X.shape[1]), X.mean(axis=(0, 1)), X.var(axis=(0, 1))


def pool_moments(parts: Sequence[Moments]) -> Moments:
    """The moments of the union of several parts from each part's moments alone [CGL79], written
    with the parts' shares of the pooled count so no intermediate product can overflow:

        n = n_a + n_b,   f_a = n_a / n,   f_b = n_b / n,   mean = mean_a + (mean_b - mean_a) f_b,
        var = f_a var_a + f_b var_b + (mean_b - mean_a)^2 f_a f_b

    folded left from the first part, so one part comes back unchanged (bit for bit).

    parts   : (count, mean [C], var [C]) per part
    returns : (count, mean [C], var [C]) of the union
    """
    expect(len(parts), "pool_moments needs at least one part")
    shape = np.shape(parts[0][1])
    expect(
        not (
            any(
                np.shape(m) != shape or np.shape(v) != shape or (not (np.isfinite(c) and c > 0))
                for (c, m, v) in parts
            )
        ),
        f"every part needs a finite positive count and moments of shape {shape}",
    )
    n, mean, var = parts[0]
    for nb, mb, vb in parts[1:]:
        tot = n + nb
        expect(np.isfinite(tot), "the pooled record count overflows")
        fa, fb = n / tot, nb / tot
        d = mb - mean
        mean = mean + d * fb
        var = fa * var + fb * vb + d * d * (fa * fb)
        n = tot
    return n, mean, var


def fedavg(arrays: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """The federated average of one parameter tensor [MCM17]:  theta = sum_k (n_k / n) theta_k,
    accumulated in float64 and cast back to the parameters' dtype (one client returns its own
    tensor exactly).

    arrays  : one tensor per client, all the same shape
    weights : n_k per client (record counts), any positive scale
    returns : the averaged tensor in the dtype of arrays[0]
    """
    expect(
        len(arrays) and len(arrays) == len(weights),
        f"need one weight per client tensor, got {len(arrays)} tensors and {len(weights)} weights",
    )
    w = np.asarray(weights, np.float64)
    expect(
        w.shape == (len(arrays),), f"need one scalar weight per client, shape ({len(arrays)},), got {w.shape}"
    )
    expect((np.isfinite(w) & (w > 0)).all(), "client weights must be finite and positive")
    shape = np.shape(arrays[0])
    expect(
        not (any(np.shape(a) != shape for a in arrays)), f"every client tensor must have the shape {shape}"
    )
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
    A = np.asarray(A)
    expect(A.ndim == 2 and A.shape[0] == A.shape[1], f"need a square [N, N] adjacency, got shape {A.shape}")
    attackable = np.asarray(attackable, bool)
    expect(attackable.shape == (A.shape[0],), f"need a [{A.shape[0]}] attackable mask")
    expect(
        np.isfinite(heavy) and 0 < heavy <= np.finfo(np.float32).max,
        f"heavy must be a finite positive float32 weight, got {heavy}",
    )
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
    expect(
        K >= 1 and set(np.unique(assignment).tolist()) == set(range(K)),
        f"the assignment must use exactly the clients 0..{K - 1}",
    )
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


def hop_distance(A: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Hops from the nearest source bus to every bus by breadth-first search [VLX07]; -1 where no
    path exists.

    A       : [N, N] 0/1 adjacency
    sources : bus indices at distance 0
    returns : [N] int
    """
    dist = np.full(len(A), -1, np.int64)
    dist[sources] = 0
    frontier, hop = np.asarray(sources), 0
    while len(frontier):
        hop += 1
        frontier = np.flatnonzero((A[frontier] > 0).any(axis=0) & (dist < 0))
        dist[frontier] = hop
    return dist


def halo_nodes(assignment: np.ndarray, A: np.ndarray, k: int, depth: int) -> tuple[np.ndarray, int]:
    """Client k's compute buses: its own buses, then every other client's bus within `depth` hops
    of them (by `hop_distance` over the whole grid), nearest first and in bus order within a hop,
    held as read-only context [FED26].

    assignment : [N]
    A          : [N, N] 0/1 adjacency
    returns    : (bus indices, own buses first; the number of own buses)
    """
    _check_graph(assignment, A)
    expect(depth >= 0, f"depth must be >= 0, got {depth}")
    owned = np.flatnonzero(assignment == k)
    expect(len(owned), f"client {k} owns no bus")
    dist = hop_distance(A, owned)
    halo = np.flatnonzero((dist >= 1) & (dist <= depth) & (assignment != k))
    halo = halo[np.argsort(dist[halo], kind="stable")]  # nearest first, bus order within a hop
    return np.concatenate([owned, halo]), len(owned)


def _index_array(c: np.ndarray, d: int) -> bool:
    """A one-dimensional integer array of state columns inside 0..d-1."""
    c = np.asarray(c)
    if c.ndim != 1 or not np.issubdtype(c.dtype, np.integer):
        return False
    return not len(c) or (c.min() >= 0 and c.max() < d)


def _check_blocks(blocks: Sequence[tuple[np.ndarray, np.ndarray]], d: int) -> None:
    """At least one block; each block's columns a 1-D integer array inside 0..d-1, the blocks
    disjoint; one basis row per column."""
    expect(
        len(blocks) and all(_index_array(c, d) for (c, _) in blocks),
        f"need at least one block of integer state columns inside 0..{d - 1}",
    )
    cols = np.concatenate([np.asarray(c) for c, _ in blocks])
    expect(len(np.unique(cols)) == len(cols), "the blocks' state columns must be disjoint")
    expect(
        not (any(np.ndim(V) != 2 or np.shape(V)[0] != len(c) for (c, V) in blocks)),
        "each basis needs one row per state column of its block",
    )


def block_diagonal_basis(blocks: Sequence[tuple[np.ndarray, np.ndarray]], d: int) -> np.ndarray:
    """Per-client subspace bases placed on their own state coordinates [EST26], [FED26]:

        V[cols_k, K_0 + ... + K_{k-1} : ... + K_k] = V_k,   zero elsewhere

    so V is orthonormal whenever every V_k is and the column sets are disjoint.

    blocks  : (cols_k, V_k [len(cols_k), K_k]) per client
    d       : the full state dimension
    returns : [d, sum K_k]
    """
    _check_blocks(blocks, d)
    out = np.zeros((d, sum(np.shape(V)[1] for _, V in blocks)))
    j = 0
    for c, V in blocks:
        out[np.asarray(c), j : j + V.shape[1]] = V
        j += V.shape[1]
    return out
