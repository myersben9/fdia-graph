"""The split of a system's buses into K clients (utilities) for federated training [FED26].

The paper's partition is spectral clustering of the bus adjacency (scikit-learn, random_state 42),
which matched the cached partitions of the federated localization paper exactly on IEEE 14, 118
and 300 at K = 2 and 3 when checked by hand; the test suite pins IEEE 14. Needs scikit-learn for K >= 2: pip install "fdia-graph[federated]".
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..formulas.federated import attackable_affinity, cut_edge_count, halo_nodes, interior_boundary
from ..models.federated import Partition
from ..models.validation import expect


def bus_adjacency(edge_index: np.ndarray, N: int) -> np.ndarray:
    """The 0/1 undirected bus adjacency [N, N] of a branch list [2, E], without self-loops
    (parallel branches collapse to one edge)."""
    ei = np.asarray(edge_index)
    expect(
        ei.ndim == 2 and ei.shape[0] == 2 and np.issubdtype(ei.dtype, np.integer),
        f"edge_index must be an integer [2, E] array, got shape {ei.shape}",
    )
    expect(not (ei.size) or (ei.min() >= 0 and ei.max() < N), f"edge_index names a bus outside 0..{N - 1}")
    A = np.zeros((N, N), np.float64)
    A[ei[0], ei[1]] = 1.0
    A[ei[1], ei[0]] = 1.0
    np.fill_diagonal(A, 0.0)
    return A


def spectral_partition(
    edge_index: np.ndarray,
    N: int,
    K: int,
    random_state: int = 42,
    attackable: Optional[np.ndarray] = None,
    heavy: float = 8.0,
) -> Partition:
    """K clients by spectral clustering of the bus adjacency [VLX07], as in the federated paper;
    `attackable` (a [N] bool mask) biases the cut away from attackable buses (`heavy` its weight).
    K = 1 puts every bus in one client without clustering."""
    expect(1 <= K <= N, f"K must be between 1 and the {N} buses, got {K}")
    A = bus_adjacency(edge_index, N)
    if K == 1:
        assignment = np.zeros(N, np.int64)
    elif K == N:  # one bus per client: nothing to cluster
        assignment = np.arange(N, dtype=np.int64)
    else:
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError as e:
            raise ImportError(
                "a K >= 2 partition needs scikit-learn: pip install 'fdia-graph[federated]'"
            ) from e
        affinity = A if attackable is None else attackable_affinity(A, attackable, heavy)
        sc = SpectralClustering(
            n_clusters=K, affinity="precomputed", assign_labels="kmeans", random_state=random_state
        )
        assignment = sc.fit_predict(affinity).astype(np.int64)
    return partition_from_assignment(assignment, edge_index, attackable)


def partition_from_assignment(
    assignment: np.ndarray, edge_index: np.ndarray, attackable: Optional[np.ndarray] = None
) -> Partition:
    """A Partition from a given client-of-every-bus array (e.g. one saved with a paper's runs)."""
    assignment = np.asarray(assignment)
    N = int(edge_index.max()) + 1 if edge_index.size else len(assignment)
    expect(
        assignment.ndim == 1
        and len(assignment)
        and len(assignment) >= N
        and np.issubdtype(assignment.dtype, np.integer),
        f"assignment must be one integer client per bus ({N} buses)",
    )
    assignment = assignment.astype(np.int64)
    K = int(assignment.max()) + 1
    expect(
        assignment.min() >= 0 and len(np.unique(assignment)) == K,
        f"clients must be numbered 0..{K - 1} with none empty",
    )
    A = bus_adjacency(edge_index, len(assignment))
    interior, boundary = interior_boundary(assignment, A, K)
    on_boundary = None
    if attackable is not None:
        mask = np.asarray(attackable, bool)
        expect(
            mask.shape == assignment.shape,
            f"the attackable mask must be one flag per bus, shape {assignment.shape}",
        )
        on_boundary = int((boundary.any(axis=0) & mask).sum())
    return Partition(K, assignment, interior, boundary, cut_edge_count(assignment, A), on_boundary)


def compute_nodes(p: Partition, edge_index: np.ndarray, k: int, halo: int = 0) -> tuple[np.ndarray, int]:
    """Client k's compute buses, its own first and then a `halo`-hop ring of other clients' buses
    as read-only context, and the number of its own buses (`formulas.federated.halo_nodes`)."""
    return halo_nodes(p.assignment, bus_adjacency(edge_index, len(p.assignment)), k, halo)


def check_partition(p: Partition, N: int) -> None:
    """A Partition fit for a system of N buses: one client per bus, the clients numbered 0..K-1 with
    none empty (a hand-built Partition is not checked by its constructor)."""
    a = np.asarray(p.assignment)
    expect(
        a.ndim == 1 and np.issubdtype(a.dtype, np.integer),
        f"the partition's assignment must be a 1-D integer array, got {a.dtype} {a.shape}",
    )
    expect(len(p.assignment) == N, f"the partition covers {len(p.assignment)} buses, the system has {N}")
    labels = np.unique(p.assignment)
    expect(
        np.array_equal(labels, np.arange(p.K)),
        f"the partition must number its clients 0..{p.K - 1}, got {labels.tolist()}",
    )
