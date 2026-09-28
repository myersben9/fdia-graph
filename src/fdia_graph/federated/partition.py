"""The split of a system's buses into K clients (utilities) for federated training [FED26].

The paper's partition is spectral clustering of the bus adjacency (scikit-learn, random_state 42),
which matched the cached partitions of the federated localization paper exactly on IEEE 14, 118
and 300 at K = 2 and 3 when checked by hand; the test suite pins IEEE 14. Needs scikit-learn for K >= 2: pip install "fdia-graph[federated]".
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ..formulas.federated import attackable_affinity, cut_edge_count, halo_nodes, interior_boundary
from ..models.federated import Partition
from ..models.inputs import AssignmentSpec, ClientCount, EdgeList, PartitionOnGrid


def bus_adjacency(edge_index: np.ndarray, N: int) -> np.ndarray:
    """The 0/1 undirected bus adjacency [N, N] of a branch list [2, E], without self-loops
    (parallel branches collapse to one edge)."""
    ei = EdgeList(edge_index, N).edge_index
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
    ClientCount(N, K)
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
    spec = AssignmentSpec(assignment, edge_index, attackable)
    assignment = spec.assignment.astype(np.int64)
    K = int(assignment.max()) + 1
    A = bus_adjacency(spec.edge_index, len(assignment))
    interior, boundary = interior_boundary(assignment, A, K)
    on_boundary = None
    if spec.attackable is not None:
        on_boundary = int((boundary.any(axis=0) & spec.attackable).sum())
    return Partition(K, assignment, interior, boundary, cut_edge_count(assignment, A), on_boundary)


def compute_nodes(p: Partition, edge_index: np.ndarray, k: int, halo: int = 0) -> tuple[np.ndarray, int]:
    """Client k's compute buses, its own first and then a `halo`-hop ring of other clients' buses
    as read-only context, and the number of its own buses (`formulas.federated.halo_nodes`)."""
    return halo_nodes(p.assignment, bus_adjacency(edge_index, len(p.assignment)), k, halo)


def check_partition(p: Partition, N: int) -> None:
    """Deprecated: build `models.inputs.PartitionOnGrid(p.assignment, p.K, N)`, which checks it."""
    warnings.warn(
        "check_partition is deprecated; PartitionOnGrid(p.assignment, p.K, N) checks a partition",
        DeprecationWarning,
        stacklevel=2,
    )
    PartitionOnGrid(p.assignment, p.K, N)
