"""Federated training over K clients (utilities) that never pool their records [FED26].

Built step by step from the federated localization paper's harness: the partition of buses into
clients comes first; the federated localizers and the federated state-estimation fits follow.

    from fdia_graph.federated import spectral_partition
    part = spectral_partition(ds.edge_index_np, ds.N, K=3)   # needs the [federated] extra for K >= 2
"""

from __future__ import annotations

from ..models.federated import Partition
from .partition import bus_adjacency, compute_nodes, partition_from_assignment, spectral_partition

__all__ = ["Partition", "bus_adjacency", "compute_nodes", "partition_from_assignment", "spectral_partition"]
