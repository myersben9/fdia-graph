"""Federated training over K clients (utilities) that never pool their records [FED26].

Built from the federated localization paper's harness: the partition of buses into clients
(`spectral_partition`) and the paper's localizers trained by FedAvg (`FedBusMLP`, `FedBusCNN`,
drop-in LocalizerBase methods), and the estimator's subspace prior fitted per client
(`RegionalPrior`).

    from fdia_graph.federated import FedBusCNN
    loc = FedBusCNN(K=3).fit(train, val=val)   # needs the [federated] extra
    rep = loc.score(test)
"""

from __future__ import annotations

from ..models.federated import Partition, RoundLog
from .localizer import FedBusCNN, FedBusMLP, FederatedLocalizer
from .partition import (
    bus_adjacency,
    check_partition,
    compute_nodes,
    partition_from_assignment,
    spectral_partition,
)
from .se import RegionalPrior

__all__ = [
    "FedBusCNN",
    "FedBusMLP",
    "FederatedLocalizer",
    "Partition",
    "RegionalPrior",
    "RoundLog",
    "bus_adjacency",
    "check_partition",
    "compute_nodes",
    "partition_from_assignment",
    "spectral_partition",
]
