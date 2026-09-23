"""What the federated layer passes around: how the buses split into clients."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class Partition(NamedTuple):
    """Buses split into K clients (utilities): the client of every bus, each client's interior
    (all neighbours its own) and boundary buses, and the number of cross-client edges."""

    K: int
    assignment: np.ndarray  # [N] int, client of every bus
    interior: np.ndarray  # [K, N] bool
    boundary: np.ndarray  # [K, N] bool
    cut_edges: int
    attackable_boundary: Optional[int] = None  # attackable buses on some client's boundary, when known

    def owned(self, k: int) -> np.ndarray:
        """Client k's buses in increasing order."""
        return np.flatnonzero(self.assignment == k)


class RoundLog(NamedTuple):
    """One federated round: the clients' mean local training loss, each client's, and the bytes
    the round moved (every client's weights up to the server and the average back down)."""

    round: int
    loss_mean: float
    loss_per_client: list[float]
    bytes_up: int
    bytes_down: int
