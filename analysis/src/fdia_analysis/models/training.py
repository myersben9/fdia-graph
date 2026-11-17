"""What the learned localizers pass between their fitting pieces (a centralized fit and a
federated client share them)."""

from __future__ import annotations

from typing import NamedTuple


class OptimConfig(NamedTuple):
    """The optimizer and loss settings of a learned localizer's training loop."""

    lr: float
    weight_decay: float
    batch_size: int
    pos_weight: float
