"""Federated fitting of the state estimator's learned part [EST26], [FED26].

The learned part of the proposed estimator is the subspace prior: a low-rank basis of benign
operating states. `RegionalPrior` fits one basis per client from that client's own state
coordinates (the angles of its non-slack buses and the voltages of all its buses) and places them on
a block diagonal, so no client's states leave it. It is exact for one client (the SubspacePrior
itself) and loses the cross-area correlation otherwise; how much that costs is what the regional
prior measures. Everything else the estimator learns is already local: each meter's sigma comes
from that meter's own residuals, the benign mean from each coordinate's own history.

The solve itself stays at the control center (the classical EMS arrangement); a localizer trained by
FedAvg can gate it: `GatedPrior(gate=FedBusCNN(...).fit(train, val))`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..formulas.estimation import whitened_svd_basis
from ..formulas.federated import block_diagonal_basis
from ..models.federated import Partition
from ..se.methods import SubspacePrior


class RegionalPrior(SubspacePrior):
    """The subspace prior (optionally with Huber reweighting) fitted per client of `partition`:
    each client's basis keeps `rank_frac` of its own state coordinates; the estimate is restricted
    to their block-diagonal union. Every other knob is SubspacePrior's."""

    def __init__(
        self, partition: Partition, rank_frac: float = 0.5, reweight: Optional[str] = None, **kw: Any
    ) -> None:
        super().__init__(rank_frac=rank_frac, reweight=reweight, **kw)
        self.partition = partition

    def state_columns(self, k: int) -> np.ndarray:
        """Client k's coordinates of the 2N-1 state: the angles of its non-slack buses (their
        positions in `keep`), then the voltages of all its buses."""
        own = self.partition.owned(k)
        angles = np.flatnonzero(np.isin(self.keep, own))
        return np.concatenate([angles, len(self.keep) + own])

    def _fit_states(self, x_benign: np.ndarray) -> None:
        if len(self.partition.assignment) != self.N:
            raise ValueError(
                f"the partition covers {len(self.partition.assignment)} buses, the system has {self.N}"
            )
        blocks = []
        for k in range(self.partition.K):
            cols = self.state_columns(k)
            # the whole state in order (one client) is passed as is: LAPACK picks singular-vector
            # signs by memory layout, and a copy would flip one relative to SubspacePrior
            X = x_benign if np.array_equal(cols, np.arange(x_benign.shape[1])) else x_benign[:, cols]
            blocks.append((cols, whitened_svd_basis(X, self.rank_frac)[1]))
        self.VK = block_diagonal_basis(blocks, x_benign.shape[1])
        self.K = self.VK.shape[1]
