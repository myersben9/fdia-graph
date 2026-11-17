"""What the analysis packages hand back that is neither a score table nor a setting: the truth an
estimator is scored against."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdia_graph.models.base import Bundle


@dataclass(frozen=True, eq=False)
class TrueState(Bundle):
    """The per-record truth an estimator is scored against."""

    x: np.ndarray  # [n, 2N-1] the true state: non-slack angles (rad), then every voltage magnitude (pu)
    thsl: np.ndarray  # [n] the slack angle reference per record (rad)
