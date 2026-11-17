"""The static description of a system: the per-branch pi model and the admittance matrices built
from it. Nothing here changes from one scan to the next."""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class BranchModel(NamedTuple):
    """The per-branch pi model [MP19], one entry per branch, per unit; the shapes a shard stores."""

    r: np.ndarray  # series resistance
    x: np.ndarray  # series reactance
    b: np.ndarray  # charging susceptance
    g: np.ndarray  # charging conductance (transformer iron losses)
    tap: np.ndarray  # turns ratio, 1 (or 0, read as 1) for lines
    shift_deg: np.ndarray  # phase shift in degrees
    status: Optional[np.ndarray] = None  # 1 in service, 0 out; None = all in service


class Admittances(NamedTuple):
    """The nodal and branch admittance matrices of one topology, complex per unit."""

    ybus: np.ndarray  # [N, N]
    yf: np.ndarray  # [E, N] from-end: I_f = Yf @ V
    yt: np.ndarray  # [E, N] to-end: I_t = Yt @ V
