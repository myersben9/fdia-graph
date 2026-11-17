"""The pool of operating points a timeline walks, one frame per point: an array [T, N, 4] in memory,
or `PoolStates`, which keeps a year of solved voltages as they are and gives a frame's state when
it is asked for."""

from __future__ import annotations

from typing import Union

import numpy as np

from fdia_graph.systems.network import Network


class PoolStates:
    """Solved voltages `vm` [T, N] p.u. and `va` [T, N] degrees read as the states a timeline walks:
    `states[t]` is [N, 4] = |V|, P MW, Q MVAr, angle, and a slice is [T, N, 4]. The injections are
    those the admittances of `net` give for the voltages (generation positive). An unsolved frame
    is NaN."""

    def __init__(self, vm: np.ndarray, va: np.ndarray, net: Network) -> None:
        self.vm, self.va, self.net = vm, va, net

    def __len__(self) -> int:
        return len(self.vm)

    def __getitem__(self, frames: Union[int, slice]) -> np.ndarray:
        vm, va = self.vm[frames], self.va[frames]
        with np.errstate(invalid="ignore"):  # an unsolved frame keeps its place with NaN
            p, q = self.net.injections(vm, va)
        return np.stack([vm, p, q, va], axis=-1)


# the pool a timeline walks: an array [T, N, 4] (an IEEE pool stores its injections) or solved voltages
States = Union[np.ndarray, PoolStates]
