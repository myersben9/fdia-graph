"""A scan's load and generation, read from its stored state.

The attacker's local false state (its area, the local solve, the operating limits) is in
`engine.attacks`."""

from __future__ import annotations

import numpy as np

from ..formulas.attacks import bus_load, element_loads, generator_output
from ..models.frames import OperatingLimits  # noqa: F401  re-exported: defined here before the models package
from ..models.grid import NODE
from .base import GridBase


class PhysicsMixin(GridBase):
    """Read a scan's load and generation. Mixed into FdiaGenerator."""

    def true_load(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's active load per load element [n_loads] (MW), `formulas.attacks.bus_load`: the
        stored injection plus the co-located generation at this scan's scale, not the base case's.
        Not the load at the slack bus (see `bus_load`), which no attack targets."""
        return element_loads(bus_load(Xt, self.load_base, self.gen_base), self.load_bus, self.load_p0)

    def true_reactive_load(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's reactive load per load element [n_loads] (MVAr): the bus's stored reactive
        injection plus its generators' reactive output (`scan_reactive_generation`), split over the
        bus's load elements by base reactive share (`formulas.attacks.element_loads`)."""
        return element_loads(
            Xt[:, NODE.q_inj] + self.scan_reactive_generation(Xt), self.load_bus, self.load_q0
        )

    def scan_reactive_generation(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's generator reactive output per bus [N] (MVAr), `formulas.attacks.generator_output`,
        zero at a bus with no generator (a zero-MW condenser counts as one)."""
        q = generator_output(Xt, self.load_base, self.gen_base)[:, 1]
        return np.where(self.has_gen, q, 0.0)

    def scan_generation(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's active generation per bus [N] (MW), `formulas.attacks.generator_output`;
        zero where the bus has no generator."""
        return generator_output(Xt, self.load_base, self.gen_base)[:, 0]
