"""A scan's load and generation, read from its stored state.

The attacker's local false state (its area, the local solve, the operating limits) is in
`engine.attacks`."""

from __future__ import annotations

import numpy as np

from ..constants.channels import NodeChannel
from ..formulas.attacks import bus_load, element_loads, generator_output
from .base import GridBase


class PhysicsMixin(GridBase):
    """Read a scan's load and generation. Mixed into FdiaGenerator."""

    def true_load(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's active load per row of the load table [n_loads] (MW), `formulas.attacks.bus_load`;
        zero on a row with no base active load. Not the load at the slack bus (see `bus_load`)."""
        return element_loads(bus_load(Xt, self.load_base, self.gen_base), self.load_bus, self.load_p0)

    def true_reactive_load(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's reactive load per row of the load table [n_loads] (MVAr): the bus's reactive
        injection plus its generators' reactive output; zero on a row with no base reactive load."""
        bus_q = Xt[:, NodeChannel.Q_INJ] + self.scan_reactive_generation(Xt)
        return element_loads(bus_q, self.load_bus, self.load_q0)

    def scan_reactive_generation(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's generator reactive output per bus [N] (MVAr), `formulas.attacks.generator_output`,
        zero at a bus with no generator (a zero-MW condenser counts as one)."""
        q = generator_output(Xt, self.load_base, self.gen_base)[:, 1]
        return np.where(self.has_gen, q, 0.0)

    def scan_generation(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's active generation per bus [N] (MW), `formulas.attacks.generator_output`;
        zero where the bus has no generator."""
        return generator_output(Xt, self.load_base, self.gen_base)[:, 0]
