"""A scan's load and generation, read from its stored state.

A pool stores one net injection per bus. How it divides into load and generation depends on how the
pool was made, and a `LoadSplit` says it: `ScaledSplit` for a pool that scaled the case's load and
generation by one factor per bus (the released IEEE pools), `NetSplit` for a pool whose units were
dispatched (an OPF: the New York pool). The attacker's local false state (its area, the local solve,
the operating limits) is in `engine.attacks`."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..constants.channels import NodeChannel
from ..constants.grid import LOAD_FLOOR_MW, BusType
from ..formulas.attacks import element_loads, generator_output
from ..models.choices import LoadRule
from ..systems.case import Case
from .base import Part


class LoadSplit(ABC):
    """How a pool state's net injection divides into load and generation at each bus. `base_load`
    [N, 2] (P MW, Q MVAr) is the load the generator's load table is built from: a bus with none
    carries no load an attack can pretend."""

    base_load: np.ndarray

    @abstractmethod
    def generation(self, X: np.ndarray) -> np.ndarray:
        """The generator output [..., N, 2] (P MW, Q MVAr) behind pool states X [..., N, 4]."""

    def load(self, X: np.ndarray) -> np.ndarray:
        """The active load [..., N] (MW) behind pool states X [..., N, 4]: the load-positive
        injection plus the bus's generation. Not the load at the slack bus of a scaled pool, whose
        injection also nets the external grid's output."""
        return X[..., NodeChannel.P_INJ] + self.generation(X)[..., 0]

    @abstractmethod
    def idle_units(self, case: Case) -> np.ndarray:
        """[G] bool: the units of `case` that inject nothing in any frame of the pool. A scaled pool
        leaves a unit on a PQ bus at its base output, so one with none is idle (all of IEEE-200's); a
        dispatched pool sets every unit's output itself, so none is."""

    @classmethod
    def of(cls, rule: str, case: Case, X: np.ndarray) -> LoadSplit:
        """The split the setting `rule` (`LoadRule`) names, on `case` and its pool X [T, N, 4]."""
        return NetSplit.of_pool(case, X) if rule == LoadRule.NET else ScaledSplit.of_case(case)


class ScaledSplit(LoadSplit):
    """The split of a pool that scaled a bus's load and generation by one factor
    (`formulas.attacks.generator_output`): `base_load` [N, 2] is the case's load and `gen_base` [N, 2]
    its generation (the Q column zero). The load is P_inj + s P_gen_base = s P_load_base; adding the
    base generation instead would give s P_load_base + (1 - s) P_gen_base at a generator bus. Assumes
    no static generator on a load bus (the pool does not scale them)."""

    def __init__(self, base_load: np.ndarray, gen_base: np.ndarray) -> None:
        self.base_load, self.gen_base = base_load, gen_base

    @classmethod
    def of_case(cls, case: Case) -> ScaledSplit:
        """The split on the base state of `case`: its loads, and the output of the units that hold a
        voltage, summed over co-located units in row order (np.add.at is unbuffered)."""
        gen_base = np.zeros((len(case.bus_id), 2))
        gens = np.flatnonzero(case.bus_type[case.unit_bus] == BusType.PV)
        np.add.at(gen_base[:, 0], case.unit_bus[gens], case.unit_p[gens])
        return cls(np.stack([case.pd, case.qd], axis=1), gen_base)

    def generation(self, X: np.ndarray) -> np.ndarray:
        return generator_output(X, self.base_load, self.gen_base)

    def idle_units(self, case: Case) -> np.ndarray:
        static = case.bus_type[case.unit_bus] == BusType.PQ
        return static & (case.unit_p == 0) & (case.unit_q == 0)


class NetSplit(LoadSplit):
    """The split of a pool whose units were dispatched, where no one factor ties load to generation:
    a bus with a unit (`has_unit` [N]) is all generation, any other bus all load. The load at a
    unit's own bus is not seen, so it is never a target."""

    def __init__(self, has_unit: np.ndarray, base_load: np.ndarray) -> None:
        self.has_unit, self.base_load = has_unit, base_load

    @classmethod
    def of_pool(cls, case: Case, X: np.ndarray) -> NetSplit:
        """The split on `case` with the mean load of the pool X [T, N, 4] as the base load (a frame
        with no solution is NaN and left out); a bus whose mean load is not above `LOAD_FLOOR_MW`
        has none."""
        has_unit = np.zeros(len(case.bus_id), bool)
        has_unit[case.unit_bus] = True
        mean = np.nanmean(X[:, :, [NodeChannel.P_INJ, NodeChannel.Q_INJ]], axis=0)
        loaded = ~has_unit & (mean[:, 0] > LOAD_FLOOR_MW)
        return cls(has_unit, np.where(loaded[:, None], mean, 0.0))

    def idle_units(self, case: Case) -> np.ndarray:
        return np.zeros(len(case.unit_bus), bool)

    def generation(self, X: np.ndarray) -> np.ndarray:
        injected = -X[..., [NodeChannel.P_INJ, NodeChannel.Q_INJ]]
        return np.where(self.has_unit[:, None], injected, 0.0)


class ScanLoads(Part):
    """Read a scan's load and generation: the generator's `scan_loads`."""

    def true_load(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's active load per row of the load table [n_loads] (MW), `LoadSplit.load`; zero
        on a row with no base active load."""
        return element_loads(self.g.split.load(Xt), self.g.load_bus, self.g.load_p0)

    def true_reactive_load(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's reactive load per row of the load table [n_loads] (MVAr): the bus's reactive
        injection plus its generators' reactive output; zero on a row with no base reactive load."""
        bus_q = Xt[:, NodeChannel.Q_INJ] + self.scan_reactive_generation(Xt)
        return element_loads(bus_q, self.g.load_bus, self.g.load_q0)

    def scan_reactive_generation(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's generator reactive output per bus [N] (MVAr), `LoadSplit.generation`, zero at
        a bus with no generator (a zero-MW condenser counts as one)."""
        return np.where(self.g.has_gen, self.g.split.generation(Xt)[:, 1], 0.0)

    def scan_generation(self, Xt: np.ndarray) -> np.ndarray:
        """This scan's active generation per bus [N] (MW), `LoadSplit.generation`; zero where the
        bus has no generator."""
        return self.g.split.generation(Xt)[:, 0]
