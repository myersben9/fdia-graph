"""The AC solves of a Case, behind one interface with two engines.

    Solver.from_engine(engine, case)  ->  IpoptSolver or PandapowerSolver

`Solver.opf` is the one solve both engines have: the AC point of least cost for given loads. The power
flow is `PandapowerSolver.power_flow` only, and the New York dispatch's own OPFs (`project`, `dispatch`)
are `IpoptSolver`'s only. An engine's module is imported only when that engine is asked for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from fdia_graph.constants.grid import BusType
from fdia_graph.errors import ConfigError
from fdia_graph.models.choices import SolveEngine
from fdia_graph.models.grid import OpfPoint
from fdia_graph.models.inputs import UnitCosts
from fdia_graph.systems.case import Case
from fdia_graph.systems.network import Network


class Solver(ABC):
    """One engine's solves of one Case. Holds the case, its admittances (`network`) and the last solved
    state `vm` (p.u.), `va` (degrees). `opf` does not start from it: its result depends on its
    arguments only. `IpoptSolver.project` and `dispatch` do start from it."""

    def __init__(self, case: Case) -> None:
        self.case = case
        self.network = Network(case)
        self.vm, self.va = case.vm.copy(), case.va.copy()
        self._ref = int(np.flatnonzero(case.bus_type == BusType.REF)[0])

    @classmethod
    def from_engine(cls, engine: str, case: Case) -> Solver:
        """The solver of `case` on the engine `engine`, a `SolveEngine` or its name."""
        choice = SolveEngine(engine)  # a name that is no engine is refused here
        if choice is SolveEngine.IPOPT:
            from fdia_graph.data.ipopt_solver import IpoptSolver

            return IpoptSolver(case)
        if choice is SolveEngine.PANDAPOWER:
            from fdia_graph.data.pandapower_solver import PandapowerSolver

            return PandapowerSolver(case)
        raise ConfigError(f"no solver is written for the engine {choice}")

    @abstractmethod
    def opf(self, pd: np.ndarray, qd: np.ndarray, costs: UnitCosts) -> OpfPoint | None:
        """The AC point of least cost `costs` for the bus loads `pd`, `qd [N]` (MW, MVAr), within the
        case's unit and voltage limits; None when the engine finds no optimum. The same arguments give
        the same point, whatever was solved before."""

    def _net_load(self, pd: np.ndarray, hvdc_p: np.ndarray) -> np.ndarray:
        """The bus loads (MW) with each HVDC link's transfer `hvdc_p [D]` added."""
        p = pd.copy()
        # a lossless link draws at its from end and injects at its to end
        np.add.at(p, self.case.dc_f, hvdc_p)
        np.add.at(p, self.case.dc_t, -hvdc_p)
        return p
