"""The pandapower objects the engine reads, as types.

pandapower ships no type information and is an optional dependency, so the engine names here the
parts of a network it actually reads: the element tables, the result tables of a power flow, and the
internal PYPOWER case with its pandapower-to-PYPOWER bus lookup. A real `pandapowerNet` satisfies
`PandapowerNet` structurally; nothing here is imported at run time beyond typing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class PpcTables(TypedDict):
    """The internal PYPOWER case after a power flow: base power (MVA) and the bus and branch matrices."""

    baseMVA: float
    bus: np.ndarray
    branch: np.ndarray


class PandapowerNet(Protocol):
    """A pandapower network, as far as the engine reads one."""

    bus: pd.DataFrame
    line: pd.DataFrame
    trafo: pd.DataFrame
    load: pd.DataFrame
    gen: pd.DataFrame
    sgen: pd.DataFrame
    ext_grid: pd.DataFrame
    shunt: pd.DataFrame
    res_bus: pd.DataFrame
    res_line: pd.DataFrame
    res_shunt: pd.DataFrame
    sn_mva: float
    _ppc: PpcTables
    _pd2ppc_lookups: dict[str, np.ndarray]
