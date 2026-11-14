"""NYISO's internal transmission interfaces on the reduced case, with their real 2019 hourly limits and flows.

NYISO's real-time dispatch constrains the flow across each interface (a set of branches between zones),
not single lines, and NYgrid applies those limits to the signed sum of each interface's branches in its
reduced case (`updateOpCond.m` L420-430), from NYISO's published hourly limits and flows
(`interflowHourly_2019.csv`). Decision N48: the overload attacks target these interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fdia_graph.errors import DataConditionError
from fdia_graph.generation.nygrid.case import Case

# updateOpCond.m L420-430: each interface's branches as (from bus, to bus, sign) of the reduced case, as
# mapped by end buses in out/ratings_check/interface_branch_map.csv. The sign makes the from-end P of
# every branch count in the interface's positive direction.
NYISO_INTERFACES: dict[str, list[tuple[int, int, int]]] = {
    "CENTRAL EAST - VC": [(41, 45, -1), (40, 44, -1), (37, 38, -1), (37, 43, -1)],
    "MOSES SOUTH": [(47, 48, -1), (44, 48, -1), (46, 49, -1)],
    "UPNY CONED": [(73, 74, 1), (74, 77, -1)],
    "SPR/DUN-SOUTH": [(78, 81, 1), (78, 82, 1)],
}


@dataclass(frozen=True)
class Interface:
    """One interface: its `branches` [n] (positions into the case's branches) and their `signs` [n], and
    over the pool's frames [T] its real hourly `limit` (MW, positive direction) and real `flow` (MW),
    each hour's value held over its twelve five-minute frames, NaN where NYISO gives none."""

    name: str
    branches: np.ndarray
    signs: np.ndarray
    limit: np.ndarray
    flow: np.ndarray

    def model_flow(self, pf: np.ndarray) -> np.ndarray:
        """The interface flow (MW) from the from-end active flows `pf [..., L]` of every branch."""
        return pf[..., self.branches] @ self.signs


def hourly_interfaces(
    case: Case, time: np.ndarray, data: str | Path, names: list[str] | None = None, year: int = 2019
) -> list[Interface]:
    """The interfaces named in `names` (default all of `NYISO_INTERFACES`) over the frames `time [T]`
    (seconds, the pool's clock), their limits and flows from NYgrid's `interflowHourly_{year}.csv` in
    `data`."""
    ends = {(int(case.bus_id[case.f[k]]), int(case.bus_id[case.t[k]])): k for k in range(len(case.f))}
    table = pd.read_csv(Path(data) / f"interflowHourly_{year}.csv")
    table["t"] = pd.to_datetime(table.TimeStamp, format="%m/%d/%Y %H:%M")
    hour = pd.DatetimeIndex(np.asarray(time).astype("datetime64[s]")).floor("h")
    out = []
    for name in names or list(NYISO_INTERFACES):
        branches, signs = [], []
        for f, t, s in NYISO_INTERFACES[name]:
            if (f, t) not in ends:
                raise DataConditionError(f"{name}: branch {f}-{t} is not in the case in that orientation")
            branches.append(ends[(f, t)])
            signs.append(float(s))
        rows = table[table.InterfaceName == name].set_index("t")
        if rows.empty:
            raise DataConditionError(f"{name}: no rows in interflowHourly_{year}.csv")
        out.append(
            Interface(
                name,
                np.array(branches, int),
                np.array(signs),
                rows.PositiveLimitMWH.reindex(hour).to_numpy(float),
                rows.FlowMWH.reindex(hour).to_numpy(float),
            )
        )
    return out
