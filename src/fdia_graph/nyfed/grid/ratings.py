"""Branch ratings from the limits NYgrid itself sets. `npcc.mat` carries no branch ratings."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fdia_graph.errors import DataConditionError
from fdia_graph.nyfed.grid.case import Case

# updateOpCond.m L418-465 sets NYISO's interface limits on the reduced case as limits on the signed sum
# of each interface's branches. Three interfaces are a single branch, and two of those have a fixed
# limit, so for them the interface limit is a limit on that one branch (MW, both directions):
# F-G (reduced branch 4, buses 37-39) and H-I (reduced branch 68, buses 74-78).
_SINGLE_BRANCH = {(37, 39): 5400.0, (74, 78): 8450.0}


def line_ratings(case: Case) -> Case:
    """`case` with every branch rating NYgrid gives, and 0 where no public source gives one.

    Rated branches and their sources:

    - the ties to PJM, ISO-NE and IESO that modifyMPC.m L54-67 rates (MVA), already set by `modify_mpc`:
      66-134, 67-138, 81-125, 75-124, 60-140, 29-37, 35-73, 48-100, 54-102, 54-103. `hvdc_links`
      removes 81-125, so a case with the HVDC links has nine of them.
    - the single-branch interfaces with a fixed limit (updateOpCond.m L460 and L462, as `_SINGLE_BRANCH`):
      F-G 37-39 at 5400 MW and H-I 74-78 at 8450 MW.

    Left unrated, because the source gives no per-branch number:

    - E-G, the single branch 38-77 that modifyMPC.m adds, has an hourly limit of 5650 MW minus that
      hour's Central East positive limit forward and 1600 MW back (L459), not a fixed rating.
    - Dysinger East, West Central, Total East, Moses South, Central East, UPNY-ConEd, Sprain Brook/
      Dunwoodie South and I-K limit the sum of several branches (L420-430, L453-465). A sum limit does
      not say how much each branch may carry, so none of their branches takes a share of it.
    - every other branch: neither NYgrid nor npcc.mat rates it, and no public rating is known.
    """
    ends = np.stack([case.bus_id[case.f], case.bus_id[case.t]], axis=1)
    rating = case.rating.copy()
    for (a, b), limit in _SINGLE_BRANCH.items():
        k = np.flatnonzero(((ends[:, 0] == a) & (ends[:, 1] == b)) | ((ends[:, 0] == b) & (ends[:, 1] == a)))
        if len(k) != 1:
            raise DataConditionError(f"interface branch {a}-{b} found {len(k)} times in the case")
        rating[k] = limit
    return replace(case, rating=rating)
