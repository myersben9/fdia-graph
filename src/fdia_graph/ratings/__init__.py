"""Branch thermal ratings for the overload attack of [WU26] (eqs. 24-25), stored with the package.

The IEEE cases pandapower ships rate every line and transformer at 9,900 MVA, MATPOWER's
placeholder for "no limit", so no flow in the data comes near it. The ratings here are the `rate_a`
of the PGLib-OPF v23.07 versions of IEEE-14, 118 and 300 (IEEE PES Power Grid Library, CC BY 4.0,
https://github.com/power-grid-lib/pglib-opf), one JSON file per case with the source file's name,
URL and sha256, the rows as (from bus, to bus, rate_a MVA) in MATPOWER bus numbers. The generator
matches them to its branches by their end buses (`formulas.attacks.branch_ratings`).
"""

from __future__ import annotations

import json
import os
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
PGLIB_SYSTEMS = (14, 118, 300)  # the cases with ratings; the overload attack is refused elsewhere


def pglib_branches(system: int) -> Optional[list[tuple[int, int, float]]]:
    """The PGLib-OPF branch rows (from bus, to bus, rate_a MVA) of an IEEE case, None when the
    package holds no ratings for it."""
    if system not in PGLIB_SYSTEMS:
        return None
    with open(os.path.join(_HERE, f"pglib_case{system}_ieee.json"), encoding="utf-8") as fh:
        rows = json.load(fh)["branches"]
    return [(int(f), int(t), float(r)) for f, t, r in rows]
