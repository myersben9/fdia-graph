"""The power systems: the IEEE cases the toolkit builds, the bus types and the "no limit" ratings."""

from __future__ import annotations

from enum import IntEnum


class BusType(IntEnum):
    """MATPOWER's bus types, the `type` column of a case file's bus table [MP19]."""

    PQ = 1  # a load bus: P and Q given
    PV = 2  # a generator bus: P and |V| given
    REF = 3  # the reference (slack) bus: |V| and the angle given


# The ladder of systems: bus count -> the pandapower.networks builder of the case. Transmission systems
# only (>= 110 kV, meshed). Every one has a pool and a timeline at every data release.
IEEE_CASES = {
    14: "case14",
    30: "case30",
    57: "case57",
    89: "case89pegase",
    118: "case118",
    145: "case145",
    200: "case_illinois200",
    300: "case300",
}
# The cases the package holds PGLib-OPF line ratings for (`ratings/`); "pglib" ratings are refused elsewhere.
PGLIB_SYSTEMS = (14, 118, 300)
UNRATED_MVA = 9900.0  # MVA: MATPOWER's "no limit" rating, which PGLib-OPF keeps on a few branches
# MVA, MW or MVAr: the placeholder limit the New York solver gives pandapower where a limit does not
# enter the solve (pandapower's converter fails on a zero rating).
NO_LIMIT = 9999.0
