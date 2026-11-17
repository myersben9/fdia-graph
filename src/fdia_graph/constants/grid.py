"""The power systems: the IEEE cases the toolkit builds, the bus types and the "no limit" ratings."""

from __future__ import annotations

from enum import IntEnum


class BusType(IntEnum):
    """MATPOWER's bus types, the `type` column of a case file's bus table [MP19]."""

    PQ = 1  # a load bus: P and Q given
    PV = 2  # a generator bus: P and |V| given
    REF = 3  # the reference (slack) bus: |V| and the angle given


class BusColumn(IntEnum):
    """The columns of a PYPOWER bus table that `Case.from_pandapower` reads [MP19]."""

    GS = 4  # shunt conductance (MW at 1 pu)
    BS = 5  # shunt susceptance (MVAr at 1 pu)


class BranchColumn(IntEnum):
    """The columns of a PYPOWER branch table that `Case.from_pandapower` reads [MP19], per unit."""

    F_BUS = 0  # the from bus, as a row of the bus table
    T_BUS = 1  # the to bus
    BR_R = 2  # series resistance
    BR_X = 3  # series reactance
    BR_B = 4  # charging susceptance
    TAP = 8  # turns ratio (pandapower writes 1 for a line)
    SHIFT = 9  # phase shift (degrees)
    BR_STATUS = 10  # 1 in service, 0 switched off
    BR_G = 23  # shunt conductance, pandapower's own column: a transformer's iron losses


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
# The element tables of a pandapower net, as `Case.from_pandapower` reads one (`models.inputs.ModelledNet`):
PANDAPOWER_IN_SERVICE = ("bus", "line", "trafo", "load", "gen")  # every row must be in service
PANDAPOWER_SCALED = ("load", "gen", "sgen")  # their `scaling` column must be 1
# the tables with grid elements a `Case` has no place for: each must be empty
PANDAPOWER_NOT_READ = (
    "trafo3w",
    "impedance",
    "ward",
    "xward",
    "dcline",
    "switch",
    "storage",
    "motor",
    "asymmetric_load",
    "asymmetric_sgen",
    "svc",
    "tcsc",
    "ssc",
    "vsc",
    "line_dc",
)
# The cases the package holds PGLib-OPF line ratings for (`ratings/`); "pglib" ratings are refused elsewhere.
PGLIB_SYSTEMS = (14, 118, 300)
UNRATED_MVA = 9900.0  # MVA: MATPOWER's "no limit" rating, which PGLib-OPF keeps on a few branches
# MVA, MW or MVAr: the placeholder limit the New York solver gives pandapower where a limit does not
# enter the solve (pandapower's converter fails on a zero rating).
NO_LIMIT = 9999.0
