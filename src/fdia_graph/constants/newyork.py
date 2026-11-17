"""The New York grid: NYISO's zones and neighbors, the fields of the case file, the pinned inputs and
their addresses, NYgrid's dispatch numbers, the fitted load model and the operating rules.

"NYgrid" is the open New York model of the Anderson lab at Cornell (pinned at `NYGRID_COMMIT`);
`updateOpCond.m` and `allocateLoad.m` are its MATLAB files, cited by line. "II-B" to "II-D" are
sections of [LI21], the bottom-up load method. An "N" number (N31) is a decision named in
docs/concepts/new_york_grid.md.
"""

from __future__ import annotations

from enum import IntEnum

# ---- the zones and the neighbors -----------------------------------------------------------------
ZONES = tuple("ABCDEFGHIJK")  # the NYISO load zones; in a case file a bus's zone is 1 for A to 11 for K
FILE_AREA = 1  # a case file is one system with one area: its bus `area` column is 1 on every row
NY_AREA = 2  # `Case.area` of every New York bus (NYgrid's NPCC area number)
# Where a bus is, is its zone code in the case file: 1 to 11 a NYISO load zone, then one code per neighbor.
# Each neighbor of New York as (the code of its buses in a case file's bus `zone` column, its name, the area
# its buses have in a `Case`: NYgrid's NPCC area number, which the stored files and the code use).
REGIONS = ((12, "New England", 1), (13, "Ontario", 4), (14, "MISO", 5), (15, "PJM", 6))
ZONE_LEGEND = f"1 to {len(ZONES)} for the NYISO load zones {ZONES[0]} to {ZONES[-1]}, " + ", ".join(
    f"{code} for {name}" for code, name, _ in REGIONS
)
# NYISO's zone names in its load feed (the "pal" files) -> the zone letter, as NYgrid's loadHourly table maps them.
ZONE_NAME = {
    "WEST": "A",
    "GENESE": "B",
    "CENTRL": "C",
    "NORTH": "D",
    "MHK VL": "E",
    "CAPITL": "F",
    "HUD VL": "G",
    "MILLWD": "H",
    "DUNWOD": "I",
    "N.Y.C.": "J",
    "LONGIL": "K",
}

# ---- the case file -------------------------------------------------------------------------------
# The mpc fields beyond MATPOWER's own: the bus names, each unit's fuel, and the HVDC links.
BUS_NAME, GEN_FUEL, DCLINE = "bus_name", "genfuel", "dcline"
# The mpc fields a case file must have (it may also have `DCLINE`), and the standard MATPOWER ones passed over.
CASE_FIELDS_NEEDED = (
    "version",
    "baseMVA",
    "bus",
    "gen",
    "branch",
    BUS_NAME,
    "bus_latlon",
    GEN_FUEL,
    "gen_ramp",
)
CASE_FIELDS_PASSED_OVER = ("gencost", "gentype", "areas", "dclinecost")
# The fields of a `Case` (`generation.nygrid.case`) read as they are from one column of the case file, as
# floats: field -> (mpc table, MATPOWER's name of the column).
CASE_COLUMNS = {
    "kv": ("bus", "BASE_KV"),
    "vmin": ("bus", "VMIN"),
    "vmax": ("bus", "VMAX"),
    "r": ("branch", "BR_R"),
    "x": ("branch", "BR_X"),
    "b": ("branch", "BR_B"),
    "shift": ("branch", "SHIFT"),
    "rating": ("branch", "RATE_A"),
    "pmin": ("gen", "PMIN"),
    "pmax": ("gen", "PMAX"),
    "qmin": ("gen", "QMIN"),
    "qmax": ("gen", "QMAX"),
    "dc_pmin": ("dcline", "PMIN"),
    "dc_pmax": ("dcline", "PMAX"),
    "pd": ("bus", "PD"),
    "qd": ("bus", "QD"),
    "gs": ("bus", "GS"),
    "bs": ("bus", "BS"),
    "vm": ("bus", "VM"),
    "va": ("bus", "VA"),
    "unit_p": ("gen", "PG"),
    "unit_q": ("gen", "QG"),
    "unit_vg": ("gen", "VG"),
}
HQ_BUS = 48  # updateOpCond.m L367-377: Hydro-Quebec's import is a unit at bus 48, -1100 to 1500 MW
# updateOpCond.m L407-412: the four HVDC links of the reduced case, as from bus, to bus and limit (MW).
HVDC_LINKS = ((21, 80, 530.0), (124, 79, 660.0), (125, 81, 660.0), (125, 81, 660.0))

# ---- the inputs and where they are fetched from --------------------------------------------------
NYGRID_COMMIT = "47698b6c7823ae7b1bc6935e5601d5b64b8f918e"  # the commit of NYgrid every input is read at
NYGRID_ZIP_URL = f"https://codeload.github.com/AndersonEnergyLab-Cornell/NYgrid/zip/{NYGRID_COMMIT}"
TERRITORIES_URL = "https://data.ny.gov/api/geospatial/awza-4vgu?method=export&format=GeoJSON"  # NYS DPS
TERRITORIES_FILE = "nys_electric_territories.geojson"  # the file the utility territories are saved as
# Each NY bus located on the territories (N12), shipped with the package.
BUS_LOCATIONS_FILE = "ny_bus_locations.csv"
# The bottom-up load model's sources [LI21]: NREL's building stock profiles (the bucket, and the key prefix
# of the ResStock and ComStock county files in it, the `{}`), EIA-861 sector sales of 2019, the IAC
# industrial plants, and the census county centroids.
OEDI_URL = "https://oedi-data-lake.s3.amazonaws.com"
STOCK_KEY_PREFIX = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021/{}_amy2018_release_1/timeseries_aggregates/by_county/state=NY/"
EIA_861_URL = "https://www.eia.gov/electricity/data/eia861/archive/zip/f8612019.zip"
IAC_URL = "https://itac.university/storage/ITAC_Database.zip"
GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2019_Gazetteer/2019_Gaz_counties_national.zip"
)
# EIA-861 utility numbers of the client utilities (Sales_Ult_Cust_2019.xlsx).
EIA_UTILITY = {
    "CHGE": 3249,
    "ConEd": 4226,
    "LIPA": 11171,
    "NGRID": 13573,
    "NYSEG": 13511,
    "ORU": 14154,
    "RGE": 16183,
}
# Floor area of each DOE commercial reference building, ft2 (Deru et al., 2011, the prototypes of [23] in
# [2]). ComStock reports energy per represented floor area, so one prototype building is that area.
REFERENCE_FT2 = {
    "fullservicerestaurant": 5500,
    "hospital": 241351,
    "largehotel": 122120,
    "largeoffice": 498588,
    "mediumoffice": 53628,
    "outpatient": 40946,
    "primaryschool": 73960,
    "quickservicerestaurant": 2500,
    "retailstandalone": 24962,
    "retailstripmall": 22500,
    "secondaryschool": 210887,
    "smallhotel": 43200,
    "smalloffice": 5500,
    "warehouse": 52045,
}

# ---- NYgrid's hourly inputs and its dispatch ------------------------------------------------------
# The fuel categories of NYISO's fuel-mix feed, in the order `NygridHour.fuel_mw` holds them.
FUELS = ("Dual Fuel", "Hydro", "Natural Gas", "Nuclear", "Other Fossil Fuels", "Other Renewables", "Wind")
# The interfaces NYgrid reads, by their names in NYISO's interface-flow feed (updateOpCond.m L248-262 and
# L445-451), schedules into NY are positive.
INTERFACE_NAMES = (
    "SCH - PJ - NY",
    "SCH - PJM_HTP",
    "SCH - PJM_NEPTUNE",
    "SCH - PJM_VFT",
    "SCH - HQ - NY",
    "SCH - HQ_CEDARS",
    "SCH - NE - NY",
    "SCH - NPX_1385",
    "SCH - NPX_CSC",
    "SCH - OH - NY",
    "DYSINGER EAST",
    "WEST CENTRAL",
    "TOTAL EAST",
    "MOSES SOUTH",
    "CENTRAL EAST - VC",
    "UPNY CONED",
    "SPR/DUN-SOUTH",
)
# updateOpCond.m L420-430: the branches of each internal NYISO interface as (from bus, to bus, sign) of
# the reduced case, as mapped by end buses in out/ratings_check/interface_branch_map.csv. The sign makes
# the from-end P of every branch count in the interface's positive direction.
INTERFACE_BRANCHES: dict[str, list[tuple[int, int, int]]] = {
    "CENTRAL EAST - VC": [(41, 45, -1), (40, 44, -1), (37, 38, -1), (37, 43, -1)],
    "MOSES SOUTH": [(47, 48, -1), (44, 48, -1), (46, 49, -1)],
    "UPNY CONED": [(73, 74, 1), (74, 77, -1)],
    "SPR/DUN-SOUTH": [(78, 81, 1), (78, 82, 1)],
}
RELEASE_INTERFACE = "SPR/DUN-SOUTH"  # the NYISO interface the release's Am attacks (N48)
EXT_PRICES = ("NPX", "O H", "PJM", "H Q")  # proxy buses pricing the NE, IESO, PJM and HQ units
# External areas (the areas of `REGIONS`) and the proxy bus pricing them (updateOpCond.m L497-500).
AREA_PRICE = {1: "NPX", 4: "O H", 5: "PJM", 6: "PJM"}
# A unit's fuel in NYgrid's unit table -> the column of its weekly fuel-price table that prices it
# (oil and coal at the upstate price; gas is priced by zone).
FUEL_PRICE = {"Coal": "coal_NY", "Fuel Oil 2": "FO2_UPNY", "Kerosene": "FO2_UPNY", "Fuel Oil 6": "FO6_UPNY"}
# The columns of NYgrid's daily nuclear table, in the unit order of the case file: Ginna (53),
# FitzPatrick, Nine Mile 1, 2 (50), Indian Point 2, 3 (74).
NUCLEAR_COLS = (
    "GinnaGen",
    "FitzPatrickGen",
    "NineMilePoint1Gen",
    "NineMilePoint2Gen",
    "IndianPoint2Gen",
    "IndianPoint3Gen",
)
NIAGARA_BUS, ST_LAWRENCE_BUS = 55, 48  # hydro buses with their own rule (updateOpCond.m L143-152)
NIAGARA_CAP_MW, ST_LAWRENCE_CAP_MW = 2460.0, 856.0  # MW: their capacities
# updateOpCond.m L485-491 draws nuclear costs 1 + 2 rand and hydro costs 10 rand ($/MWh) after rng("default")
# every hour, so they are fixed numbers: MATLAB's default stream is Mersenne Twister seeded 5489, the first
# 16 draws of numpy's RandomState(5489), in the unit order of the case file.
NUCLEAR_COST_USD_PER_MWH = (2.6294, 2.8116, 1.2540, 2.8268, 2.2647, 1.1951)
HYDRO_COST_USD_PER_MWH = (2.7850, 5.4688, 9.5751, 9.6489, 1.5761, 9.7059, 9.5717, 4.8538, 8.0028, 1.4189)

# ---- the load model -------------------------------------------------------------------------------
POWER_FACTOR = 0.98  # NYgrid's load power factor (allocateLoad.m), so P4 still compares with MATLAB
POOL_COUNTIES = 5  # II-D: the five prototype locations nearest the bus
SWAP_PAIRS = (100, 100, 50)  # II-D: hour pairs permuted per residential, commercial, industrial pick
# N31 within-hour terms (the docstring of `generation/nygrid/loads.py`). FLUCT_* were fitted on NYISO 2018, zone
# H left out, and SWITCH_* on NYISO 2019 zone D with the model's mean industrial load there (125.43 MW,
# its one NYSEG load bus). The two fitting functions were removed after release 0.21.0 (there
# `_fluctuation_fit` and `_switching_fit` in `fdia_graph/nyfed/scenario/loads.py`); the constants are
# their results.
# The residual's own spread is 0.2554 size^-0.516 (R^2 0.61 over the ten zones); FLUCT_C is the AR(1)'s.
FLUCT_C, FLUCT_BETA, FLUCT_PHI = 0.32547, 0.51590, 0.37889
SWITCH_RATE = 0.0012584  # pulses per MW of mean industrial load per hour
# size / mean industrial load: the 5% threshold, the generalized Pareto shape and scale, the largest step
SWITCH_U, SWITCH_XI, SWITCH_SCALE, SWITCH_MAX = 0.21958, 0.36568, 0.058239, 2.5035
SWITCH_DUR_MU, SWITCH_DUR_SIGMA = 4.61943, 1.63243  # log minutes
SWITCH_OFF = 0.51719  # the share of pulses that start with a drop

# ---- operating rules ------------------------------------------------------------------------------
HYDRO_RAMP_PER_MIN = 0.10  # share of nameplate per minute (USBR, Hydro Model Data Project, 2013, Tables 1-2)
# MW per minute: NY real-time interchange ramp per interface (ISO-NE CROP.31001, 300 MW per 15 min)
IMPORT_RAMP_MW_PER_MIN = 300.0 / 15
BASEPOINT_BAND = 0.03  # share of the upper operating limit (NYISO MST Rate Schedule 3-A, 15.3A.1)
Q_SHARE = 0.312  # |Q| per MW of capability at power factor 0.95 (NYISO Manual 2, 3.6; FERC Order 827)


# ---- the pool file --------------------------------------------------------------------------------
class SolveKind(IntEnum):
    """How a frame of the pool was solved: the `solve_kind` code, one per frame."""

    FAILED = 0  # the dispatch or the power flow did not converge
    SOLVED = 1  # dispatch and AC power flow converged
    NO_HOUR = 2  # the hour the chunk starts from has no AC point (the projection failed): no start
    NO_LOAD = 3  # NYISO has no load for the frame (the spring-forward gap and a data gap), nothing solved


POOL_ARRAYS = ("vm", "va", "p", "q", "solve_kind")  # the per-frame datasets of a pool file
