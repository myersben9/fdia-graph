# Test systems

A test system is a power grid the package can build a timeline on. Every system is the same
class, a `Case`, and goes through the same stages ([How the package flows](flow.md)).

## What a `Case` is

`Case` (`systems/case.py`) is one grid as a frozen record of arrays: the buses, the branches,
the generating units, the HVDC links if any, and a base operating state. Powers are in MW and
MVAr, impedances in per unit on `base_mva`, angles in degrees. Branch ends are positions into
the bus arrays, not bus numbers.

There are three ways to make one.

| Call | Reads |
|---|---|
| `Case.from_ieee(118)` | the case of that bus count that pandapower ships, with its base power flow solved |
| `Case.from_pandapower(net)` | a pandapower net whose power flow is solved |
| `Case.from_matpower(path)` | a MATPOWER case file laid out like the New York one |

Each of them ends with `case.check`, which refuses a bus type that is not MATPOWER's 1, 2 or 3,
a case without exactly one reference bus, and a list of names, coordinates, fuels or ramps with the wrong number of rows.
`from_pandapower` refuses a net that holds something a `Case` does not model, such as a
three-winding transformer or an element out of service. Every refusal is a `DataConditionError`
that says what is wrong.

```python
from fdia_graph.systems.case import Case, nygrid_case
from fdia_graph.systems.network import Network

ieee = Case.from_ieee(14)      # needs fdia-graph[generate]
new_york = nygrid_case()       # needs fdia-graph[ny]
net = Network(new_york)        # admittances, built once
print(len(ieee.bus_id), len(new_york.bus_id))  # 14 140
```

## The systems

| System | Source case | Buses | Branches |
|---|---|---|---|
| `ieee14` | `case14` | 14 | 20 |
| `ieee30` | `case30` | 30 | 41 |
| `ieee57` | `case57` | 57 | 80 |
| `ieee89` | `case89pegase` | 89 | 210 |
| `ieee118` | `case118` | 118 | 186 |
| `ieee145` | `case145` | 145 | 453 |
| `ieee200` | `case_illinois200` | 200 | 245 |
| `ieee300` | `case300` | 300 | 411 |
| New York | `case_nygrid.m` | 140 | 227 |

The first eight source cases are builders of `pandapower.networks`, listed in `IEEE_CASES`
(`constants/grid.py`). A branch is a line or a transformer. The New York grid has 271 units.

`ratings/` holds PGLib-OPF line ratings for `ieee14`, `ieee118` and `ieee300`. The cases
pandapower ships carry none.

## The New York case file

`src/fdia_graph/systems/case_nygrid.m` is the project's own test system. It is one MATPOWER
case file, shipped inside the package. A change to the grid is an edit to that file.

It is derived from NYgrid, the open New York model of the Anderson lab at Cornell (MIT
License), at one pinned commit. The starting point is NYgrid's 140-bus NPCC case: New York and
its neighbors. The file's header lists every change:

- the changes of NYgrid's own `modifyMPC.m`: some bus types, loads made positive, three lines
  between PJM and Ontario removed, ten tie-line ratings and two lines added;
- the fixed unit list of NYgrid's hourly case: thermal, nuclear and hydro units, the units of
  the neighboring regions, and the Hydro-Quebec import;
- the units of zone J (New York City) split evenly over buses 81 and 82;
- the 765 kV Massena-Marcy line, from bus 47 to bus 43.

The file is the network with the units' limits. It is not a solved operating point: the New
York units are at zero output, and the loads and the dispatch come from the pool.

Of the 140 buses, 46 are in New York. Ten of the 227 branches have a rating. There is no HVDC
link, and the base is 100 MVA.

### The zone column says where a bus is

| Zone code | Where | Buses |
|---|---|---|
| 1 to 11 | the NYISO load zones A to K | 46 |
| 12 | New England | 36 |
| 13 | Ontario | 31 |
| 14 | MISO | 10 |
| 15 | PJM | 17 |

The file is one system, so it has one area: the bus area column is 1 on every row. The package
reads a bus's region from the zone column. A zone outside 1 to 15, or an area other than 1, is
refused. The codes and their names are in `constants/newyork.py` (`ZONES`, `REGIONS`).

Beyond MATPOWER's own fields the file has `mpc.bus_name`, `mpc.genfuel`, `mpc.bus_latlon`
(degrees) and `mpc.gen_ramp` (MW per minute). It has no `mpc.gencost`.

`pipelines/newyork/make_case.py` is the recipe of the file: it makes the file again from
NYgrid's `Data/` folder. A test checks that the recipe still reproduces the shipped file
exactly.

A change to the grid is an edit to the file, one row per line. No code patches the case on the
way in. A variant in code is a copy, never a change in place: `hvdc_links(case)` or
`case.with_voltage_margin(0.02)`.

### What the reader refuses

`Case.from_matpower` reads the file with matpowercaseframes, the reader pandapower uses for
`.m` files. A branch with status 0 is left out, and a tap ratio of 0, MATPOWER's mark of a
line, is read as 1. A file whose parts disagree is refused with a `DataConditionError` that
names the file and what is wrong:

- a bus number that is not whole, or is on two rows;
- a unit or a branch on a bus the case does not have;
- a zone number outside 1 to 15, or an area other than 1;
- an indexed assignment such as `mpc.bus(1, 3) = 500;`;
- a field that is not one of the case's. The standard `mpc.gencost`, `mpc.gentype`,
  `mpc.areas` and `mpc.dclinecost` are passed over.

`tests/nygrid/test_case_file.py` has one test per rule.

### Areas in the code

The file has one area, but `Case.area` is derived from the zone code on read: 2 for a New York
bus (`NY_AREA`), and the source model's own area number for each neighbor.

| Zone code | `Case.zone` | `Case.area` |
|---|---|---|
| 1 to 11 | `A` to `K` | 2 |
| 12, New England | `""` | 1 |
| 13, Ontario | `""` | 4 |
| 14, MISO | `""` | 5 |
| 15, PJM | `""` | 6 |

### Reading the file

In this package:

```python
from fdia_graph.systems.case import CASE_FILE, Case, nygrid_case

case = nygrid_case()                    # the shipped file
same = Case.from_matpower(CASE_FILE)    # or any file laid out like it
```

In MATPOWER the file is an ordinary case function:

```matlab
mpc = loadcase('case_nygrid');
```

In pandapower, build the net from the `Case`. `PandapowerSolver` does it
([Solving](solving.md)):

```python
from fdia_graph.data.pandapower_solver import PandapowerSolver

net = PandapowerSolver(case).net    # 140 buses, 271 generators
```

pandapower's own converter, `from_mpc`, fails on this file in pandapower 3.4.0. The converter
has a bug on a branch that joins two voltage levels and has no rating, and the file has 83 of
them. `PandapowerSolver` avoids it by giving every branch a placeholder rating.

## The other files of `systems/`

| File | Holds |
|---|---|
| `fixes.py` | two variants of the New York case, each a copy: `externals_at_base` and `hvdc_links` |
| `interfaces.py` | `Interface`: a New York transmission interface, its branches with its hourly limit and flow |
| `ownership.py` | `Ownership`: which utility owns each New York bus |
| `states.py` | `PoolStates`: solved voltages read frame by frame as the states a timeline walks |
| `tables.py` | the reader of the CSV input tables |
| `pandapower_types.py` | the pandapower net, named as types for `from_pandapower` |

## Which systems have released datasets

| System | Released | How to get a timeline |
|---|---|---|
| the eight `ieee` systems | a pool and a timeline of 72,000 frames each | `fg.load("ieee118")` downloads it |
| New York | nothing yet | build the pool ([Build the operating points](../guides/operating_points.md)), then the timeline ([Generate a timeline](../guides/generation.md)) |

You can also generate your own timeline on any of the eight `ieee` systems with `fg.generate`
([the tutorial](../getting-started/generate.md)).
