"""The test systems: a grid as a `Case` with its `Network`, whichever grid it is.

`case.py` holds `Case`, one grid's buses, branches and units with a base operating state, read from a MATPOWER
case file (`Case.from_matpower`) or from a pandapower net (`Case.from_pandapower`, whose input `pandapower_types.py`
names as types). `Case.from_ieee` gives an IEEE case pandapower ships, `nygrid_case()` the New York grid of the
case file `case_nygrid.m` beside it, and `fixes.py` holds that grid's two variants. `network.py` holds `Network`,
a case's admittances and the AC quantities they give for a state. `ownership.py` says which utility owns each
bus (`Ownership`), from the CSV tables `tables.py` reads. Import each name from its module; importing this
package alone loads nothing.
"""
