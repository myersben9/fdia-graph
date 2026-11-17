"""Where the load data and the operating points come from, for every test system.

Load data: `profiles.py` reads a load series from a file or fetches it from an operator's feed, as a
load-scaling vector; `inputs.py` downloads and reads the New York grid's public input tables
(`config.py` holds that build's settings and makes its random streams), `load_sources.py` fetches the
sources of its bottom-up loads and `loads.py` builds each bus's load from them (the buses' places are the
packaged `ny_bus_locations.csv`).

Operating points, the pool a timeline is written from: `ieee_pool.py` solves an IEEE case over a load
profile (`generate_states`); for the New York grid `dispatch.py` sets each hour's dispatch and `pool.py`
builds the year of frames (`PoolBuilder`, the `Pool`) and reads and writes the pool file. A solve runs on
one of two engines behind `solver.py`'s `Solver`: `ipopt_solver.py` (the package's OPF, on the Ipopt of
`ipopt.py`) or `pandapower_solver.py`; a `SolveSettings` given to a pool builder picks it.

Import each name from its module; importing this package alone loads neither pandapower nor Pyomo.
"""
