# Solving: power flow and optimal power flow

A timeline starts from a pool: the AC operating point of every frame
([How the package flows](flow.md)). Solving is the step that builds the pool. There are two kinds
of solve, and every model in the package is AC.

## The two kinds

**Power flow.** The loads and each generator's output are given. The solve finds the voltage at
every bus that satisfies the AC equations. Nothing is chosen.

**Optimal power flow (OPF).** The generator outputs are not given. The solve chooses them to
make a cost as small as possible, while the AC equations and the limits of the units and the
voltages hold. It is an optimization problem and needs a nonlinear solver.

## Where each is used

Building the pool:

| Test system | Solve | Code |
|---|---|---|
| IEEE cases | one power flow per step, in pandapower | `generate_states` in `data/ieee_pool.py` |
| New York | one OPF per frame, in Pyomo with Ipopt | `AcSolver.dispatch` in `data/solver.py` |
| New York | then one power flow at the OPF's outputs, in pandapower | `AcSolver.power_flow` |

`PoolBuilder` (`data/pool.py`) runs the New York solves in time order. First,
`AcSolver.project` solves one OPF per hour: the AC point closest to the source model's hourly
outputs. The nuclear, hydro and external units follow these hourly points. Then each frame is
dispatched. A thermal unit's output may differ from its output in the frame before by at most
its ramp. A New York frame that does not solve keeps its place in the pool with an empty state.
An IEEE step whose power flow does not converge is dropped from the pool.

After the pool exists, a benign scan needs no solve: it is read from the stored state. Only the
attacks solve again, and only on a small area of the grid.

| Attack design | Solve | Code |
|---|---|---|
| `At` and `Am` on the IEEE cases | a local AC solve in numpy | `local_ac_solve`, `local_flow_solve` in `formulas/network.py` |
| `At` on New York | the same local solve | `Attacker.false_state` in `engine/attacks/attacker.py` |
| `Am` on New York | a local OPF, in Pyomo with Ipopt | `OpfFdia` in `engine/attacks/opf.py` |

`data/ipopt.py` has one function, `ipopt`. Every OPF gets its solver from it. It finds
Ipopt, checks that it starts, and raises `SolverUnavailable` with the fix when it does not.

`Case.from_ieee` also solves the base power flow of an IEEE case once, in pandapower.

## What to install

| You want to | Install |
|---|---|
| load a released timeline | `pip install fdia-graph` |
| generate an IEEE timeline | `pip install "fdia-graph[generate]"` (pandapower) |
| build the New York pool or timeline | `pip install "fdia-graph[ny]"` (pandapower, Pyomo), and Ipopt |

Ipopt is a program, not a Python package, so pip does not install it:

```bash
conda install -c conda-forge ipopt
```

Check that the package finds it:

```python
from fdia_graph.data.ipopt import ipopt

print(ipopt().version())
```

## The choice is fixed per test system

Today the kind of solve is part of each test system's code, not a setting. An IEEE pool is
built with power flows and the New York pool with OPFs, and no argument switches one for the
other.
