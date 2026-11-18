# Solving: power flow and optimal power flow

A timeline starts from a pool: the AC operating point of every frame
([How the package flows](flow.md)). Solving is the step that builds the pool
([Build the operating points](../guides/operating_points.md) has the commands). There are two kinds
of solve, and every model in the package is AC.

## The two kinds

**Power flow.** The loads and each generator's output are given. The solve finds the voltage at
every bus that satisfies the AC equations. Nothing is chosen.

**Optimal power flow (OPF).** The generator outputs are not given. The solve chooses them to
make a cost as small as possible, while the AC equations and the limits of the units and the
voltages hold. It is an optimization problem and needs a nonlinear solver.

## Where each is used

Building the pool, with each builder's own choice
([the setting](#the-solve-setting) changes it):

| Test system | Solve | Code |
|---|---|---|
| IEEE cases | one power flow per step, in pandapower | `generate_states` in `data/ieee_pool.py` |
| New York | one OPF per frame, in Pyomo with Ipopt | `IpoptSolver.dispatch` in `data/ipopt_solver.py` |
| New York | then one power flow at the OPF's outputs, in pandapower | `PandapowerSolver.power_flow` in `data/pandapower_solver.py` |

The two engines are two classes behind one interface, `Solver` (`data/solver.py`).
`IpoptSolver` is the package's own OPF, a Pyomo model solved by Ipopt. `PandapowerSolver`
builds a pandapower net from the `Case` and runs pandapower's power flow and its OPF.

Only one solve is shared by both engines: `opf(pd, qd, costs)`, the AC point of least cost for
the bus loads, with every unit and every voltage within the case's limits and no branch limit.
`power_flow` is `PandapowerSolver`'s only. `project` and `dispatch`, the New York dispatch's own
OPFs, are `IpoptSolver`'s only.

The released IEEE power flow does not go through `Solver`. `generate_states` runs it on the net
pandapower ships, as it always has, because the released pools depend on that net bit for bit.

`PoolBuilder` (`data/pool.py`) runs the New York solves in time order. First,
`IpoptSolver.project` solves one OPF per hour: the AC point closest to the source model's hourly
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

## The solve setting

The pool-building step of every test system takes one setting, `SolveSettings`
(`models/config.py`). It has two fields:

| Field | Values | Meaning |
|---|---|---|
| `method` | `"power_flow"`, `"opf"` | how each step is solved |
| `engine` | `"pandapower"`, `"ipopt"` | what runs the solve |

Leave the setting out and each builder solves as the first table says: `generate_states` with
a power flow in pandapower, `PoolBuilder.from_settings` with an OPF in Ipopt. The generator
reads the pool and does not solve again.

```python
import numpy as np

from fdia_graph.data.ieee_pool import generate_states
from fdia_graph.models.config import SolveSettings

profile = np.sin(np.arange(288) / 46.0)
pool = generate_states(118, profile, solve=SolveSettings(method="opf", engine="ipopt"))
```

What each choice does:

| Test system | `method` | `engine` | Result |
|---|---|---|---|
| IEEE | `power_flow` | `pandapower` | the default: the released pools |
| IEEE | `opf` | `ipopt` | a least-cost OPF per step |
| IEEE | `opf` | `pandapower` | the same problem, in pandapower |
| New York | `opf` | `ipopt` | the default: the dispatch described above |
| New York | `opf` | `pandapower` | refused |
| New York | `power_flow` | `pandapower` | refused |
| any | `power_flow` | `ipopt` | refused |

### An IEEE pool by OPF

An IEEE OPF step scales the loads as the power-flow step does, then chooses each generator's
output. The limits are the case's own. The cost is the case's own cost curve, read by
`Case.pandapower_costs` into a `UnitCosts`: the linear term ($/MWh) and the quadratic term
($/MW^2h) of each unit. The constant term is not read. It does not change the optimum, and it
is not in the cost an OPF reports. Branch ratings are not enforced, and the voltage at the
reference bus is free within its limits.

Every step is solved from the same start, the case's base state. The pool is therefore the
same for any number of `workers`.

A step with no optimum is dropped, as a step whose power flow does not converge is. At their
base loads IEEE-57, IEEE-145, IEEE-200 and IEEE-300 have no optimum within the case's limits in
either engine. An OPF pool in which no step has an optimum raises `DataConditionError`.

### Why a choice is refused

Each refusal is a `ConfigError`.

- A power flow in Ipopt: the Ipopt engine solves an OPF only. pandapower runs every power
  flow.
- New York with a power flow: `PoolBuilder` builds by OPF only. The source model gives hourly
  generator outputs, and a power flow at those outputs leaves voltages far below the case's
  limits. A power flow at given outputs is `PandapowerSolver(case).power_flow`.
- New York with pandapower's OPF: the dispatch limits each unit's change between frames by
  its ramp and holds the hydro and external units near their schedule by a penalty.
  pandapower's OPF solves plain least cost only.

The New York power flow was measured on 40 hours of 2019, every 219th hour, each from a flat
start at the source model's outputs and the case's voltage setpoints. All 40 converged. The
lowest bus voltage was between 0.72 and 0.82 p.u., where the case's lower limit is 0.9, and the
reference bus made up between 1,001 and 2,532 MW. No hour was within the voltage limits.

One New York operating point at plain least cost can be solved in either engine, with
`IpoptSolver(case).opf(pd, qd, costs)` or `PandapowerSolver(case).opf(pd, qd, costs)`.

## The two OPF engines compared

`tests/nygrid/test_solve_engines.py` solves the same case, loads and costs in both engines
and checks that they agree.

| Case | Cost, Ipopt ($/h) | Cost, pandapower ($/h) | Time, Ipopt | Time, pandapower |
|---|---|---|---|---|
| IEEE-14 | 8,081.526 | 8,081.527 | 0.09 s | 0.40 s |
| IEEE-30 | 574.517 | 574.517 | 0.09 s | 0.33 s |
| IEEE-118 | 129,668.852 | 129,668.854 | 0.18 s | 0.71 s |
| New York, 70 % load | 417,428.790 | 417,428.797 | 0.22 s | 3.69 s |

The IEEE rows use the case's base loads and its linear and quadratic cost. The New York row
uses a flat linear cost per fuel. On these four the cost agrees within 0.01 %, the losses
within 0.01 MW and every unit's output within 0.1 MW. The times are one solve on one desktop
machine, after the model is built.

The engines agree on every unit's output only where the optimum is one point. IEEE-89 has
linear costs and several units at the same cost: the two costs agree to 2 parts in 10 million,
and single units differ by up to about 20 MW. Over a load profile on IEEE-14, a unit near its
lower limit differs by up to 0.3 MW between the engines at the same cost.

pandapower's OPF of the net it ships gives 8,081.527 $/h on IEEE-14, the same problem. On
IEEE-30 it gives 578.486 $/h against 574.517 here. That net's OPF also enforces branch
ratings, which this problem does not: with the ratings lifted it gives 575.352 $/h. The rest
of the difference is not accounted for.

pandapower's OPF is fragile on the New York case. At the case's base load, and at 80 %, 90 %
and 105 % of it, it does not converge where Ipopt does. The comparison therefore uses 70 % of
the base load. Even there it depends on the machine: it converges on Windows and not on the
Linux test runner, where the test reports that it was skipped. The IEEE comparisons run
everywhere Ipopt is installed.
