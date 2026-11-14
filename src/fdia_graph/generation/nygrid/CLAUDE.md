# The New York code in the SDK

The New York grid's code lives in the SDK's own packages, by job. Its modules are listed
in `DEFERRED` in `tools/readability.py` until each is brought under the SDK's rules; until then the rules below
hold for them.

| Job | Where |
|---|---|
| the system: NYgrid's case, its fixes, interfaces, reduced network, loads, dispatch and the pool | `generation/nygrid/` |
| the timeline written from a pool, and its HDF5 files | `generation/nygrid/timeline.py`, `schedule.py`, `io.py` |
| the meter table, the measurement function and meter noise | `engine/sensing/` |
| the attack families (`AttackFamily`, `Attacker`, Am, the OPF and interface attacks, Ac, At) | `engine/attacks/` |
| the operator under attack | `operator/` |
| the residual test and its alarm rules | `se/detect/` |
| the utility clients and their data views | `federated/utilities.py`, `dataset/clients.py` |
| study helpers | `fdia_graph.results.studies` |
| the Ipopt every OPF runs on | `engine/solvers.py` |

## Rules that hold

- pandapower is imported only in `generation/nygrid/solver.py`. Optimization problems are Pyomo models solved by
  `engine.solvers.ipopt()`. h5py only in `generation/nygrid/io.py` and `dataset/`. Random generators come only from
  `generation.nygrid.config.rng()` and are passed in.
- Records are frozen dataclasses of numpy arrays. A class owns state, cached work or a shared interface. Stateless
  math is a plain function, and a formula the SDK already has in `formulas/` is called, not rewritten.
- Measurements are the flat vector `z` with the `MeterPlan` table. A client's data is `z[:, meters.owner == c]`.
  A `ClientView` never exposes another client's meters.
- Every change keeps the golden snapshots of `tests/nygrid/test_nygrid_golden.py` (and the SDK's frozen references)
  unless the change is meant to move a number, and says so in `CHANGELOG.md`.

## Work

- Tests are in `tests/nygrid/` and read the inputs (`data/`, `out/`) under `NYGRID_ROOT`. Run them before saying a
  change is done, and report the numbers it produced.
- Never commit or push unless asked. Never add AI attribution to commits.
