# The New York code in the SDK

The New York grid's code lives in the SDK's own packages, by job. Its modules are listed
in `DEFERRED` in `tools/readability.py` until each is brought under the SDK's rules; until then the rules below
hold for them.

| Job | Where |
|---|---|
| the system: the case file, its variants, interfaces, reduced network, loads, dispatch and the pool | `generation/nygrid/` |
| the timeline written from a pool, and its HDF5 files | `generation/nygrid/source.py` (`NewYorkSource` and `SingleEpisodeSource`, the grid's steps of the one path `TimelineSource.generate`; its plan stage is `generation/plan.py`), written by the writer every timeline uses (`TimelineWriter`, `generation/write.py`) in the one layout (`schema.py`); `io.py` is the pool file |
| the meter table, the measurement function and meter noise | `engine/sensing/` |
| the attack families (`AttackFamily`, `Attacker`, Am, the OPF and interface attacks, Ac, At) | `engine/attacks/` |
| the alarm rules | `se/detect/` of the analysis toolkit (`analysis/src/fdia_analysis/se/detect/`) |
| the utility clients and their data views | `generation/nygrid/ownership.py`, `dataset/clients.py` |
| the Ipopt every OPF runs on | `engine/solvers.py` |

## Rules that hold

- One path generates a timeline, for the IEEE cases and for this grid: `TimelineSource.generate`
  (`generation/source.py`). What is this grid's own (the onset masks and the survey, the splits, designs, frames)
  is a method of `NewYorkSource`; never a second driver, and no mode flag: a variant is a subclass. The file
  layout is not this grid's own: one layout (`schema.py`) and one writer serve every grid.
- `generation/nygrid/source.py` is measured by `tools/readability.py` like the SDK's own code; the other modules
  here are still listed in `DEFERRED`.
- The grid is the MATPOWER case file `case_nygrid.m` in `generation/nygrid/`, read by `nygrid_case()`
  (`case.py`, with matpowercaseframes). A change to the grid is an edit to that file; no code assembles or patches the case on the way
  in. `pipelines/newyork/make_case.py` is the recipe that first made the file from NYgrid's `Data/` folder.
- pandapower is imported only in `generation/nygrid/solver.py`. Optimization problems are Pyomo models solved by
  `engine.solvers.ipopt()`. h5py only in the timeline writer every grid shares (`TimelineWriter`, `generation/write.py`), the pool readers (`generation/__init__.py`, `generation/nygrid/io.py`) and `dataset/`; a source holds the writer (`source.writer`), never the open file. Random generators come only from
  `generation.nygrid.config.rng()` and are passed in.
- Records are frozen dataclasses of numpy arrays. A class owns state, cached work or a shared interface. Stateless
  math is a plain function, and a formula the SDK already has in `formulas/` is called, not rewritten.
- Measurements are the flat vector `z` with the `MeterPlan` table. A client's data is `z[:, meters.owner == c]`.
  A `ClientView` never exposes another client's meters. In a file the readings are the layers of `schema.py`:
  `MeterPlan.scatter` lays a value per meter out as layers, `MeterPlan.from_masks` and `MeterPlan.gather` read the
  table and the flat vector back, exactly. No New York path or attribute is spelled outside `schema.py`.
- Every change keeps the golden snapshots of `tests/nygrid/test_golden.py` (and the SDK's frozen references)
  unless the change is meant to move a number, and says so in `CHANGELOG.md`.

## Work

- Tests are in `tests/nygrid/` and read the inputs (`data/`, `out/`) under `NYGRID_ROOT`. Run them before saying a
  change is done, and report the numbers it produced.
- Never commit or push unless asked. Never add AI attribution to commits.
