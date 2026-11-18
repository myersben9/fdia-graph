# The New York code in the SDK

The New York grid's code lives in the SDK's own packages, by job. Its modules are listed
in `DEFERRED` in `tools/readability.py` until each is brought under the SDK's rules; until then the rules below
hold for them.

| Job | Where |
|---|---|
| the grid: `Case` and its readers, `Network`, the case file and its variants, the utility that owns each bus, the interfaces | `systems/` |
| the build: the inputs, loads, dispatch and the pool | `data/` |
| the timeline written from a pool, and its HDF5 files | `generation/newyork.py` (`NewYorkSource` and `SingleEpisodeSource`, the grid's steps of the one path `TimelineSource.generate`; its plan stage is `generation/plan.py`; the original generator's source, for a pool of its own kind, is `generation/search.py`), written by the writer every timeline uses (`TimelineWriter`, `generation/write.py`) in the one layout (`schema.py`); `data/pool.py` reads and writes the pool file |
| the meter table, the measurement function and meter noise | `sensing/` |
| the attack families (`AttackFamily`, `Attacker`, Am, the OPF and interface attacks, Ac, At) | `engine/attacks/` |
| the alarm rules | `se/detect/` of the analysis toolkit (`analysis/src/fdia_analysis/se/detect/`) |
| the utility clients and their data views | `systems/ownership.py`, `dataset/clients.py` |
| the Ipopt every OPF runs on | `data/ipopt.py` |

## Rules that hold

- One path generates a timeline, for the IEEE cases and for this grid: `TimelineSource.generate`
  (`generation/source.py`). What is this grid's own (the onset masks and the survey, what a file records) is a method of
  `NewYorkSource`, its designs are `PoolDesigner`'s, and the shared steps are `TimelineSource`'s, the cut of the splits
  included (`split_codes`, by the `split_by` setting; whole weeks are this grid's default). Never a second driver: a
  choice any grid can make is a setting, a variant of this grid is a subclass. The file
  layout is not this grid's own: one layout (`schema.py`) and one writer serve every grid.
- `generation/newyork.py`, `systems/case.py`, `systems/network.py` and `sensing/` are measured by
  `tools/readability.py` like the SDK's own code; the other New York modules are still listed in `DEFERRED`.
- The grid is the MATPOWER case file `case_nygrid.m` in `systems/`, read by `nygrid_case()`
  (`systems/case.py`, with matpowercaseframes). A change to the grid is an edit to that file; no code assembles or patches the case on the way
  in. `pipelines/newyork/make_case.py` is the recipe that first made the file from NYgrid's `Data/` folder.
- pandapower is imported only in `data/pandapower_solver.py` and, lazily, in `data/ieee_pool.py` and `Case.from_ieee` (`systems/case.py`). Optimization problems are Pyomo models solved by
  `data.ipopt.ipopt()` (`data/ipopt_solver.py`, `engine/attacks/opf.py`); the one exception is `PandapowerSolver.opf`, pandapower's own OPF, kept as the second engine of the solve setting. h5py only in the timeline writer every grid shares (`TimelineWriter`, `generation/write.py`), the pool readers (`generation/__init__.py`, `data/pool.py`) and `dataset/`; a source holds the writer (`source.writer`), never the open file. Random generators of the New York build come from
  `data.config` and are passed in: `rng()` for the meters and the placement, and `stream_key()` for
  each design, whose generator the design process makes from the key. The shared path keys three streams by the
  seed itself: the week split (`TimelineSource.stream("split")`: the seed and the stream's name, the key `rng()` gives
  a build stage), a frame's meter jitter (seed, `JITTER_STREAM`, pool frame) and an episode's move
  (seed, `RELOCATE_STREAM`, split, slot, move).
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
