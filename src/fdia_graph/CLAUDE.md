# The New York code in the SDK

The New York grid's code lives in the SDK's own packages, by job. Some of its modules are listed
in `DEFERRED` in `tools/readability.py` until each is brought under the SDK's rules; until then the rules below
hold for them.

| Job | Where |
|---|---|
| the grid: `Case` and its readers, `Network`, the case file and its variants, the client that owns each bus | `systems/` |
| the build: the inputs, loads, dispatch and the pool | `data/` (`data/pool.py` reads and writes the pool file); the scripts `pipelines/newyork/run_pool.py` and `make_case.py` |
| the timeline written from a pool | `generation/search.py`: `SearchSource.for_pool`, the one source every grid uses, on the one path `TimelineSource.generate` (`generation/source.py`); written by the writer every timeline uses (`TimelineWriter`, `generation/write.py`) in the one layout (`schema.py`) |
| the meter table, the measurement function and meter noise | `sensing/` |
| the attack families: the search-based At and Am of every grid, and `AttackFamily`, the base of a family that designs an attack vector | `engine/attacks/` |
| the alarm rules | `se/detect/` of the analysis toolkit (`analysis/src/fdia_analysis/se/detect/`) |
| the clients and their data views | `systems/ownership.py`, `dataset/clients.py` |
| the Ipopt every OPF of the build runs on | `data/ipopt.py` |

## Rules that hold

- One path generates a timeline, for the IEEE cases and for this grid: `TimelineSource.generate`
  (`generation/source.py`), through one source, `SearchSource` (`generation/search.py`). This grid has no source of
  its own: it is a case and a pool file handed to `SearchSource.for_pool`, with coverage meters
  (`MeterPlan.from_coverage`) and the search-based designs. Never a second driver: a choice any grid can make is a
  setting (`split_by`, `loads`), a design family of the caller's is the `families` argument, and the owners of the
  buses are the `own` argument. The file layout is not this grid's own: one layout (`schema.py`) and one writer serve
  every grid.
- `systems/case.py`, `systems/network.py`, `generation/` and `sensing/` are measured by `tools/readability.py` like
  the SDK's own code; the modules still listed in `DEFERRED` are not yet.
- The grid is the MATPOWER case file `case_nygrid.m` in `systems/`, read by `nygrid_case()`
  (`systems/case.py`, with matpowercaseframes). A change to the grid is an edit to that file; no code assembles or patches the case on the way
  in. `pipelines/newyork/make_case.py` is the recipe that first made the file from NYgrid's `Data/` folder.
- pandapower is imported only in `data/pandapower_solver.py` and, lazily, in `data/ieee_pool.py` and `Case.from_ieee` (`systems/case.py`). Optimization problems are Pyomo models solved by
  `data.ipopt.ipopt()` (`data/ipopt_solver.py`); the one exception is `PandapowerSolver.opf`, pandapower's own OPF, kept as the second engine of the solve setting. h5py only in the timeline writer every grid shares (`TimelineWriter`, `generation/write.py`), the pool readers (`generation/__init__.py`, `data/pool.py`) and `dataset/`; a source holds the writer (`source.writer`), never the open file. Random generators of the New York build come from
  `data.config` and are passed in: `rng()`, keyed by the seed and the stage's name. The shared path keys its
  streams by the seed itself: the week split (`TimelineSource.stream("split")`: the seed and the stream's name, the
  key `rng()` gives a build stage), a frame's meter jitter (seed, `JITTER_STREAM`, pool frame), an episode's design
  (seed, `DESIGN_STREAM`, split, slot, move) and an episode's move (seed, `RELOCATE_STREAM`, split, slot, move).
- Records are frozen dataclasses of numpy arrays. A class owns state, cached work or a shared interface. Stateless
  math is a plain function, and a formula the SDK already has in `formulas/` is called, not rewritten.
- Measurements are the flat vector `z` with the `MeterPlan` table. A client's data is `z[:, meters.owner == c]`.
  A `ClientView` never exposes another client's meters. In a file the readings are the layers of `schema.py`:
  `MeterPlan.scatter` lays a value per meter out as layers, `MeterPlan.from_masks` and `MeterPlan.gather` read the
  table and the flat vector back, exactly. No New York path or attribute is spelled outside `schema.py`.
- The layout still allows flows at both branch ends (`LayerSet.both_ends`) and currents as magnitude and angle
  (`CURRENT_POLAR` in `schema.py`, `MeterKind.I` and `IA`), and the readers accept such files. `SearchSource` writes from-end
  flows and rectangular currents.
- Every change keeps the SDK's frozen references (`tests/golden/`, `tests/test_timeline_digests.py`) unless the
  change is meant to move a number, and says so in `CHANGELOG.md`.

## Work

- Tests are in `tests/nygrid/` and read the inputs (`data/`, `out/`) under `NYGRID_ROOT`; the timeline tests run on
  the two checked-in days of the pool (`tests/nygrid/data/days`). Run them before saying a change is done, and report
  the numbers it produced.
- Never commit or push unless asked. Never add AI attribution to commits.
