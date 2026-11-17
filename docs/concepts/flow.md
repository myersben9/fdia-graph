# How the package flows

A timeline is the meter readings of one grid, frame by frame, with attacks placed in it and
labelled per bus. The package builds one in the same stages for every test system. What differs
between systems is the settings, and how the operating points are solved.

![Six stages, top to bottom. 1. Test system: Case, Network. 2. Operating points: generate_states for the IEEE cases, PoolBuilder for New York. 3. Meters and attacks: MeterPlan, NoiseModel, the attack designs. 4. Timeline file: TimelineSource.generate, TimelineWriter. 5. Load: fg.load returns FdiaGraph. 6. Analyse: fdia_analysis.](../figures/diagrams/flow.png)

The sections follow the picture; box 3 is two sections, the meters and then the attacks. Paths
are under `src/fdia_graph/` unless a section says otherwise.

Three folders serve every stage. `formulas/` holds the mathematics as pure functions, one per
equation. `models/` holds every bundle of values that is passed or returned. `constants/` holds
the fixed values, each with its unit and its source.

## Test system

The grid: its buses, branches and generating units, with a base operating state.

- `Case` (`systems/case.py`) holds one grid as a frozen record of arrays.
- `Network` (`systems/network.py`) builds the admittances of a `Case` once. It gives the
  injections, flows and currents of any state.

`Case.from_ieee(118)` gives an IEEE case and `nygrid_case()` gives the New York grid. Both return
the same class. [Test systems](test_systems.md) lists every system.

You now have: a `Case` and its `Network`, which describe the grid.

## Operating points

One AC operating point per frame, in time order. The set is called the pool.

- IEEE cases: `generate_states` in `profiles.py` scales the loads along a load profile and solves
  one power flow per step. `load_profile` and `fetch_profile` read the profile.
- New York grid: `PoolBuilder` in `generation/nygrid/pool.py` solves one optimal power flow per
  five-minute frame, with `AcSolver` (`generation/nygrid/solver.py`).

`generation/nygrid/` holds everything only the New York grid needs: its loads, its dispatch,
its pool and its timeline source.

The IEEE pools are released, so `fg.generate` downloads one and you do not build it.
[Solving](solving.md) explains the two kinds of solve, and what happens to a frame that does
not solve.

You now have: the pool, the true state of the grid at every frame.

## Meters and noise

What the operator reads from a state. The three classes are in `sensing/` and serve every grid.

- `MeterPlan` (`meters.py`): every meter as one row of a table, with its kind, its place, its
  owner and its accuracy class.
- `Measurement` (`measure.py`): the measurement function h(x), the readings a state gives, and
  its Jacobian.
- `NoiseModel` (`noise.py`): each meter's accuracy class, split into a constant bias and a jitter
  drawn again at every scan.

A benign scan is h(x) of a pool state plus noise. It needs no new solve.

You now have: one benign scan per frame, the readings with no attack.

## Attacks

An attack is an episode: a run of consecutive frames under one design. `engine/` holds the
generator of the IEEE cases (`core.py`) and, under `engine/attacks/`, the attack designs of
every grid. There are two families.

- `At`, the slow ramp: a false load that rises a little each frame.
- `Am`, the overload: the reported flow of a line or an interface rises to its limit.

Each attacked frame is the reading of one false AC state. The attacker re-solves a small area of
the grid and holds every voltage outside it true. The attack vector is h(x_false) - h(x_true),
added to the benign scan.

| Grid | `At` | `Am` |
|---|---|---|
| IEEE cases | `ramp_design` | `am_overload_design` |
| New York | `Ramp` | `InterfaceFdia` |

The IEEE designs are methods of `FdiaGenerator` (`engine/core.py`), which gets them from
`AttackMixin`. The New York designs are classes on one `Attacker` (`engine/attacks/attacker.py`).
A timeline asks every family the same thing, the episode of a window (`EpisodeFamily.designed`,
`engine/attacks/family.py`); the IEEE designs answer as the families `SearchRamp` and
`SearchOverload` (`engine/attacks/search.py`).

You now have: the episodes, each with its frames, its buses and its attack vectors.

## The timeline file

`generation/` holds the stages that walk a timeline: the plan, the designs, the frames and the
writer. `TimelineSource.generate` in `generation/source.py` is the one path every timeline takes: open
the file, cut the train, validation and test splits, place the episodes, design them, walk the
frames in time order, and finish the file.

- The steps are written once. A grid is a subclass that builds the generator, the pool and the
  designer of its families, and says what its files record: `IeeeSource` (`generation/ieee.py`)
  and `NewYorkSource` (`generation/nygrid/source.py`).
- `TimelineWriter` (`generation/write.py`) is the one writer of the HDF5 file.
- `schema.py` defines every group, dataset and attribute name of the file, once.
- Settings arrive as models from `models/config.py`: `TimelineSettings` for an IEEE timeline,
  `NewYorkSettings` for a New York one. `NewYorkRelease` holds what the released New York
  timeline is built from.

`fg.generate` starts an IEEE timeline through `generate_timeline` in `timeline.py`. The script
`pipelines/newyork/build_timeline.py` starts the New York one.
[The generation guide](../guides/generation.md) has every step and every setting.

You now have: one HDF5 file. `fg.generate` also registers it under a name, for `fg.load`.

## Loading

`fg.load(name, split="train")` returns an `FdiaGraph`, the dataset class in `dataset/`. It reads
the static graph once and slices records on request, so the whole file is never in memory.

- `registry.py` turns a name and a data release into a file. `download.py` fetches the file to
  the cache and checks its sha256.
- `dataset/` has one concern per file: `base.py` (shared state), `graph.py` (the static graph),
  `physics.py` (admittances and clean flows), `records.py` (one record, batches), `export.py`
  (whole-split arrays) and `sequence.py` (windows and episodes).
- `clients.py` reads a New York timeline as federated clients, one per utility.
- `errors.py` defines every error the package raises on purpose.

The package's `__init__.py` imports the heavy helpers only when they are called, so `fg.load`
never imports torch or pandapower.
[The data dictionary](../reference/DATA_DICTIONARY.md) says what every array means.

You now have: the timeline as records, batches, windows and episodes.

## Analysis

The second toolkit, `fdia_analysis`, measures things on a loaded timeline. Its code is under
`analysis/src/fdia_analysis/`. Each package has a page with its results.

| Package | Question | Page |
|---|---|---|
| `se` | what is the true state? | [State estimation](../analysis/estimation/README.md) |
| `localization` | which buses are attacked? | [Localization](../analysis/localization/README.md) |
| `trust` | which meters should be secured? | [Trusted meters](../analysis/trust/README.md) |
| `federated` | can clients train without pooling records? | [Federated](../analysis/federated/README.md) |

`se/detect/` holds the alarm rules, which turn a detector's score per scan into a first alarm,
and `results/` the code of the results store, where every measured number is kept.

## Two toolkits, one direction

The generation toolkit is the distribution `fdia-graph`, the package `fdia_graph` under `src/`.
The analysis toolkit is the distribution `fdia-analysis`, the package `fdia_analysis` under
`analysis/`.

`fdia_analysis` imports `fdia_graph`: it reads timelines through `fdia_graph.dataset` and uses
the generation toolkit's formulas and records. Nothing in `fdia_graph` imports `fdia_analysis`,
and `tests/test_toolkit_boundary.py` checks it. The generation toolkit installs and runs alone.
[Install and data versions](../guides/install.md) lists every extra.

## Where next

Read in this order.

1. [The landing page](https://github.com/myersben9/fdia-graph#three-steps): install, then load,
   generate and analyse in three steps.
2. [New York walkthrough](../getting-started/new_york_walkthrough.md): one attack followed
   through every stage.
3. [Test systems](test_systems.md) and [Solving](solving.md): the grids, and how their operating
   points are solved.
4. [Timeline generation](../guides/generation.md): how a timeline is built, step by step.
5. [Glossary](../reference/GLOSSARY.md) and [Data dictionary](../reference/DATA_DICTIONARY.md):
   every term, every array and its shape.
6. [Modules](../reference/MODULES.md): what every file does, in one line.
7. [Concepts to code](../reference/CONCEPTS_TO_CODE.md) and [Formulas](../reference/FORMULAS.md):
   the function behind each idea and equation.
8. [Class map](../reference/CLASS_MAP.md) and [Examples](../reference/EXAMPLES.md): the class
   diagrams, and training examples to copy.
9. [State estimation guide](../guides/state_estimation.md): fitting and scoring an estimator.
10. The four study pages in the table above: what was measured, on what, and the result.
11. [Contributing](https://github.com/myersben9/fdia-graph/blob/main/CONTRIBUTING.md): the rules
    and the pull-request flow.
