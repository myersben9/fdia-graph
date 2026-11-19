# Changelog

Every release lists what a user of the package can see change. "No user-visible change" means
the public API, the generated files and the numbers are the same as the previous release.

## Unreleased

- **The original generator reads a dispatched pool, and picks its targets on the grid's own buses.**
  New setting `loads` of `TimelineSettings` (`LoadRule`): `"scaled"` (default) reads a bus's load and
  generation by the one factor a scaled pool applied to both, as before; `"net"` reads a pool solved by
  an OPF, where a bus with a unit is all generation and any other bus all load. The two rules are the
  classes `ScaledSplit` and `NetSplit` (`engine/physics.py`), and `FdiaGenerator` takes one as `split`.
  `Case.own_bus` marks the grid's own buses (every bus of an IEEE case, the 46 New York buses of the New
  York case); ramp targets, the units an overload may move and an overload's target branches are held
  to them, and `Attacker` reads the same mark where it read the meter plan's owners.
  `SearchSource.for_pool(net, pool, settings)` runs the original generator on a New York pool file.
  Removed: `formulas.attacks.bus_load` (now `LoadSplit.load`), `FdiaGenerator.load_base` and `gen_base`
  (now `split.base_load`, `ScaledSplit.gen_base`), `Attacker.owner` (now `Attacker.own`);
  `formulas.attacks.operating_limits` takes the pool's generator output in place of the pool and the
  base tables. Every IEEE timeline and every New York release timeline is byte for byte what it was; a
  file records `loads` only when it is not the default.

- **Docs: the six remaining wide diagrams are redrawn for a phone (no change to the packages).**
  `concepts_one_record`, `localization_flow`, `se_estimators`, `se_gated_prior`, `se_jacobian_weighting`
  and `trust_flow` are now top to bottom, at most 8 nodes and 900 px wide, checked against the code, with
  alt texts that describe them. `NOT_YET_REDRAWN` in `tools/prereview.py` and the stylesheet's exempt
  list hold `pipeline` only. What changed in the pictures:

  | Diagram | Change |
  |---|---|
  | `concepts_one_record` | shows the three layers and the labels of one record, with the function that makes each; the list of attack families is gone |
  | `se_estimators` | the steps of `SEBase` and the step each estimator changes; the page cites `ac_measurement` in `formulas/measurement.py`, where it is |
  | `se_jacobian_weighting` | the change is taken against the previous frame's estimate, as the code does, not against its clean state |
  | `se_gated_prior` | adds the secured meters, which keep their weight; the gate is any fitted localizer |
  | `localization_flow` | one chain from `fit` to `score`, in the order of the code |
  | `trust_flow` | adds the alarm level, which `fit` calibrates before the selection |

  `docs/concepts/solving.md` has a new section, "Repeating a solve": how closely a New York pool
  repeats on the reference machine and on the Linux test runner, and what the test compares.

- **Docs: one guide for every test system, three tutorials and three new concept pages (no user-visible
  change to the packages).** The New York grid no longer has pages of its own; it is a test system in the
  pages every system shares.

  | Kind | Pages |
  |---|---|
  | tutorials | `docs/getting-started/load.md`, `generate.md`, `analyse.md` |
  | guides | `docs/guides/generation.md` (rewritten), `operating_points.md`, `extending.md`, `troubleshooting.md` |
  | concepts | `docs/concepts/meters.md`, `attacks.md`, `layers.md` |

  The five New York pages are removed and their content folded into the pages above, into
  `docs/concepts/test_systems.md` and into `CONTRIBUTING.md` (the New York tests and snapshots). The
  loading part of `docs/reference/EXAMPLES.md` is the load tutorial. The generation flow diagram is redrawn
  for every system, and the New York pipeline diagram is removed. `NOT_YET_PLAIN` in `tools/prereview.py`
  is empty: no docs page holds an internal decision code. Docstrings that cited a moved page cite its new
  place.
  The example `examples/nygrid/custom_attack_family.py` labels its episode with a family name of
  `fg.FAMILIES` ("Ad"; it was "Ab", which a timeline cannot write) and also designs through
  `EpisodeFamily.designed`, the call a timeline makes.

- **The original generator is built on a `Network`, and `IeeeSource` builds it for an IEEE case (every IEEE
  file is byte for byte the same, and so are the New York release's files).**
  `fdia_graph.generation.search.SearchSource(net, X, settings, seed, time)` is the source `IeeeSource` was, on a
  `Network` and its pool; `IeeeSource(system, states, settings, seed, time)` is unchanged for callers.

  | | Before | Now |
  |---|---|---|
  | the generator of an IEEE case | `FdiaGenerator(system, seed, meters, max_load_mw, plan)` | `FdiaGenerator(Network.from_ieee(n), seed, meters, max_load_mw, plan)` |
  | the generator of the New York grid | `FdiaGenerator.from_network(net, plan, seed, spec)` | `FdiaGenerator(net, seed, plan=plan, spec=spec)`: the one constructor |
  | a source with no default path | `NewYorkSource` refused `generate()` | every source but `IeeeSource` refuses it: "a timeline needs the path of its file" |
  | a family the generator does not make | refused when the source is built | refused when the `TimelineSettings` are built, with the same message |

  - **`FdiaGenerator.from_network` is removed, with no alias, and `FdiaGenerator(...)` takes a `Network`
    where it took an IEEE system.** `Network.from_ieee(n)` is the network of the IEEE case of `n` buses.
    `GeneratorSpec.system` is `GeneratorSpec.net`.
  - The original designs are built for a pool that scales each bus's load and generation by one factor
    (`generate_states`); they are not offered on the New York pool. The meter placement is not a setting: the
    search-based designs run on coverage meters and the New York release's designs on owned meters
    (`docs/guides/generation.md`).

- **How a timeline's splits are cut is a setting of every test system, `split_by` (the default of each is
  what it did before: the IEEE and New York files made without the setting are byte for byte the same).**
  `"time"` cuts three blocks in time order, `"week"` draws whole weeks into the three splits.

  | | Before | Now |
  |---|---|---|
  | an IEEE timeline | three blocks in time order, always | `split_by="time"` by default; `fg.generate(..., split_by="week", time=times)` cuts whole weeks |
  | a New York timeline | `NewYorkSettings.split_by`, `"week"` by default | the same setting, read by the shared step (`TimelineSource.split_codes`) |
  | a pool without time | not asked | `split_by="week"` is refused with the reason (`ConfigError`, declared on the model `SplitRule`) |
  | each frame's time (`data/time`) | written by a New York file | written by any file whose source has the frames' time |
  | an episode and a gap in time | not asked (no IEEE file had time; the New York pool has no gap) | with the frames' time, no episode is placed across a step that differs from the median step |

  - `fg.generate`, `generate_timeline` and `IeeeSource` take `time`, each frame's time `[T]` (an IEEE pool
    stores none, so the caller passes it): numpy datetime64 values, one per frame, strictly increasing.
    Numbers, strings, Python datetime objects, timezone-aware times and NaT are refused when the source is built
    (`SplitRule`).
  - `TimelineSettings.split_by` is recorded in a file only when it is not `"time"`, the value every file
    made before the setting meant (the rule `WhenSet(legacy="time")` of `models.validation`), so the recorded
    settings and their hash of those files are unchanged.
  - A split by weeks does not check that every split gets a week: a pool of fewer weeks than splits leaves a
    split empty, as the New York days' pool of the tests does.

- **The pool-building step takes one solve setting, `SolveSettings(method, engine)`, for every test system
  (the default of each builder is what it did before: the IEEE and New York pools built without the setting
  are byte for byte the same).** `method` is `"power_flow"` or `"opf"`, `engine` is `"pandapower"` or
  `"ipopt"`. `generate_states(..., solve=SolveSettings("opf", "ipopt"))` builds an IEEE pool by a
  least-cost AC OPF per step (the case's limits and its own cost curve, linear and quadratic), in either
  engine; the pool is the same for any number of `workers`, and a case with no optimum at any step raises
  `DataConditionError`. `PoolBuilder.from_settings(case, loads, dispatch, renew, ramp, solve=None)` builds
  the New York pool; it refuses a power flow and pandapower's OPF with the reason (`ConfigError`, declared on
  the model `NewYorkSolve`), and `SolveSettings` refuses a power flow in Ipopt. `docs/concepts/solving.md`
  has the table.
  - The two engines are classes of one interface, `fdia_graph.data.solver.Solver`:
    `fdia_graph.data.ipopt_solver.IpoptSolver` (the package's OPF, with `Compensation`) and
    `fdia_graph.data.pandapower_solver.PandapowerSolver` (the power flow, and pandapower's OPF). Both have
    `opf(pd, qd, costs)`, and a solve returns an `OperatingPoint` or, from an OPF, an `OpfPoint`
    (`fdia_graph.models.grid`), read by attribute or as the dict it was.
  - **`AcSolver` is removed, with no alias.** Its `project` and `dispatch` are `IpoptSolver`'s, its
    `power_flow` and `net` are `PandapowerSolver`'s.
  - `PandapowerSolver.net` carries the case's bus shunts and each branch's shunt conductance (as bus shunts),
    where `AcSolver.net` carried neither. Both are zero on the New York case, so its net is the same.
  - A power flow given no start now starts flat where it started from a DC solution; the pool build always
    gives a start. `power_flow` and `opf` return None only when pandapower reports no convergence; any other
    error is raised.
  - `Case.pandapower_costs(net)` reads each unit's linear and quadratic cost from a pandapower net into a
    `UnitCosts` (`fdia_graph.models.inputs`); the constant term is not read.
  - `tests/nygrid/test_solve_engines.py` compares the two OPF engines on IEEE-14, 30 and 118 and on the
    New York case: the cost agrees within 0.01 % and every unit within 0.1 MW.

- **The load data and the operating points get their own package, `fdia_graph.data`; the folder
  `generation/nygrid/` and the module `profiles.py` are gone (a move: no function or class changes, and no
  file changes: thirty IEEE timelines are byte for byte the files the previous code writes, and the five
  New York digests are unchanged).** `fg.generate_states`, `fg.load_profile` and `fg.fetch_profile` are
  called as before. A module imported by its old path is imported by its new one; no old path is kept.

  New York is a test system like an IEEE case, so no folder holds "the New York code". What was in
  `generation/nygrid/` is three general things: where the load data comes from and how the operating points
  are solved (`data/`, for every test system), a fact about the grid (its interfaces, `systems/`) and the
  grid as a timeline source (`generation/newyork.py`, beside `generation/ieee.py`). The imports point one
  way: `constants`, `models`, `formulas`, `systems`, `sensing`, `data`, then `engine`, `generation` and
  `dataset`. `data` imports nothing of `engine`, `generation` or `dataset`, and `systems` and `sensing`
  nothing of `data` (`tests/test_models_package.py`).

  | Before | Now |
  |---|---|
  | `fdia_graph.profiles`: `load_profile`, `fetch_profile`, `LoadSource`, `RawSeries`, `CsvColumn`, `IsoFolder`, `LoadFeed`, `NyisoArchive`, `GridstatusFeed`, `nyiso_pal_month`, `nyiso_zone_load` | `fdia_graph.data.profiles` |
  | `fdia_graph.profiles`: `generate_states`, `K_DEFAULT`, `SIGMA_DEFAULT`, `CLIP_DEFAULT` | `fdia_graph.data.ieee_pool` |
  | `fdia_graph.generation.nygrid.config` (`Settings`, `rng`, `stream_key`) | `fdia_graph.data.config` |
  | `fdia_graph.generation.nygrid.inputs` | `fdia_graph.data.inputs` |
  | `fdia_graph.generation.nygrid.load_sources` | `fdia_graph.data.load_sources` |
  | `fdia_graph.generation.nygrid.loads` | `fdia_graph.data.loads` |
  | `fdia_graph.generation.nygrid.dispatch` | `fdia_graph.data.dispatch` |
  | `fdia_graph.generation.nygrid.solver` (`AcSolver`) | `fdia_graph.data.solver` |
  | `fdia_graph.generation.nygrid.pool` (`Pool`, `PoolBuilder`, `peak_ratings`) | `fdia_graph.data.pool` |
  | `fdia_graph.generation.nygrid.io` (`read_pool`, `write_pool`) | `fdia_graph.data.pool` |
  | `fdia_graph.engine.solvers` (`ipopt`) | `fdia_graph.data.ipopt` |
  | `fdia_graph.generation.nygrid.interfaces` (`Interface`, `hourly_interfaces`) | `fdia_graph.systems.interfaces` |
  | `fdia_graph.generation.nygrid.source` (`NewYorkSource`, `SingleEpisodeSource`, `PoolDesigner`, `Survey`) | `fdia_graph.generation.newyork` |
  | the packaged `generation/nygrid/ny_bus_locations.csv` | `data/ny_bus_locations.csv` |

  `engine.solvers` moved because the pool's solver (`data/solver.py`) and the attack OPFs (`engine/attacks/`)
  both run on it, and `data` may not import `engine`. A pool file is read by its path, so an existing
  `pool.h5` reads as before. The survey cache keeps its `_v2` chunks: a chunk pickles `models.plan.Episode`
  with numbers and text, and no pickled class moved. The rules of the New York modules not yet under the
  SDK's gates are `src/fdia_graph/CLAUDE.md`.

- **One path generates a timeline on any grid, and New York adopts the original generator's noise, walk
  and move rule (a New York file changes; an IEEE file does not: the two IEEE digests are unchanged, and
  thirty-two IEEE timelines are byte for byte the files the previous code writes).** `fg.generate`,
  `generate_timeline`, `IeeeSource(system, states, settings, seed)` and
  `NewYorkSource(pool, attacker, families, own, settings, build)` are called as before.

  The steps are written once, in `TimelineSource` (`generation/source.py`): opening the file, the truth
  layers, cutting the splits, placing, the design stage with its moves, the walk and finishing the file. A
  source is built from a generator over its grid, its pool, the designer of its attack families, its
  settings, the names of the families it places and a seed
  (`TimelineSource(g, X, designer, settings, placed, seed)`); `IeeeSource` and `NewYorkSource` build those
  and say only what is their own.

  "Design an attack for this window" has one shape. A family is an `EpisodeFamily`
  (`engine/attacks/family.py`) with one method, `designed(onset, X, rng)`: the episode over the window of
  true states `X` `[L, N, 4]`, or None. What it returns is a `DesignedEpisode`, which the walk emits by one
  loop (`begin`, a `frame` per scan, `record`). The original search designs are the families `SearchRamp`
  and `SearchOverload` (`engine/attacks/search.py`: each scan re-solved, with the same random draws in the
  same order; a ramp is still designed as the walk reaches it, now inside `RampEpisode.begin`), and a New
  York family (`AttackFamily`, unchanged for those who write one: `design(window, rng)` gives an `Episode`)
  yields a `VectorEpisode`, the true scan plus the episode's row of its stored attack vector. A source's
  `Designer` (`generation/design.py`) holds the families and the processes their designs run on:
  `SearchDesigner` on an IEEE pool, `PoolDesigner` on the New York pool.

  | | Before | Now |
  |---|---|---|
  | the noise of a New York frame | one stream (`rng(build, "noise")`) run over the frames in time order | the frame's own stream, keyed (seed, `JITTER_STREAM`, pool frame), as on an IEEE timeline: the benign scans do not depend on the attacks placed, and one pool frame reads the same in a timeline and in a single-episode file |
  | a New York frame under an episode | `float32(z_benign + a)` from the float64 benign scan | the stored float32 benign scan plus `a`, as an IEEE attacked frame is built (`FdiaGenerator.frame_from_vector`) |
  | how New York frames are written | `NewYorkSource._write_block`, `block` frames at a time (`NewYorkSettings.block`) | frame by frame, by the walk every timeline uses; `NewYorkSettings.block` is removed |
  | a New York slot with no attack, or on an unsolved frame | kept no episode; its frames stayed benign | moved to another free onset of its split, where its family's mask allows, up to `REDRAWS` times, then given up with a warning: the rule of an IEEE timeline |
  | the move rule | `AmDesigner.design_split`, `_round`, `_move`, `_new_onset` (IEEE Am only) | `generation.design.SplitDesign(seed, span, rec, masks).run(slots, designs)`, both grids |
  | `free_onset(rng, length, others, span)` | block splits | any split, with an optional family mask |
  | `open`, `_clean`, `frames`, `cut_splits`, `lengths`, `design`, `walk`, `finish`, `close` | written in `IeeeSource` and again in `NewYorkSource` | written once, in `TimelineSource` |
  | the design processes | `AmDesigner` (IEEE) and `NewYorkSource._designs` with `_Task`, `_Tried`, `_Designing` | `Designer` with `DesignTask` and `Tried`, one for both grids |
  | `Walk(ctx, writer)`, `Walk.walk_split(plan, span, episodes, rec)`, `ramp_episode`, `overload_episode`, `record_episode` | one block split at a time, one loop per family | `Walk(g, X, writer, first)`, `Walk.run(t, until, pending, spans, rec)` and `Walk.episode`: any stretch, splits of blocks or of whole weeks, one loop |
  | an episode's family in the plan and design stages | an integer code on IEEE, a name on New York (`Slot[Family]`, `Placement[Family]`, `models.choices.family_label`) | its name on both (`Slot.family: str`); a designed episode carries its file code |
  | the episode lengths | `models.plan.Schedule` (IEEE), `NewYorkSettings.lengths_for` | `lengths_for(families)` on both settings models; `Schedule` is removed |
  | the pool as the walk reads it | an array `[T, N, 4]` | an array, or solved voltages read frame by frame (`systems.states.PoolStates`) |
  | a frame from its scan | built field by field in three places | `Frame.from_scan` (`models/frames.py`) |
  | the tampered meters of an attack vector | `Attacker.tampered` | `NoiseModel.tampered` (`sensing/noise.py`, with `wrap_angles`), which `Attacker.tampered` and `frame_from_vector` both call |
  | the generator on a grid that is no IEEE case | not possible | `FdiaGenerator.from_network(net, plan, seed, spec)`; `FdiaGenerator(system, ...)` opens an IEEE case as before |
  | `TimelineWriter.add_episode(episode, tamper, lines)` | three arguments | `add_episode(record)`, an `EpisodeRecord` (`models/plan.py`) |
  | `models.frames.FrameContext`, `engine/records.py` (`attack_frame`) | the walk's context record; the benign scan as a frame | removed: the walk holds the generator and the pool, and the benign frame is `FdiaGenerator.benign_frame` |
  | `NewYorkSource._split`, `._episodes`, `._recorded()`, `._graph()` | private | `split`, `episodes` (the file's episode rows), `recorded()`, `graph()`, the names of the shared steps |
  | `NewYorkSource.summary()` | `skipped_unsolved`, `missed` counted placed slots | `moved` and `given_up` per split and family; `missed` and `unsolved_attempts` count design attempts (a slot is tried again at every onset it moves to) |

  With the onset survey on, a waiting family is placed only at onsets whose design the survey already
  holds, so it never moves; the rule then acts on the families that start anywhere (the ramp). A verbose
  New York build no longer prints a line per stretch written or per fifty designs; the survey still reports
  its progress. The survey cache keeps its `_v2` chunks.

  What changes in a New York dataset: every noisy reading (`data/node_x`, `data/edge_x`, `data/pmu_i` and
  their `benign/` twins) is a new draw of the same noise model, and through the rounding of the benign scan
  the attack `z - z_benign` moves in its last float32 digit. Where every placed design finds an attack (the
  three stored digests) nothing else moves: the true state (`clean/`), the splits, the placed and designed
  episodes, the labels, the tamper masks and every attribute are the same. Where a design fails, the
  timeline now holds an episode at another onset in place of benign frames, so it carries more attacked
  frames, closer to `attacked_frac`, and its `episode_redraws` attribute is no longer always zero. No New
  York dataset is released, so no published number changes.

- **Docs: three concept pages, and the roadmap page is retired (no change to the code of either
  package).** `docs/concepts/flow.md` follows the package stage by stage, from the test system to the
  analysis, and holds the reading order; `docs/concepts/test_systems.md` says what a `Case` is, lists the
  nine test systems and describes the New York case file as one of them; `docs/concepts/solving.md` says
  where a power flow and where an optimal power flow is solved, and what each needs installed.
  `docs/guides/install.md` lists every extra and the data pins. The roadmap page and its two diagrams
  are deleted: its file tables are the generated `docs/reference/MODULES.md`, its tables of the
  `constants/` and `models/` modules open the Models part of `docs/reference/DATA_DICTIONARY.md`, and its
  test and release commands are a table of `CONTRIBUTING.md`. The site's tabs are Home, Getting started,
  How-to guides, Concepts, Studies, Reference, Contributing (with the review checklist) and Changelog.
  The glossary's `pool`, `split` and `onset` hold for the New York grid too, and the reference pages
  carry no decision codes. `tools/prereview.py` has a new check, plain pages: it fails a page under
  `docs/`, or the README, with a fenced Mermaid block or a decision code (one code or a range of them);
  one page is exempt on its `NOT_YET_PLAIN` list. The install tables say what the base install does
  without PyTorch (it opens a timeline and exports whole splits as numpy arrays; a single record, a
  batch and the DataLoader need the `torch` extra), and the benchmarks page names no pull request.

- **Docs: a landing page that says what the package is, and diagrams a phone can read (no change to the
  code of either package).** `README.md` presents the two toolkits, the test systems (the eight IEEE
  cases and the New York grid), three steps (load, generate, analyse) and one top-to-bottom picture of
  the flow, `docs/figures/diagrams/flow.png`. Its loading table moved to `docs/reference/EXAMPLES.md`, the
  list of extras to `docs/guides/install.md`. `tools/render_mermaid.py` draws a hand-written flow diagram for a
  phone (20 px type, a white card 900 px wide), and `tools/prereview.py` fails one that is not
  `flowchart TB`, has more than 8 nodes or is wider than 900 px; seven diagrams are drawn that way, seven
  are exempt on its `NOT_YET_REDRAWN` list (the old `pipeline.png` among them, which the published PyPI
  page loads until the next release), the docs site shows a phone diagram at most 450 px wide
  (`docs/stylesheets/extra.css`), and the class and module diagrams keep the wide rendering. The
  New York walkthrough's snippet reads `SolveKind.SOLVED`, and links to files outside `docs/` are full
  GitHub addresses. The New York guide names the pool file as the pipeline scripts do
  (`out/pool_2019_v3.h5`). The generation guide carries no decision codes, and the recipes of its
  overload table are named in plain words ("with the load and generator bounds"; the labels in
  `results/queries.py`, the stored runs unchanged). The package description on PyPI follows the new
  summary.

- **The grid and the meters have their own packages, `fdia_graph.systems` and `fdia_graph.sensing`
  (breaking for the module paths below; a move only: no class is renamed, no logic changes and no stored
  byte changes: the five whole-file digests are unchanged, and thirty-two IEEE timelines are byte for byte
  the files the previous code writes).** `fg.load`, `fg.generate` and `generate_timeline` are used as
  before. New York is a test system like IEEE-14: its `Case`, its `Network` and its case file sit beside
  the IEEE cases, and the meter plan, the measurement function and the noise model are no part of the
  engine. `systems` and `sensing` import nothing of `engine`, `generation` or `dataset` (a test holds them to
  it), so the generator (`engine/core.py`, `engine/base.py`) no longer imports `generation.nygrid`;
  `engine/attacks/interface.py` still reads `Interface` from `generation.nygrid.interfaces`. The old module
  paths are gone, with no alias.

  | | Before | Now |
  |---|---|---|
  | `Case`, `nygrid_case`, `CASE_FILE` | `fdia_graph.generation.nygrid.case` | `fdia_graph.systems.case` |
  | `Network` | `fdia_graph.generation.nygrid.network` | `fdia_graph.systems.network` |
  | `externals_at_base`, `hvdc_links` | `fdia_graph.generation.nygrid.fixes` | `fdia_graph.systems.fixes` |
  | the New York case file `case_nygrid.m` | in the package folder `generation/nygrid` | `src/fdia_graph/systems/case_nygrid.m` (the wheel ships it there) |
  | `PandapowerNet`, `PpcTables` | `fdia_graph.engine.pandapower_types` | `fdia_graph.systems.pandapower_types` |
  | the IEEE case of a bus count | built inside `FdiaGenerator._open_case` | `Case.from_ieee(buses)`, which `_open_case` calls |
  | `Ownership`, `assign_clients` | `fdia_graph.generation.nygrid.ownership` | `fdia_graph.systems.ownership` |
  | `read_rows` | `fdia_graph.generation.nygrid.tables` | `fdia_graph.systems.tables` |
  | `MeterPlan`, `plan_meters` | `fdia_graph.engine.sensing.meters` | `fdia_graph.sensing.meters` |
  | `Measurement` | `fdia_graph.engine.sensing.measure` | `fdia_graph.sensing.measure` |
  | `NoiseModel` | `fdia_graph.engine.sensing.noise` | `fdia_graph.sensing.noise` |

  The New York survey cache (`survey_cache`) keeps its `_v2` chunks: a chunk pickles `models.plan.Episode`
  records, seconds and miss reasons, and none of the moved classes, so an existing cache still loads.
  The layout gate's
  page for both packages is `docs/guides/generation.md`. The case file's bytes are unchanged, so the comment
  that opens it still names the reader by its old module path, and the recipe
  `pipelines/newyork/make_case.py` writes that same line.

- **One grid representation, one measurement function, one meter table and one noise model for the IEEE
  cases and New York (breaking for the internal names below; no stored byte changed: the five whole-file
  digests are unchanged, and thirty IEEE timelines are byte for byte the files the previous code writes:
  all eight systems, the row-reduction and the rule-chosen area on IEEE-118, a two-worker run, "delta"
  ratings, four meter coverages and several seeds).** `fg.load`, `fg.generate` and
  `generate_timeline` are used as before. The IEEE generator `FdiaGenerator` is now built on the classes
  of the New York grid; pandapower solves the base power flow once and the net is not kept. By
  `python tools/count_lines.py`, `fdia_graph` goes from 13,850 to 13,814 lines and from 8,628 to 8,561
  lines of code outside the constants (column "other code").

  | | Before | Now |
  |---|---|---|
  | the grid of an IEEE case | a pandapower net kept by the generator (`g.base`), its PYPOWER tables, a bus lookup and a `BranchModel` | a `Case` read by `Case.from_pandapower` and its `Network`: `g.case`, `g.net` |
  | the meters | `models.grid.MeterPlan` (sets of buses) and masks built on every call | the table `engine.sensing.meters.MeterPlan`, placed by `MeterPlan.from_coverage` (`from_placement` at given places): `g.plan`, `g.masks`, `g.pmu_buses`, `g.flow_metered` |
  | the readings of a state, h(x) | flows and currents computed in `MeasurementMixin.emit_from_state` | `Measurement.h`, which gains the kinds `IR` and `II` (a current phasor as real and imaginary part) and reads the stored injections of an IEEE pool |
  | bias and jitter | `models.grid.MeterBias`, `FdiaGenerator._meter_bias` and `_current_bias`, `formulas.noise.bias_jitter_split`, `jitter_sigma`, `biased_current` | the plan's `bias` column and `NoiseModel.sample` |
  | the rated accuracy of a scan | `formulas.noise.accuracy_sigma` | `NoiseModel.sigmas` |
  | the static graph of a file | built by `IeeeSource._graph` and by `NewYorkSource._graph` | `Case.static_graph()`, to which New York adds its ratings, bus facts and clients |
  | the bus admittance of the local solves | pandapower's `makeYbus` (`g._Ybus`) | `g.net.ybus`, from `Network(case, matpower_order=True)` |
  | the generator's settings | `FdiaGenerator(system, seed, vbus_frac, pmu_frac, flow_frac, max_load_mw, meter_model)` | `FdiaGenerator(system, seed, meters, max_load_mw, plan)`: `meters` is a `MeterSettings`, `plan` a `MeterPlan` used as given |
  | the load table `g.load_bus`, `g.load_p0`, `g.load_q0` | one row per load element of the net (`load_bus` uint32) | one row per loaded bus, in bus order (`load_bus` int64); the eight cases have one load element per loaded bus, so the rows are the same |
  | the meter masks | `g.meter_masks()`, `g.current_mask()` (None without PMU currents) | `g.masks`, a `MeterLayers` (`node`, `edge`, `current`) |
  | the PMU currents of a scan | optional: `Scan.i_x`, `Frame.i_x`, `AttackVector.current` and `g.currents_from_states` could be None, and `LayerSet.currents` said whether a file has them | always there: the fields are required, the guards and `LayerSet.currents` are gone |
  | the rated accuracy where no meter sits | the class of the channel | zero (`g.layers(g.noise.sigmas(z))` lays out one value per meter) |
  | the file attribute `meter_model` | written when the plan has PMU currents | written on every IEEE file |
  | the jitter of a scan | `g._jitter` (a stream), `g._i_jitter`, `formulas.noise.jitter_sigma` | `NoiseModel.jitter(z)`, the standard deviation `NoiseModel.sample` draws with |

  - **Same draws, same arithmetic.** An IEEE plan draws what the generator drew, in the same order: the
    voltmeter buses, the PMU buses, the flow meters, one bias for every bus and branch (P, Q, |V|, angle,
    P flow, Q flow, then the four current channels of every branch), and per frame one jitter per meter
    from the frame's own stream, bus by bus, then the flows, then the currents. The rows of the plan are
    in that order.
  - **`Case` gains `g`**, a branch's shunt conductance (a transformer's iron losses on IEEE-118, 145 and
    300; zero on a MATPOWER branch), which `Network` passes to `branch_admittances`. The New York
    admittances are unchanged.
  - **No matrix comes from pandapower.** The stored IEEE files were made with the bus admittance of
    pandapower's `makeYbus`, and an attacked frame depends on its last bit and on the order a product
    runs over it. `makeYbus` sums the from-side and the to-side terms of a bus apart and returns the
    matrix by column; `Network` sums every term in one pass and holds it by row, which differs in the
    last bits of a few diagonal entries (at most 5.7e-14 pu on IEEE-300, none on IEEE-30). So
    `branch_admittances` takes `sides_apart` and `Network` takes `matpower_order`, which the IEEE
    generator turns on: its `Network.ybus`, `yf` and `yt` then equal `makeYbus`'s bit for bit on all
    eight systems (`tests/test_ieee_case.py`), and the generator holds this one bus admittance
    (`g.net.ybus`). New York keeps the one-pass sum its files were made with.
  - **`Case.from_pandapower` refuses what a `Case` does not model**, with a `DataConditionError` naming
    the element (`models.inputs.ModelledNet`): a bus, line, transformer, load or generator out of
    service (a static generator too, unless it produces nothing: IEEE-200 has eleven of those), a
    `scaling` other than 1, a branch the solved case has switched off, rows in an element table no field
    is read from (`trafo3w`, `impedance`, `ward`, `xward`, `dcline`, `switch`, `storage`, `motor`, the
    asymmetric and power-electronic elements), and a solved case with another number of buses. Before,
    such a net was read with the element missing or misread. `pipelines/newyork/make_case.py`
    (`write_case`) refuses a branch with a shunt conductance, which a MATPOWER case file cannot hold.
  - **The `generate` extra lists scipy**, which the engine imports itself (it came with pandapower).
  - **The readability gate measures the shared classes.** `engine/sensing/`, `generation/nygrid/case.py`
    and `generation/nygrid/network.py` leave the deferred list of `tools/readability.py`; the search for
    the smallest PMU set is a class (`_PmuSearch`) and the New York meter groups are built in one pass,
    with the same result (the three New York digests are unchanged).
  - **Removed, with nothing in their place:** the branch status of a generator and its N-1 `Outage`
    (every IEEE file records the intact case, as before), the meter model without PMU currents (no
    setting selected it), `constants.channels.NODE_DRAW_ORDER` (the order is `constants.meters.BUS_KINDS`),
    `GeneratorOptions.meter_model` (the meter model is `MeterSettings.meter_model`, which refuses an
    unknown one), the generator methods `_live_edges` and `_live_adjacency` (every branch is in service:
    `g.ei`), and the generator attributes `base`, `pp`, `NET`, `branch`, `meters`, `bias`, `SD`, `SDj`,
    `contingency`, `edge_gs`, `edge_bs`, `edge_is_trafo`, `bus_shunt_g`, `bus_shunt_b`, `n_lines`, `_Ybus`,
    `_Yf`, `_Yt`, `_ppc_row`, `_from_bus_ppc`, `_ppc_branch`, `_n_ppc_buses`, `_base_mva`,
    `_attackable_mask`, `_stealthy_mask` (the positions `attackable_pos` and `stealthy_pos` stay),
    `_injection_buses`, `_jitter`, `_i_jitter` and `_bias_sd`.

- **Shorter docstrings and comments in `fdia_graph` (no user-visible change: no line of code changed).**
  A module docstring is at most about 10 lines, a class's 6, a function's 1 to 3 (CONTRIBUTING.md); each
  keeps its equation, source key, units and the invariants a caller must respect. By
  `python tools/count_lines.py`, `fdia_graph` goes from 14,897 to 13,850 lines with its 9,194 lines of
  code unchanged. The file layout and knobs that the docstring of `timeline.py` listed are in
  `docs/guides/generation.md`, and the fit of the within-hour load terms is in docs/guides/new_york.md (since moved to `docs/guides/operating_points.md`).

- **Less code to read in `fdia_graph` (breaking for the internal names below; no behaviour and no stored
  byte changed: the five whole-file digests are unchanged, an IEEE timeline is byte for byte the file the
  previous code writes, and every refused input raises the same error with the same message).**
  `fg.load`, `fg.generate` and every flat keyword of `generate_timeline` are used as before. The rules for
  inputs are as they were: every check is declared on a model and run by the one engine, and
  `models.inputs`, `models.validation` and the settings models of `models.config` are unchanged. By `python tools/count_lines.py`,
  `fdia_graph` goes from 15,107 to 14,897 lines (column "lines") and from 9,404 to 9,194 lines of code
  (column "code"); `fdia_analysis` is unchanged.
  - **The record bundles share one base.** `models.fields.RecordFields` declares every field of a record,
    a batch and an exported split once; it replaces the nine field-group mixins. The keys, their order
    and what each bundle requires are unchanged, and the data dictionary no longer names a group per row.
  - **A mixin's bases are the mixins whose methods it calls.** `PhysicsMixin` and `MeasurementMixin` each
    have `GridBase` as their one base (neither calls the other), `AreaMixin` has `MeasurementMixin`, and
    `FalseStateMixin` has `AreaMixin` and `PhysicsMixin`; the other attack mixins are as before. So
    `GridBase` declares the shared attributes only: the ten method signatures it repeated for the type
    checker and the attributes only `engine/core.py` reads are gone. Every attribute of
    `FdiaGenerator` is defined by the class that defined it before.
  - **Long statements are plain steps.** A statement the formatter split over three lines around one
    long expression names its part first (about forty of them), and a comment that forced such a split
    sits above its statement. A repeated body is written once: the three hourly tables of
    `generation.nygrid.inputs` (`_hourly`) and the three `as_record` bodies of `models.config` (`_record`).
    No arithmetic is reordered, no operand is evaluated in another order and no random draw moves.
  - **Two tables.** `constants.newyork.CASE_COLUMNS` names the MATPOWER column of each `Case` field read
    as it is (before: 23 keyword arguments of one constructor call), and `schema.BRANCH_MODEL_PHYSICS`
    the `graph/` array of each `BranchModel` field (before: the same six names typed in two modules). A
    test holds each table to the fields of its record, and a case file that lacks a column is refused by
    the column's name ("the case file has no column 'BR_R'").

    | Before | Now |
    |---|---|
    | `models.fields.ScanFields`, `LabelFields`, `RecordIds`, `TemporalFields`, `CleanFields`, `GraphFields`, `StreamLayers`, `PmuCurrentFields` | `models.fields.RecordFields` |
    | `models.fields.PreviousFrameFields` | the `prev_*` fields of `models.data.ArraysBundle` |
    | `FdiaGenerator(MeasurementMixin, PhysicsMixin, AttackMixin)` | `FdiaGenerator(AttackMixin)` |
    | `AreaMixin(GridBase)` | `AreaMixin(MeasurementMixin)` |
    | `FalseStateMixin(AreaMixin)` | `FalseStateMixin(AreaMixin, PhysicsMixin)` |
    | the method signatures on `engine.base.GridBase` (`emit_from_state`, `true_load`, ...) | removed: the mixin that defines the method is a base of the mixin that calls it |
    | "Field groups: ..." and "(group)" on each row of `DATA_DICTIONARY.md` | removed |

- **Constants have one home: the `fdia_graph.constants` package (breaking for the names below; no
  behaviour and no stored byte changed: the five whole-file digests are unchanged, and an IEEE timeline
  is byte for byte the file the previous code writes).** `fg.load`, `fg.generate`, `fg.FAMILIES` and
  `fg.STEALTHY_FAMILIES` are used as before.
  - **`.node()` / `.edge()` are removed** from a record, a batch and a split; write
    `rec.node_x[..., NodeChannel.THETA]` (`from fdia_graph.constants.channels import NodeChannel`).
  - **`fdia_graph.constants`**, one module per subject, each constant with its unit and source:
    `channels`, `families`, `meters`, `grid`, `time`, `streams`, `numerics`, `records`, `files`,
    `releases` and `newyork`. The modules import nothing from the rest of the toolkit. The
    module-level constants of 37 modules moved there or to `schema.py`; a name is imported from its
    module of `constants`, no longer from the module that held it.
  - **Columns are enums.** `constants.channels.NodeChannel` (`V`, `P_INJ`, `Q_INJ`, `THETA`),
    `EdgeChannel` (`P_FROM`, `Q_FROM`, and `P_TO`, `Q_TO` in a file with both branch ends),
    `CurrentChannel` (`RE_FROM`, `IM_FROM`, `RE_TO`, `IM_TO`) and `PolarCurrentChannel` (a New York
    file's magnitude and angle) index the last axis of an array: `node_x[..., NodeChannel.P_INJ]`.
    They replace `models.grid.NODE`, `EDGE`, `CURRENT`, `NodeColumns`, `EdgeColumns`, `NodeIndex`,
    `EdgeIndex` and `CurrentIndex`.
  - **Codes are enums.** `constants.meters.MeterKind` replaces the meter-kind numbers `V`, `VA`, `P`,
    `Q`, `PF`, `QF`, `I`, `IA` and the tuple `KINDS` of `engine.sensing.meters`, `KV`, `KPF`, `KQF` of
    `engine.attacks.interface` and `P_KIND`, `Q_KIND` of `engine.attacks.opf`; `constants.grid.BusType`
    replaces `PQ`, `PV`, `REF` of `generation.nygrid.case`; `constants.newyork.SolveKind` replaces
    `SOLVED`, `FAILED`, `NO_HOUR`, `NO_LOAD` of `generation.nygrid.pool`.
  - **`schema.py`** also holds the legends a file carries (`NODE_FEAT_LEGEND`, `FLOW_NAMES`,
    `CURRENT_FEAT_LEGEND`, `IEEE_TAMPER_RULE`, `GRAPH_LEGEND` and the New York ones) and the window of
    the swing feature. The shapes and types of the datasets are tables of the one class that creates
    them: `TimelineWriter.FRAME_LAYERS`, `CURRENT_LAYERS`, `TEMPORAL_LAYERS`, `EPISODE_COLUMNS`,
    `MIN_COLUMNS`, `AM_COLUMNS`, `CHUNK_ROWS` and `FLOW_WIDTH` (before: `_LAYERS`, `_CURRENT_LAYERS`,
    `_TEMPORAL`, `_EPISODE_COLUMNS`, `_MIN_COLUMNS`, `_AM_COLUMNS`, `_CHUNK_ROWS` and `_FLOW` of
    `generation.write`).
  - **`REGIONS`** (the neighbors of New York) is plain data, `(zone code, name, area)` per neighbor; the
    `Region` record of `generation.nygrid.case` is removed.
  - **`tools/layout.py`** refuses a module-level UPPER_CASE name of `fdia_graph` outside `constants/`
    and `schema.py` (one assigned under an `if` or a `try` as well) unless its `EXEMPT` table lists it
    with the reason (the knobs that have no settings model yet, among them), and refuses a new one in
    `fdia_analysis` (`EXEMPT_ANALYSIS` lists the ones it has). **`tools/count_lines.py`** prints each
    toolkit's lines, code lines, and the code inside and outside the homes of constants.
  - **Every renamed name.** A name not listed kept its name and moved to the module of `constants`
    for its subject (`FAMILIES`, `FAMILY_CODE`, `BENIGN_CODE`, `STEALTHY_FAMILIES` and
    `GENERATED_FAMILIES` of `models.choices` are in `constants.families`; `ACCURACY_CLASS`,
    `JITTER_FRAC`, `WU26_NOISE` and the noise floors of `formulas.noise` in `constants.meters`).

    | Before | Now |
    |---|---|
    | `engine.attacks.stealthy.RAMP_FAMILY`, `AM_FAMILY` | `constants.families.AT_CODE`, `AM_CODE` |
    | `models.choices.HELDOUT_FAMILIES`, `dataset.base._HELDOUT_TRAIN_EXCLUDE` | `constants.families.HELDOUT_CODES` (the codes; the tuple of names is removed) |
    | `models.choices.SPLITS` | removed: `Split.values()`, `len(Split)` |
    | `models.inputs.FAMILY_WORDS` | private to `models.inputs` (`_FAMILY_WORDS`) |
    | `formulas.trust.TOL` | `constants.numerics.ZERO_TOL` |
    | `engine.attacks.rref.ZERO` | `constants.numerics.RELATIVE_ZERO` |
    | `engine.attacks.rref.V_STEP`, `THETA_STEP` | `constants.numerics.JACOBIAN_V_STEP`, `JACOBIAN_THETA_STEP` |
    | `formulas.network._BACKTRACK_HALVINGS` | `constants.numerics.BACKTRACK_HALVINGS` |
    | `engine.attacks.opf.MIN_BUDGET` | `engine.attacks.opf.SUPPORT_BUDGET` (not the `min_budget` of `SearchSettings`) |
    | `generation.plan.WEEK` | `constants.time.WEEK_SECONDS` |
    | `generation.nygrid.load_sources.HOURS`, `QUARTERS` | `constants.time.HOURS_PER_YEAR`, `QUARTERS_PER_YEAR` |
    | `generation.nygrid.loads.FRAME_MIN` | `constants.time.FRAME_MINUTES` |
    | `generation.nygrid.pool._HALF_HOUR` | `constants.time.HALF_HOUR` |
    | `profiles.JITTER_RHO` | `constants.time.LOAD_JITTER_RHO` |
    | `formulas.temporal.SWING_WINDOW` | `schema.SWING_WINDOW_SCANS` |
    | `generation.write.TAMPER_RULE` | `schema.IEEE_TAMPER_RULE` |
    | `generation.nygrid.source.TAMPER_RULE`, `NY_GRAPH_LEGEND` | `schema.NEW_YORK_TAMPER_RULE`, `NEW_YORK_GRAPH_LEGEND` |
    | `engine.measurement._NODE_DRAW_ORDER` | `constants.channels.NODE_DRAW_ORDER` |
    | `engine.sensing.meters.ACCURACY` | `constants.meters.KIND_ACCURACY` |
    | `engine.sensing.meters._LAYER` | `constants.meters.METER_SLOT` |
    | `dataset.clients._BUS_KINDS`, `_ANGLE_KINDS` | `constants.meters.BUS_KINDS`, `ANGLE_KINDS` (members of `MeterKind`) |
    | `engine.core._CASE` | `constants.grid.IEEE_CASES` |
    | `generation.nygrid.solver._NO_LIMIT` | `constants.grid.NO_LIMIT` |
    | `dataset.base._BATCH_SCALARS`, `models.data._IDS` | `constants.records.ID_FIELDS` |
    | `models.data._SCAN`, `_TEMPORAL`, `_CLEAN`, `_BENIGN`, `_PMU_I` | `constants.records.SCAN_FIELDS`, `TEMPORAL_FIELDS`, `CLEAN_FIELDS`, `BENIGN_FIELDS`, `CURRENT_FIELDS` |
    | `dataset.base._CLEAN_LAYERS` | `constants.records.CLEAN_FIELDS` |
    | `dataset.base._BENIGN_LAYERS`, `_CURRENT_LAYERS` | `schema.BENIGN_PATH`, `CURRENT_PATH` |
    | `dataset.base._PREV_FIELDS` | `constants.records.PREV_SOURCE_FIELD` |
    | `models.data._PREV` | `constants.records.COMMON_PREV_FIELDS` |
    | `dataset.base._UNIT_KIND` | `constants.records.UNIT_KIND` |
    | `dataset.export._INT_KEYS`, `_FLOW_COLUMNS` | `constants.records.INT_FIELDS`, `FLOW_COLUMNS` |
    | `registry._SHA256`, `_ALIASES`, `_REPO`, `_DATA_TAG_PREFIX` | `constants.releases.SHA256`, `ALIASES`, `REPO`, `DATA_TAG_PREFIX` (the pinned release is `PINNED_RELEASE`) |
    | `generation.nygrid.io._POOL_ARRAYS` | `constants.newyork.POOL_ARRAYS` |
    | `generation.nygrid.fixes._HVDC` | `constants.newyork.HVDC_LINKS` (a tuple) |
    | `generation.nygrid.case.NEEDED`, `PASSED_OVER` | `constants.newyork.CASE_FIELDS_NEEDED`, `CASE_FIELDS_PASSED_OVER` |
    | `generation.nygrid.inputs.INTERFACES` | `constants.newyork.INTERFACE_NAMES` |
    | `generation.nygrid.interfaces.NYISO_INTERFACES` | `constants.newyork.INTERFACE_BRANCHES` |
    | `generation.nygrid.inputs.NYGRID_ZIP`, `TERRITORIES` | `constants.newyork.NYGRID_ZIP_URL`, `TERRITORIES_URL` |
    | `generation.nygrid.inputs.GEOJSON`, `LOCATIONS` | `constants.newyork.TERRITORIES_FILE`, `BUS_LOCATIONS_FILE` |
    | `generation.nygrid.inputs._ZONE_NAME`, `_FUEL_PRICE`, `_NUCLEAR_COLS` | `constants.newyork.ZONE_NAME`, `FUEL_PRICE`, `NUCLEAR_COLS` |
    | `generation.nygrid.load_sources._EIA`, `_IAC`, `_OEDI`, `_GAZETTEER` | `constants.newyork.EIA_861_URL`, `IAC_URL`, `OEDI_URL`, `GAZETTEER_URL` |
    | `generation.nygrid.load_sources._STOCK` | `constants.newyork.STOCK_KEY_PREFIX` (a key prefix in the OEDI bucket, not a whole address) |
    | `generation.nygrid.dispatch._AREA_PRICE` | `constants.newyork.AREA_PRICE` |
    | `generation.nygrid.dispatch._NIAGARA`, `_ST_LAWRENCE` | `constants.newyork.NIAGARA_BUS`, `ST_LAWRENCE_BUS` |
    | `generation.nygrid.dispatch._NIAGARA_CAP`, `_ST_LAWRENCE_CAP` | `constants.newyork.NIAGARA_CAP_MW`, `ST_LAWRENCE_CAP_MW` |
    | `generation.nygrid.dispatch._NUCLEAR_COST`, `_HYDRO_COST` | `constants.newyork.NUCLEAR_COST_USD_PER_MWH`, `HYDRO_COST_USD_PER_MWH` |
    | `generation.nygrid.source.ATTACKED_FRAC` | removed: the default of `NewYorkSettings.attacked_frac` |
    | `engine.attacks.minimize._BLAS`, `_BLAS_LOCK`, `_BLAS_HELD`, `_BLAS_USERS`, `generation.design._STATE` | lower case, being state and not constants: `_blas_pools`, `_blas_lock`, `_blas_held`, `_blas_users`, `_worker_state` |

- **One writer owns a timeline file, one record is an episode, and the layers of indirection around
  them are removed (breaking for the internal names below; no behaviour and no stored byte changed: the
  five whole-file digests are unchanged, and an IEEE timeline is byte for byte the file the previous
  code writes).** `fg.generate`, `fg.load`, `generate_timeline`,
  `TimelineSource.generate`, `IeeeSource`, `NewYorkSource` and `SingleEpisodeSource` are called as before.
  - **`generation.write.TimelineWriter`** is the open file of a timeline for both grids and the only
    place a timeline file is written: `TimelineWriter(path, dims, clean, spec)`, `write_graph`, `header`,
    `store`, `flush`, `add_episode`, `finish` (the masks, the columns, the episodes, the attributes and,
    on a file that carries them, the temporal features) and `close`. A source holds it as
    `source.writer`. It replaces `start_timeline`, `create_layers`,
    `finish_timeline`, `timeline_attrs`, `placement_attrs`, `_write_masks`, `_write_episodes`,
    `_write_min_rows` and `_write_am_rows` of `generation.write`; `TimelineBuffers`, `layers` and
    `CleanSlice` of `generation.emit` (`CleanSlice` is now in `generation.write`); `header_attrs`,
    `_write_graph`, `_base_attrs`, `_static_physics`, `GRAPH_LEGEND` (now in `generation.write`) and
    `_CHUNK_ROWS` of `generation`; and `TimelineSource._finish_file`. `generation.write.search_attrs` is
    part of `IeeeSource._recorded`. `write_temporal_layers` is `TimelineWriter.write_temporal_layers`
    (a static method, which `trust.secured_copy` calls on its own file) and `ragged` is private to the
    writer.
  - **The steps of the walk are methods of `generation.emit.Walk`**: `walk_split`, `benign_run`,
    `ramp_episode`, `overload_episode`, `emit_benign` and `relocate_later` (static) were functions of
    `generation.emit`. `emit.clean_slice` is `IeeeSource._clean`. `generation.design.build_generator` is
    `FdiaGenerator.from_spec`; the worker functions of `generation.design` are static methods of
    `DesignPool` and `AmDesigner`. `engine.attacks.episodes.ramp_step` is the method
    `EpisodeDesignMixin.ramp_step` only.
  - **A New York survey cache (`survey_cache`) written before this change is not read**: its chunks
    pickled `engine.attacks.family.Episode`, and the chunks are now named `_v2`. The survey runs again
    and saves new chunks beside the old ones.
  - **`models.plan.Episode`** is the one record of an attack episode: what a New York family designs
    (it moved from `engine.attacks.family`) and a row of every file's episode table. `models.data.EpisodeRow`
    and the walk's `generation.emit._Episode` are gone; `Walk.buf` is `Walk.writer`, and `Walk.T` is gone.
  - **`models.plan.Planned`** (a placed episode with its design, its index and its moves) replaces
    `Pending` and `Designed`. `Schedule.build` is gone: `Schedule(...)` takes the same arguments.
    `generation.ieee._Planned` and `IeeeSource._generator` are gone (the generator comes from
    `FdiaGenerator.from_spec`); `AmDesigner` is a `DesignPool`.
  - **The designs of an episode are methods of the generator**: `EpisodeDesignMixin.ramp_design` and
    `am_overload_design` hold the code of `RampDesigner` and `OverloadDesigner`, which are gone with
    `DESIGNERS`, `Designer` and `EpisodeDesignMixin.designer`. `MinimizeMixin.min_tamper` chooses between
    the candidate search and the row reduction itself: `SupportStrategy`, `SearchSupport`,
    `MinimizeMixin.support_strategy` and `RrefSupport.find` are gone (`RrefSupport.rref_support` stays).
  - **The loader**: the capability tests are written in the `dataset.base.CAPABILITIES` table (the ten
    one-line functions are gone); `RecordsMixin._add_benign` and `_add_currents` are one method, `_add_frame_layers`.
  - `models.inputs.GeneratedFamilies` is a `FamilySelection` and `SupportedSystem` a `SystemRef`.
  - `formulas.network`: the two Newton solves share the halving of a step (`_halved_steps`).

- **The analysis toolkit's names are imported from the module that defines them: the eight tables of
  re-exported names in `fdia_analysis` are removed (breaking for these import paths; no behaviour, no
  stored byte and no result number changed).** The same change as the generation toolkit's, below. Each
  package's `__init__.py` is now its docstring; `import fdia_analysis` and `fdia_analysis.__version__`
  are unchanged.
  - `fdia_analysis.se` exports no name: `WLS`, `AdaptiveWeighting`, `ResidualRemoval`, `SubspacePrior`,
    `JacobianWeighting` and `GatedPrior` come from `fdia_analysis.se.methods`, `SEBase` from `se.base`,
    `JacobianFeatures` from `se.jacobian`. `fdia_analysis.se.detect` exports none either: `AlarmRule` and
    `PersistenceRule` come from `se.detect.rules`.
  - `fdia_analysis.localization`: `SwingThreshold`, `DeltaThreshold` and `ResidualLocalizer` come from
    `localization.methods`, `BusCNN`, `BusMLP`, `LearnedLocalizer`, `full14` and `kcl_residual` from
    `localization.learned`, `LocalizerBase` from `localization.base`.
  - `fdia_analysis.federated`: `FedBusCNN`, `FedBusMLP` and `FederatedLocalizer` come from
    `federated.localizer`, `spectral_partition`, `partition_from_assignment`, `bus_adjacency` and
    `compute_nodes` from `federated.partition`, `RegionalPrior` from `federated.se`; `Partition` and
    `RoundLog` from `fdia_analysis.models.federated`, which defines them.
  - `fdia_analysis.trust`: `TrustedMeters` and `TrustSelector` come from `trust.base`, `TrustedMetersDQN`
    from `trust.meter_dqn`, `secured_copy` from `trust.secured`.
  - `fdia_analysis.results`: `Run` comes from `results.run`, `Store` from `results.store`, `QUERIES`,
    `fill`, `query` and `stale` from `results.docs`, `cell`, `figure_data`, `pivot` and `table` from
    `results.render`; `METRICS`, `Better`, `Metric`, `Provenance` and `Record` from
    `fdia_analysis.models.results`, which defines them.
  - `fdia_analysis.models` exports no name (`__all__` and `PUBLIC` are gone): `from fdia_analysis.models
    import PriorConfig` becomes `from fdia_analysis.models.config import PriorConfig`, and likewise
    `models.inputs`, `models.scores`, `models.data`, `models.grid`, `models.federated`, `models.training`
    and `models.results`. The bundles the data dictionary lists for this toolkit are the ones
    `models/scores.py` defines, now in that file's order (the same eleven tables, each unchanged).
  - `fdia_analysis.formulas` exports no name: a formula is imported from `formulas.estimation`,
    `formulas.federated`, `formulas.linalg`, `formulas.measurement`, `formulas.metrics`,
    `formulas.projection` or `formulas.trust`.
  - As for any package, `fdia_analysis.se.methods` (or any other submodule) is an attribute of its package
    only once that submodule has been imported.
- **Names only the tests reached are removed (breaking for these names; no behaviour and no stored byte
  changed).** None was named in the docs or used by either toolkit.
  - `FdiaGraph.edge_attr_np` is gone: `ds.edge_attr.numpy()` holds the same values.
  - `fdia_graph.engine.records.stealthy_state`, `with_region` and `is_feasible` are gone: they were
    aliases of the `FalseStateMixin` methods of the same names, which every caller already used.
  - `fdia_graph.schema.ATTR_KEYS` and `fdia_graph.schema.CURRENT_LAYERS` are gone (the tests that compared
    a file against them build the two sets themselves, from `schema.Attr` and the four `PMU_I*` paths).
  - `fdia_graph.timeline.DEFAULT_FAMILIES` is gone: it was `models.choices.GENERATED_FAMILIES`.
  - `fdia_graph.generation.nygrid.loads.SWITCH_INDUSTRIAL_MW` is gone: a constant no code read.
- **A name is imported from the module that defines it: the tables of re-exported names are removed
  (breaking for these import paths; no behaviour and no stored byte changed).** `fg.load`, `fg.generate`,
  the profile helpers, `fg.FdiaGraph` and the rest of `import fdia_graph as fg` are unchanged, and so is
  `fdia_graph.timeline.generate_timeline`.
  - `fdia_graph.models` exports no name (`__all__` and `PUBLIC` are gone): `from fdia_graph.models import
    RecordBundle` becomes `from fdia_graph.models.data import RecordBundle`, and likewise `models.base`
    (`Bundle`), `models.config` (the settings), `models.inputs`, `models.fields`, `models.frames`,
    `models.grid` (`NODE`, `EDGE`, `CURRENT`, `MeterPlan`, ...), `models.plan` and `models.assets`. The
    bundles the data dictionary lists are the ones `models/data.py` defines.
  - `fdia_graph.formulas` exports no name: a formula is imported from `formulas.network`, `formulas.noise`,
    `formulas.attacks`, `formulas.temporal` or `formulas.trust`.
  - `fdia_graph.generation.nygrid` exports no name: `nygrid_case`, `Case` and `ZONES` come from
    `generation.nygrid.case`, `Network` from `.network`, `read_pool` and `write_pool` from `.io`, `Pool`,
    `PoolBuilder`, `peak_ratings` and the solve kinds from `.pool`, `NewYorkSource`, `SingleEpisodeSource` and
    `Survey` from `.source`, `Settings` and `rng` from `.config`, and so on by module; `Dataset` and
    `ClientView` from `fdia_graph.dataset.clients`, `ipopt` from `fdia_graph.engine.solvers`.
  - `fdia_graph.engine.sensing` exports no name (`meters.MeterPlan`, `meters.plan_meters`, `meters.KINDS`,
    `measure.Measurement`, `noise.NoiseModel`), and `fdia_graph.engine` none: `FdiaGenerator` is
    `fdia_graph.engine.core.FdiaGenerator`.
  - `fdia_graph.generation` no longer serves `DesignPool`, `IeeeSource` and `TimelineSource`: they are
    `generation.design.DesignPool`, `generation.ieee.IeeeSource` and `generation.source.TimelineSource`.
  - The module `fdia_graph._lazy` (`attach`) is removed with the tables it served.
  - The errors have one home, `fdia_graph.errors`, which now defines them: the module
    `fdia_graph.models.errors` is removed, and `ConfigError` and `MissingCapability` are no longer names of
    `fdia_graph.models.validation`. The classes are the same ones, so an `except` clause is unaffected;
    their `__module__` is now `fdia_graph.errors`, which is what a traceback, a `repr` and a pickle name.
  - `fdia_graph.formulas.<module>` and `fdia_graph.models.<module>` are attributes of the package only once
    that submodule has been imported, as for any package: `import fdia_graph.formulas` alone no longer
    makes `fdia_graph.formulas.network` reachable; `import fdia_graph.formulas.network` does.
  - The module-level `__all__` lists of `models.fields`, `formulas.trust`, `engine.solvers`,
    `generation.nygrid.tables` and `fdia_analysis.se.detect.rules` are removed. Nothing star-imported
    these modules; `from fdia_graph import *` keeps its list.
  - `import fdia_graph` loads six fewer modules than before (four formula modules, `models.frames` and
    `models.plan`, which only the tables pulled in) and, as before, neither pandapower, torch, pandas, scipy nor Pyomo.
- **One file layout: a New York timeline is written in the layout of every timeline (breaking for New York
  files; IEEE files are unchanged).** The New York grid wrote its own layout with its own writer and reader.
  It now writes the groups, datasets and attributes named in `fdia_graph.schema`, through the writer the IEEE
  cases use (`generation/write.py`, `generation/emit.py`), and `FdiaGraph` opens the file. No number moved:
  the readings, the benign readings, the tampered meters and the attacked buses gathered back from a new
  file were compared, bit for bit, with the arrays the old layout stored, on the three digest files (twelve
  comparisons, all equal; a one-time proof, run before the three New York digests were re-recorded and
  then removed), and the two IEEE whole-file digests pass unchanged. The three New York digests are
  re-recorded, since every path in them changed.
  - Old New York files stop reading. There is no converter and no fallback: `Dataset` refuses a file in the
    earlier layout with `NotATimeline`, and fdia-graph 0.21 reads it. Regenerate the file to use it with
    this version.
  - Paths. `frames/z [T, M]` is now three layers with their masks, zero where no meter sits:
    `data/node_x [T, N, 4]` (|V|, P, Q, angle), `data/edge_x [T, E, 4]` (P and Q flow at the from end, then
    at the to end) and `data/pmu_i [T, E, 4]` (current magnitude and angle at the from end, then at the to
    end), with `data/node_m`, `data/edge_m`, `data/pmu_i_m`. `frames/z_benign` is `benign/node_benign`,
    `benign/edge_benign`, `benign/pmu_i_benign`. `labels/meter_tampered` is `attack/node_tamper`,
    `attack/edge_tamper`, `attack/pmu_i_tamper`; `labels/bus_attacked` is `data/y`. `frames/time`, `split`,
    `family` and `episode` are `data/time`, `data/split`, `data/family` and `data/seq_id`. `frames/vm` and
    `frames/va` are columns 0 and 3 of `clean/node_clean`.
  - Family codes. `data/family` and `episodes/family` carry the one table of `models/choices.py`: 7 for Am
    and 5 for At, where a New York file had 1 for Am and 2 for At (and `episodes/family` held the names as
    text). The New York table `FAMILIES` of `generation.nygrid.io` and the file attributes `family_codes`,
    `split_codes` and `kind_codes` are removed.
  - Types. The labels are uint8 (were bool); the true state is float32 in `clean/node_clean` (`vm` and `va`
    were float64); `episodes/onset`, `length` and `target_branch` are int32 (were int64); the episode buses
    are `episodes/bus_idx` (int32) with offsets `episodes/bus_ptr` (were `buses`, `buses_ptr`, int64).
  - Added. `clean/node_clean` also holds the P and Q injections of the solved state, and `clean/edge_clean`
    its exact flows at the metered branch ends. `graph/` holds the edge index, the branch model and bus
    shunts that give the admittance matrix (it equals the case's, `tests/nygrid/test_dataset.py`), the bus
    type, voltage limits, base kV, base load and generator flag, and `bus_client`. The root carries the
    attributes of every timeline that apply (`kind`, `system`, `N`, `E`, `T`, `baseMVA`, `seed`, the
    legends, the split and episode counts, `settings` and `settings_hash`) and three of its own:
    `client_names`, `current_form` and the wider `edge_feat` (`P_from,Q_from,P_to,Q_to`). Also
    `data/stealthy`, `data/timestep` and the empty `attack/mag*` tables, as in every timeline.
  - What identifies a file. `settings` (JSON) and `settings_hash` record `NewYorkSource.record`: the
    settings of the timeline (the new `NewYorkSettings.as_record`) and, under `release`, the knobs of the
    `NewYorkRelease` the source was built for (`families`, `interfaces`, `hold_external`, `v_margin`,
    `gen_v_tol`, `load_cap`, `band`; the new `NewYorkRelease.as_record`), so two files that differ in a
    release knob have different hashes. A single-episode file records its episode (`episode`: family,
    onset, lag and the pool frames it covers) in place of the attacked share, the splits and the survey;
    it has no `target_attacked_frac`, and its `split_frac` is 1, 0, 0. A family passed as `using` is not
    recorded.
  - Line ratings. `graph/edge_rating` (MVA, New York only) holds the ratings the attacks were designed
    against. The release rates a branch the case file leaves unrated at its peak flow over the pool (N8),
    so the case file alone does not give them.
  - The integers of `graph/` (`edge_index`, `bus_type`, `bus_client`) are int32 on every platform, the
    type of an IEEE file's `edge_index`; they were the platform's own integer (int64 on Linux).
  - Removed. `case/` (the unit table, the HVDC links, the names and places of the buses and the base
    state, which are the case file's, `systems/case_nygrid.m`; the line ratings moved to
    `graph/edge_rating`), `meters/` (the table
    is rebuilt from the masks; the accuracy class is the table `ACCURACY`, the bias is drawn from the seed
    and is not stored, the owner is the owner of the meter's bus), `case/tie_lines` (a branch whose ends
    have two owners) and `episodes/clients` (the owners of an episode's tampered meters).
  - What a New York file declares differently from an IEEE one. Its flow layers are four wide. Its PMU
    currents are polar (`current_form = "polar"`; an IEEE file stores real and imaginary parts and carries
    no such attribute); the estimators' pseudo-measurements, which read real and imaginary parts, refuse a
    polar file with `MissingCapability` (the new capability `rectangular_currents`). Its labels keep the New
    York rule, written in the `tamper` attribute of `attack/`: a meter is tampered where the attack moves
    its reading by more than its accuracy-class standard deviation, and a bus is attacked where a meter at
    it is tampered. It has no `data/temporal_delta` and no `data/swing`.
  - `FdiaGraph` on a New York file. The pandas export has a column per stored flow, so a four-wide file
    also gives `P_to_e{n}` and `Q_to_e{n}` (an IEEE table is unchanged). A `units="pu"` view gives the
    angles of polar PMU currents in radians, as it gives theta, in records and exports. An estimator
    refuses a file whose `system` is no IEEE case with `MissingCapability` (it was a `KeyError`). The
    refusal of `windows()` on a view that is not consecutive now names a week split, whose weeks
    interleave with the other splits'.
  - Code. `MeterPlan` gains `scatter` (a value per meter laid out as the layers), `from_masks` (the table a
    file's masks describe, in `plan_meters` row order) and `gather` (the flat vector read back from the
    layers). `Dataset` reads schema paths; it gains `z(frames, benign=False)` and `tampered(frames)` and no
    longer has `case` or `kinds` (the kinds are the constant `KINDS` of `engine/sensing/meters.py`, and
    `ClientView` no longer takes them). `TimelineWriter`, `open_timeline` and `FAMILIES` are removed from
    `fdia_graph.generation.nygrid` with no alias; `read_pool` and `write_pool` stay. The writer's functions
    take what both grids have: `start_timeline`, `finish_timeline(f, buf, columns)`, `timeline_attrs(header,
    buf, knobs, record)`, `placement_attrs(rec, split, ...)`, and `LayerSet` (`models/frames.py`) says which
    layers a grid's file carries. What every source does when it finishes a file is written once, in
    `TimelineSource` (`record`, the settings a file is identified by, and `_finish_file`).
  - A New York days file (576 frames) is 4.2 MB, from 2.9 MB: the layers hold a zero where no meter sits and
    the masks are stored per frame, as in an IEEE file.

- **Two toolkits in one repository (breaking for the imports of the analysis packages).** The package was
  generation and analysis in one; it is now two distributions that live side by side here. `fdia-graph`
  (the package `fdia_graph`, under `src/`) generates a timeline and loads it. `fdia-analysis` (the package
  `fdia_analysis`, under `analysis/`, with its own `pyproject.toml`, tests and README) is what is measured
  on a timeline. The analysis toolkit depends on the generation toolkit (it reads the files through
  `fdia_graph.dataset` and uses its formulas and records); nothing in `fdia_graph` imports `fdia_analysis`,
  which `tests/test_toolkit_boundary.py` checks in the source and by generating and loading a timeline in an
  interpreter where `fdia_analysis` cannot be imported. No generated file changes (the timeline digests of
  the IEEE cases and of New York are the same, the settings hash stored in a file included) and no result
  number changes (no results block of the docs changed). No alias is kept for a moved name.
  - What a caller sees, in short. The imports of the analysis packages change from `fdia_graph.*` to
    `fdia_analysis.*` (`from fdia_graph.se import WLS` becomes `from fdia_analysis.se.methods import WLS`). The `se`
    and `federated` extras move to `fdia-analysis` (`pip install "fdia-graph[se]"` becomes
    `pip install "fdia-analysis[se]"`). The two toolkits share one version and are released together:
    `fdia-analysis` X.Y.Z requires exactly `fdia-graph==X.Y.Z`, so upgrading one upgrades the other.
  - What moved, package by package: `fdia_graph.se` is `fdia_analysis.se` (with `fdia_analysis.se.detect`),
    `fdia_graph.localization` is `fdia_analysis.localization`, `fdia_graph.federated` is
    `fdia_analysis.federated`, `fdia_graph.trust` is `fdia_analysis.trust`, and `fdia_graph.results` (the
    results store's code) is `fdia_analysis.results`. Their classes and functions keep their names and
    behaviour: `from fdia_analysis.se.methods import WLS`,
    `from fdia_analysis.localization.methods import SwingThreshold`.
  - What stays in `fdia_graph`: `fg.load`, `fg.generate`, `FdiaGraph`, the profiles, the registry and the
    download, the file protocol (`schema`), `generation` (the New York grid included), `engine`, `dataset`,
    `ratings`, `errors` (every named error, the ones the analysis packages raise too), and the models and
    formulas the generator or the loader use. `import fdia_graph as fg` and everything under `fg.` are
    unchanged.
  - The models only the analysis packages use are in `fdia_analysis.models`. The settings (from
    `models.config`): `SolveConfig`, `FitOptions`, `HuberConfig`, `RemovalConfig`, `PriorConfig`,
    `JacobianWeightingConfig`, `GateConfig`, `LocalizerConfig`, `LearnedConfig`, `TrainerConfig`,
    `PerBusReport`, `TrustConfig`, `DqnConfig`, `FederatedSettings`, with the field types `Fraction`, `Share`,
    `Norm` and `Tolerance`. The fixed sets (from `models.choices`): `Calibrate`, `Reweight`, `Features`,
    `FrOver`, `Buses`, `Kcl`, `Reduce`. What the analysis formulas accept (from `models.inputs`):
    `ShapedArray`, `SubBatch`, `EdgeList`, `ClientCount`, `ClientGraph`, `Halo`, `AssignmentSpec`,
    `PartitionOnGrid`, `ClientUpdates`, `MomentParts`, `FeatureBlock`, `Affinity`, `StateBlocks`,
    `LabelGrids`, `TauSearch`, `RankedLabels`, `Aggregation`, and the rule `IntegerDtype` (from
    `models.validation`, now beside the models that declare it in `fdia_analysis.models.inputs`). The score tables (all of `models.scores`): `ErrorPair`, `EstimatorScores`,
    `OverallMetrics`, `BenignMetrics`, `FamilyMetrics`, `LocalizerScores`, `PerBusMetrics`, `PerBusScores`,
    `GridScores`, `TrustScores`, `JacobianOutputs`. And `TrueState` (from `models.data`), `PseudoLinks` and
    `PseudoVoltages` (from `models.grid`), `Partition` and `RoundLog` (all of `models.federated`),
    `OptimConfig` (all of `models.training`), and the results store's records (all of `models.results`:
    `Record`, `Provenance`, `RunRecords`, `Metric`, `METRICS`, `Better`, `Leaf`, `KEYS`, `leaves`,
    `as_values`). Each is in the module of the same name (`fdia_analysis.models.config`, and so on).
    The loader's bundles are the ones `fdia_graph.models.data` defines; the score tables are the ones
    `fdia_analysis.models.scores` defines.
  - The formulas only the analysis packages use are in `fdia_analysis.formulas`. Whole modules:
    `formulas.estimation` (`normal_matrix`, `wls_step`, `wls_step_batched`, `weighted_objective`,
    `residual_covariance_diag`, `critical_measurements`, `floored_covariance`, `normalized_residual`,
    `huber_weights`, `whitened_svd_basis`, `gate_weights`, `pmu_pseudo_links`, `pmu_pseudo_phasors`,
    `pmu_pseudo_voltages`), `formulas.linalg` (`guarded_inverse`, `condition_number`,
    `batched_normal_matrices`), `formulas.metrics` (`perbus_counts`, `perbus_counts_at`,
    `perbus_f1_from_counts`, `tau_from_counts`, `perbus_rates`, `micro_prf`, `sample_f1`, `strict_accuracy`,
    `average_precision`), `formulas.projection` (`bus_incidence`, `weighted_pseudoinverse`, `leverage`,
    `weak_directions`, `explained_unexplained`, `direction_coefficients`, `weak_move`, `meters_to_buses`,
    `meter_positions`) and `formulas.federated` (`Moments`, `channel_moments`, `pool_moments`, `fedavg`,
    `attackable_affinity`, `interior_boundary`, `cut_edge_count`, `hop_distance`, `halo_nodes`,
    `block_diagonal_basis`). Parts of modules: the estimator's measurement function and its Jacobian,
    `ac_measurement` and `ac_jacobian`, left `formulas.network` for the new
    `fdia_analysis.formulas.measurement`; `attack_subspace`, `sparse_basis`, `attack_cost` and
    `greedy_trusted_meters` left `formulas.trust` for `fdia_analysis.formulas.trust`, which builds on the
    `rref` that stays in `fdia_graph.formulas.trust` beside `sparsest_rows` (the attack generator uses both);
    that module's tolerance is now the public `TOL`.
  - Three names moved inside `fdia_graph`, to where the generation code that uses them is.
    `accuracy_class_sigma` is in `formulas.noise` (it was in `formulas.estimation`; the meter noise model
    uses it). `Ownership` and `assign_clients`, which utility owns which New York bus, are in
    `generation.nygrid.ownership` (the module was `federated.utilities`). `config_hash`, the hash of the
    settings a timeline stores (`settings_hash`), is in `schema` (it was in `results.run`, which now imports
    it); the hash of a given settings record is the same.
  - Install: `pip install fdia-graph` for the loader, with `[torch]`, `[pyg]`, `[generate]`, `[iso]`, `[ny]`
    and `[all]` as before; `pip install fdia-analysis` for the analysis toolkit, which installs `fdia-graph`,
    with the extras `[se]` (pandapower, scipy), `[torch]`, `[federated]` (torch, scikit-learn) and `[all]`.
    The `[se]` and `[federated]` extras of `fdia-graph` are gone, and its `[all]` no longer brings
    scikit-learn; the install hints in the analysis toolkit's error messages name `fdia-analysis`.
  - In the repository: each toolkit has its tests (`tests/`, `analysis/tests/`), which share the tiny generated
    timeline through the fixtures of the root `conftest.py` and the helpers of `tests/helpers.py`; `pytest` at
    the root runs both on this checkout's source.
    `tools/prereview.py` is still the one command for every gate, now over both packages (`tools/sources.py`
    names them once for every tool), the file map and the data dictionary list both, the module diagram
    draws both with every arrow between them pointing from analysis to generation, and `tools/layout.py`
    resolves a cited name in the toolkit it says (`fdia_analysis.se.base`) or, with none said, in either.
    CI installs both and tests both; `publish.yml` (which runs on a version tag only) builds both
    distributions and uploads both, since they share the tag's version (`tests/test_versions.py`, and
    `tools/release.py` before it tags).

- **One path generates a timeline, for the IEEE cases and for the New York grid (breaking for the New York
  build's callers).** `TimelineSource.generate(out)` (`generation/source.py`) makes every timeline in the same
  order: start the file, prepare, cut the splits, then split by split place the episodes (`generation/plan.py`),
  design them and walk the frames in time order, then finish the file. The New York grid no longer has a driver of
  its own. No generated value changes: the five pinned timelines (two IEEE-14, the New York days, two single
  episodes) are the same, dataset by dataset, and their snapshots were not rewritten.
  - The grid is a class hierarchy. `TimelineSource` is the system a timeline is generated on, and a subclass
    owns only what differs between the grids: the pool, what it must know before the splits (`prepare`), how the
    splits are cut (`cut_splits`), which onsets a family may take (`onset_masks`), the stream the onsets are drawn
    from (`placement_rng`), how an episode is designed and what happens when no design exists (`design`), how the
    frames are emitted (`walk(until)`) and the file layout (`open`, `finish`, `close`). The subclasses are
    `IeeeSource` (`generation/ieee.py`: blocks in time order, no mask, an Am with no design is moved),
    `NewYorkSource` (`generation/nygrid/source.py`: whole weeks, the families' masks and the onset survey, an
    episode with no design is left benign) and, under it, `SingleEpisodeSource` (one episode placed by hand).
    The placement itself is one function for both, `place_split`. The two file layouts are unchanged.
  - What a caller of the New York build now calls: `NewYorkSource.production(pool_path,
    NewYorkRelease(...)).generate(path)`, then `source.summary()` for what was placed and built. The arguments
    `production_builder` took are the fields of `NewYorkRelease` (`models/config.py`: `families`, `interfaces`,
    `workers`, `survey`, `survey_cache`, `root`, `hold_external`, `v_margin`, `gen_v_tol`, `load_cap`, `band`),
    checked when it is built; the build `Settings` (the seed) is the third argument. One episode alone:
    `SingleEpisodeSource(source, onset, family, lo, hi, using=..., lag=...).generate(path)`, then its
    `summary()`. `pipelines/newyork/build_timeline.py` and `build_single_episodes.py` do exactly that.
  - `NewYorkSource(pool, attacker, families, own, settings, build)` takes the timeline's settings as a
    `NewYorkSettings` (`length`, `attacked_frac`, `split`, `split_by`, `workers`, `block`, `verbose`, `survey`,
    `survey_cache`); the measurement is the attacker's and the noise model is made from its meter plan.
    `AttackFamily.onset_mask` is part of the family interface (None by default: the family may start anywhere).
  - Removed, with no alias: `TimelineBuilder` and its module `generation/nygrid/timeline.py` (its `build`,
    `build_episode`, `_slots`, `_designs`, `_write` and its worker pool), `production_builder`,
    `TimelineBuilder.episode_lengths` (now `NewYorkSource.lengths`), and the module `generation/nygrid/build.py`:
    `fetch_inputs` and the pinned URLs are in `generation/nygrid/inputs.py`. `timeline.py` no longer holds the
    IEEE set-up (`_generator`, `_knobs`, `_recorded`, `_write`): it is in `IeeeSource`.
  - `fg.generate` and `generate_timeline("ieee14", ...)` are unchanged: `generate_timeline` is
    `IeeeSource(system, states, settings, seed).generate(out)`. An `IeeeSource` generates one timeline (its
    generator's random stream is used up); a second `generate` is a `ConfigError`.
  - Behaviour that changed:
    - **A failed run leaves no file, on either grid.** A timeline is written beside `out` as `<out>.part` and
      takes the name `out` only when it is whole. Before, the file was created at the start of the run: a rerun
      destroyed the file already there at once, and a run that failed left a stub that opened as a timeline.
    - **Design workers are spawned on every OS** (`generation.design.DesignPool`, one pool for both grids,
      started once per timeline). A script that passes `workers` above 1 must start under
      `if __name__ == "__main__":`, on Linux too (before, IEEE workers were forked there). After a failed run
      the workers are stopped at once instead of finishing the designs in flight. A single episode is designed
      in the caller's process.
    - **New York frames are written in stretches**, each as soon as every week before it is planned, instead
      of one pass after every design. The noise stream still runs over the frames in time order, so no value
      depends on where a stretch or a block ends; `tests/nygrid/test_stretches.py` pins that against a
      one-pass build on a synthetic pool, with no data and no solver.
    - `NewYorkSource.generate()` with no path is a `ConfigError` (the IEEE default path is unchanged).
    - The warning for an episode dropped by a jammed placement reads the same on both grids: "split train:
      placements jammed 10 times, 1 Am episode(s) dropped" (the New York one did not give the count of tries).
    - A single episode with no attack still raises `NoAdmissibleTarget`, now from `generate`; an unsolved window
      or a window outside the file's frames is refused when the `SingleEpisodeSource` is built.
    - The release build asks for each unit's output in the pool (`unit_p`) only when it builds Am.
  - Records (`models/plan.py`): `SplitSpan` may list its frames (a split of whole weeks) and says where an
    episode fits (`onsets_fit`); `Designed` and `Placement` serve both grids' families; `Schedule.lengths`.
    `place_split(rng, lengths, frac, span, rec, masks)` takes the lengths and the fraction instead of a
    `Schedule`; `AmDesigner.design_split` returns the split's episodes as `Designed` records and `walk_split`
    takes them; `planned_until` says how far the walk may run once a split is planned.
  - `generation/nygrid/source.py` is no longer in `DEFERRED` of `tools/readability.py`: it meets the limits of
    the SDK's own code. `fdia_graph.generation` exports `TimelineSource`, `IeeeSource` and `DesignPool`;
    `fdia_graph.generation.nygrid` exports `NewYorkSource`, `SingleEpisodeSource` and `Survey`.

- **The simulated grid operator and the [WU26] defense reproduction are no longer part of the package (breaking
  for their imports).** Neither was reached by `fg.*`, by a pipeline or by a tool: they served study code, which
  is not kept in this repository. Both are removed with no alias. The operator was never in a release. Release
  0.21.0 still holds the reproduction, under its earlier module names (`trust/wu26.py`, `wu26_defense.py`,
  `defense.py`, `solution1.py`, `wu_dqn.py`) with the `wu` extra and its page in the docs (the `wu26` folder). No generated file
  changes: the pinned IEEE and New York timelines are the same, byte for byte.
  - `fdia_graph.operator` is gone: `operator_day`, `OperatorDay`, `PoolFrames`, `pool_frames`, `node_lbmp`,
    `respond`, `Operator`, `NyisoOperator`, `Twin`, `Response`, `OperatorPolicy`, `OperatorMode`, `AlarmAnswer`,
    `Entry`, `Dispatch`, `MODES`, `AFTER`, `ENTRY`, `entry_share`, `summarize`, `AcSced`, `Step`, `shortage_usd`,
    `loss_factors`, `ac_shift`, `ac_power_flow`, `open_branch`, `TrueGrid`, `tie_meters`, `neighbour_reads`,
    `tie_gap`, `interchange_alarm`, `seen_interfaces`, `area_damage`, `attack_surface`, `zone_levers`,
    `neighbor_prices`, `FEEDS`, `NEIGHBORS` and `PROXY`. The `Dispatch` removed here is the operator's policy
    choice (least movement or least cost), not the hourly `Dispatch` record `fdia_graph.generation.nygrid` still
    exports.
    `fdia_graph.generation.nygrid` no longer exports `operator_day`, `respond` and `summarize`.
  - What only the operator used: `OperatorOpf`, `OpfSolution` and `SEGMENTS` in `generation/nygrid/solver.py`
    (`AcSolver` and `Compensation` stay), `PoolBuilder.frame_limits`, `Interface.model_flow`,
    `se/detect/estimate.py` (`wls`, `WlsFit`, `exact_state`, `reference_bus`), `se/detect/fleets.py`
    (`FLEET_STREAM`, `fleet_bias`, `with_fleet`) and `fdia_graph.results.studies` (`study_days`, `variant`, `load_calib`,
    `add_dispatch_argument`, `add_study_arguments`, `run_parallel`). `fdia_graph.se.detect` keeps the alarm rules,
    `AlarmRule` and `PersistenceRule`.
  - `fdia_graph.trust` keeps the trusted-meter selection (`TrustSelector`, `TrustedMeters`, `TrustedMetersDQN`,
    `secured_copy`). Gone from it: `trust/faithful_attack.py` (`Wu26Network`, `wu26_network`, `wu26_snapshots`,
    `solve_window`, `incremental_freeze`), `trust/faithful_defense.py` (`Wu26DefenseEnv`, `wu26_solution1`),
    `trust/pmu_environment.py` (`DefenseEnv`, `WuDefenseEnv`), `trust/pmu_row_reduction.py` (`TrustedPMUs`) and
    `trust/pmu_dqn.py` (`TrustedPMUsDQN`).
  - Gone from `fdia_graph.models`: `Wu26Attack`, `WuDefenseConfig`, `WuDqnConfig`, `WindowAttack`, `WindowSlots`,
    `ChosenAction` and `SameDefense`; the choices `MeterModel.WU26` (`"wu26"`) and `SupportMethod.WU_L1`
    (`"wu_l1"`), which the generator already refused; the errors `NoSuchBranch` and `SnapshotDrawFailed`.
  - Gone from `engine/attacks/overload.py`: the case-study tables `WU26_SCENARIOS`, `WU26_PMUS` and
    `WU26_ATTACK_AREA`, and `OverloadMixin.wu26_area`, `wu26_branch` and `wu26_buses`. The tests that run on the
    paper's scenarios read them from `tests/case_studies.py`. The overload attack, the trust schedule
    (`TrustSchedule`) and the row-reduction support (`support_method="rref"`) are unchanged.
  - The `wu` extra is removed, and `cyipopt` leaves the `all` extra: nothing else needs it.
  - The analysis page "[WU26] defense reproduced" (the `defense_reproduction` folder) and its figures are
    removed, with the results it rendered: the experiments `wu26.*`, `wu.*`, `minlp.*` and `search.solver_compare` (their tables under `results/` and
    their rows in `results/runs.csv`) and their queries in `results/queries.py`. The trusted-meter results
    (`trust.selection`, `trust.secured_localization`, `trust.secured_estimation`) stay.
  - The example `custom_operator.py` is removed, and the New York pages no longer describe an operator.
    `tests/nygrid` loses the operator's tests, its snapshots (`golden/respond_*.json`) and the attack days they
    read (`data/days/am_48220_*.h5`, `sd_south_2019-07-05.h5`, `loads_2019_v2_days.npz`).

- **Code that only its own tests reached is removed (breaking for imports).** These names were used by nothing
  but their own tests, so they are removed with no alias; released versions keep them. One data-dictionary
  line, one checklist line and one tool list that named them are updated. No generated file and no number
  changes.
  - `registry.latest_release` is removed. `load(release=None)` never called it: it reads the pinned data
    release, as before.
  - The dataset accessors `edge_status`, `edge_is_trafo`, `bus_type`, `bus_vmin`, `bus_vmax`, `bus_base_kv`,
    `bus_is_zero_inj`, `bus_has_gen`, `bus_base_pd`, `bus_base_qd` and `bus_attackable` are removed.
    `ds.edge_attr`, the `ds.branch_*` columns and `ds.bus_shunt_g` / `ds.bus_shunt_b` stay.
  - `models.TimelineKnobs` is removed: `TimelineSettings` holds the same settings and is what
    `generate_timeline` reads.
  - `models.BranchColumns`, `models.BranchIndex`, `models.BRANCH` and `models.CurrentColumns` are removed. The
    column order of `pmu_i` is `models.CurrentIndex` (the constant `CURRENT`).
  - The validation rules `AllFinite` and `AllPositive` are removed.
  - The results-block queries `table` and `reduction` are removed; `value` stays, and the repository's tables
    come from the queries in `results/queries.py`. In a `value` block the arguments `rows`, `cols`, `base`,
    `new` and `order` are now tag filters like any other name; no block uses them.
- **The New York case file has one area; the zone column says where a bus is.** `case_nygrid.m` is one system
  for an OPF, with one area: its bus `area` column is 1 on every row. It used to carry
  NYgrid's NPCC area numbers (1 New England, 2 and 3 New York, 4 Ontario, 5 MISO, 6 PJM). Existing files are
  read as before, and no measurement, label, state, episode or operator result changes.
  - The region of a bus is its zone code in the file: 1 to 11 for the NYISO load zones A to K as before, then
    12 New England, 13 Ontario, 14 MISO, 15 PJM (12 used to mean any bus outside New York). The legend is in
    the file's header and above `mpc.bus`.
  - The `Case` record keeps its fields. `Case.zone` is the letter `A` to `K`, empty outside New York, and
    `Case.is_ny` is unchanged. `Case.area` is derived from the zone code on read: a neighbor's bus keeps
    NYgrid's number (New England 1, Ontario 4, MISO 5, PJM 6), and every New York bus is 2. The reader, the
    recipe's writer and the legend in the file follow one table of codes, names and areas, `REGIONS`, in
    `generation/nygrid/case.py`, with `FILE_AREA` (1) and `NY_AREA` (2); `OUTSIDE` is removed.
    `generation/nygrid/dispatch.py` and `operator/` still tell the neighbors apart by the area numbers and
    are not changed here. `Case.from_matpower` refuses a file whose area column is not 1 on every row
    with a `DataConditionError` that names the column.
  - One stored value differs in a New York timeline file generated from now on: `case/area` is 2 where it
    was 3, on the 10 downstate buses (NYgrid's case splits New York into areas 2 and 3; the file does not,
    and nothing in the package reads the difference). Every other dataset is the same bytes; the three New
    York digest snapshots were rewritten for that key alone. Files written before, with 3 stored, load and
    behave as before; the day files under `tests/nygrid/data/days` are unchanged.
  - `pipelines/newyork/make_case.py` is where NYgrid's area numbers become the zone codes; a test checks
    that each neighbor's buses and units are exactly the ones its area number gives.
- **New York code that nothing built is removed (breaking for New York imports).** The release builds Am against
  a NYISO interface (`InterfaceFdia`) and At (`Ramp`). The code below was reached by no pipeline, example, tool or
  doc, only by its own tests. It is removed with no alias. Release 0.21.0 still holds it, under
  `fdia_graph.nyfed` (`attacks/msfdia.py`, `grid/ratings.py`, the fits and `validate` in `scenario/loads.py`);
  `unit_q_limits` was never released. No generated file changes: the pinned New York timeline and episodes are
  the same, byte for byte.
  - The line-target Am and the Ap family: `engine/attacks/msfdia.py` (`MsFdia`, `NEAR_LIMIT`) is gone, and
    `OpfFdia` no longer attacks a line. `OpfFdia` is now the base class of the OPF attacks (the support search
    and the per-scan OPF) and `InterfaceFdia` is its one subclass. Its arguments `ratings`, `n_lines`, `tries`,
    `target_order`, `kappa`, `reactive_step`, `bound_rise` and `wait_share` are gone; the step bound now holds
    over the exit only, as the release always built it.
  - `Attacker.false_flow_state` is removed. `Attacker.false_state` (the ramp's solve) gives the same states.
  - `production_builder` builds `"Am"` and `"At"`. Asking it for `"Ap"` or any other family, or for Am with
    `interfaces=()`, raises a `ConfigError` that names the families it builds; before, `"Ap"` built a
    line-target attack and an empty `interfaces` built Am against branch ratings. The family code 3 (`Ap`) stays
    in `frames/family_codes`, so files written with it still read.
  - `MISSES` is removed from `engine/attacks/family.py`; the three miss reasons (`"no_target"`, `"infeasible"`,
    `"solver_error"`) are unchanged and listed in the module's docstring.
  - `generation/nygrid/load_fit.py` (`fluctuation_fit`, `switching_fit`), `generation/nygrid/load_metrics.py`
    with `LoadModel.validate`, `generation/nygrid/ratings.py` (`line_ratings`) and `unit_q_limits` are removed.
    The load constants the fits produced stay in `loads.py`, and the tie ratings stay in the case file.

- **The New York grid is one MATPOWER case file (breaking for New York imports).** The grid every New York pool
  and timeline is built on is `src/fdia_graph/systems/case_nygrid.m`, shipped with the package, in place
  of code that assembled it from NYgrid's `npcc.mat` each time. A change to the grid is now an edit to that file.
  No generated file changes: the file reads back to the same arrays, bit for bit, as the code built.
  - `nygrid_case()` returns the grid's `Case` (140 buses, 227 branches, 271 units). `production_builder`, the
    tests and `pipelines/newyork/run_pool.py` start from it, so the case no longer needs NYgrid's `Data/` folder
    (the hourly data still does).
  - `Case.from_matpower(path)` reads a MATPOWER version-2 case file with matpowercaseframes, the reader
    pandapower uses for `.m` files (a new dependency of the `ny` and `all` extras), taking each column by
    MATPOWER's name. The bus `zone` column says where each bus is, as a number (the entry above),
    `mpc.bus_name` and `mpc.genfuel` are MATPOWER's own, and `mpc.bus_latlon` and `mpc.gen_ramp`
    hold the coordinates and the ramps. `Case.check` refuses a file whose parts disagree, and an indexed
    assignment or an unknown field is refused instead of passed over, each a `DataConditionError` naming the
    file and what is wrong. The file's writer is in `pipelines/newyork/make_case.py`.
  - `ZONES` moved from `generation/nygrid/inputs.py` to `generation/nygrid/case.py`.
  - `read_npcc`, `modify_mpc`, `nygrid_units`, `split_zone_j` and `restore_765kv` are removed from the package
    with no alias, and `generation/nygrid/npcc.py` with them. They are how the file was made, and live in
    `pipelines/newyork/make_case.py`, which makes the file again from NYgrid's `Data/` folder; a test checks that
    it reproduces the checked-in file exactly. `hvdc_links` and `externals_at_base` stay in
    `generation/nygrid/fixes.py`.
  - `HQ_BUS` moved from `generation/nygrid/npcc.py` to `generation/nygrid/case.py`.

- **One copy of the shared constants and helpers (breaking for imports).** Names the IEEE generator and the New
  York builder each held a copy of now have one home, and the old names are removed with no alias. No generated
  file changes: every value and every floating-point expression is the one it was.
  - `SPLITS` (`("train", "val", "test")`) is in `models/choices.py`, beside `Split`, and derived from it. The copy in
    `generation/nygrid/io.py` is removed, and `fdia_graph.generation.nygrid` no longer exports it.
  - `generation.write.ragged` (variable-length rows as a pointer array and a flat array) also writes the New
    York episode table's `buses` and `clients`; `generation/nygrid/io.py` no longer builds them by hand.
  - `ACCURACY_CLASS` and `POWER_NOISE_FLOOR_MW` moved from `engine/base.py` to `formulas/noise.py`, beside
    `PMU_CURRENT_CLASS` and `CURRENT_FLOOR_PU`. The New York table `engine.sensing.meters.ACCURACY` (per meter
    kind, angles in degrees) is built from those four names instead of repeating their numbers.
  - `JITTER_FRAC` (0.25) moved from `engine/sensing/meters.py` to `formulas/noise.py`; it is the default of
    `bias_jitter_split` and what the IEEE generator passes, in place of a bare 0.25. `fdia_graph.engine.sensing`
    no longer exports it.
  - `generation.nygrid.inputs.download_once(url, path)` is the one download-if-missing of the New York inputs
    (to a `.part` file first, so an interrupted download is never taken for the file); the copies in
    `generation/nygrid/build.py` and `load_sources.py` are removed.
  - The module diagram follows `from fdia_graph.x import y` as well as relative imports (`tools/class_diagrams.py`
    followed only the relative ones), so it now shows what the New York modules import: `generation`, `engine`,
    `dataset` and `federated` read each other in both directions.
  - `formulas.temporal.wrap_degrees(d)` is the one wrap of an angle difference to (-180, 180].
    `engine.attacks.attacker.wrap_angles` (which picks the angle meters) and `ClientView.temporal` both call it.
  - `AM_LINE_TRIES` (8) is in `engine/attacks/overload.py` only; `engine/attacks/msfdia.py` imports it.
  - `engine.FAM_ID` (and `engine.core.FAM_ID`) is removed: the family codes are `models.choices.FAMILY_CODE`.
  - `RAMP_CODE` and `AM_CODE` of `engine/attacks/episodes.py` are removed: the same two codes are `RAMP_FAMILY`
    and `AM_FAMILY` in `engine/attacks/stealthy.py`.

- **One plan stage for every timeline (breaking for New York imports).** Where a timeline's splits are cut and
  where its episodes go is decided in `generation/plan.py` for the IEEE systems and for New York alike;
  `generation/nygrid/schedule.py` is removed.
  - `place_episodes(rng, lengths, frac, n, allowed=None)` is the one placement: it returns
    `Slot(onset, family, length)` records with the counts requested and dropped per family. The IEEE walk and the
    New York builder both call it. `split_weeks` and `week_index` live beside it; New York's `split_frames` is
    `split_column(split_bounds(T, frac), T)`.
  - `fdia_graph.generation.nygrid` no longer exports `place_episodes`, `split_frames`, `split_weeks` or
    `week_index`, nor `Slot`: the one `Slot` record, for both generators, is in `models/plan.py`.
    `plan.episode_counts` and `_Schedule.weights` are removed (the placement computes its weights from the
    lengths).
  - The plan and walk records have plain names, live in `fdia_graph.models` like every other record, and the
    stages take them instead of tuples: `Slot`, `SplitSpan` (a split's code and its frames), `Pending` (an
    episode not yet written), `Designed`, `Schedule`, `Placement` and `GeneratorSpec` in `models/plan.py`
    (`design.build_generator(spec)` builds a worker's generator from the spec), `FrameContext` in
    `models/frames.py`; the walk's state is the class `Walk` in `generation/emit.py`. The split names `SPLITS`
    are in `models/choices.py`. No generated file changes.
  - **One rule when a placement jams.** After ten whole placements fail, one episode of the family that could
    not be placed is dropped. The IEEE generator used to drop one of the family with the most episodes. Only a
    timeline that already warned "placements jammed" can differ. `place_episodes` reports
    what it dropped and its caller warns.
  - **Episode shares are exact.** The episodes of a split are shared between the families by largest remainder
    over the weights 1 / length as exact fractions. In floating point a tie between two families' remainders
    fell differently on the two generators (the IEEE one normalised the weights first). Counts can differ by
    one episode between families only where their lengths differ and the remainders tie: on the IEEE side for
    example `ramp_len=12`, `am_len=60`, `attacked_frac=0.05` on a 6,000-frame split got 12 At and 3 Am episodes
    and now gets 13 and 2 (the tie goes to the family named first, as documented). Families of equal length,
    the default, are unaffected, and so are the New York lengths (16, 16 and 96 frames).
  - The whole-timeline digests, IEEE and New York, are unchanged.
  - `capacity` (the most windows a family's allowed onsets can hold) is public in `generation/plan.py`, with
    `WEEK` and `STRATUM`; `HIT_STRATUM` is in `generation/nygrid/source.py`.

- **The study scripts leave the repository.** The folder `experiments/` (the estimation, localization, trust,
  federated and defense-reproduction studies, the data-release statistics, the design-speed measurement, and the
  federated study's per-run files) is removed: the repository holds the package, its tests, its tooling and its
  docs. The runs those scripts wrote stay in the results store (`results/`), and the pages under
  `docs/analysis/` still have their tables rendered from it; the pages now say the scripts are kept elsewhere, and
  their figures and CSV sidecars stay as last drawn (they can no longer be redrawn from this repository).
  The speed benchmark is repository tooling and is `tools/bench.py` again.
- **Family-name aliases and old import paths are removed (breaking).** Under the same rule as the deprecated
  logic: one name per family, one home per name, nothing kept for an old spelling.
  - **Families.** `"Ao"`, `"SLS"`, `"ramp"` and `"LRA"` are no longer accepted where a family is named
    (`fg.load(families=...)`, `generate_timeline(families=...)`, `family_ids`): write `"Aq"`, `"At"`, `"Al"`.
    `models.choices.FAMILY_ALIAS` is removed.
  - **Import paths.** A name is imported from the module that defines it, or from its package's public names
    (`fdia_graph`, `fdia_graph.models`, `fdia_graph.se`, ...). The copies other modules kept "where it was read
    before" are gone:
    - from `dataset.base` (and, where `fdia_graph.dataset` only passed them on, from it): `RecordBundle`, `BatchBundle`, `ArraysBundle`, `Summary`
      (in `models.data`); `FAMILIES`, `STEALTHY_FAMILIES`, `Split`, `Order`, `Units`, `RecordFormat`, and
      `Format`, `Label`, `Layer` of `dataset.export` and `dataset.sequence` (in `models.choices`; `fg.FAMILIES`
      and `fg.STEALTHY_FAMILIES` stay);
    - from `fdia_graph.timeline`: `AttackDesign` (in `models.frames`), `is_feasible` and `ramp_profile` (in the
      engine), `KIND` and `write_temporal_layers` (in `generation.write`), `RAMP_FAMILY`, `BENIGN_CODE`,
      `FAMILY_CODE` (in `models.choices` and `engine.attacks.stealthy`), and the plan stage's `SPLITS`,
      `REDRAWS`, `PLACE_TRIES`, `episode_counts`, `largest_remainder`, `split_column` (in `generation.plan`);
    - from `fdia_graph.engine`: `Scan` and `OperatingLimits` (in `models.frames`), `INTACT` and `Outage` (in
      `models.grid`), `POWER_NOISE_FLOOR_MW` of `engine.core` (in `engine.base`), and `AM_FAMILY`, `RAMP_FAMILY`,
      `STEP_HALVINGS`, `StealthyMixin` of `engine.records` (in `engine.attacks.stealthy`);
    - `schema.FAMILIES`, `schema.FAMILY_ALIAS` and `schema.STEALTHY_FAMILIES`, `se.base.Calibrate`,
      `profiles.system_id` (in `registry`),
      `formulas.projection.Reduce`, `localization.base.Buses` and `FrOver`, `federated.localizer.Kcl` (in
      `models.choices`).
  - The whole-timeline digests are unchanged by this entry: no generated number moves.

- **No DC model is left (breaking; New York operator numbers move).** The DC dispatch `operator.sced` is removed
  with `DcSced` (and its `implied` and `prices`), `shift_factors`, `outage_factors`, the objective `cost_lf`
  and the constants `OBJECTIVES`, `ROOM_MW`, `UNIT_ADDER_WEIGHT` and `ENERGY_CAP`; `operator.run.AC_DISPATCH`
  goes too. The operator dispatches by the AC
  optimal power flow that was `dispatch="ac"` and `"ac_cost"`: `Dispatch` is now `"movement"` (the default) and
  `"cost"`, both AC, and `"ac"`, `"ac_cost"` and `"cost_lf"` are refused. `Step`, `shortage_usd`, `loss_factors`
  and the dispatch constants live in `operator.acsced`; `fdia_graph.operator` exports `AcSced`.
  - `respond` needs the bus reactive loads of the pool's load series (`pool_frames`: `NYGRID_LOADS`, or the
    default file beside the data), which only the AC dispatches needed before.
  - `TrueGrid(day)` screens each 345 kV outage by an AC power flow (`TrueGrid(day, ac=True)` before); the `ac`
    argument and the DC outage factors are gone. `zone_levers(day)` uses the AC sensitivities (`ac_shift`); its
    `ac` argument is gone. `area_damage`'s payments are the AC bus prices, attacked less benign.
  - A `respond` run is an AC OPF per scan, about a minute per day on the test machine against five seconds for
    the DC dispatch. The `respond_*` snapshots under `tests/nygrid/golden/` are regenerated under AC and cut
    from 26 to 14. They compare within relative 1e-4 and absolute 1e-2 on the machine that recorded them
    (`respond_recorded_on.json`), where a refactor must repeat them, and within 5 percent or 2 (MW or dollars)
    elsewhere: on another platform the least-movement dispatch lands on another of its near-equal answers (the
    first Linux run moved the same MW to 0.05 percent and the dollars to half a percent). The 14 are: every operator and loop on the window past the limit,
    the trusting closed loop on the window at the limit, and the seven options on the window past the limit.
  - Load the units cannot follow is no longer priced at the 1,000 $/MWh offer cap (`ENERGY_CAP`, the DC
    least-cost dispatch's): the AC dispatch leaves it as a slack at `SLACK_WEIGHT`.
  - With `delay` the benign twin now moves: it dispatches each scan for the scan before against the current
    basepoint (14,439 MW-scans on the snapshot day at `delay=1`), where the DC benign twin moved nothing. This
    is how the AC dispatch already behaved; it is now the only behaviour.
  - Tests of `DcSced` alone are removed (its lossless prices, its shortage and energy caps, the DC shift and
    outage factors); relieving an interface read over its limit, ramp-limited recovery and shedding only when
    allowed and only where it relieves are tested on `AcSced.step`.
- **The attack-area rule reads AC flow sensitivities.** Rule 3's spread term of `AreaMixin.area_score` (used only
  with `area_rule="rules"`; the default region is the buses within `hops`) is the mean absolute sensitivity of
  the goal lines' active flow to the area's injections at the window's first true state
  (`formulas.network.flow_sensitivities`, checked against a re-solved power flow), no longer DC transfer
  factors from the reactances. On IEEE-14 and IEEE-118 the two agree to a correlation above 0.996 and the term
  moves by 2 to 5 percent on the candidate regions tried; `AreaMixin._ptdf` is removed.

- **Deprecated logic is removed (breaking).** Main holds current code only; anything below still works in
  `fdia-graph==0.21.0`, and nothing here keeps an alias or a warning period.
  - **Record shards and stream files.** `fg.load` and `FdiaGraph` read timelines only: a file without the `kind`
    attribute `timeline` raises `NotATimeline`, and a data release before v0.8.0 (`release="v0.7.2"`) is refused
    by name. The registry drops the v0.7.x layout (`ml_only_ieee{C}.h5`, npz pools, the bare `v0.7.x` tags, their
    sha256), `registry.is_timeline_release`, `STREAM_RELEASE` and the environment variable
    `FDIA_GRAPH_STREAM_RELEASE`; `registry.dataset_file` takes the system alone, `registry.pool_spec` names the
    HDF5 pool only, and `registry.release_tuple` and `release_name` raise for a release before v0.8.0
    (`models.inputs.FIRST_RELEASE`). `fg.load(include_gaps=...)` and `FdiaGraph(include_gaps=...)` go with the gap
    rows and `schema.GAP`, and so do `ds.is_timeline` and `Capability.TIMELINE` (the `timeline` capability). The
    new error is `errors.NotATimeline`. The pinned timeline releases v0.8.0, v0.8.1 and v0.8.3 read as before.
  - **The stream entry points.** `fg.generate_stream`, `fg.load_stream`, `fg.windows`, the module
    `fdia_graph.streams` and the models `Stream`, `StreamSummary` and `StreamLike` are removed: `fg.generate`
    writes the timeline, `fg.load(name, order="time")` reads it and `ds.windows(W, stride, label)` windows it.
  - **Dict-style settings.** `generate_timeline(am_attack={...})` and `redundancy={...}` are refused: pass
    `OverloadSettings(...)` and `MeterSettings(...)`. `OverloadSettings.of` is removed.
  - **`load_profile`'s old keywords.** `load_profile(source, path=, column=)` with a string or an array is
    refused: pass `IsoFolder`, `CsvColumn`, `RawSeries` or any object with `loads()`. `ProfileSource` keeps its
    `source` field alone (`path`, `column`, `text`, `series` and `kind` are removed).
  - **`edge_reactance` and `ds.edge_*`.** `ds.edge_reactance`, `ds.edge_reactance_np`, `schema.EDGE_REACTANCE`,
    `schema.Attr.EDGE_REACTANCE_DEPRECATED`, the generator's `x_react`, the `edge_reactance` key of `ds.export()` and the
    `ds.edge_r` ... `ds.edge_shift` spellings are removed: the branch physics is `ds.branch_*` and `ds.edge_attr`.
    A newly written timeline no longer carries `graph/edge_reactance` or the attribute
    `edge_reactance_deprecated`; every other dataset of the file is unchanged.
  - **`fdia_graph.operator.response`.** The re-export module is removed: import each name from the module that
    defines it (`operator.day`, `ties`, `truegrid`, `policy`, `run`, `summary`); the package `fdia_graph.operator`
    exports its public names as before.

- **The repository has one place per kind of file.** Files moved; nothing in the package changed. Each study is a
  folder of `experiments/` (`estimation`, `localization`, `trust`, `federated`, `defense_reproduction`, `benchmarks`,
  `data_release`): `run.py` measures and writes to the results store, `report.py` draws the figures of its page. The
  scripts came from docs/<package>/run.py and `make_report.py`, from tools/reproduce_*.py (now
  `experiments/defense_reproduction/run.py`, `run_faithful.py`, `run_faithful_dqn.py`), from tools/bench.py (now
  `experiments/benchmarks/run.py`) and from tools/experiments/. The New York build tools and the MATLAB reference
  are `pipelines/newyork/` (was tools/nygrid/). The docs are sorted by kind: `docs/getting-started/`,
  `docs/guides/`, `docs/concepts/`, `docs/analysis/` (one page per study with its figures, docs/se/ now
  `docs/analysis/estimation/`, docs/wu26/ then the `defense_reproduction` folder beside it, since removed) and `docs/reference/`; the four
  pages of docs/nygrid/ are docs/concepts/new_york_grid.md, docs/getting-started/new_york_walkthrough.md,
  docs/guides/extending_new_york.md and docs/guides/troubleshooting_new_york.md (since folded into the pages every test system shares). The run scripts keep their caches
  in `experiments/<study>/cache/` (move a cache from docs/<package>/results/cache/ to keep it).
  `tools/site_index.py` was scripts/site_index.py. `tools/` holds repository tooling only.
- **Removed with the move:** the design documents of docs/plans/ (in the git history: `git show
  91c15ad:docs/plans/README.md`), the review ledger (tools/review_ledger.py and its CSV), the CodeRabbit
  configuration, and the review-bot code of `tools/pr.py` with its `request-review` command; a merge needs green CI
  and the owner's approval.
- **The layout gate maps the repository.** `tools/layout.py` holds the role of every folder at the root and under
  `docs/` (`REPO`, `DOCS`), prints them as the first section of `docs/reference/MODULES.md`, and fails a folder that
  has no role or a role whose folder is gone. The format and lint gates now cover `experiments/` and `pipelines/`.
- **File names say what a file does, and a gate keeps them and the docs current.** `tools/layout.py --check` (in the
  pre-review and CI) fails on a file name that repeats its folder's name, differs from a sibling's by a short ending,
  or carries a digit, a citation key or an author's name; on a Python file without a docstring; on a module its
  package's page does not name; and on a module or name a doc or docstring cites that does not exist.
  `docs/reference/MODULES.md`, generated from the docstrings, gives every file of the repository in one line (the
  module table of `CONCEPTS_TO_CODE.md` moves there). The pre-review's cited-paths check reads every Markdown file,
  not only the changed ones.
- **The [WU26] equation registry is removed.** `schema.WU26_EQUATIONS`, the tool that read it (the paper-to-code
  equation map), its gate and the generated equations table of `FORMULAS.md` go: nothing else read the registry, and
  the file-protocol module is no place for one paper's equation list. The `[WU26 eq. N]` citations stay in the
  docstrings.
- **Renamed modules.** `trust.defense` is
  `trust.pmu_environment`, `trust.solution1` `trust.pmu_row_reduction`, `trust.wu_dqn` `trust.pmu_dqn`, `trust.wu26`
  `trust.faithful_attack`, `trust.wu26_defense` `trust.faithful_defense`, `trust.dqn` `trust.meter_dqn` and
  `engine.pp_types` `engine.pandapower_types`. The names `fdia_graph.trust` exports are unchanged. The scripts
  `tools/wu26_*.py` are `experiments/defense_reproduction/run.py`, `experiments/defense_reproduction/run_faithful.py` and
  `experiments/defense_reproduction/run_faithful_dqn.py`; the docs' `run_<folder>.py` scripts are `run.py`; the New York tests drop their
  `nygrid_` prefix (the test folders are packages, so a name may repeat across folders). The old module paths are
  removed, not aliased: code imports from `fdia_graph.trust`, whose names are unchanged.
- **No aliases for moved private names.** The tables that kept old private names of `timeline`, `engine.records` and
  `operator.response` importable with a warning (`_ONSET_DRAWS`, `_Schedule`, `_TrueGrid`, ...) are removed with the
  `_moved` module: a name with a leading underscore was never public, so it carries no promise when it moves.

- **The decisions registry is removed.** `schema.DECISIONS`, `models.Decision`, the generated decisions page
  and the `[D9]` / `[E13]` tags go; each choice is stated in the docstring of the code that makes it.

- **The New York grid in the SDK's own packages.** `fdia_graph.nyfed` is gone: the system source is
  `generation.nygrid`, the meters `engine.sensing`, the attack families `engine.attacks`, the operator
  `fdia_graph.operator`, the residual test `se.detect`, the utility clients `federated.utilities`, the client views
  `dataset.clients` and the study helpers `results.studies`. Its tests, tools, docs and examples are under `nygrid`
  folders, and the environment variables are `NYGRID_ROOT`, `NYGRID_LOADS` and `NYGRID_UPDATE_GOLDEN`.
  The old `fdia_graph.nyfed` path is removed without a warning period. The duplicated network, accuracy-class, flow-solve, placement and
  NYISO-reader code now calls the SDK's formulas and generation functions.
- **Constructors take settings models.** Estimators, localizers, federated localizers and trust selectors take their
  `models.config` settings (`SolveConfig`, `PriorConfig`, `LearnedConfig`, `FederatedSettings`, `TrustConfig`,
  `DqnConfig`, ...) instead of flat keywords.

- **The New York grid's study scripts and design log leave the repository.** `pipelines/newyork` keeps the build tools
  (`run_pool.py`, `build_timeline.py`, `build_single_episodes.py`, `matlab/`); the study drivers, their result files
  under `results/` and the design log are removed. The library, its defaults and its outputs are unchanged.

- **An Ipopt that ends abnormally is a failed solve (New York grid).** `OperatorOpf.solve` returns None when Ipopt does
  not exit normally (killed, out of memory), so the AC operator retries from the AC point instead of ending the study
  that ran it.

- **The AC operator solves its relief only where an interface binds (New York grid).** `AcSced.step` solves the
  dispatch without interface limits (for `relief_usd`) only when an interface row binds, the only case where it can
  differ, with the commitment the dispatch used (it allowed every quick-start unit, so a congestion-free scan could
  show relief) and without moving the next scan's starting point. A scan that moves without congestion takes one
  solve, not two; `relief_usd` under the AC operator changes, the other numbers do not.

- **The AC operator holds the units' voltage schedule (New York grid).** `AcSced` now holds each unit bus's voltage
  near the AC point's (`V_WEIGHT`, the `OperatorOpf` voltage term), as the true grid's power flow holds it. Its voltages
  were free, so under least movement its model flows could part from the true grid's and the closed loop read that
  gap as a flow to relieve: one benign day of 24 moved a median of 224 MW. With it the benign twin holds on all 24
  (`ac` and `ac_cost`) and the least-cost results change by at most 1 MW of transfer on the three days rechecked;
  least-movement results from the earlier code change and are rerun.

- **An AC operator (New York grid).** `OperatorPolicy(dispatch="ac")` and `dispatch="ac_cost"` run the operator's dispatch as an AC optimal power flow each scan (`AcSced`, Pyomo and
  Ipopt, `OperatorOpf`): the loads it reads (the true loads plus what the attack changed in its New York injection
  meters; the accounting operator adds the false interchange back over New York's loads by their size), the units
  within the pool's own bounds (`PoolFrames`, `pool_frames`, `PoolBuilder.frame_limits`) and their five-minute ramps
  from its previous dispatch, hydro and external units under the pool's anchor, bus voltage and reactive limits, the
  four interfaces' AC flows within their limits, and the shed, shortage and balance prices of `DcSced`. `ac` moves the
  least generation; `ac_cost` clears at least cost and prices every bus at its AC balance dual (`Twin.price`). Its
  benign twin stays at the AC points (both solve the pool's problem). The closed loop re-solves on what the operator
  reads after its own dispatch (true AC flows less its AC model), not on shift factors, the N-1 screen is an AC power
  flow per outage (`TrueGrid(day, ac=True)`, `ac_power_flow`), shedding and the zone levers use AC sensitivities
  (`ac_shift`). The DC
  dispatches and their results are unchanged.

- **A least-cost dispatch that keeps the AC operating points (New York grid).** `DcSced(objective="cost_lf")` and
  `OperatorPolicy(dispatch="cost_lf")` (`--dispatch cost_lf` on every study tool) run the market dispatch around the
  AC operating point, as an RTO's incremental dispatch is: each unit's cost counts per MW delivered to the reference
  bus (`loss_factors`, the power flow Jacobian's delivery factors at the frame's true AC state), the balance is
  written on the delivered change, the first scan ramps from the basepoint, offline quick-start units start only
  when the committed units leave a constraint unmet, and the AC point's implied values (`DcSced.implied`: the least
  bus and unit adders that make the AC point this model's least-cost dispatch) carry what the AC dispatch priced
  and the DC model does not see. The plain least-cost dispatch moves its benign twin 1.2 to 4.1 GW off the AC
  points on at-limit days; this one keeps it there. Each run now reports every bus's price (`Twin.price`,
  `DcSced.prices`), and `area_damage`'s payments use it. The `movement` and `cost` dispatches are unchanged.

- **Shunt compensation and reactive limits as a sensitivity (New York grid).** `AcSolver(case, compensation=...)` takes a `Compensation`: switched shunt banks the dispatch sets frame by frame, a generator voltage schedule held by a penalty, an operating voltage band and unit reactive limits (`unit_q_limits`, power factor 0.85 lagging and 0.95 leading on Pmax where NYgrid left plus or minus 9999 MVAr). `PoolBuilder` applies its limits and records each bank's output (`shunt_q`), which enters the power flow and the pool's injections. Without a `Compensation` the model, the pool and every released file are unchanged.

- **Overruns counted against the benign twin (New York grid).** `summarize`'s `over_scans`, `over_run_scans`,
  `over_mwh` and `over_iface` now count only the scans on which the attacked run is further over a true interface
  limit than its benign twin, and that excess energy. The least-cost dispatch holds a binding interface at its
  shift-factor limit, so its true flow can pass the limit with no attack, which the old counts charged to the
  attack. `benign_margin_min_mw` reports the benign twin's tightest margin. Results of the least-movement dispatch,
  whose benign twin never passed a limit, are unchanged.

- **The least-cost operator from the command line (New York grid).** `experiments.add_dispatch_argument` (and
  `add_study_arguments`) declares `--dispatch cost` once for a driver that runs the operator. The default stays the
  least-movement dispatch, so every existing result is unchanged.

- **Market prices that a market could set (New York grid).** The least-cost dispatch (`DcSced(objective="cost")`)
  now prices the flow it leaves over an interface on the 2019 transmission shortage curve (350, 1,175 and 4,000
  $/MWh, `SHORTAGE`) and load it cannot match at the energy offer cap (`ENERGY_CAP`, 1,000 $/MWh), instead of the
  least-movement dispatch's slack weight of 10^6. A shadow price no longer reaches the slack weight on a congested
  day, so the load payments `area_damage` reports stay at prices a market could set. The default dispatch is
  unchanged.

- **A least-cost dispatch with prices, and the idealized operator (New York grid).** `DcSced(objective="cost")` and
  `OperatorPolicy(dispatch="cost")` dispatch as a market clears, each MW moved charged `MOVE_USD`, and report each
  interface's shadow price and the energy price (`Step.congestion`, `Step.energy`, and per scan on `Twin`). Under
  it `area_damage` prices what each zone's and utility's load pays at the moved prices (`payments_usd`). The default least-movement dispatch is unchanged: every golden value holds, and the operator
  snapshots gain only the two new arrays, empty under it.

- **Who bears an attack's damage (New York grid).** New `fdia_graph.nyfed.operator.areas`: `area_damage` splits an
  operator run, attacked less benign, by New York zone, by utility and by neighboring area. Each zone and utility
  gets the energy its units lose, the fuel saved and the sales lost at the zone's real price, the load shed, and the
  transfer cut into it and margin lost on the interfaces that feed it (`FEEDS`). Each neighbor gets the inadvertent
  interchange it supplied, priced at its NYISO proxy bus (`PROXY`, `neighbor_prices`), and the system row gives the
  attack's net cost once the borrowed energy is priced. `attack_surface` counts the meters an attack tampers per
  zone and utility, and `zone_levers` measures how far each zone's plausible false load and output could move each
  interface. `node_lbmp` reads NYISO's hourly price at any zone or proxy bus. No change to existing results.

- **The New York grid's tests run in CI (New York grid).** A new workflow, `nyfed.yml`, installs the `[ny]` extra
  and Ipopt from conda-forge, downloads NYgrid's public inputs with `fetch_inputs` (cached under the pinned commit)
  and runs `tests/nygrid` on pull requests that touch the New York grid, on main and weekly. Before it, no CI job
  installed Pyomo, so the whole folder, the examples' test included, was left out of every run. Two damage-study days,
  the single-episode day and the 2019 pool's frames of those days are checked in under `tests/nygrid/data/days`
  (6.7 MB), so the operator's tests and 33 of the 35 golden snapshots run there too; the tests that need the whole
  pool (the attack families, the builder, the timeline and the interface attack's design) still run only where it
  is built. `tests/nygrid/inputs.py` names every input in one place and says which tests it lets run, in place
  of a path block repeated in nine test files. No user-visible change.

- **Docs to read the New York grid by, and to extend it (New York grid).** docs/getting-started/new_york_walkthrough.md follows one
  attack through every stage with the call that does each step (each snippet checked on the 2019 data);
  docs/guides/extending_new_york.md names the extension points (an attack family, an alarm rule, an operator, a study), each
  with a runnable example under `examples/nygrid/` that CI runs; docs/concepts/new_york_grid.md opens with a rendered
  diagram of the stages and its principles as they now hold, in place of an ASCII diagram that predated the operator
  and the detectors.

- **The load model in one module per job (New York grid).** `scenario/loads.py` (676 lines) keeps `LoadSeries`,
  `LoadModel` and its constants; the downloads and readers of its inputs move to `load_sources.py`, the fit of its
  within-hour constants to `load_fit.py` (public as `fluctuation_fit` and `switching_fit`, for refitting them), and
  the validation measures to `load_metrics.py`. The code moved as it was.

- **Study helpers, and the research tools as thin drivers (New York grid).** New `fdia_graph.nyfed.experiments`:
  `study_days` (a study's day files), `variant` (at the limit or damaging), `load_calib` (the interchange check's
  calibration), `add_study_arguments` (the options every tool shares) and `run_parallel` (a process pool that writes
  the CSV as results come in, sorted by file at the end, so its rows no longer depend on which worker finished
  first). The study tools under `pipelines/newyork/` use them in place of their own copies, and the interchange-check tool
  takes `--data` instead of a fixed path. The tools' results are unchanged.

- **The overload attacks read plainly (New York grid).** The OPF attack's per-support model and prepared window
  are named records (`_OpfModel`, `_Prepared`) instead of tuples unpacked by position; the interface attack keeps
  its per-support state on that model rather than writing it onto the shared window; windows are built by keyword;
  and the unused `OpfFdia.reachable` is gone. Each class's docstring now says which step of the search it replaces
  (`MsFdia` the target choice and support search, `OpfFdia` the solve, `InterfaceFdia` the goal and the design).
  Attack designs are unchanged; `OpfFdia.reachable`, which nothing called, is no longer available.

- **The operator's choices as one record, and an operator interface (New York grid).** `OperatorPolicy` holds
  `mode`, `delay`, `over`, `entry`, `closed_loop` and `after`, each checked when the record is built (`OperatorMode`,
  `AlarmAnswer` and `Entry` name the allowed values); `respond(day, policy=...)` takes one, and its keywords still
  build one, so existing calls are unchanged. `Operator` is the protocol a partner's operator implements (return a
  `Response` and `summarize` works on it), and `NyisoOperator(policy)` is this package's.

- **The operator in one module per job (New York grid).** `fdia_graph.nyfed.operator.response` (730 lines) is split
  into `day` (reading an attacked day), `ties` (the interchange check), `truegrid` (the AC power flows on the true
  grid), `policy` (the operator's choices), `run` (the scan-by-scan loop) and `summary` (its consequence), the code
  moved as it was. `response` re-exports every name, so existing imports keep working. The interchange check's alarm
  is public as `interchange_alarm`.

- **Detection as building blocks (New York grid).** New `fdia_graph.nyfed.detect`: the residual test's AC weighted
  least squares (`wls`, returning a `WlsFit` that says whether it converged), the noise-free fit of a state
  (`exact_state`), the persistence rule with its calibration (`PersistenceRule`, `PersistenceRule.fit`) behind an
  `AlarmRule` protocol any other rule can implement, and other meter fleets (`with_fleet`, `fleet_bias`).
  `TrueGrid.state` and `TrueGrid.run_state` give the true state after the operator acts, with the load it shed (the
  research tools left the shed load out; no studied day sheds load, so no number changes). The six detection tools
  under `pipelines/newyork/` now drive these instead of carrying their own copies and importing each other.

- **Settings are arguments, not patches (New York grid).** `production_builder` takes `v_margin` and `gen_v_tol`,
  the attacker's voltage sensitivities (N71, N75); `v_margin` widens the limits on the attacker's own copy of the case
  (`Case.with_voltage_margin`). Before, the damage study widened the shared case's arrays in place, so a timeline
  written from that builder recorded the widened limits. `InterfaceFdia.with_overshoot` returns a variant as a copy,
  and `max_overshoot` searches on one, instead of swapping `overshoot` in and out; `build_episode(using=...)` designs
  with such a variant. `hourly_dispatch(hold_externals=False)` replaces the AC ablation's monkeypatch.

- **A public surface for the New York grid.** Every `fdia_graph.nyfed` package (`grid`, `scenario`, `sensing`,
  `attacks`, `timeline`, `operator`, and `fdia_graph.nyfed` itself) now has a docstring that says what its stage does,
  what goes in and comes out, and how it fails, and an `__all__` of its public names, loaded on first use (reading a
  dataset still never imports pandapower or Pyomo). Helpers the research tools reached into became public:
  `operator.response.TrueGrid` and `entry_share` (the old private names `_TrueGrid` and `_entry_share` are gone),
  `Attacker.absorb` and `TimelineBuilder.episode_lengths`.

- **Named errors and a troubleshooting page (New York grid).** `fdia_graph.nyfed` raises the SDK's named errors:
  `ConfigError` for a wrong argument, `DataConditionError` (or `NoAdmissibleTarget`, `NotFitted`) for data that does
  not meet a condition, instead of bare `ValueError`s and asserts. Both subclass `ValueError`, so existing handlers
  keep working. The input CSV tables are read through one reader that closes them (`fdia_graph.nyfed.tables`), and
  the research tools no longer silence warnings. docs/guides/troubleshooting_new_york.md lists what each stage reports when
  it fails and what to do.

- **A failed power flow reads as unknown, not safe (New York grid).** The operator's `Twin` gains `pf_failed [T]`,
  the scans whose AC power flow on the true grid failed, and `summarize` reports `pf_failed_scans` and
  `pf_failed_benign_scans`. Before, those scans' NaN margins counted as within the limit with no trace. `respond`
  now rejects an unknown `after` (before, a typo ran as an operator that never answers the alarm), and an unknown
  `mode`, `after` or `entry` raises `fdia_graph.errors.ConfigError` (a ValueError, as before).

- **Why a design found no attack (New York grid).** `AttackFamily.design_explained` returns the episode or the
  reason there is none (one of `fdia_graph.nyfed.attacks.episode.MISSES`: `no_target`, `infeasible`,
  `solver_error`), also kept in `last_miss`, and `TimelineBuilder.build` reports `missed` and, with the onset
  survey, `survey_missed`: the count of each reason per family. `build_episode`'s error names the reason. Before, a
  timeline with few attacks gave no way to tell an empty window from a failing solver.

- **A missing solver now says so (New York grid).** `fdia_graph.nyfed.solvers.ipopt` finds Ipopt in the environment,
  checks that it starts, and raises the new `fdia_graph.errors.SolverUnavailable` with the fix when it does not. Before,
  an Ipopt that could not start (a missing DLL, exit code 0xC0000135) made every attack design read as infeasible, so a
  timeline built with no overload attacks and no error. The attack OPF now reads only an abnormal stop on its own model
  as an infeasible support, and counts those in `solver_errors`.

- **Answering a residual alarm (New York grid).** `respond(after=...)` models the operator's response to an alarm
  (switch to the benign scans, freeze with the AGC on the falsified ties, freeze with it on the common source) (N78).

- **The residual test with the meters' noise (New York grid).** `respond(alarm_at=...)` models an operator that falls
  back at the first residual alarm (N77).

- **The closed loop (New York grid).** `respond(closed_loop=True)` has the operator read the true flows after its
  dispatch (N76).

- **Voltage checks and the overload record (New York grid).** `production_builder(gen_v_tol=...)` holds generator
  voltages near their schedule (N75), and `summarize` reports how long an interface
  stays past its true limit.

- **Entry patterns and many meter fleets (New York grid).** `respond(entry=...)` chooses where the inadvertent
  interchange enters (N72).

- **Where the inadvertent interchange lands, and the certificate's reach (New York grid).** The shortfall the trusting operator leaves is now taken up by the synchronous units outside New York in proportion to their capacity rather than by the reference bus, and every scan reports the tightest true interface margin (`margin`, summary `margin_min_mw` and `margin_lost_mw`) (N69). `production_builder(v_margin=...)` widens the attacker's voltage limits (N71). A failed power flow now leaves the margin unknown.

- **Three operators and a calibrated interchange check (New York grid).** `respond(mode=...)` runs the
  trusting operator, whose AGC reads the falsified New York-end tie meters, the accounting operator, which takes
  its load and ACE from the agreed interchange (BAL-005), and the oracle upper bound (N66; replaces
  `defended`). The neighbours' tie meters carry their own accuracy-class noise (`neighbour_reads`); `respond(calib=...)` takes the alarm's calibration (N67). The consistent attacker's area stays inside New York.
  The target interface comes first, the shed clock runs per interface, and the oracle falls back at decision time.

- **The interchange-consistent attacker and the interchange check (New York grid).** `Attacker(hold_external=True)`
  (and `production_builder(..., hold_external=True)`) keeps every bus outside New York at its true injection, so
  the attack cannot move New York's net interchange (N62). The operator gets the matching check, `tie_gap`: the interchange read at the
  New York ends of the ties less what the neighbours meter. `respond(..., defended=True)` stops acting on the
  tampered scans from the first alarm (N65). The operator now enforces all four mapped interfaces, each with its
  own 15-minute clock and shortage price. It may start the quick-start units of zones J and K, and it caps
  regulation at 275 MW, reporting the rest as an ACE excursion (N63, N64). Shed load takes its reactive demand
  with it, and a reused day keeps its meter count.

- **The damaging attack and the operator's escalation (New York grid).** `InterfaceFdia(overshoot=...)`
  takes the reported interface flow past its limit, designed under every bound as a real attack, and
  `max_overshoot` finds the largest overshoot an onset allows by bisection (N58). The operator now
  escalates as NYISO's emergency procedures do (N59): redispatch first, the flow it cannot relieve priced
  at the 2019 transmission shortage curve (`shortage_usd`, N60), and load relief after 15 minutes over the
  limit at the buses where shedding relieves the interface. `Step` carries the shed load and the cost of
  relieving the interface.

- **The grid operator under attack (`fdia_graph.nyfed.operator`).** `DcSced` is the operator's real-time
  dispatch on what it sees: a DC dispatch on shift factors around the pool basepoint, with unit limits,
  five-minute ramps from its previous dispatch and the interface limit (the benign model flow plus the
  hour's real 2019 headroom). It moves as little as it must, then at least cost. `respond(day, delay,
  over)` runs it on a day's attacked scans and on their benign twin, in closed loop on the dispatch. The
  true load never changed, so regulation picks up the generation the dispatch removed, and an AC power
  flow gives the true interface flow and the worst single 345 kV outage. `summarize` reports the attack's
  consequence, attacked less benign: dispatch moved, regulation (MW, MWh, by utility), unrelieved flow,
  the true interface and N-1 changes, production cost and the energy settled at the real zonal LBMP (N50 to N57).

## 0.21.0

- **New York grid build in the package.** `fdia_graph.nyfed.build`: `fetch_inputs(root)` downloads the
  inputs at their pinned versions (NYgrid at 47698b6, NYISO's 2019 five-minute load, the NYS DPS utility
  territories, and the bus locations now shipped as `grid/ny_bus_locations.csv`), and
  `production_builder(pool_path, ...)` returns the release `TimelineBuilder` with every decided bound.
  `tools/nyfed/build_timeline.py` and `tools/nyfed/build_single_episodes.py` run them; a single-episode
  file rebuilt this way is identical to the one built before. New guide: docs/guides/new_york.md.

- **New York grid (`fdia_graph.nyfed`, the `[ny]` extra).** The New York federated grid package, moved in
  from its own repository as written: the modified NPCC-140 of NYgrid in AC, the 2019 five-minute
  operating-point pool from a ramp-constrained AC dispatch, utility clients from the state's service
  territories, the hybrid SCADA and PMU meter plan with accuracy-class noise, the overload attack built
  on [WU26] eqs. (12)-(25) against NYISO interface limits (`InterfaceFdia`), the slow ramp, the
  timeline builder (`TimelineBuilder.build`, `build_episode` for a single-episode file) and the loader
  with per-client views. Its plan and architecture are docs/nyfed; its tests (tests/nyfed) read the
  NYgrid checkout and the built pool under `NYFED_ROOT` and skip without them. The package keeps its own
  rules until it is brought under the SDK's (`DEFERRED` in tools/readability.py). No change to the rest
  of the package.

- `tools/wu26_faithful.py --tau` runs a scenario at another per-snapshot weight; the records carry a
  `tau` tag and the `wu26.faithful` table shows one row set per weight. New `tools/wu26_dqn.py` trains
  [WU26]'s DQN over `Wu26DefenseEnv` in parallel sessions and tests each on fresh windows, stored as
  `wu26.dqn` (docs/wu26, "Solution 2").

- **[WU26] reproduced as the paper states it (`trust.wu26`, the `[wu]` extra).** A faithful attack and
  defense beside the dataset generator's:
  - `wu26_network(case, pmus)`: the paper's meter plan ([29] Sec. 2.2), every bus's SCADA reading its
    injection and the flow of each incident branch at its end, so every branch metered at both ends;
    PMUs read |V|, the angle and their incident branch currents; no SCADA voltmeter (new
    `MeterModel` value `"wu26"`, refused by the generator's `MeterSettings` and `GeneratorOptions`).
  - `solve_window`: eq. (12) read literally, the l1 of the window's summed attack ([37] p. 1898's l1)
    plus `tau` times the per-snapshot l1, as one IPOPT problem per window, with the boundary held,
    the voltage band, the goal of eqs. (24)-(25) as written ([E19]: each target at `rho` times its true
    flow by the window's end, the flow never falling between snapshots, no per-snapshot target; the
    generator's D9 ramp is unchanged), and changes below the
    paper's noise counted as untampered (new `SupportMethod` value `"wu_l1"`, refused by the
    generator's `OverloadSettings`). Settings: the validated `Wu26Attack` (`rho`, `tau`, the trust
    region `dv`/`da`, the voltage band, `max_iter`; all ours where the paper is silent, [E18]).
  - `incremental_freeze`: trust read incrementally (eqs. 29-30, [E1] revised): a trusted PMU keeps the
    offset its bus had at the snapshot before its slot.
  - `Wu26DefenseEnv`: the trusted-PMU MDP on this attack, the cost the devices tampered at any snapshot
    (Fig. 12's count, unit "devices") or eq. (33)'s net l0 over channels (unit "channels") ([E3]
    revised); `wu26_solution1`: Solution 1's row reduction on this meter plan. `TrustedPMUsDQN` now
    trains on any `trust.DefenseEnv` (`WuDefenseEnv` or `Wu26DefenseEnv`).
  - `tools/wu26_faithful.py` runs the paper's scenarios at its scale into the results store
    (`wu26.faithful`, query `wu26.faithful`); new metric `reference_hits`. Decisions E16 to E19
    added; new named error `NoSuchBranch`; `WindowAttack` in `fdia_graph.models`.

- **Replication against [WU26] recorded.** `docs/wu26/README.md` gains a section stating what the
  package implements as the paper states it, what agrees, what does not, and what the paper leaves
  unspecified, with every hypothesis tested rendered from the results store (queries `wu.hypotheses`
  and `wu.table5`). Seven scratchpad experiments are added to the store: `wu.data_recipe`,
  `wu.l1_attack`, `wu.table5_search`, `wu.attacker_estimate`, `wu.multirate`,
  `wu.independent_samples` and `wu.local_opf`. Their scripts were prototypes and are not part of the
  package. New metrics in `models.results.METRICS`: `pmu_devices`, `jaccard_scada`, `jaccard_pmu`,
  `bdd_false_alarm_pct`, `bdd_detection_pct`, `bdd_detection_matched_pct`, `state_dev_pu` and
  `state_dev_deg`.

- **Hand-written loops and formulas replaced by library calls and array operations.** Generated
  files are byte-identical to the previous release at the same seed; every replacement is pinned to
  the code it replaced (`tests/test_package_equivalence.py`).
  - Generation: the temporal-feature pass, the meter-noise draw (one vector draw per layer in the
    same order as the per-channel draws), the meter masks, the per-bus scale factors, the case
    limits (`np.add.at`) and the swing scale run as array operations instead of Python loops.
  - Metrics: new `formulas.micro_prf`, `formulas.sample_f1`, `formulas.strict_accuracy` and
    `formulas.metrics.perbus_counts_at` (every candidate threshold at once); the localizers' and the
    federated localizer's threshold searches and score blocks use them (checked against
    scikit-learn in the tests).
  - `formulas.leverage` takes the hat-matrix diagonal without forming the matrix;
    `guarded_inverse` inverts through its Cholesky factor; the SE `geo` score uses
    `scipy.stats.gmean`; the DQN replay buffer is a `collections.deque`; both DQN selectors build
    their network with `trust.dqn.q_network`; `trust.secured_copy` restores secured meters in one
    indexing step; the dataset's derived series admittance uses `formulas.series_admittance`;
    `profiles` removes shunt draws with `np.subtract.at` and the CSV export concatenates once.
  - Docstrings that promised bit-for-bit reproduction against the removed frozen suite now name
    what pins the behaviour.
  - `formulas.ac_jacobian` is built from pandapower's `dSbus_dV` and `dSbr_dV` (the [se] extra)
    instead of by hand; its signature is now `(vm, theta, Ybus, Yf, Yt, branch, lut)`, taking the
    ppc branch matrix and `Yt` in place of `from_bus` and `n_ppc`. It equals the previous kernel to
    within floating-point rounding on all eight systems, and the estimators' fitted Jacobian with it.
  - `formulas.subnetwork` and `formulas.hop_distance` use scipy's unweighted shortest path through
    the new `formulas.hop_distances` and `formulas.edge_adjacency`; the interiors, boundaries and hop
    counts equal the breadth-first search they replace.

- **The attack search is a plain candidate loop; the certifier is gone.**
  - `MinimizeMixin.min_tamper` solves the area, then the goal's buses grown one area bus at a time,
    smallest first, at most `min_budget` candidates, and keeps the cheapest; it stops early only at one
    device on one channel, the least an attack tampers. Each candidate's false state is the existing
    least-norm Gauss-Newton solve of the AC measurement equations (`local_flow_solve`,
    `local_ac_solve`). Replacing it with `scipy.optimize.least_squares` (trust-region reflective, or
    Levenberg-Marquardt with least-norm rows) was tried and kept out: on the pinned IEEE-14 searches it
    lost an attack the least-norm solve finds, found dearer ones on others and ran slower everywhere
    (experiment `search.solver_compare`, rendered in docs/wu26/README.md).
  - **Removed:** the convex-relaxation certifier (`engine.attacks.certify`, `relax_cuts`,
    `formulas.relax`, the `[certify]` extra with cvxpy and pyscipopt, `Certificate`, `BoundClaim`,
    `CertifyOptions`, `CertifiableLimits`, `CutFamily`, `CertifyVerdict`, `NoOperatingLimits`, the
    `certify.ablation` results); the search's proof bookkeeping (`MinimizerResult.proven`,
    `lower_bound`, `unsolved`, the goal-forced lower bound and the files' `episodes/min_proven`,
    `min_lower_bound` and `min_unsolved`); the support per trust slot (`TrustSchedule.per_slot`,
    `WuDefenseConfig.per_slot`, `MinimizerResult.plan`, decision E13 now "one support held"), which
    never changed an answer in any run. `MinimizerResult` is (support, devices, channels, evaluated) and
    `goal_state` returns the false state alone. [WU26] proves no minimum either.
  - The pinned IEEE-14 searches find the same supports, devices and channels, and a generated timeline's
    every data layer is byte-identical; only `episodes/min_evaluated` drops where an `At` search stops at
    one device on one channel. The search's attack vectors are float64 throughout (float32 before the
    comparison with noise; no answer changed).
  - **[WU26]'s attack area** [D18]: `FrameKnobs.area` takes the paper's area bus by bus
    (`OverloadMixin.wu26_area`: IEEE-14's every bus but the slack, p. 659; IEEE-118's Fig. 9 network,
    `WU26_ATTACK_AREA`, which nothing used before), and `SearchSettings.area_rule` / `FrameKnobs.area_rule`
    (new choice `AreaRule`) picks, with "rules", the region Section III-A's principles favour
    (`AreaMixin.rule_area`: rules 1 and 2 checked, 3 and 4 a score of ours with a size charge, new model
    `AreaScore`); "hops" (the default) is unchanged.
  - **Delta ratings** [E14]: `OverloadSettings(rating_source="delta", rating_delta=0.10)` rates each
    target line at its true flow at the window's last snapshot plus `rating_delta` per unit (ours, a
    scale matched to [WU26] Fig. 4's attack magnitudes; new `RatingSource.DELTA`,
    `OverloadMixin.window_ratings`). The generation defaults are unchanged. An Am episode's recorded
    ratings are its goal's end targets.
  - `tools/wu26_harness.py` takes `--ratings` ("+0.10" by default, or margins such as "1.1") and runs on
    the paper's area; its rerun (experiment `wu26.reproduction`) and the MINLP prototype's experiments
    (`minlp.prototype`, `minlp.region_reading`, `minlp.anchoring`, `minlp.rating_delta`,
    `minlp.cap_trust_grid`) are rendered in docs/wu26/README.md: IEEE-118 comes closest to the paper's
    band (the defense adds devices on its scale, the measurement count rises less), IEEE-14 does not
    reproduce Table II, and why. New metric `feasible`; `lower_bound` is now a solver's dual bound.

- **Generation in stages, typed settings, the paper mapped to the code.**
  - The timeline writer is a pipeline of stages with typed hand-offs in the new package
    `fdia_graph.generation` (the old module's `generate`, `as_v_first` and `_load_states` stay where they
    were): `plan` (splits, episode counts and onsets), `design` (the overload Am of each split),
    `emit` (the frames in time order) and `write` (the datasets, episodes, attributes and
    `write_temporal_layers`); `timeline.generate_timeline` orchestrates them. `timeline`'s old
    private names whose signature did not change (`_split_bounds`, `_place_split`, `_TimelineBuffers`,
    `_block_scale`, ...) still resolve, with a DeprecationWarning naming their new home, until 0.22;
    `_walk_split`, `_finish_timeline` and `_run_episode` changed shape and are gone. The engine's two branch points are strategy objects instead of base
    classes (decision B4): the support of an attack window comes from `SearchSupport` (the
    fewest-tamper search) or `RrefSupport` ([WU26]'s row reduction, formerly `RrefMixin`; its
    `area_jacobian` and `ladder` now take the generator through the strategy), chosen by
    `MinimizeMixin.support_strategy`; an episode's design comes from `RampDesigner` (At) or
    `OverloadDesigner` (Am), chosen by `EpisodeDesignMixin.designer`. `OverloadMixin` no longer
    inherits the row reduction, and `am_overload_design` takes an optional `rng` instead of `prev`
    (a flow goal has no stealth bound, [D11]).
  - **The overload Am designs run in parallel** (`workers=N`): a split's Am episodes are designed
    before it is walked, each drawing its line order from a stream keyed by the seed, the split, the
    episode and the move, so the file is byte-identical for any number of workers (experiment
    `generation.parallel_design`, `tools/experiments/design_speed.py`). A script passing `workers`
    above 1 on Windows must start under `if __name__ == "__main__":`. Because the Am draws no longer
    come from the walk's stream, a timeline with Am differs from one written by the previous version
    at the same seed; At-only timelines, the benign frames, the clean truth and the splits are
    unchanged byte for byte.
  - **An infeasible Am moves anywhere in its split**: earlier or later, uniform among every free onset
    (a ramp At, whose design reads the frame before it, still moves later). Before, a move could only
    go later, which gave up an Am episode that had no later room left.
  - **Typed settings**: `TimelineSettings` nests `RampSettings`, `SearchSettings`, `OverloadSettings`,
    `MeterSettings` and `SplitSettings`; `generate_timeline(system, states, settings=None, seed,
    out, **knobs)` and `generate(..., settings=None, **knobs)` take it and every old flat keyword over
    it (`TimelineSettings.of`), and refuse an unknown one. A dict for `am_attack` or `redundancy`
    still works and raises a `DeprecationWarning`: dicts go in 0.22; pass an `OverloadSettings` or a
    `MeterSettings`. A new `workers` knob. The file records the settings as JSON (attribute
    `settings`) with their hash (`settings_hash`); the registry records them as nested values.
    `TimelineKnobs` stays for one minor version.
  - **The paper-to-code map**: one tag form for [WU26]'s equations (`[WU26 eq. 28]`) and our decisions
    (`[D9]`, `[E13]`), the registries `schema.WU26_EQUATIONS` and `schema.DECISIONS` (the new model
    `Decision`), and `tools/equation_map.py`, which writes `docs/reference/DECISIONS.md`, the
    "[WU26] equations to code" section of `FORMULAS.md` and the module table of `CONCEPTS_TO_CODE.md`
    and, with `--check` in the pre-review and CI, fails on a stale doc, an unknown or old-form tag, or
    an equation with no implementing function or no test.
  - **A rerun with an arm skipped leaves no stale number**: `Run(..., replaces=True)` (and
    `Store.write(..., replaces=(system,))`) drops the experiment's earlier records on the run's system,
    even when the rerun wrote none there;
    the SE, localization and federated docs harnesses pass it.

- **Results as data** (`fdia_graph.results`, the results store `results/`): measured numbers are records
  with their provenance, not text. New models `Record`, `Provenance`, `RunRecords`, `Metric` and `Leaf`
  (`fdia_graph.models.results`, with the metric registry `METRICS`), each checked on construction; `Run`
  (a harness's run, written on exit with the SDK version, git commit, data release, seed and settings
  hash; a failed run writes nothing), `Store` (one tidy CSV per experiment plus `runs.csv`; `query`,
  `latest`, `one`, which raises the new named error `NoSuchResult`), renderers (`table`, `cell`,
  `pivot`, `figure_data` for a figure's CSV sidecar) and results blocks in markdown
  (`<!-- results: ... -->`), filled by `tools/results_docs.py --write` and checked by its `--check` in
  the pre-review and CI. The docs harnesses (`docs/se`, `docs/localization`, `docs/federated`,
  `docs/trust`), `tools/wu26_harness.py` and `tools/bench.py` write through `Run`; the harness JSONs
  are migrated into the store and removed, and the guides, READMEs, plan tables and this section cite
  experiments (`results/README.md`). `tools/experiments/release_stats.py` records a data release's
  statistics. `tools/wu26_harness.py` now gives IEEE-118 one configuration step per snapshot
  (10 steps, not its 11 PMUs).

- **[WU26] reproduction harness** (`tools/wu26_harness.py`, `docs/wu26/README.md`): 1-minute attack windows
  rebuilt from the 5-minute pool by AC power flow, Solution 1 and the DQN played on each window's
  trusted-PMU MDP, and Table II, the trusted orders, the decision times and Figs. 11-12 written with
  the CSV each figure is drawn from. Tables III-V and Fig. 13 are not reproduced (see the README).
- **Split first, then attack each split (generation).** `generate_timeline` cuts the timeline
  chronologically into train, val and test before any episode is placed (`split=`, default
  60/20/20, now validated by the new model `SplitSettings`), so every episode lies inside one split
  and each split's numbers are known in advance. Per split, the whole number of episodes whose
  frames come closest to `attacked_frac` of its frames is shared between the families by largest
  remainder and placed at uniform onsets inside the split, never overlapping and never cut (no
  episode is clipped any more, so every At episode is exactly `ramp_len` frames, 60 by default).
  A placement that jams is retried whole 10 times, then one episode is dropped (`RuntimeWarning`).
  An episode with no feasible design at its onset (no admissible ramp, or no line pair an overload
  reaches stealthily) moves to a later free onset of its split, up to 20 times, instead of leaving
  its frames benign; past that it is given up (`RuntimeWarning`). New file attributes record what
  was asked and got: `split_frac`, `split_sizes`, `split_attacked_frac`, and per split (rows) and
  family (columns, `placed_families`) `episodes_requested`, `episodes_built`, `episode_redraws`,
  `episodes_dropped`, `episode_shortfall`. `fallback_benign` now counts only frames inside a built
  episode whose scan could not be built. Gone: `timeline._frame_split` and its boundary-moving,
  `_draw_episodes`, `_place_episodes`. On a 72,000-frame timeline with 60-frame episodes every split
  is exactly half attacked (360, 120 and 120 episodes).
- **Every frame's meter jitter comes from its own stream** keyed by (seed, timestep)
  (`MeasurementMixin._jitter_stream`, set by the timeline walk; `jitter_keyed` attribute). The
  benign frames, the meter plan and the split column of timelines on one seed and pool that place
  different families (`("Am",)`, `("At", "Am")`, `("At",)`, the three release variants) are
  therefore byte-identical, and every emission of frame t draws the same noise as its benign twin.
  Emission outside a timeline (no key) draws from the generator's own stream as before. Generated
  data changes for the same seed.
- **Fix: an overload `Am` window is snapshots kappa+1 ... kappa+T** after the untouched reference
  kappa [WU26, eq. 25]. The goal's share at the window's k-th snapshot is k/T (was (k-1)/(T-1)), so
  the first labelled snapshot already carries 1/T of the way to the rating; before, its goal was
  the true flow and that frame, labelled `Am`, tampered nothing. A new test checks that every frame
  labelled attacked tampers at least one meter. The fewest-tamper optima of the pinned IEEE-14
  searches and the [WU26] scenario counts are unchanged; two pinned overload searches fail one
  candidate solve more or fewer (`tests/test_search_speed.py`).
- **Review process: one automated review per pull request.** The repository ruleset no longer has
  Copilot review every push or drafts (`review_on_push` and `review_draft_pull_requests` off): a
  pull request is reviewed once, when it is opened or marked ready. `tools/pr.py merge` now
  requires CI green on the head, one review from every required bot on any commit of the pull
  request, and a reply to every bot finding (inline in its thread; the findings a review lists in
  its body, a conversation comment after that review) (`status` and `wait` show `reviewed_commits` and
  `unanswered`). `tools/pr.py create` opens a draft and the new `tools/pr.py ready` marks it ready
  (which asks for the one review). CONTRIBUTING.md, the review checklist and the pull-request flow
  diagram describe the draft, ready, one-review, one-fix-push order.

- **Removed: single-snapshot generation, the v0.8.3 recipe and the frozen suite.** The generator
  makes the multi-snapshot families `At` and `Am` only, on the hybrid meters; every published data
  release (v0.7.1 to v0.8.3) still loads read-only, with its family names (`Aq`, `Ad`, `As`, `Ar`,
  `Al`), aliases, the `heldout=` protocol and per-family scores. The removed code is at commit
  `aa77d0b`; fdia-graph 0.20 is the last version that generates the removed families.
  - Generation: `families=` accepts `At` and `Am` (new `models.inputs.GeneratedFamilies`; an older
    family raises before any work). Gone: the in-place `Ad`/`As`/`Ar` (`engine/attacks/corrupt.py`),
    the load redistribution behind `Al` and the v0.8.3 `Am` (engine/attacks/redistribution.py),
    `single_shot_design`, `am_design` / `am_step`, the `Aq`/`Al` frames, and the knobs
    `attack_intensity`, `am_rate`, `am_direction`, `corrupt_len`, `replay_tau` and
    `am_attack="redistribution"` (with `AmDirection`, `LEGACY_FAMILIES`, `DEPRECATED_FOR_GENERATION`,
    `ONE_FRAME_FAMILIES`, `generation.NOISE_FLOOR` and the models `Band`, `TamperTarget`,
    `Redistribution`, `AmDesign`). `FrameKnobs` keeps the fields new generation uses, all with
    defaults. `generate_stream` keeps its entry point with new generation's families and defaults; its
    arguments after `families` are keyword-only, so a 0.20 positional call raises instead of shifting.
  - The v0.8.3 meter model: `meter_model` accepts `"hybrid"` only, now also the engine's default.
    Files written with the old meters load unchanged.
  - Dead or retired code: the full pandapower re-solve (`PhysicsMixin.resolve_states`, `solve`,
    `_pin_generation`, `MeasurementMixin.emit`, `state_from_net`, `ResolvedPool`),
    `FdiaGenerator.centrality_probs`, `nl` and `load_genP`, the N-1 line-outage screen
    (`fg.line_outage_candidates`, `FdiaGenerator(outage=)`, `LineCandidate`, `OutageRef`,
    `GridIslanded`), `fg.pyg_stream` and `fg.torch_windows` (torch_data.py; `ds.windows` and
    `fg.load(..., format="pyg")` replace them), the exporters `ds.to_numpy` / `to_torch` / `to_tf` /
    `to_pandas` (`ds.export(format=...)`), `dataset.check_split` / `check_units` / `check_order` and
    `federated.check_partition` (the models check these), `SplitFractions`, `StreamSystem`, and the
    private `SEBase._normal_matrices` and `_cond`. `load_stream` and `streams.windows` stay: they
    read the v0.7 stream files. The `ds.edge_*` aliases stay (`edge_gs` and `edge_bs` are requested
    names).
  - Tests: the frozen suite (tests/test_frozen.py, tests/frozen/, tools/freeze_reference.py, the
    strict mode of `tools/prereview.py`) is gone; the pinned search answers
    (`tests/test_search_speed.py`), [WU26]'s scenarios and the unit tests guard the generated data.
    The shared test timeline is new generation's (`At` and `Am`); old releases are tested on
    checked-in files (`tests/test_old_releases.py`, `tests/data/README.md`).
  - Docs: the finished plans (data models, field groups, one dataset, readability, reference study,
    restructure, validation), FUTURE_DATASETS.md, the v0.8.0 to v0.8.3 release build scripts and
    five unlinked figures.
  - Fixed: `load_stream` at v0.7.1 and v0.7.2 reads the stream files again. Those files carry the
    frames only, and the release's graph sidecar (`graph_ieee{N}.npz`: `edge_index`, `edge_attr`,
    `node_m`, `edge_m`) that 0.18 stopped attaching is attached again; the tests read a slice of the
    published IEEE-118 stream with its sidecar, and one timeline of each of v0.8.0, v0.8.1 and
    v0.8.3 written by the SDK that built it.
  - `Stream` carries the PMU branch currents of a hybrid-meter file (`pmu_i`, `pmu_i_benign`, the
    static mask `pmu_i_m`), so `generate_stream` and `stream_of` no longer drop them. The benchmark
    table records the fixture recipe per row and `tools/bench.py --check` compares only rows of the
    same machine and recipe.
  - Known behaviour made visible by the new test timeline, unchanged here: an overload `Am` whose
    window has no stealthy design falls back to benign for the whole episode (`fallback_benign`), and
    its first snapshot can tamper nothing, its drift-free goal there being the true flow (D9).
- [WU26]'s Solution 1, the trusted PMUs by row reduction (Sec. IV-D1, p. 657 steps 1-5;
  docs/plans/WU_DEFENSE_PLAN.md, PR C): `trust.TrustedPMUs(env).fit()` on a `WuDefenseEnv` trusts one
  PMU per configuration step, a PMU reading the sparsest attack the column-exchange reduction finds on
  the attack area's Jacobian at the step's snapshot (the trusted buses left out), and records the order,
  the rewards, the costs and the reductions' own time. Ours: the chase is restricted to attacks that
  move a target line and that a PMU on offer reads, the PMU reading most of the attack's channels is
  trusted, and the first PMU on offer when no open attack touches one. `RrefMixin.area_jacobian` now also
  names the device of every row (`channel_devices`). On [WU26]'s IEEE-14 scenarios the order differs from
  the paper's Fig. 6 while the final trusted set and the defended cost match (experiment
  `wu26.reproduction`, docs/wu26/README.md).
- [WU26]'s Solution 2, the trusted PMUs by a deep Q-network trained with Algorithm 1 (Sec. IV-D2;
  docs/plans/WU_DEFENSE_PLAN.md, PR D): `trust.TrustedPMUsDQN(envs, WuDqnConfig()).fit()` over one or more
  `WuDefenseEnv` windows, `order(env)` the trained policy's greedy sequence (eq. 38). The loop is
  Algorithm 1's: epsilon-greedy actions (eq. 34), line 8's break without a stored transition, one
  minibatch update per iteration by the squared loss to r + gamma max Q_target (eqs. 35-37), the target
  network copied every 20 iterations; new validated `WuDqnConfig` with the stated hyperparameters as
  defaults (gamma 0.9, Adam at 0.005, buffer 2,500, minibatch 25, target every 20, epsilon
  e^(-0.002 ep), 250 episodes). Ours: the network (two hidden layers of 128 ReLU units, torch), masking
  PMUs already trusted, terminal transitions bootstrapping nothing, and the state scaled by its largest
  magnitude over the training environments' first states. New input model `SameDefense` (the training
  environments offer as many PMUs and take as many steps).

- [WU26]'s trusted-PMU configuration as a Markov decision process (docs/plans/WU_DEFENSE_PLAN.md, PR B):
  `trust.WuDefenseEnv(g, states, goal, k, WuDefenseConfig(pmus, slots))` on one overload window, with
  `reset`, `step` and `valid`. An action trusts one more PMU at the next configuration step (Sec. IV-D2,
  "one PMU per step"); the reward is the rise in the fewest-tamper search's cost under the schedule so
  far (eq. 33 per step, eqs. 28-32), in measurements by default or devices (`WuDefenseConfig.unit`,
  new choice `CostUnit`, the plan's E3); the state is what Fig. 1 lists, the node and flow readings at
  the step's snapshot, the target lines' load rate and the trusted mask. Every schedule's answer is
  cached (`solves` counts the searches), since a training run revisits the same schedules. Ours, where
  the paper is silent: the readings are those of the attack under the schedule so far without noise, an
  attack the schedule makes infeasible costs every attackable channel or device, and Algorithm 1's line
  8 (a defended cost below the undefended one ends the episode without a reward) is kept and counted in
  `breaks`. The search is the cost oracle rather than the row reduction (`support_method="rref"`), which
  matched Table II and Fig. 12 worse (the plan's section 3). The search minimizes the unit the reward
  counts: new `FrameKnobs.objective` ("devices", the default and what generation uses, or "channels",
  which puts the tampered measurements first as eqs. 28 and 33 count them; the device bound then no
  longer prunes). The environment refuses a step outside the window (`WindowSlots`) and an action that
  is not a PMU still on offer (`ChosenAction`); its state reads only metered channels (the load rate
  from the full flows), and the infeasible cost counts the plan's metered channels or the devices that
  hold one. New validated models `WuDefenseConfig`, `WindowSlots` and `ChosenAction`.

- [WU26]'s own attack construction as the overload attack's support method (`OverloadSettings.support_method`, "search" by default or "rref"; new choice `SupportMethod`, `FrameKnobs.support_method`, file attribute `support_method`). The paper states it twice: the attacker "can utilize the transpose of the measurement matrix h, applying multiple elementary row transformations and column exchanges to find the optimal attack vector solution [25]" (p. 655, [YAN17]), and Solution 1's steps 1-5 (p. 657). New `formulas.trust.sparsest_rows` (the reduction, the column exchanges with their tracking order, the rows mapped back) and `engine.attacks.rref.RrefMixin.rref_support`: at each snapshot the Jacobian of the attack area's attackable channels (scaled by [WU26]'s noise, a trusted PMU's bus left out, eqs. 27 and 29) with respect to |V| and angle, for each target line the sparsest row whose state change moves its flow, the window's support the union over the snapshots (eq. 28), the magnitudes from the AC local flow solve so the goal, the limits and the D16 bounds hold. Ours: the chase starts from the transpose's own rows (the single-variable attacks), it chases the sparsest goal-moving row rather than the sparsest outright, and a support that cannot reach the goal grows by the next sparsest rows. On [WU26]'s scenarios the search matches Table II better: at k = 1.1 the trust schedule raises the search's devices 20% and 29% (the paper 24% to 35%) and the reduction's -14% and +13%; on IEEE-118 the search's devices rise 20% (the paper 16.4% mean) and the reduction's fall. The reduction is 0.5 to 0.8 s per 20-snapshot IEEE-14 window against the exhaustive search's 16 to 33 s, and 2.4 to 3.2 s per 10-snapshot IEEE-118 window against 2.8 to 9.6 s at budget 256. [WU26]'s reference now lists the paper's authors and pages (S. Wu, Q. Wang, J. Hu, Y. Ye, Y. Tang; pp. 651-665).

- [WU26]'s trusted-PMU defense as a constraint on the fewest-tamper search (docs/plans/WU_DEFENSE_PLAN.md,
  PR A): `min_tamper(..., trust=TrustSchedule(buses, slots))`. A PMU trusted at slot s keeps its bus's
  |V| and angle true at every snapshot t >= s (eqs. 26-32: the deviation is zero on its two secure rows,
  and trust accumulates, eqs. 30-31); its branch currents stay untrusted, as eq. (27) leaves them in the
  nonsecure set. New model `TrustSchedule` (validated: one slot per bus, each PMU once, non-negative
  indices). Without a schedule the search and its answers are unchanged. Measured on [WU26]'s IEEE-14
  scenarios with the PMUs trusted at the paper's snapshots 2, 4, 6 and 8: experiment `wu26.prototype`,
  rendered in docs/plans/WU_DEFENSE_PLAN.md. The paper's trust order and the swapped one give the same
  answer.

- Development only, no user-visible change: the test suite runs in parallel. pytest-xdist joins the
  `test` and `dev` extras, CI runs the three suites with `-n auto`, and `tools/prereview.py` does so
  (at most 8 workers) when the interpreter has it. The test extra's pip cache is kept between CI runs,
  and a pull request that changes only Markdown or `docs/` (not the data dictionary, which a test
  checks) skips the suites' install and run steps, the jobs still reporting success. The malformed-input
  tests' ids no longer carry memory addresses, which differed between workers.

- The fewest-tamper search (`MinimizeMixin.min_tamper`) holds BLAS to one thread while it runs and
  restores the caller's setting after, through threadpoolctl (added to the `generate` and `all`
  extras; without it the search runs on whatever BLAS is set to). Its products are too small for
  threads to pay: on a many-core machine starting them cost more than the arithmetic. The limit is process-wide:
  while any search runs, other BLAS work in the process is on one thread too, and overlapping searches
  share one limit, so the caller's setting comes back when the last of them leaves. No knob is added. Measured on the same fixed episodes as the entry below, before and after side by side: median
  seconds per episode in docs/reference/BENCHMARKS.md (experiment `search.speed`, `pr=164`). With the
  default threads the search now runs as fast as it did with the whole process on one thread. Every
  episode, in all four runs, returns the same target lines, support, device and channel counts, proof,
  candidates solved and failed solves, and the pinned searches of tests/test_search_speed.py are
  unchanged.

- The fewest-tamper search is faster and finds the same supports, bit for bit. The flow solve keeps
  the full Ybus products of the released solve, since a search's answer can turn on the last bit of a
  solve near its convergence limit (a product over the region's rows alone changed the answer of 4 of
  30 pinned IEEE-14 searches), and does less around them.
  - Each voltage vector tried costs one product Ybus V, which the residual and, for an accepted
    vector, the next iteration's Jacobian share (`local_flow_solve`, `local_ac_solve`); the solve's
    slices of Ybus and Yf are cut once per support and reused by the active set's re-solves
    (`local_flow_solve(blocks=...)`, `formulas.network._flow_blocks`).
  - The bus adjacency is cached (`AreaMixin._boundary`) rather than rebuilt from the edge list for
    every candidate, and the generator and free-injection buses are built once.
  - Each snapshot's true flows and currents are computed once per window, and the attack vector is
    evaluated on the support's branches only; every other branch reads the same on both sides.
  - The tamper count is a union of boolean masks instead of Python sets.

  Measured on a fixed set of episodes, before and after side by side: median seconds per episode in
  docs/reference/BENCHMARKS.md (experiment `search.speed`, `pr=163`). Every episode, in both thread
  settings, returns the same target lines, support, device and channel counts, proof, candidates
  solved and failed solves as before, and the flow solves on IEEE-14 and 118 (with and without the
  load cap) return the same false states bit for bit.
  Exhaustive searches on fixed IEEE-14 windows, with and without the load cap and with two-line goals,
  are pinned to their earlier answers (tests/test_search_speed.py).

- `SEBase.estimate` builds the measurement vectors, the `pmu_pseudo` slots included, one `chunk` at a
  time, and reads the pseudo values alone (new `formulas.estimation.pmu_pseudo_phasors`); the
  propagated covariances are computed only for the measured calibration's scans in `fit`. On a
  72k-frame IEEE-300 timeline this drops several whole-split float64 tensors (the [n, N, 2, 2]
  covariance alone about 691 MB). The estimates are unchanged, bit for bit.
- An overload episode drives two lines at once by default, as [WU26]'s case studies do
  (docs/plans/WU_MSFDIA_PLAN.md, D17): `OverloadSettings.n_lines` (2, or 1), passed as
  `am_attack={"n_lines": 1}`; the pair is drawn inside one attack area so one held support reaches
  both, at most `AM_LINE_TRIES` pairs are tried and an episode with none stays benign. The file
  records `n_lines`, and `episodes/am_*` holds one row per target line with the new
  `am_target_mva` (the goal at the window's end); `AmOverloadDesign.rating` became `ratings`, and
  `overload_step` returns the flow reached on every target line. New tables `WU26_SCENARIOS`,
  `WU26_PMUS` (IEEE-118's from the paper's Fig. 9) and `WU26_ATTACK_AREA`, with
  `OverloadMixin.wu26_branch` and `wu26_buses`; new tests `tests/test_wu_scenarios.py`. Measured: experiment `generation.am_overload`, rendered in docs/guides/generation.md.

- The overload attack bounds the edge of its support (docs/plans/WU_MSFDIA_PLAN.md, D16): the
  generator limits (22)-(23) apply to every generator whose reported output the attack changes, the
  support's edge included (pinned in the solve, P and Q apart, and checked by `within_limits` over the
  support and its edge), and every load bus it moves, in the support or on its edge, shows an active
  change of at most `load_cap` times its true load [YUA11] (new `OverloadSettings.load_cap`, 0.5 by
  default, in (0, 1], passed as `am_attack={"load_cap": ...}` and recorded as the file attribute
  `load_cap`). `local_flow_solve` holds injections at edge buses as well
  (`formulas.network._injection_jacobian`); `FalseStateMixin.touched_buses`; `FrameKnobs.load_cap`.

- The overload attack's line ratings default to the operating pool (docs/plans/WU_MSFDIA_PLAN.md,
  D15): S_max of each branch is 1.25 times its peak true apparent flow over the pool the timeline
  walks, so `Am` runs on every system of the ladder. The PGLib-OPF ratings stay available,
  `generate_timeline(am_attack={"rating_source": "pglib"})` (IEEE-14, 118 and 300; `NoLineRatings`
  elsewhere). New model `OverloadSettings` (`rating_source` "pool" or "pglib", `rating_margin` > 1),
  passed as a dict through `am_attack`; new choice `RatingSource`;
  `OverloadMixin.use_line_ratings`, `MeasurementMixin.all_flows_from_states`; new file attributes
  `rating_source` and `rating_margin` when the overload attack runs. Measured: experiment `generation.am_overload`, rendered in docs/guides/generation.md.

- Review fixes to the hybrid meters and the overload attack: a generator pinned at one limit keeps
  its other component free (the active set pins P and Q apart); an overload frame's magnitude at a
  generator bus is its apparent change against the true generator output (at a load bus, the
  active change against the true load; `OverloadMixin.pretended_change`); a PMU current channel's
  systematic bias scales with its end phasor's magnitude, the scale of `current_sigma`
  (`formulas.noise.biased_current`, `current_magnitude`), which changes hybrid emission only.

- `Am` follows [WU26]'s model, eqs. (12)-(25), exactly (docs/plans/WU_MSFDIA_PLAN.md, D11): no
  bound on how far the attack moves between snapshots, which the paper does not have; its noise
  (0.03 pu SCADA, 0.01 pu PMU, currents included) only decides which changes the tamper count
  ignores. The bound was what made the overload attack infeasible on IEEE-14. `At` keeps its
  rated-accuracy bound, and `stealth_scale` is now `At`'s alone. The rest of the `Am` path follows the paper too (D14, below).
- `Am`'s generators are free (D14): [WU26, eqs. 13-14] let every injection be tampered and (22)-(23)
  bound only the generator output, so a generator bus in the support has free P and Q injection
  inside its limits, like an attackable load (`FalseStateMixin.free_injection_buses`,
  `generator_buses`); a support needs an end bus of each goal line and one free injection. The flow
  solve enforces the limits (21)-(23) by an active set (a generator pinned at the limit it would
  pass, a bus voltage held at its limit, the least-norm solve repeated;
  `formulas.network.local_flow_solve(vm_fixed=...)`). `FlowGoal` takes one or more lines
  (`FlowGoal.more`, `lines`, `targets_at`; `overload_goal(window, line, *more)`,
  `solve_flow_local(line=[...])`): one held support, each line to its own rating. Generated episodes
  drive two lines by default since D17 (`OverloadSettings.n_lines`, one line when set to 1). A zero-injection bus stays held at zero, a rule of ours. The labels of an
  overload frame are the free-injection buses of the support. On [WU26]'s IEEE-14 metering and two-line scenarios with the PGLib-OPF ratings (the paper
  does not state its limits) the attack is feasible and tampers more devices than the paper's, with
  changes of several pu, since the ratings are 3 to 19 times the true flows (the plan's D14 table).
- `meter_model` is its own knob (D12), a field of the new meter-plan model `MeterSettings` that
  `generate_timeline(redundancy=...)` takes beside the coverage fractions (no new parameter on the
  entry point): `"hybrid"` by default, and the v0.8.3 recipe (the frozen test
  timeline, the v0.8.3 build script, `generate_stream`) pins `"v083"`.
- New timeline field `prev_pmu_i` (D13): the previous emitted frame's PMU branch currents, offered
  by `export` on a hybrid-meter file beside `prev_node_x` and `prev_edge_x`. `JacobianWeighting`
  takes `pmu_pseudo=True` and reads it.

- The hybrid meter model (docs/plans/WU_MSFDIA_PLAN.md, decision D10), what new generation's meters
  read: a SCADA voltmeter reads `|V|` only and the angle is a PMU channel, and every PMU reads the
  current phasor of each in-service branch at its bus [WU26, eqs. 19-20], the real and imaginary
  part at each end, per unit on the base current (`formulas.network.branch_currents`). Their noise
  is IEEE C37.118.1's 1% total vector error taken as three standard deviations of the current's
  magnitude plus a 1e-5 pu floor (`formulas.noise.current_sigma`, `PMU_CURRENT_CLASS`), one rule
  the emitter (with the jitter part) and the fewest-tamper search (with the whole class) share
  [C37118]. `"hybrid"` is new generation's default; the v0.8.3 recipe passes
  `redundancy={"meter_model": "v083"}` (D12); `FdiaGenerator`'s own default stays `"v083"`. A hybrid file gains `data/pmu_i`, `data/pmu_i_m`,
  `benign/pmu_i_benign` and `attack/pmu_i_tamper` and the attributes `meter_model`,
  `current_feat` and `current_units`; a v0.8.3-meter file has none of them and loads unchanged.
  Records, batches and `export` carry `pmu_i`, `pmu_i_m` and `pmu_i_benign` when the file has them,
  in per unit on every view (new capability `pmu_currents`). The stealthy families write
  `h(x^a) - h(x)` on the current channels too; the fewest-tamper search counts a tampered current
  in the PMU of the bus at its end, and bounds its step by the PMU accuracy class for `At` (D7); for `Am`
  it counts a current beyond 0.01 pu (D8, D11, `formulas.noise.paper_current_sigma`). Under either meter model the
  search charges an angle channel only at a PMU bus. New option `pmu_pseudo` (off by default) on `WLS`,
  `AdaptiveWeighting`, `ResidualRemoval`, `SubspacePrior`, `GatedPrior` and `JacobianWeighting`: the pseudo voltage
  phasors of [WU26, eq. (3)] at the far end of every PMU-metered branch fill the `|V|` and angle
  slots no meter reads, weighted in the measured calibration by their first-order propagated
  variance (`formulas.estimation.pmu_pseudo_links`, `pmu_pseudo_voltages`). New models
  `MeterModel`, `CurrentColumns`, `CurrentIndex`, `CURRENT`, `PmuCurrentFields`, `PseudoLinks` and
  `PseudoVoltages`; `AttackVector` is a named tuple with an optional `current`. The v0.8.3 recipe and the strict frozen suite are bit-exact.
- The documentation states what the stealthy families take from [WU26]: its constraints (13)-(18)
  and (21)-(23) (the SCADA measurements, the PMU voltage magnitudes and angles, and the operating
  limits; and, in new generation, whose PMUs read branch currents, the current phasors (19)-(20) as well). New generation also solves
  its objective, eq. (12), for `At` and `Am`, and `Am` is its overload attack (below); the released
  files' `Aq`, `Al` and `Am` drew their targets at random and drove no line to its limit.
- New generation makes the multi-snapshot families of [WU26], `At` and `Am`, both on the
  fewest-tamper search (docs/plans/WU_MSFDIA_PLAN.md, decisions D2 and D6): `generate_timeline` and
  `fg.generate` default to `families=("At", "Am")`, `min_tamper=True` and `am_attack="overload"`.
  `Am` is the overload attack of [WU26, eqs. 24-25]: a rated branch whose flow is metered and whose
  true flow stays below its rating over the window is driven, snapshot by snapshot, until the flow
  the tampered measurements carry before noise reaches the rating, on the support that tampers the
  fewest devices. An episode tries at most eight eligible branches in a random order (each try is a
  full search), a bounded heuristic: when none of them admits an attack the episode stays benign and
  is counted. Each snapshot's false state frees the attackable loads of the support and holds
  every other bus's injection, the least-norm voltage change that meets the flow
  (`formulas.network.local_flow_solve`, `FalseStateMixin.solve_flow_local`). The ratings are the
  `rate_a` of the PGLib-OPF v23.07 versions of IEEE-14, 118 and 300 (CC BY 4.0), stored in
  `fdia_graph.ratings` and matched to the branches by their end buses
  (`formulas.attacks.branch_ratings`, `OverloadMixin.line_ratings`); pandapower rates every branch
  9,900 MVA, the "no limit" placeholder. Other cases refuse `Am` with the new named error
  `NoLineRatings`. Each overload episode writes its branch, rating, reached noiseless flow and
  emitted flow under `episodes/` (`am_*`); new models `FlowGoal` and `AmOverloadDesign`; new knob
  `stealth_scale` (a multiplier on the stealth bound, 1 by default; `At`'s alone since D11). `Am`'s
  tamper count (and, until D11, its stealth bound) use [WU26]'s own case-study noise, 0.03 pu on SCADA channels and 0.01 pu on PMU channels in
  the stored units (`formulas.noise.WU26_NOISE`, `paper_sigma`; the plan's D8), while `At` keeps the
  meters' rated accuracy (D7). Snapshot t's goal is the true flow plus a linear share of what
  separates the window's last true flow from the rating (D9), so the attack rides on the load's own
  drift instead of cancelling it and reaches the rating at the last snapshot. Generating `Aq`, `Ad`,
  `As`, `Ar` or `Al` warns (`DeprecationWarning`) and is refused from 0.22; released files holding
  them load unchanged. `timeline.LEGACY_FAMILIES` with `am_attack="redistribution"` and
  `min_tamper=False` reproduces data release v0.8.3 (its build script and the frozen test timeline
  use it; the strict frozen suite is bit-exact). Under the meters' rated accuracy (D7) no overload
  window was stealthy, since moving one
  line's flow moves the injections and flows around its ends by several times that change, beyond
  the rated accuracy of the small loads there. The reported noiseless flow reaches the rating exactly.
- The fewest-tamper search of [WU26, eq. 12], behind `generate_timeline(min_tamper=True)` (on for
  new generation; `min_tamper=False` walks the released files' recipe). Each At episode is held on the support (the buses
  its false state moves) that tampers the fewest devices over the episode, a device being one SCADA
  terminal or one PMU per bus and a change under a meter's accuracy-class sigma not counted. A
  support is held for the whole episode and must solve at every snapshot inside the operating
  limits and move no metered channel by more than its accuracy-class sigma from one snapshot to the
  next, the first snapshot measured from the frame before the episode (its attack vector when that
  frame was attacked, since episodes may be adjacent). The search solves candidate supports in the
  attacker's area smallest first, each closed over the zero-injection buses of its boundary,
  starting from the region the episode was accepted on; a local solve that does not converge proves
  nothing, so those candidates are counted (`min_unsolved`) and the optimum over held supports is
  claimed only among converged solves, when it exhausts the area or reaches the devices the goal
  forces above their accuracy-class sigma, settles after `min_budget` candidates otherwise, and
  treats a support moving no device beyond its sigma as no attack, and records `min_devices = -1`
  when no held support meets every constraint (the episode then runs on its region). Each search is
  written under `episodes/` (`min_support_*`, `min_devices`, `min_unsolved`, `min_channels`,
  `min_proven`, `min_evaluated`, `min_lower_bound`). New formulas `formulas.noise.jitter_sigma`
  (the emitter's per-scan noise rule, emission only) and `accuracy_sigma` (a meter's rated accuracy
  at a reading, the measured calibration's rule through `accuracy_class_sigma`, what the search
  counts against), `formulas.attacks.tampered_channels` and `tampered_devices`; new models
  `LoadGoal` and `MinimizerResult`; `MeasurementMixin.meter_masks` gives the masks without a random
  draw.
- Every attack is built in one place, `engine.attacks`, a package whose modules compose the
  generator's `AttackMixin`: `area` (the attacker's area), `false_state` (the local solve, the
  operating limits, the attack vector and the tamper set), `redistribution` (the `Al` and `Am`
  redistribution), `stealthy` (the `Aq`/`At`/`Al`/`Am` frames), `episodes` (what an episode
  attacks, drawn at onset, typed as `RampDesign` and `AmDesign`) and `corrupt` (`Ad`/`As`/`Ar` and
  the replay buffer). This code was spread over `engine/records.py`, `engine/physics.py`,
  `engine/attacks.py` and `timeline.py`; `records` now emits a benign scan and hands an attacked
  family to the mixin, and `timeline` decides when and where each episode runs. No user-visible
  change: the generated files are bit for bit the same. The public names keep their old import
  paths; a moved private name (such as `engine.records._stealthy_frame` or `timeline._AmShape`)
  still imports from its old module with a `DeprecationWarning` naming its new home, until 0.22.
- Every annotation carries its real type. The engine's pandapower network and its internal case are
  `engine.pp_types.PandapowerNet` and `PpcTables`; torch modules and tensors, HDF5 files and
  datasets, sparse admittance matrices and the stream mappings are named; the column views are
  `NodeColumns`/`EdgeColumns`/`BranchColumns` of arrays and the index constants `NODE`/`EDGE`/`BRANCH`
  are `NodeIndex`/`EdgeIndex`/`BranchIndex` of ints (the same fields, still tuples); a stream's
  episodes are `models.data.EpisodeRow` and its summary `StreamSummary`. `Any` stays only where the
  value can be anything (the validation engine's raw input, keyword pass-throughs, mixed-value
  staging dicts), each place listed in `ANY_ALLOWED` in `tools/readability.py` with its reason, and
  the readability gate refuses an unlisted one. No behaviour changes.

- `load_profile` takes a load source that reads itself: `IsoFolder(iso, directory)` (an operator's
  CSV export; `models.config.IsoExport` checks the operator and refuses one with no known export
  format, ERCOT, when the folder is described), `CsvColumn(path, column)` or `RawSeries(values)`, or
  any object with a `loads()` method (`profiles.LoadSource`). The old form, `load_profile(source,
  path=, column=)` with a string or an array, still works for one minor version and raises a
  `DeprecationWarning`.
- `fetch_profile` reads from the operator's feed in `profiles._FEEDS`: `NyisoArchive` (built in, no
  dependency) for NYISO, `GridstatusFeed` for CAISO and ERCOT, which imports `gridstatus` itself and
  names the install when it is missing; `models.config.ProfileFetch` checks the operator and
  `resample_min`. The downloaded data and the returned vector are unchanged.
- Every input is checked in one place (the validation plan, in the git history at `aa77d0b`). Each consumer's settings
  are one model in `fdia_graph.models.config` (`LoadOptions`, `WindowSpec`, `HuberConfig`,
  `PriorConfig`, `LearnedConfig`, `FederatedSettings`, `TimelineKnobs` and the rest), whose fields
  declare their rules (`Annotated[float, Positive()]`, `OneOf(Units)`); one engine,
  `models.validation`, checks them when the model is built. The public signatures are unchanged:
  the keyword arguments build the model. Every fixed set of values is a `Choice` enum in
  `models.choices`, still importable where it was used. A dataset view is checked through one table,
  `ds.require(...)` over `dataset.base.CAPABILITIES`. A condition only the data reveals raises a
  named error from `fdia_graph.errors` (`NoBenignRecords`, `NoAttackedRecords`, `GridIslanded`, ...).
  The formulas and the parsers keep their signatures and build their input model
  (`models.inputs`: `ClientUpdates`, `ClientGraph`, `LabelGrids`, `FamilySelection`, `SystemRef`,
  `ReleaseName`, `OutageRef`, `StatePool`, ...), so no function body checks its arguments.
  Every error is still a `ValueError`; the messages now have one shape, "<Model>.<field> <rule>, got
  <value>". The loader's record `format` ("torch" or "pyg") is now checked too: a typo such as
  "pygg" used to fall through to the torch records silently. `check_split`, `check_units` and
  `check_order` still work for one minor version and raise a `DeprecationWarning`. The private
  helpers the models replace are gone (`_check_knobs`, `_check_settings`, `_check_frac`,
  `_check_graph`, `_check_blocks`, `_index_array`, `_is_int`, `_FORMATS`, `_ISOS`, `_FIELD_FLAG`);
  `dataset._FAMILY_ALIAS` and `profiles.system_id` still resolve.
  Integer settings (`npass`, `iters`, `n_calib`, `layers`, `hidden`, `k`, a window's `T`, the
  federated `partition_clients` and `epochs`, the episode lengths, `max_test`) now refuse a float
  at construction; before, a value such as `npass=1.5` passed and failed later with a raw
  `TypeError`. A malformed value of any setting (a string where a number belongs) is a
  `ConfigError` with the same message as an out-of-range one.
  The deprecated `pyg_stream` and `torch_windows` refuse an unknown `layer` with a `ConfigError`
  before the stream loads; before, it was a raw `KeyError` after loading. `fetch_profile`'s
  `resample_min` must be a whole number of minutes: `1.5` used to be truncated to a 1-minute
  cadence without a word, and is now a `ConfigError`. `partition_from_assignment` checks the
  assignment and the branch list before sizing the grid from them, so a scalar assignment or a
  non-integer `edge_index` is a `ConfigError`, not a raw `TypeError`, and so is a negative bus
  count.
  A profile date is a 'YYYY-MM-DD' string, a date or a datetime on every Python version (from
  3.11 '20240131' and the integer 20240131 used to parse too). `load_profile` refuses a source
  that is not a `LoadSource`, an operator name, a CSV path or a 1-d series of loads (a scalar
  used to pass as a one-point profile); `generate`'s `states` must be an array or a path; and
  `export(fields=...)` takes a list or tuple of names, so a lone string, a mapping or a set is a
  `ConfigError`. `RawSeries` takes a non-empty 1-d series: a scalar used to pass as a one-point
  profile and a 2-d array was flattened without a word.
  `CsvColumn` checks its path and column when built (a missing column used to surface as a
  pandas `KeyError` on read), and a dataset name that is neither a name nor a bus count, such
  as `fg.load(["ieee14"])`, is a `ConfigError` instead of a raw `TypeError`.
  A profile date span that ends before it starts, and an `IsoFolder` directory that is not a
  path, are refused when given; `generate(states=pathlib.Path(...))` now reads the file instead
  of failing with an `AttributeError`.
  Arguments reach their models unconverted, so a malformed one is a `ConfigError`, not the raw
  error of a conversion done first: `families=3`, `pool_moments(None)`, `fedavg(None, ...)`,
  `block_diagonal_basis(3, d)`, `score(ds, scores="bad")`, `score(ds, xhat="bad")` and a
  non-numeric in-memory state pool. A test refuses `Model(tuple(arg))`-style calls in the package.
  A Huber `c`, a removal threshold, `am_rate` and the learned localizer's `lr` and `pos_weight`
  must be finite as well as positive, and a solver `tol` finite and non-negative. The learned
  localizer checks its whole training setup (`dropout`, `lr`, `weight_decay`, `batch_size`,
  `epochs`, `pos_weight`, `seed`) on `LearnedConfig` when it is built. `generate_timeline` checks
  `families` on `FamilySelection` before it builds the case, and the admissible-target check is a
  model, `models.inputs.AdmissibleTargets`, which raises `NoAdmissibleTarget` as before (a model
  names its failure type in `Validated.error`; the named data errors now live in `models.errors`
  and `fdia_graph.errors` re-exports them). `TimelineKnobs.am_len` defaults to None, resolved to
  `ramp_len` by the model. `check_partition` is deprecated in favour of building
  `PartitionOnGrid`, and the internal `check_window_args` and `check_targets` are gone
  (`WindowSpec`, `AdmissibleTargets`). A value no model can even read (a number past the float
  range, a mapping where a sequence belongs) is a `ConfigError`, never a raw `OverflowError`,
  `KeyError` or `AttributeError`; `tests/test_malformed_inputs.py` sweeps every config and input
  model with values of the wrong kind and the public entry points with one bad argument each.
  Loose input is read by a parser model, so no function outside `models/` dispatches on a type:
  `ProfileSource` (what `load_profile` was handed), `DateSpan` (`fetch_profile`'s dates; a date it
  cannot read is a `ConfigError` before any download), `StateSource` (an in-memory pool or a path)
  and `DatasetName` (the registry's lookup key). `tools/readability.py` now refuses
  `isinstance(...)` outside `fdia_graph.models` as it refuses a bare `raise ValueError`.
- The estimator solve path no longer takes the slack angle. `_solve(z, w)`, `_w_solve(z, w)`,
  `_nres(x, z)` and the rest solve at the fitted reference (`ref_angles`, through the new
  `_h_ref(x)`); a custom `SEBase` subclass that overrides `_solve` drops its `thsl` argument.
  `JacobianFeatures.previous_estimate` returns the state alone. The numbers are unchanged.
- Detectors calibrate from measurements only. `SEBase.fit(ds, calibrate="measured")` sets every
  meter's sigma from its accuracy class (`formulas.estimation.accuracy_class_sigma`, the classes now
  one definition, `engine.base.ACCURACY_CLASS`, shared with the noise model), takes the angle
  reference from the case and the linearization point from the mean estimate of the benign training
  scans, and reads no clean layer. The Jacobian feature sets, `ResidualLocalizer` and
  `TrustSelector` fit their estimator this way; the state estimators keep `calibrate="truth"` as the
  benchmark default. Residual-based sigmas were tried first and rejected: a meter's constant bias
  is absorbed into the state, so they shrink on the biased meters and collapse the fit. The frozen
  localization reference moves for the residual arm.
## 0.20.0

- Data release v0.8.3 is the default (`fg.load("ieee118")`; v0.8.1 stays readable with
  `release="v0.8.1"`). The eight timelines are regenerated from the unchanged pools (byte-identical)
  with seed 123 and the v0.8.1 knobs, built by `examples/_build_timelines_v083.py`: Aq and Al
  episodes are one frame, and the stored `temporal_delta` and `swing` are functions of the observed
  frames alone. Each file holds about 24,600 episodes instead of about 16,100, keeps 72,000 frames,
  exactly half attacked, no frame falling back to benign, and every family present.
  `data-v0.8.2`, an intermediate build with one-frame Aq only, is superseded.
- Engine: parameter groups that travelled through every attack function are data models
  (`models.frames`): `Band(floor, cap)`, the plausibility band of an in-place tamper
  (`FrameKnobs.band`); `TamperTarget(buses, branches)`, where an in-place attack writes; and
  `AttackDesign(targets, mult, interior)`, a stealthy attack's design on one scan.
  `FdiaGenerator.corrupt(scan, buses, kind, replay, band)` tampers a `Scan` in place and returns
  `(weak, mags)`; `attack_frame`, `is_feasible` and `stealthy_state` take an `AttackDesign`, and
  `with_region` fills in a design's interior. The random-draw order is unchanged: every frozen
  reference matches bit for bit.
- Features are computed from measurements only. The swing feature's scale is now the recent change
  of the observed frames, not of the noiseless pool: `timeline.write_temporal_layers` writes
  `temporal_delta` and `swing` after the walk from the observed injections alone, and
  `trust.secured_copy` recomputes them the same way. Every estimate takes one angle reference,
  `SEBase.theta_ref`, fixed at fit time from the training split's slack angle (the case's reference
  angle on the released pools; a slack angle that varies is refused), so estimating, the Jacobian transform, the residual localizer's scores and the
  trust scoring read no clean layer. Two uses of the truth remain, by design: fitting an estimator
  calibrates the meter weights, the benign mean state and the angle reference on the training
  split's clean layer, and
  `SEBase.score` compares estimates with the truth to report the error.
  `write_temporal_layers` runs in blocks with bounded memory. Takes effect in the data with the next
  release; scores on existing files are unchanged.
- Al is a single-snapshot attack: every Al episode of a generated timeline is one frame, as Aq.
- N-1 timeline generation is disabled: `fg.generate` and `generate_timeline` take no outage;
  `line_outage_candidates` remains a screening aid and `FdiaGenerator(outage=)` stays for engine use.
- The tiny test timeline moves to seed 1, the first seed that puts every family in both its train
  and test split under the new episode lengths; the frozen references are rewritten from it.
- Localization: `features="full14+prev"` (16 channels) and `"full14+prev+jac"` (24) append the
  previous frame's swing to the papers' 14, for `BusCNN`, `BusMLP` and the federated localizers
  (each bus's own reading, so client-local). A timeline export offers it as `prev_swing` (file
  row - 1). On a timeline the frame after a one-frame attack carries the same jump back with the
  opposite sign, labelled benign (31% of IEEE-118's benign training frames); a one-frame model
  cannot tell the two apart and stops trusting the swing. With the previous swing, the zero-shot
  BusCNN on v0.8.1 goes 0.78 -> 0.84, 0.55 -> 0.83 and 0.49 -> 0.81 macro-F1 on IEEE-14, 118 and 300
  (seeds 123 to 125), As 0.56 -> 0.95 and Ar 0.32 -> 0.83 on 118, at a lower false-alarm rate.
- Aq is a single-snapshot attack: every Aq episode of a generated timeline is one frame, as
  `corrupt_len=1` already made Ad, As and Ar. Through v0.8.1 Aq ran as 15 to 44 frame episodes, so
  a change-based feature saw only an episode's first frame (on IEEE-118 the zero-shot CNN recalled
  0.39 of the attacked buses on that frame and 0.00 on the rest). The tiny test timeline moves to
  seed 23, the first seed that puts every family in both its train and its test split, and the
  frozen references are rewritten from it. Takes effect with the next data release.
- The Jacobian features never read the true state. `JacobianFeatures` took each frame's change
  against the previous pool timestep's clean state, truth an operator does not hold; it now takes it
  against the previous frame's estimate, dz = z_t - h(x_hat_{t-1}), the fitted estimator's plain
  solve of the frame emitted just before (the slack angle stays the shared angle reference). A
  timeline export offers that frame's readings on request as `prev_node_x`, `prev_edge_x` and
  `prev_timestep` (`models.PreviousFrameFields`), read from file row - 1 whatever split or family it
  belongs to. `BusCNN` / `BusMLP` / the federated localizers with a Jacobian feature set and
  `JacobianWeighting` follow; a record shard is refused (its rows are not consecutive frames). On
  the tiny timeline `JacobianWeighting` moves in the fourth significant digit; the stealthy re-solve
  now raises the explained energy at an episode's first frame, as the temporal features do. A test
  rewrites the clean layer and checks the features do not change. The guides' Jacobian rows are
  rerun on it: the zero-shot CNN with the Jacobian block reads 0.820, 0.809 and 0.812 on IEEE-14, 118
  and 300 (0.873, 0.896 and 0.840 with the truth reference), its gain now on the unseen `As` and `Ar`
  rather than on `Aq`; the federated guide's tables are marked pending a rerun.
- Docs: the SE, localization and trust guides rerun on the v0.8.1 timelines, every table and quoted
  number regenerated from the new JSON. Directions that changed: the CNN gate now lowers the
  estimator's error on IEEE-14 (0.049 to 0.044) and 300, residual removal beats WLS on 300, and the
  zero-shot CNN with the Jacobian block reads 0.873, 0.896 and 0.840.
- Docs: `docs/federated/`, the federated localization papers' protocol on the v0.8.1 timelines.
  `FedBusCNN` and `FedBusMLP` with `full14+jac`, zero-shot, K = 1 to 3, seeds 123 to 125, in the
  paper's Table IV layout with per-family F1 and FR over every record and over benign records. The
  run script saves each run and skips saved ones; `make_report.py` builds the tables, figures and
  CSV sidecars from the JSON.
- `FedBusCNN` and `FedBusMLP` accept the Jacobian feature sets (`features="full14+jac"`, `"jac"`):
  the 14 channels stay per client (local power balance by default) and the 8-channel Jacobian block
  is the whole system's estimator applied to every meter's change, computed once centrally per pass
  and shared with every client (appended after the 14 channels for `full14+jac`, the whole input for
  `jac`); that block is the one feature a client does not build from its own meters. One client equals the centralized `BusCNN(features="full14+jac")`.
- Localization, the federated paper's tables: `LocalizerBase.score_perbus(ds, buses=, fr_over=)`
  returns per-bus F1, detection rate, false-alarm rate and AUPRC (`models.scores.PerBusScores`,
  `PerBusMetrics`) over every record and per family (that family plus benign), on the active buses
  or, for a learned localizer, the training-attackable ones; `fr_over="all"` is the paper's Table IV
  false-alarm rate, `"benign"` equals `score()["all"].macro_fr`. `LearnedLocalizer.tune_grid_threshold`
  and `score_grid` give record-level detection at a validation-tuned grid threshold
  (`models.scores.GridScores`). New formulas `perbus_rates` and `average_precision` [DG06] (equal
  to scikit-learn's). Existing outputs are unchanged.
- `fdia_graph.federated.RegionalPrior`: the proposed estimator's subspace prior fitted per client
  of a `Partition` (each client's basis from its own non-slack angles and voltages, placed on a block
  diagonal by `formulas.federated.block_diagonal_basis`), so no client's states leave it; one client
  is `SubspacePrior` exactly. With `GatedPrior(gate=FedBusCNN(...))` the federated localizer gates the
  estimator. The meter sigmas and the benign mean were already local; the solve stays central.
- `fdia_graph.federated`: the federated localization paper's localizers trained by FedAvg [MCM17].
  `FedBusMLP` and `FedBusCNN` are `LocalizerBase` methods (`fit(train, val)`, `localize`, `scores`,
  `score` as for `BusMLP`/`BusCNN`) over a K-way partition: each client reads only its own buses,
  builds its power-balance channel from its own flow meters by default (`kcl="local"`; `"global"`
  is the paper's central feature, which reads a neighbour's tie-line meters), trains on its own
  buses (optionally with a read-only halo) and scores them; only pooled moments, model weights and
  per-bus confusion counts cross a client boundary. `history` logs every round's loss and bytes.
  With K = 1 and no clip the fit equals the centralized one weight for weight (tested for both
  encoders). The Jacobian feature sets are supported with one central block (the entry above).
- `fdia_graph.federated`, first part: the split of a system's buses into K clients.
  `spectral_partition(edge_index, N, K)` is the federated localization paper's partition (spectral
  clustering of the bus adjacency, random_state 42 [VLX07]) and matched its cached partitions
  exactly on IEEE 14, 118 and 300 at K = 2 and 3 (the tests pin IEEE 14); `attackable=` biases the cut away from attackable buses. It
  returns a `models.federated.Partition` (client of every bus, interior and boundary buses, cut
  edges); `compute_nodes` gives a client's buses plus an optional halo of other clients' buses.
  `formulas.federated.fedavg` [MCM17], `interior_boundary`, `cut_edge_count`,
  `attackable_affinity` and `halo_nodes` carry the arithmetic. A `federated` extra (torch,
  scikit-learn) and CI installs it.
- Learned localizers, no number changes (the first step of `fdia_graph.federated`): the training
  loop is `localization.learned.LocalTrainer` (optimizer and batch order persist across `run`
  calls, optional owned-bus loss and gradient clipping for a federated client), prediction is
  `predict`, standardization pools channel moments (`formulas.federated.channel_moments`,
  `pool_moments` [CGL79]) and the validation threshold picks from per-bus confusion counts
  (`formulas.metrics.perbus_counts`, `tau_from_counts` [KEC25]), so a federated fit can sum them
  across clients. `models.training.OptimConfig` carries the training knobs. BusMLP and BusCNN
  weights, scores, thresholds and standardization are bit-identical to 0.19.0.
## 0.19.0

- Data release v0.8.1 is the default (`fg.load("ieee118")`; v0.8.0 stays readable with
  `release="v0.8.0"`): the eight timelines regenerated with this release's generator, the same pools
  (byte-identical), seed and knobs, built by `examples/_build_timelines_v081.py`. The stealthy
  families change on every system but IEEE-200, whose file is byte-identical to v0.8.0 (no bus there
  holds both a load and a generator, and its static generators produce nothing). Every file keeps
  72,000 frames, exactly half attacked, no frame falling back to benign, every family present, and
  no stealthy frame labelling a generator bus.
- Generator, the stealthy families change. Aq, At, Al and Am no longer target a load on a generator
  bus (zero-MW condensers included), the local attacker's rule in [BOY22]: a generator bus is too
  risky to falsify. `FdiaGenerator.stealthy_pos` is their target set; Ad, As and Ar keep
  `attackable_pos`. Attackable loads left to the stealthy families: IEEE-14 8 of 11, 30 18 of 20,
  57 35 of 41, 89 29 of 29, 118 54 of 99, 145 11 of 36, 200 108 of 108, 300 156 of 191. The frozen
  timeline is re-frozen. The data dictionary states that `attack/mag` is unsigned (a load rise and
  a drop of the same size record the same value).

- Loading and fitting, no number changes. `fg.load("IEEE118")` works like `"ieee118"` (a locally
  registered name stays case-sensitive). `latest_release` skips a tag it cannot parse instead of
  falling back to the pinned default, and its docstring no longer claims `load(release=None)` calls
  it. Two jobs downloading the same asset at once write separate temp files; the second keeps the
  first's installed copy instead of corrupting it. `fit` refuses a dataset whose slack disagrees
  with the case's. The solve's reduced normal-matrix inverse is built once in `fit` rather than on
  every chunk. The recent-change scale's docstring says "plus 1e-3", what the code does.
- Tests, no package change. `tests/test_public_coverage.py` tests directly what the suite reached only
  through whole pipelines: the replay policy and its buffer, `batched_normal_matrices`,
  `local_ac_solve` on a two-bus grid, `sparse_basis`, `per_bus_sequences`, `read_episodes`,
  `latest_release` online and offline, the `TrustSelector` base, the removal guard, and
  `tune_threshold`.
- Generator, the stealthy families change. The engine recovered a scan's load at a bus holding
  both a load and a generator as the stored injection plus the BASE generation, but the pools scale
  a bus's load and generation by one factor, so the load came out as s L + (1 - s) G instead of
  s L: at s = 0.75 the median load was 1.6x the true one on IEEE-118 (10 of its 99 attackable
  loads) and 2.0x on IEEE-300 (23 of 191), at times negative, so an Aq step there moved the wrong
  amount and a "raise" could lower the load. Every load-recovery site now uses
  `formulas.attacks.bus_load` (the `generator_output` construction the limit check already used),
  through `FdiaGenerator.true_load` and `scan_generation`. Static generators that produce anything
  count as injection buses (IEEE-89 buses 1, 65, 69, 79 and eight on IEEE-300 were treated as
  zero-injection junctions and pulled into attack regions). The attack-redistribution sign: the
  engine's redistribution LOWERS the target line's |flow| in the false state (checked on 77 of 77
  draws), so `am_direction="mask"` now keeps its sign and `"induce"` flips it (they were swapped);
  `"both"`, the default, maps a given draw to the same sign as before. The frozen
  timeline is re-frozen (341 of its 1000 frames change: the corrected stealthy draws spend the
  random stream differently, so later noise shifts too). The released v0.8.0 timelines were
  generated before this fix; regenerating them is a data release.
- CI, no package change. `publish.yml` runs the test suite on the tagged commit and publishes only
  when it passes. The suite also runs on Python 3.9 (the lowest supported) and on Windows, both
  required by `tools/pr.py merge`; the Ubuntu job installs the `[se]` extra by name and reports
  coverage. `ruff format --check` covers `tests` and `tools`. Every action is pinned to a commit
  SHA, kept current by Dependabot; `slow.yml` runs the `FDIA_SLOW` IEEE-118 check weekly and on
  demand.
- Fixes the new CI jobs found. The estimators failed on Python 3.9: its SciPy has no
  `lapack.dtrcon`, which `guarded_inverse` called for its condition estimate; it now falls back to
  Hager's 1-norm estimate (the same value LAPACK returns, a few O(kÂ²) solves) when SciPy lacks it (`[se]` still asks only scipy>=1.8). `tools/readability.py`
  no longer fails on a file on another drive than the repository (Windows CI's temp dir).
- Repository, no package change. `_publish_pypi.sh` is gone: publishing goes through a version tag and
  `publish.yml` only (no manual upload path). Four figures in `docs/` that nothing referenced
  (the README uses the copies in `docs/figures/`) are gone. `.gitignore` loses a duplicated block,
  ignores the pytest and ruff caches, and names `examples/_build_timelines_v080.py`, the v0.8.0
  build script, as tracked on purpose.
- Estimators, numbers change. `ResidualRemoval` is classical largest-normalized-residual removal:
  one meter per record per pass (the largest above the threshold), re-solved, until none exceeds it;
  a removal the observability guard refuses keeps that meter. It used to remove every meter above
  the threshold at once, taking out the honest neighbours a single gross error smears onto. Meter
  sigma (and the trust selectors' alarm level) is calibrated on benign records spread evenly over
  the train split instead of the first ones, which on a timeline are one early load regime. The
  guarded weighted solve also considers its final Newton step. Measured on the IEEE-14 v0.8.0
  timeline (geometric-mean angle error, test split): removal 0.0665 to 0.0552 degrees (replay
  0.0645 to 0.0224), WLS 0.0966 to 0.0961, Huber 0.0613 to 0.0605, prior + Huber 0.0499 to 0.0503,
  Jacobian weighting 0.0826 to 0.0800, gated arms within 0.2 percent. The frozen SE reference moved
  by at most 1.6e-7 relative (its train split has fewer than 600 benign frames).
- Estimators, fixes. Every public path that solves a dataset (`estimate`, `score`,
  `ResidualLocalizer`, `TrustSelector.score`, `JacobianFeatures.fit`, `JacobianWeighting`) refuses a
  `units="pu"` view with a ValueError; `fit` did before, but estimating or scoring a per-unit view
  converted twice and returned errors about ten times too large. `GatedPrior` and
  `JacobianWeighting` supply their per-record weights through one hook (`_record_weights`) instead
  of overriding `estimate`, so `ResidualLocalizer(estimator=GatedPrior(...))` scores the gated
  estimator rather than its ungated parent; the Huber passes live once in `SEBase` and
  `JacobianWeighting` gains `tol` (it ran all 40 passes) and builds its Jacobian features once in
  `fit`. `ResidualLocalizer` no longer refits an estimator that is already fitted
  (`SEBase.is_fitted`). The localizer hook is `_score(d, ds)`.
- `ds.windows(..., copy=False)` returns the windows as a read-only strided view of the frames, no
  memory per window (a 72k-frame IEEE-118 timeline at W=60 was about 8 GB as a copy); the default
  still returns a writable copy, now built from that view. `TrustedMetersDQN` evaluates the attack
  cost once per step instead of twice, keeps its replay buffer in a ring, and defaults to `seed=123`
  like the localizers (pass `seed=0` for the previous default; the trust guide pins it). Windows and
  the seed-0 DQN selection are bit-identical to 0.18.0.
- The trust guide's secured-copy tables for IEEE-118 (`results/secured_ieee118.json`): the DQN set
  opens the residual test on the stealthy families and lifts the learned localizer four points, the
  estimator gains within a percent at 20 meters. No package change.
- The docs site's staging step drops nav entries for pages a checkout lacks, so a backfilled older
  version on the versioned site has no link into a 404. No package change.
- Versioned docs: `.github/workflows/docs.yml` deploys the MkDocs site per package version to GitHub
  Pages with mike (tags as `latest`, main as `dev`) with a version picker on every page; the nav lists
  every page (the trust guide, the class map, the formulas, references, benchmarks and plans,
  contributing and the changelog), and `scripts/site_index.py` stages the three root pages. No
  package change.
- Trusted meters reach the estimators and the localizers: `TrustSelector.secured_copy(ds, out, name)`
  (`trust.secured_copy(selector, ds, out, name)`) writes a copy of a timeline in which the selected meters read their benign
  value on every frame, the attacker locked out of them, with the tamper masks and the stored
  temporal features following, so every estimator and localizer can be scored with a trusted set;
  `GatedPrior(secured=...)` never down-weights a secured meter (on a secured copy every gate was
  worse than no gate until this exemption, and with it the DQN's 20 meters take the proposed
  estimator from 0.050 to 0.020 degrees on IEEE-14); `formulas.projection.meter_positions` maps
  masked measurement indices to the file's layers; `docs/trust/run_secured.py` and the trust guide
  carry the tables. `formulas.temporal.SWING_WINDOW` is the writer's window, imported by
  `generation`.
- The narrative sections of the SE and localization guides (the Jacobian-informed weighting, the
  localization-gated estimation, the digest's ablation, the three readings) are re-derived on the
  v0.8.0 timelines from the results JSONs; their v0.7.2 markers are gone. On the timelines the
  Jacobian block is what makes the localizer's vector work (zero-shot 0.55 to 0.86 on IEEE-118)
  and a gate alone no longer lowers the estimator's error on any system. No package change.
- `docs/reference/CLASS_MAP.md`: one module diagram and six class diagrams of the package, drawn
  from the source by `tools/class_diagrams.py` (inheritance, the classes each class uses, the imports
  between modules) and checked in CI against the code. No package change.
- Every diagram in the docs is a rendered image (`docs/figures/diagrams/<name>.png`, source `<name>.mmd`
  beside it, `tools/render_mermaid.py`), since the GitHub mobile app and PyPI show a Mermaid block
  as code; no Mermaid block remains. No package change.
- `docs/se` and `docs/localization` IEEE-300 columns re-run on the v0.8.0 timeline (tables, figures
  and CSV sidecars); the live result tables of both guides now read the v0.8.0 release, and the
  paper-comparison table stays labeled v0.4.1. No package change.
- The README's pipeline diagram is a pre-rendered image (`docs/figures/diagrams/pipeline.png`, source
  `pipeline.mmd` beside it, `tools/render_mermaid.py`), since PyPI and the GitHub mobile app show a
  Mermaid block as code; README images use absolute URLs so the PyPI page shows them. No package change.
- Docs: the six diagrams that GitHub laid out badly are redrawn (the module map as layers, the
  generate and load pipelines as two rows, the estimator overview as columns, the trust guide with
  fit feeding score, the concepts page on the attack-vector construction with Am); no package change.

## 0.18.0

- Reviews: `tools/pr.py` waits for and requires every installed review bot (Copilot always;
  CodeRabbit and Gemini Code Assist once they have reviewed a pull request) on the head before a
  merge; `.coderabbit.yaml` carries the repository's review instructions (the physics conventions,
  the readability limits, the prose rules); CONTRIBUTING lists the three reviewers and their order,
  with the Claude Code `/code-review ultra` pass first. No package change.
- Names, no behavior: the generator's attributes are words (`_ppc_row`, `_n_ppc_buses`,
  `_from_bus_ppc`, `_base_mva`, `_target_lines`, `_primary_target_line`, `_line_flow_sign`,
  `_ptdf_load_buses`, `_solve_net`, `_injection_buses`, `_bias_sd`, `_draw_noise` and the public
  `n_lines`, whose old name `nl` still answers with a `DeprecationWarning` until 0.19),
  the dataset package's docstring maps each mixin to the questions it answers, and the data
  dictionary and the schema name the static `graph/edge_x` (the series reactance) apart from the
  flow layer of the same name.
- `fdia_graph.schema` is the file protocol: every group, dataset and attribute name of an HDF5
  file, the record-field to path map, the family table and the split codes, defined once; the
  writer, the readers and the tools spell paths through it, and `tools/readability.py` (the CI
  gate) refuses a path-shaped literal ("data/...", "graph/...") in any other module.
  `dataset.base.FAMILIES`, `STEALTHY_FAMILIES` and `timeline.KIND` stay importable. No file or
  number changes.
- One export of a split: `ds.export(fields, format="numpy" | "torch" | "tf" | "pandas", device,
  flatten_features)` replaces `to_numpy`, `to_torch`, `to_tf` and `to_pandas`, which still answer
  with a deprecation notice until 0.19. `ds.windows(..., per_bus=True)` returns one sequence per
  bus (`[n*N, W, 4]`), what `torch_windows` built, and `fg.load(name, split, order="time",
  format="pyg")` gives the graphs `pyg_stream` built with the file's chronological split, so both
  torch helpers are deprecated (retire in 0.19). Eight ways to read a view become three: `export`,
  `ds[i]` and `ds.windows`. No number changes.
- Four findings of a full-branch review: a load on the slack bus is never a target (the slack
  never enters a region, so IEEE-57 frames that scaled its 55 MW load were labelled attacked with
  no attack in the state); the tamper masks are the meters whose stored float32 reading changed
  (the attack vector's 1e-7 tolerance marked 1.8% of entries that the reading could not show); a
  branch out of service is not a hop of the attacker's region on N-1 generators; and the local
  solve builds its interior Jacobian block directly instead of slicing the n x n derivatives,
  about half the cost of a stealthy frame on IEEE-300, with identical states.
- The local power flow of a stealthy frame is a damped Newton: each step is halved until the
  mismatch drops, the full step first, so every frame the plain method solved is bit-identical. A
  step the region still cannot absorb is halved (an Aq step at most three times and never under
  the noise floor, a ramp or Am frame at most six times) so the frame stays attacked at the
  largest step with a solution, an Al frame halves its redistribution and then redraws its line (up to forty),
  and an Aq, At or Am episode tests its design (the full step, the ramp's peak, the held
  redistribution's peak) on every frame it will occupy, each with the step that frame will carry
  (an Aq episode's whole span, every frame of the ramp and of the Am rise, plateau and return),
  and redraws it, up to forty times, when no stealthy state exists on any of them, so an accepted design cannot fall back, and a span with no admissible design stays
  benign and is counted; an Am redistribution is also tried at halved sizes above the floor
  before it is redrawn. A ramp or Am frame never carries a zero step (the profile's first frame
  and return leg floor at one rate), so every frame labelled attacked carries an attack. `fallback_benign` now counts every frame the draw
  meant to attack that is benign, whether or not an episode record exists for it (an Am episode
  with no admissible redistribution left no record and was not counted). An Aq episode now draws its direction like the ramp, a load rise or a load
  drop with the same 5% to 20% scale: on a case that runs below its voltage limits (IEEE-57) a
  rise near the low buses has no admissible state and a drop is the attack that fits. Loads above `max_load_mw` (new knob, default 2000 MW) are never targets: the
  IEEE-145 case lumps whole areas into loads of 4 to 58 GW, and a 5% step on one has no local
  solution, which left 30% of that system's attack frames benign; no other ladder system has a
  load above 1.1 GW, so their files do not change. `fallback_benign` is zero on every released file. Every false state also satisfies the security and operational constraints of Wu et al.
  2026 (`OperatingLimits`, their equations 21 to 23): each bus voltage within the case's own
  limits (a bus the true state already holds outside a limit may not be made worse, IEEE-57 runs
  below its own minimum) and, for the generators of the attacked subnetwork as in the paper (a
  boundary bus is outside it, its voltage held true and its injection whatever balances the
  region), the implied output within its P and Q limits widened per bus to the range the benign
  pool ran it over (the pools scale generation with load and never enforced nameplate), the true
  output recovered exactly from the pool's common load and generation scale, and the pretended
  load change at a target bus not counted against its generator; a false state outside them is rejected and its step halved
  like an unsolvable one. The file records the widest bus limits as `v_lo` and `v_hi`.
- A stealthy frame (Aq, At, Al, Am) is the true scan plus the attack vector a = h(x_false) - h(x_true)
  of its local false state: every meter keeps its own noise draw and the tampered meters are
  shifted by exactly what the false state moves them, so `observed - benign` is the attack for
  every family and a WLS residual test flags the stealthy families at the benign rate. Until now
  the tampered meters were re-emitted from the false state, which drew their noise from the false
  reading; where a boundary bus has a structurally zero injection (the IEEE-14 synchronous
  condenser's P, a zero-injection bus) that noise is several times what the estimator calibrates
  for the meter, and the residual test flagged about a third of the Aq, Al and Am frames on the
  full IEEE-14 timeline. An Al frame whose drawn target line has no feasible redistribution in
  its region redraws the line (up to ten) instead of falling back to a benign frame. Both change
  every generated file; the frozen references are re-frozen.
- `tools/pr.py merge` requires every smoke job green by name on the head (a job that has not
  registered yet is not green), every other check finished without failure, and Copilot's review on
  that head; `wait` waits for the same set. A merge can no longer slip in while CI is still starting.
- `fdia_graph.trust`: which meters to secure so that stealthy attacks stop being stealthy, after
  the trusted-PMU defence of Wu et al. 2026. `TrustedMeters(k)` is the greedy row-reduction
  selection on the WLS Jacobian (secure, on the cheapest open attack, the meter whose protection
  raises the attack cost most), `TrustedMetersDQN(k, episodes)` the same selection learned as a
  Markov decision process with a deep Q-network (state the secured set, reward the attack-cost
  rise; needs torch). Both expose the order and the attack cost after each meter, and `score(test)`
  on a time-ordered timeline pins the secured meters to their benign reading and reports the WLS
  residual detection per family before and after (`TrustScores`). The kernel is
  `formulas.trust` (`attack_subspace`, `sparse_basis`, `attack_cost`, `greedy_trusted_meters`);
  `docs/trust/` has the guide and `run_trust.py`.
- The timeline writer draws its attack episodes so their frames sum to exactly
  `round(attacked_frac * T)`, places every one of them (longest first, each at an onset drawn
  uniformly among the onsets where it fits), and emits benign frames everywhere else. The
  benign-gap band and the rule that started the next episode with no gap are gone, so whether two
  episodes touch or a quiet stretch separates them is a property of the draw; an Am episode redraws
  its redistribution at onset rather than leaving frames benign, and only a non-converging power
  flow falls back to a benign frame (counted in `fallback_benign`). The frozen references are
  re-frozen on the tiny timeline (seed 4).
- Data releases have their own tag namespace: the assets of `v0.8.0` and later live under the
  GitHub tag `data-v0.8.0` (package versions own the bare `v0.x.y` tags, PyPI 0.8.0 is `v0.8.0`),
  while `v0.7.1` and `v0.7.2` keep their bare tags; `registry.release_tag` maps the short name
  users write (`fg.load(name, release="v0.8.0")`, `FDIA_GRAPH_RELEASE`) to the tag, the newest
  data release is picked from the tag list, and `tools/upload_assets.py` refuses a package tag.
- The localization scores in `docs/localization` are re-run on the v0.8.0 timelines and are not
  comparable with the v0.7.2 shard tables: the shard's temporal features compared an attacked
  snapshot with the benign scan before it, so every attacked record spiked, while the timeline's
  compare each frame with the frame emitted one minute earlier, so a sustained episode spikes at
  its onset and the one-frame families also spike on the benign frame after them. The per-record
  numbers are lower for the same detectors and the same data, and the tables say which reference
  they use.
- `fg.load(name)` reads the v0.8.0 timelines by default (`_RELEASE` is `v0.8.0`, the checksums of
  the final files pinned). The test suite's tiny fixture now walks the v0.8.0 pool (72,000 one-minute
  states where the v0.7.2 pool held 36,000), so the frozen references are re-frozen on it.
- The data release `v0.8.0`, step 4 of `docs/plans/ONE_DATASET_PLAN.md`: one timeline file per
  system of the ladder (`timeline_ieee{N}.h5`, 72,000 frames, the seven families, seed 123,
  `attacked_frac` 0.5) and the operating-point pools as HDF5 (`pool_ieee{N}.h5`), built by
  `examples/_build_timelines_v080.py` from the v0.7.1 NYISO pools. `fg.load(name)` reads them by
  default; `fg.load(name, release="v0.7.2")` (or `FDIA_GRAPH_RELEASE=v0.7.2`) still reads the
  record shards, since the registry now maps a release tag to the file layout it shipped
  (`registry.dataset_file`, `pool_spec`, `is_timeline_release`) and knows the sha256 of each pinned
  release. `generate(states=None)` fetches the pool through the same registry. `load_stream`
  (deprecated) returns the timeline through the loader at a timeline release and the v0.7.2
  stream file before it. `tools/upload_assets.py` publishes release assets idempotently.
- One generator path, step 3 of `docs/plans/ONE_DATASET_PLAN.md`: the record-shard writer
  (`generation.generate_shard` and its draw loop), the stream walker and its `.npz` output, the
  magnitude sidecar (`<out>.mag.npz`) and the graph sidecar read of `load_stream` are deleted, with
  the `ShardArrays` and `Record` models and `SINGLE_SHOT_ORDER`. `generate_stream` is now the
  timeline writer followed by a read of the file it wrote (`streams.stream_of` turns a time-ordered
  dataset into the stream dict), so its output is the timeline's and its `out` is the HDF5 path;
  `load_stream` still reads the v0.7.2 stream files (which embed their graph) until the v0.8.0 data
  release. The loader keeps reading the v0.7.2 record shards; `tests/data/tiny_shard_v072.h5` is a
  checked-in file in that layout that the suite loads.
- The frozen references are re-frozen on the tiny timeline the suite builds (`frozen_spec.TIMELINE_KW`,
  1000 IEEE-14 frames, every family): the file's arrays and attributes, and the estimator and
  localizer scores on its test split. The shard and stream references are gone with their writers.
- One loader for both kinds of file, step 2 of `docs/plans/ONE_DATASET_PLAN.md`. `fg.load(name)`
  reads a timeline file (`kind="timeline"`) as well as a v0.7.2 record shard. New arguments:
  `order="time"` (default) keeps the file order, chronological on a timeline; `order="random"` is
  the same records in a permutation fixed by `seed` (default 0), a view index, nothing copied, so
  two people asking for the same seed get the same record table. On a timeline every record,
  batch and export carries `benign` and `edge_benign` (the attack removed, the noise kept, in the
  requested units), `ds.windows(W, stride, label, layer)` slides windows over a time-ordered
  contiguous view (a split or the whole file; it refuses a random order, a family subset and a
  shard), `ds.episodes` is the episode table (`EpisodeTable`: onset, length, family, buses) of the
  view, and `torch_windows(dataset=ds)` / `pyg_stream(dataset=ds)` take the loaded timeline.
  `ds.is_timeline` and `ds.has_benign` say which kind a file is.
- `fg.generate(system, name, frames=None, **knobs)` now writes a timeline (the knobs of
  `timeline.generate_timeline`, plus `frames` to cap the pool timesteps walked) and registers it, so
  `fg.load(name)` and `fg.load(name, order="random")` read it back. The record-shard writer is
  gone (the entry above): the loader still reads the v0.7.2 shards, nothing writes them.
- `generate_stream`, `load_stream` and `windows(stream, ...)` warn with `DeprecationWarning` and
  retire in 0.19: the timeline file replaces the stream, `fg.load(name, order="time")` reads it, and
  `ds.windows` replaces `windows`. They keep working unchanged until then (`load_stream` reads the
  v0.7.2 stream files).
- The timeline writer, step 1 of `docs/plans/ONE_DATASET_PLAN.md`: `fdia_graph.timeline.generate_timeline`
  walks one attacked timeline over a system's operating-point pool and writes one HDF5 file
  (`kind="timeline"`) carrying every layer the streams and shards had between them: the observed,
  benign and clean scans per frame, the full static graph, per-frame `stealthy`, `seq_id` (the
  episode index) and a chronological `split` that never cuts an episode, an `episodes/` table, and
  an `attack/` group with the designed magnitude per attacked bus and the tamper masks (the meters
  the attacker wrote). Families are scheduled by inverse expected episode length so each gets about
  the same share of attacked frames; `corrupt_len=1` makes every Ad/As/Ar frame an independent draw.
  The loader for these files is step 2; `generate` and `load` are unchanged in this release.
- A seventh family, `Am` (code 7, stealthy), the multi-snapshot attack of Wu et al. 2026: the Al
  load redistribution drawn once at episode onset and applied along a ramp whose per-bus per-frame
  step stays under `am_rate` of the noise floor. `fg.FAMILIES[7] == "Am"`, `fg.STEALTHY_FAMILIES`
  now `{1, 5, 6, 7}`; the published shards and streams carry no `Am` frames.
- Every stealthy family (Aq, At, Al, Am) is now a local false state, the attacker model of Wu et al.
  2026: the attacker solves the power flow of the subnetwork within `hops` branches of the attacked
  loads (or the target line) with every other bus voltage held true (`FdiaGenerator.solve_local`,
  `formulas.network.local_ac_solve`), never the whole grid and never the slack, and writes only the
  meters that false state moves (the tamper masks). The measurement vector is exactly consistent
  with an AC state, so a WLS residual test flags these frames at the benign rate; the global
  re-solve with pinned generation is gone from the writer. `hops` (default 2) replaces `am_sigma`.
- `generate_stream` builds its frames through the timeline module's episode primitives; its
  scheduler and RNG order are unchanged and the streams are bit-identical. `generate(states=...)`
  also accepts a pool stored as HDF5 (dataset `X`).
- One definition of the measurement column orders: `fdia_graph.models.NodeColumns` (`v`, `p_inj`,
  `q_inj`, `theta`), `EdgeColumns` (`p_from`, `q_from`) and `BranchColumns` (the eight `edge_attr`
  columns), with `.of(array)` giving named views of any such array and `NODE`, `EDGE`, `BRANCH`
  the indices. Every record, batch, split and stream has `.node()` and `.edge()`; the generator,
  loader and estimator index columns through the definition instead of literals. No behaviour change.
- The shard-shaped bundles (`RecordBundle`, `BatchBundle`, `ArraysBundle`, `Stream`)
  are built from field groups (`fdia_graph.models.fields`: scan, labels, record ids, temporal,
  clean, graph, stream layers), so each shared field is declared once; `Bundle` gains `_order`
  (the dict-key order, unchanged from before) and `_required` (a missing required field raises
  `TypeError` at construction, as a missing argument did). These four bundles are keyword-only
  (a positional call raises `TypeError` instead of binding to the inherited field order); they
  are return types, and no caller in the package or its examples built them positionally.
- `fg.load_stream` now fills `system` and `attacked_frac` (they were None: the stream files carry
  the arrays only and the generator attached the two values after writing). Both are derived from
  the arrays on load, the same way `generate_stream` computes them; `fdia_graph.streams.stream_summary`
  is the shared helper.
- `ruff check` configured in `pyproject.toml` with pyflakes, import order and modern-syntax rules,
  and the package clean under them (`list[int]`-style generics and `collections.abc` imports;
  `Optional`/`Union` stay, since `typing.get_type_hints` on 3.9 cannot evaluate `X | None`). The
  CI `format` job runs it on every pull request. No behaviour change.
- `tools/bench.py` and `docs/reference/BENCHMARKS.md`: per-record timings of shard generation and
  the WLS, Huber and prior+Huber estimators on the tiny shard, appended per run with the machine
  and torch state; `--check` fails when a timing is more than 3x slower than the last row.

- `CONTRIBUTING.md` (the rules, the pull-request flow, releasing), `tools/pr.py` (create, wait,
  comments, reply, merge, with the merge rule enforced) and `tools/release.py` (tag, GitHub
  release, PyPI wait) so a second maintainer can ship; the plan documents move to `docs/plans/`.
- Clear errors at the public edges instead of silent or opaque failures: an unknown family name
  or code in `load(families=...)`, an unknown `split`, an unknown field in `to_numpy` and its
  siblings, and an invalid `label`, window length or stride in `windows` raise `ValueError`
  naming what is allowed (an unknown family used to select nothing; an invalid window label used
  to behave as "any"). `tests/test_edges.py` covers these and the estimators' and localizers'
  argument checks; `FDIA_SLOW=1` additionally runs the IEEE-118 estimator sanity test.

## 0.17.0

State estimation without torch, the formula kernel complete, the loader as concerns, and the
scheduled removals (PRs #74 to #78). What a user sees: `pip install "fdia-graph[se]"` no longer
pulls torch (add `[torch]` for faster per-record inverses); the pre-0.16 generator attribute
names are gone; every helper on the package namespace is the real function. Shards, streams and
the localizer scores are bit-identical to 0.16.0; estimator scores moved by at most 2.7e-6
relative (the closed-form Jacobian, see the entry below). No import path changes.

- State estimation no longer needs torch: `formulas.network.ac_measurement` is the estimator's
  h(x) and `ac_jacobian` its closed-form Jacobian (equal to the automatic-differentiation Jacobian
  to 1e-14), so `fit` takes no gradient; torch, when installed, only speeds up the per-record
  inverses, and the `[se]` extra no longer installs it. The frozen estimator scores moved by at most 2.7e-6 relative (six of 56 cells above
  1e-9, all in the Huber arm, whose passes amplify last-bit differences in h); shards, streams
  and every other score are bit-identical. The torch twin `SEBase._h_t` stays for callers that
  differentiate through it.
- `formulas.estimation` (the WLS step, gain matrix, residual covariance, normalized residual,
  Huber weight, the operating-point prior basis, the localization gate), `formulas.linalg` (the
  guarded inverse, condition number, per-record normal matrices) and `formulas.projection` (the
  weighted pseudo-inverse, explained/unexplained split, leverage, weak directions, meter-to-bus
  aggregation): the estimators and the Jacobian features now call these named functions, each
  with its equation and source key. Identical numbers (bit-identical frozen scores); the
  estimator's private helpers keep their names and delegate.
- Removed, as scheduled one minor version after 0.16: the generator's pre-0.16 attribute names
  (`edge_r` ... `edge_status`, `M`, `flow_meter`, `bias_pi` ... `bias_qf`, `outage`,
  `outage_pos` ... `outage_base_flow_mw`; read `g.branch`, `g.meters`, `g.bias`, `g.contingency`)
  and `fdia_graph.streams._swing_scale` (use `fdia_graph.generation._swing_scale`).
- `fdia_graph.generate`, `generate_stream`, `load_stream`, `windows`, `pyg_stream`, `torch_windows`,
  `load_profile`, `fetch_profile`, `generate_states` and `line_outage_candidates` are the real
  functions, resolved on first use (PEP 562) instead of wrappers that restated their docstrings.
  Same names, same calls, same lazy imports; `help(fg.generate)` now shows the one docstring.
- The loader is the `fdia_graph.dataset` package: `FdiaGraph` is assembled from `graph`,
  `physics`, `records` and `export` mixins over a `base` that declares the shared state, the same
  pattern as the generator. Same import path, same names, same behaviour (docs/plans/READABILITY_PLAN.md,
  step 9).

## 0.16.0

The readability and data-models series (PRs #64 to #72). What a user sees: every dict the package
returns is now a typed bundle with attributes and a docstring, still the same dict with the same
keys; every model lives in `fdia_graph.models`; the generator's old attribute names warn. Shards,
streams and every score are bit-identical to 0.15.0 (the strict frozen tests are the proof), no
data release moves, and no import path changes. The entries below are in the order the steps
merged, newest first within each series.

Data models, steps 1 and 2 (docs/plans/DATA_MODELS_PLAN.md). No user-visible change; one deprecation.

- `fdia_graph.models.Bundle`: the base of typed records, a frozen dataclass that is also a read-only
  mapping, so a bundle works wherever a dict did.
- Internal tuples and dicts are models: `Scan`, `TrueState`, `Redistribution`, `ResolvedPool`,
  `Admittances` (also returned by `formulas.network.branch_admittances`), `AssetSpec` (returned by
  `registry.resolve`, indexable as before), `DownloadTarget`, `ShardArrays`.
- `FdiaGenerator` holds `branch`, `meters`, `bias` and `contingency` models. Deprecated: the previous
  attribute names (`edge_r` ... `edge_status`, `M`, `flow_meter`, `bias_pi` ... `bias_qf`, `outage`,
  `outage_pos` ... `outage_base_flow_mw`) still work as read-only properties and warn; they are
  removed one minor version later.

Data models, step 3: the public dicts are typed bundles. No user-visible change: every one is
still the dict it was (a `dict` subclass with the same keys in the same order), so indexing,
`**`, iteration, JSON dumping of score tables and PyTorch's default collate keep working; each
also has attributes and a docstring naming its fields.

- `dataset.RecordBundle` (`FdiaGraph[i]`), `BatchBundle` (`collate`), `ArraysBundle` (`to_numpy`,
  `to_torch`, `to_tf`), `Summary` (`summary`).
- `se.base.EstimatorScores` of `ErrorPair` rows (`SEBase.score`); `localization.base.LocalizerScores`
  of `OverallMetrics`, `BenignMetrics` and `FamilyMetrics` (`LocalizerBase.score`).
- `se.jacobian.JacobianOutputs` (`JacobianFeatures.transform`; the `"global"` key is the
  `global_` attribute).
- `streams.Stream` (`generate_stream`, `load_stream`) and `engine.core.LineCandidate`
  (`line_outage_candidates`).
- `Bundle` is now a `dict` subclass rather than a `Mapping`, so `isinstance(out, dict)` and
  `json.dump(out)` hold as well; it is read-only on both sides.

Data models, step 5: every data model lives in the `fdia_graph.models` package, grouped by what
the data is (`grid`, `frames`, `data`, `scores`, `assets`), with `fdia_graph.models.PUBLIC` naming
the bundles a user receives. Every previous import path still works (each producer module
re-exports what it used to define), so no user code changes.

Data models, step 4: `docs/reference/DATA_DICTIONARY.md` gains a "Models" section listing every
public bundle and its fields, generated by `tools/models_doc.py` from the dataclasses and kept
current by a test.

Readability series, steps 6 and 7 (docs/plans/READABILITY_PLAN.md): the temporal and ramp formulas in
the kernel, and the rest of the backlog. No user-visible change (bit-identical shards and
streams, identical estimator and localizer scores).

- `formulas.temporal` (`recent_change_scale`, `temporal_delta`, `swing_zscore`) and
  `formulas.attacks.ramp_profile`, each with its equation and source key, used by both generators.
- `AttackMixin.corrupt` decides the family once and runs one short loop per family;
  `PhysicsMixin.solve` pins the generation in `_pin_generation`; `line_outage_candidates` screens a
  line with guard clauses; the loader's `__getitem__` and `collate`, the localizer's `score`, the
  learned model builders, the stream loader and the download are split the same way; the Huber
  passes of the estimators are one method.

Readability series, step 5 (docs/plans/READABILITY_PLAN.md): the generator constructor, the loader
constructor and `to_numpy` split into named steps. No user-visible change (bit-identical shards
and streams, same loader outputs).

- `FdiaGenerator.__init__` reads as a list of what it builds: the case with its contingency, the
  load tables, the meter plan, the edge index, the branch physics, the admittances, the meter
  biases; the accuracy-class split of the meter error is `formulas.noise.bias_jitter_split` and the
  series admittance comes from `formulas.network`.
- `FdiaGraph.__init__` delegates to `_read_header`, `_read_static_graph`, `_read_layers`,
  `_reference_bus`, `_record_mask` and `_preload`; `family_ids` translates family names or codes in
  one place; `to_numpy` gathers the clean layers and converts units through small helpers.

Readability series, step 4 (docs/plans/READABILITY_PLAN.md): one AC measurement function. No
user-visible change (bit-identical shards and streams; `ybus`, `yf`, `yt` and `edge_clean_full`
unchanged).

- `fdia_graph.formulas` (new, public, provisional until 1.0): `formulas.network` holds the AC
  network model as pure functions, `complex_voltages`, `series_admittance`, `branch_admittances`
  (the pi model behind `ybus`/`yf`/`yt`), `bus_injections` and `branch_flows`, each with its
  equation and source key. The generator's measurement emission and the loader's admittance
  and clean-flow computations call them; the estimator's torch measurement function is pinned to
  them by a test, so the package has one AC model.

Readability series, steps 2 and 3 (docs/plans/READABILITY_PLAN.md): one attack frame for shards and
streams, both generators split into named steps. No user-visible change: a seeded shard and a
seeded stream are bit-identical to the previous release (`tests/test_frozen.py`, strict mode).

- `fdia_graph.engine.records`: `attack_frame` builds one scan of any family from a stored
  operating point, for `generate` and `generate_stream` alike; `FrameKnobs` carries the two
  switches that differ between them; `replay_frame` and `remember_benign` name the replay policy.
- `generate` and `generate_stream` are sequences of named draws and episodes; the swing scale,
  the temporal features and the ramp profile are functions with their source keys, shared by both.
- Shard records are a named `Record` tuple and the writer reads them by field name; the writer is
  split into the attributes, graph, data and clean groups.
- No public name changed. The private stream helper `_swing_scale` now lives in `generation`;
  `streams._swing_scale` stays as a deprecated alias that warns, removed one minor version later.

Readability series, step 1 (docs/plans/READABILITY_PLAN.md): the safety net. No user-visible change.

- `tests/test_frozen.py`: a generated shard, a generated stream, and the estimator and localizer
  scores on the test shard are compared with frozen references (`tests/frozen/`): exactly in
  strict mode; otherwise arrays to 1e-7 relative and estimator and localizer scores to 1e-4
  relative, the cross-platform tolerances the test module documents. `tools/freeze_reference.py`
  rewrites the references when a change to the generator or an estimator is intended.
- `tools/readability.py`: the readability measures (complexity, nesting, closures, parameters,
  positional indexing) as a report over the package and as a gate on the functions a change
  touched; runs in CI.
- `docs/reference/FORMULAS.md` and `REFERENCES.md`: the catalogue of formulas and their sources,
  with the docstring template every formula function follows.
- `docs/plans/READABILITY_PLAN.md`, `docs/plans/RESTRUCTURE_PLAN.md`: the plans for the series.

## 0.15.0

- `edge_clean_full`: the exact clean power flow on every branch, computed on load from the clean
  state through the branch admittance matrix; in the record dict, PyG data, `collate`, `to_numpy`
  and the per-unit view.

## 0.14.1

- Per-record normal matrices are built by chunked matrix products instead of one einsum; the
  IEEE-300 estimators run in minutes instead of days. Same numbers.
