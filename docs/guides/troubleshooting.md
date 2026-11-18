# Troubleshooting

Every stage either raises a named error or records why it produced less than asked. This page
lists what each stage reports, what it means and what to do. It covers every test system; a row
that applies to one system says so.

## The kinds of error

The named errors of the package are defined in `src/fdia_graph/errors.py`. A few conditions
raise a built-in error: a missing optional package (`ImportError`), an unknown dataset name
(`KeyError`), a missing or corrupt download (`FileNotFoundError`, `OSError`), and an IEEE pool
with no converged step (`RuntimeError`).

| Error | Means | Is a |
|---|---|---|
| `ConfigError` | an argument or a setting has a value its model does not allow | `ValueError` |
| `DataConditionError` | the data read does not meet a condition | `ValueError` |
| `SolverUnavailable` | the environment cannot run the solver | `RuntimeError` |

The other named errors are subclasses of `DataConditionError`, except `MissingCapability`,
which is a `ConfigError`. So `except ValueError` catches every named error but
`SolverUnavailable`. A broken environment is not a property of the data.

A settings model checks its values when it is built, before any download or file is made. The
message names the field and what it may be.

## Environment

| What you see | Why | Fix |
|---|---|---|
| `SolverUnavailable: Pyomo is not installed` | the `[ny]` extra is missing | `pip install "fdia-graph[ny]"` |
| `SolverUnavailable: Ipopt is not found or does not start` | Ipopt is not installed, or cannot load its DLLs | see below |
| `ImportError` for pandapower | the `[generate]` extra is missing | `pip install "fdia-graph[generate]"` |
| a change you made does not show up | Python imports another copy of the package | put your checkout's `src` first on `PYTHONPATH` |
| a script starts itself again and again | it asks for several workers without a main guard | start it under `if __name__ == "__main__":` |

**Ipopt.** Install it with `conda install -c conda-forge ipopt` into the environment, and run
that environment's Python with the environment active. On Windows a missing DLL shows as exit
code `0xC0000135`. `fdia_graph.data.ipopt.ipopt()` adds the environment's `Library/bin` folder
to `PATH` when it exists. Check it:

```python
from fdia_graph.data.ipopt import ipopt

print(ipopt().version())
```

**Another copy of the package.** This usually happens in a git worktree, when an editable
install points at the main checkout.

**Workers.** The design processes and the pool workers are spawned, and a spawned process
imports the script that started it.

## Loading a dataset

| What you see | Why | Fix |
|---|---|---|
| `NotATimeline` | the file's `kind` is not `timeline`: an older record shard, or another file | load a release from v0.8.0 on |
| `NotATimeline` from `Dataset` | the file is not a New York timeline | open it with `fg.FdiaGraph` |
| `MissingCapability` | the view lacks what the caller needs, for example PMU currents | load a file that has them |
| `ConfigError` on `fg.load` | an option is not one of its choices | the message lists the choices |

## The operating points

**An IEEE case** (`fg.generate_states`, `fg.generate`):

| What you see | Why | Fix |
|---|---|---|
| `RuntimeError: no operating points converged` | no step's power flow converged | check the profile and the scaling arguments |
| `DataConditionError` from an OPF pool | no step has an optimum within the case's limits | use a power flow, or another case ([Solving](../concepts/solving.md#an-ieee-pool-by-opf)) |
| `ConfigError` on `SolveSettings` | a power flow in Ipopt | pandapower runs every power flow |
| `UnknownColumnOrder` | the pool's column order cannot be read from its values | store the columns as voltage, P, Q, angle |

**The New York case** (`pipelines/newyork/run_pool.py`):

| What you see | Why | Fix |
|---|---|---|
| `FileNotFoundError` on `data/...` | the scripts read `data/` and `out/` relative to the working folder | run from the work folder that holds `data/` |
| `DataConditionError: bus ... has no row in ...` | a bus has no location in the bus-location table | fetch the inputs again (`build_timeline.py --fetch`) |
| `DataConditionError: ... first hour has no AC point` | the span's first hour did not solve, so the chunk has no start | start the span at another hour |
| `ConfigError` on `PoolBuilder.from_settings` | a solve choice the New York pool cannot be built with | [Solving](../concepts/solving.md#why-a-choice-is-refused) |

`PoolBuilder.report()` counts every frame by its solve kind. Only `solved` frames are used
later.

| Solve kind | Meaning | What to do |
|---|---|---|
| `solved` | the dispatch and the power flow converged | nothing |
| `failed` | one of them did not converge | the frame is kept in place and counted by `report()`; many in a row point at the inputs of those hours |
| `no_hour` | the hour the chunk starts from has no AC point | rebuild that span |
| `no_load` | NYISO has no load for the frame: the spring-forward gap, or a gap in its data | nothing; nothing is solved for these frames |

`report()` before `build()` raises `NotFitted`.

## The timeline

### Any test system

| What you see | Why | Fix |
|---|---|---|
| `ConfigError: generate_timeline got unknown settings` | a keyword that is no setting | [the settings tables](generation.md#the-settings) |
| a `ConfigError` saying the pool carries no time | `split_by="week"` on a pool without times | pass `time=`, or use `split_by="time"` |
| a `ConfigError` about `SplitRule.time` | the times are not one `datetime64` per frame, increasing | [`split_by` and `time=`](generation.md#split_by-and-time) |
| a `RuntimeWarning` that a split's placements jammed | the split could not fit every episode; one was dropped | lower `attacked_frac`, shorten the episodes, or use more frames |
| a `RuntimeWarning` that an episode found no feasible design | it had no design after 20 moves (`REDRAWS`) | see below; the file records the shortfall |
| no file after a failed run | the file is written under a `.part` name and renamed only when whole | fix the error and run again |

A file records what it could not build, per split and family, in the attributes
`episodes_requested`, `episodes_built`, `episode_redraws`, `episodes_dropped` and
`episode_shortfall`. `fallback_benign` counts the frames of an episode stored benign.

### A `SearchSource` (the IEEE cases)

| What you see | Why | Fix |
|---|---|---|
| `NoAdmissibleTarget` | a family has nothing to attack on this case | drop the family, or check `max_load_mw` |
| `NoLineRatings` | `am_attack=OverloadSettings(rating_source="pglib")` on a case without PGLib-OPF ratings | use `rating_source="pool"` |
| `ConfigError: ... generates one timeline` | a second `generate` on one source | build a new source |
| `ConfigError: ... the generator makes At, Am only` | `families` names an older family | use `At` and `Am` |

An overload window with no stealthy design is moved to another onset of its split. The
measured IEEE-14 builds record their redraws and one shortfall
([Generate a timeline](generation.md#entry-point-an-ieee-case)).

### A `NewYorkSource`

| What you see | Why | Fix |
|---|---|---|
| `DataConditionError: ...: branch f-t is not in the case in that orientation` | an interface names a branch the case does not hold | check the case file and the interface data against the pinned inputs |
| `DataConditionError: ...: no rows in interflowHourly_<year>.csv` | no interface flows for that year | fetch the inputs for the year you build |
| `DataConditionError: the pool must carry each unit's output` | the pool has no `unit_p` | build the pool with `run_pool.py` |
| `ConfigError` on `NewYorkRelease` | a family other than `Am` and `At`, or `Am` with no interface | fix the field the message names |

After `source.generate(path)`, `source.summary()` says what was built.

| Key | Counts |
|---|---|
| `placed` | the slots placed, per split and family |
| `moved` | the moves of slots whose design found no attack |
| `built` | the episodes built |
| `given_up` | the slots given up after `REDRAWS` moves |
| `unsolved_attempts` | design attempts not made, because the window touched an unsolved frame |
| `missed` | why design attempts found no attack, per family |
| `survey_missed` | the same for the onset survey |

`missed` counts attempts, not slots. A slot is tried again at every onset it moves to.

| Miss reason | Meaning | What to do |
|---|---|---|
| `no_target` | nothing to attack: no interface near its limit, or no load bus for a ramp | if it covers every slot, the onset mask and the interface data disagree |
| `infeasible` | targets existed, but no support met every bound: the load cap, the basepoint band, the ramps, the voltage limits | the bounds are binding; widening them is a modelling decision, not a fix |
| `solver_error` | Ipopt stopped abnormally on a support the search tried, so `infeasible` may be wrong | the family counts them in `solver_errors` |

For one episode alone, building a `SingleEpisodeSource` raises a `DataConditionError` when the
window touches an unsolved frame. Its `generate` raises `NoAdmissibleTarget`, naming the
reason, when that one window has no attack.

## Analysis

| What you see | Why | Fix |
|---|---|---|
| `NoBenignRecords` | a fit that calibrates on benign records was given none | pass the train split unfiltered |
| `NoAttackedRecords` | a threshold tuned on validation records was given no attacked ones | pass a validation split that has both |
| `NotFitted` | a step that needs an earlier one was called first | call the earlier step |
| `SlackMismatch` | the dataset's slack bus is not the case's | use the dataset of that case |
| `VaryingReference` | the slack angle varies across the training frames | there is no one reference angle to fix |
| `MissingCapability` on `pmu_pseudo` | the file has no PMU currents | use a file generated with the hybrid meters |
| `NoSuchResult` | a results query found no record, or several | check the query's filters |
