# The New York grid: how it fails

Every stage of `fdia_graph.generation.nygrid` either raises a named error or records why it produced less than asked. This
page lists what each stage reports, what it means and what to do. The stages are those of
[the guide](../guides/new_york.md): environment, inputs, the operating-point pool, the timeline, the operator and
the detection tools. The errors live in `fdia_graph.errors`. A wrong argument is a `ConfigError` and data that
does not meet a condition is a `DataConditionError`. Both are a `ValueError`, so existing `except ValueError`
code keeps working. `SolverUnavailable` is a `RuntimeError`, because a broken environment is not a property of
the data.

## Environment

| What you see | Why | Fix |
|---|---|---|
| `SolverUnavailable: Ipopt is not found or does not start` | Ipopt is not installed, or it is installed but cannot load its DLLs. On Windows a missing DLL shows as exit code `0xC0000135` | `conda install -c conda-forge ipopt` into the environment, and run that environment's Python with the environment active, so its `Library/bin` folder is on PATH. `fdia_graph.engine.solvers.ipopt()` adds the folder when it exists |
| `SolverUnavailable: Pyomo is not installed` | The `[ny]` extra is missing | `pip install "fdia-graph[ny]"` |
| A change you made does not show up | Python imports another copy of the package, usually the editable install of the main checkout when you work in a git worktree | Put the worktree's `src` first: `PYTHONPATH=<worktree>/src` |
| `FileNotFoundError` on `data/...` | The build tools and the operator read `data/` and `out/` relative to the working folder | Run from the work folder that holds `data/`, or pass `--data` and `--pool` |

## Inputs

| What you see | Why | Fix |
|---|---|---|
| `DataConditionError: bus ... has no row in ...` | A bus of the case has no location in the bus-location table | Re-fetch the inputs (`build_timeline.py --fetch`). The table ships with the package |
| `DataConditionError: ...: branch f-t is not in the case in that orientation` | An interface definition names a branch the case does not hold | The case or the interface table changed; check both against the NYgrid commit the guide pins |
| `DataConditionError: ...: no rows in interflowHourly_<year>.csv` | No interface flows for that year | Fetch the inputs for the year you build |

## The operating-point pool

`PoolBuilder.report()` counts every frame by its solve kind. Only `solved` frames are used downstream.

| Solve kind | Meaning | What to do |
|---|---|---|
| `solved` | The dispatch and the AC power flow converged | Nothing |
| `failed` | The dispatch or the power flow did not converge for that frame | A few per year are expected (the 2019 pool solves 99.8 percent). Many in a row point at the inputs of those hours |
| `no_hour` | The hour a chunk starts from has no AC point, so the chunk has no start | Usually follows a `failed` hour; rebuild that span |
| `no_load` | NYISO has no load for the frame (the spring-forward gap and data gaps) | Nothing; these frames are skipped everywhere |

`report()` before `build()` raises `NotFitted`. A span whose first hour has no AC point raises a
`DataConditionError` naming the frame.

## The timeline

`TimelineBuilder.build()` returns a summary. When it built fewer attacks than it placed, `missed` (and, with the
onset survey on, `survey_missed`) says why, counted per family:

| Reason | Meaning | What to do |
|---|---|---|
| `no_target` | The window has nothing to attack: no line or interface near its limit, or no load bus for a ramp | Expected outside the near-limit hours. If it covers every slot, the onset mask and the interface data disagree |
| `infeasible` | Targets existed, but no support met every bound of the attacker (load cap, basepoint band, ramps, voltage limits) | The bounds are binding. Widening them is a modeling decision, not a fix |
| `solver_error` | Ipopt started but stopped abnormally on a support the search tried, so the window may hold an attack the search did not see | A few are tolerable. Many mean a numerically hard case; the family's `solver_errors` counts them |

`build_episode` raises `NoAdmissibleTarget` naming the reason when its one window has no attack, and a
`DataConditionError` when the window touches an unsolved frame.

## The operator

`respond` checks its choices before anything runs: an unknown `mode`, `after` or `entry` raises a `ConfigError`
listing the allowed values. In the result:

| What you see | Meaning |
|---|---|
| `pf_failed_scans` above zero in `summarize` (or `Twin.pf_failed`) | The AC power flow of the operator's dispatch on the true grid failed on those scans. Their interface changes and margin are NaN, which means unknown, not safe. They are not counted in `over_scans` |
| `margin_min_mw` below zero with `over_scans` zero | The benign twin is over the limit as far, or further. The least-cost dispatch holds a binding interface at its shift-factor limit, and the true AC flow can pass it with no attack. `benign_margin_min_mw` gives the benign twin's tightest margin, and `over_*` count only the attack's excess |
| `alarm_scans` NaN | The interchange check never fired in the window |
| `RuntimeError: scan t: ...` from the dispatch | The operator's dispatch linear program failed on that scan |

## The detection tools

The research tools under `tools/nygrid/` show every warning; none is silenced. Their detection thresholds come
from the calibration files under `results/`.

## The golden snapshots

`tests/nygrid/test_nygrid_golden.py` compares what the operator, the detection tools and the interface attack return
on fixed inputs with the snapshots in `tests/nygrid/golden/`. A failure names the first value that changed. If the
change is intended, rewrite the snapshots with `NYGRID_UPDATE_GOLDEN=1` and say in the pull request which numbers
moved and why. The operator and detection snapshots run on the attack days checked in under
`tests/nygrid/data/days` with NYgrid's inputs, so CI runs them; the interface attack's design also needs the whole
2019 pool under `NYGRID_ROOT` and skips without it.
