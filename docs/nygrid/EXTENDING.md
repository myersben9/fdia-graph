# The New York grid: extending it

`fdia_graph.generation.nygrid` is built to take new parts without changes to its own code. This page names each extension
point, what it must provide, and a runnable example that implements it. The examples under `examples/nygrid/` need
no data, and CI runs each of them, so they stay current with the code. For how the stages fit together see
[the architecture](ARCHITECTURE.md), for one run end to end [the walkthrough](WALKTHROUGH.md).

## What you can add

| You add | It provides | It plugs into | Example |
|---|---|---|---|
| An attack family | `design(window, rng) -> Episode \| None` (subclass `AttackFamily`) | `TimelineBuilder` beside the package's families; `design_explained` reports why a window has none | [custom_attack_family.py](../../examples/nygrid/custom_attack_family.py) |
| An alarm rule | `first_alarm(stat) -> int` (the `AlarmRule` protocol) | anywhere `PersistenceRule` is used, the detection tools included | [custom_alarm_rule.py](../../examples/nygrid/custom_alarm_rule.py) |
| An operator | `respond(day, *, calib, alarm_at) -> Response` (the `Operator` protocol) | `summarize` and the detection tools, unchanged | [custom_operator.py](../../examples/nygrid/custom_operator.py) |
| A study | a job over attack windows | `fdia_graph.results.studies`: `study_days`, `run_parallel` | the tools under `tools/nygrid/` |

## An attack family

A family receives a window of true states, `(onset, vm [L, N], va [L, N])`, and returns an `Episode`: the attack
vector `a [L, M]` added to each scan, the buses it attacks (the localization labels) and the owners of the meters
it moves. Return None where the window holds nothing to attack, and set `self.last_miss = "no_target"` when that is
why; `design_explained` then reports `no_target`, and a None left unexplained reads as `infeasible`.

An attack the operator's residual test should not see has to change the readings as h(x_false) - h(x_true) for a
false state, as the package's `Attacker` does. A family that does not, like the example's slow bias, is the naive
attack the residual test is meant to catch, and is useful as a baseline.

## An alarm rule

A rule turns a statistic per scan (the residual sum of `wls`, a tie gap, any score) into the first scan it alarms
on, or -1. Calibrate it on benign days only: `PersistenceRule.fit` takes a percentile of benign statistics, and a
rule fitted on no statistic at all is an error (`NoBenignRecords`), since it would never alarm. Report its false
alarms on benign days of other months beside its detection rate, and on other meter fleets (`with_fleet`).

## An operator

An operator runs on an `OperatorDay` and returns a `Response`: its run on the attacked scans and on their benign
twin, each a `Twin` of per-scan arrays (dispatch, regulation, the imbalance the neighbours supply, the true interface
changes and margins, and `pf_failed` where the true grid's power flow failed). `summarize` reports the attack's
consequence as attacked less benign, so an operator only has to produce both runs the same way. The package's own,
`NyisoOperator(policy)`, takes its choices from an `OperatorPolicy`, which checks every choice when it is built.

## Changing the package itself

- The golden snapshots (`tests/nygrid/test_nygrid_golden.py`) pin what the operator, the detection tools and the
  interface attack return on fixed inputs. A change that should not move a number must pass them unchanged; one
  that should must rewrite them (`NYGRID_UPDATE_GOLDEN=1`) and say in the pull request which numbers moved and why.
- A setting is an argument; never change a library object's settings or a shared record's arrays after it is built.
- Raise the named errors of `fdia_graph.errors`: `ConfigError` for a wrong argument, a `DataConditionError` for data
  that does not meet a condition, `SolverUnavailable` for an environment that cannot run the solver.
- A new module is listed in [the architecture](ARCHITECTURE.md) in the same change, and a new public name in its
  package's `__init__.py`.
