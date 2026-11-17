# The New York grid: extending it

`fdia_graph.generation.nygrid` is built to take new parts without changes to its own code. This page names each extension
point, what it must provide, and a runnable example that implements it. The examples under `examples/nygrid/` need
no data, and CI runs each of them, so they stay current with the code. For how the stages fit together see
[the architecture](../concepts/new_york_grid.md), for one run end to end [the walkthrough](../getting-started/new_york_walkthrough.md).

## What you can add

| You add | It provides | It plugs into | Example |
|---|---|---|---|
| An attack family | `design(window, rng) -> Episode \| None` (subclass `AttackFamily`) | `NewYorkSource` beside the package's families (`source.generate(path)` then places, designs and writes its episodes like theirs); `onset_mask` holds it to the onsets it may take; `design_explained` reports why a window has none | [custom_attack_family.py](../../examples/nygrid/custom_attack_family.py) |
| An alarm rule | `first_alarm(stat) -> int` (the `AlarmRule` protocol) | your own detector, in place of `PersistenceRule`: the package ships the rule and the protocol, and the code that computes the statistic and asks the rule is yours | [custom_alarm_rule.py](../../examples/nygrid/custom_alarm_rule.py) |

## An attack family

A family receives a window of true states, `(onset, vm [L, N], va [L, N])`, and returns an `Episode`: the attack
vector `a [L, M]` added to each scan, the buses it attacks (the localization labels) and the owners of the meters
it moves. Return None where the window holds nothing to attack, and set `self.last_miss = "no_target"` when that is
why; `design_explained` then reports `no_target`, and a None left unexplained reads as `infeasible`.

An attack the operator's residual test should not see has to change the readings as h(x_false) - h(x_true) for a
false state, as the package's `Attacker` does. A family that does not, like the example's slow bias, is the naive
attack the residual test is meant to catch, and is useful as a baseline.

## An alarm rule

A rule turns a statistic per scan (a residual sum, a gap between two meters, any score) into the first scan it
alarms on, or -1. Calibrate it on benign days only: `PersistenceRule.fit` takes a percentile of benign statistics,
and a rule fitted on no statistic at all is an error (`NoBenignRecords`), since it would never alarm. Report its
false alarms on benign days of other months beside its detection rate.

## Changing the package itself

- The grid is the MATPOWER case file `src/fdia_graph/generation/nygrid/case_nygrid.m`, read by `nygrid_case()`. To
  change a line, a unit or a limit, edit the file; code does not patch the case on the way in. Edit rows in place,
  or add and delete whole rows: `Case.from_matpower` reads the file with matpowercaseframes and refuses a file
  whose parts disagree, naming the file and what is wrong ([the architecture](../concepts/new_york_grid.md) lists
  the checks). A study on another grid passes its own `Case` (`Case.from_matpower` of its own file) wherever the
  package takes one; that file must follow this file's conventions: area 1 on every bus, and the zone codes
  of the legend in its header (1 to 11 the NYISO load zones, 12 to 15 the neighbors). A change to the file moves generated numbers, so it comes with the goldens and digests below.
- The golden snapshots (`tests/nygrid/test_golden.py`, and `analysis/tests/test_detect.py` for the
  persistence rule) pin what the persistence rule and the
  interface attack return on fixed inputs. A change that should not move a number must pass them unchanged; one
  that should must rewrite them (`NYGRID_UPDATE_GOLDEN=1`) and say in the pull request which numbers moved and why.
- The timeline digests (`tests/nygrid/test_timeline_digests.py`) pin every dataset and attribute of a built
  timeline and of two episodes the same way; they are rewritten with `FDIA_UPDATE_DIGESTS=1`.
- A setting is an argument; never change a library object's settings or a shared record's arrays after it is built.
- Raise the named errors of `fdia_graph.errors`: `ConfigError` for a wrong argument, a `DataConditionError` for data
  that does not meet a condition, `SolverUnavailable` for an environment that cannot run the solver.
- A new module is listed in [the architecture](../concepts/new_york_grid.md) in the same change. A new public name is
  imported from its defining module: a package's `__init__.py` holds no table of names.
