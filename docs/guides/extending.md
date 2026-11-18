# Add a design family, an alarm rule or a test system

The package has three extension points. Each is a small base class or constructor of the
package itself, not of one test system. This guide names each one, says what it must provide,
and where it plugs in today.

| You add | You write | Base |
|---|---|---|
| A design family | `design(window, rng)` | `AttackFamily` |
| An alarm rule | `first_alarm(stat)` | `AlarmRule` |
| A test system | a `Case` | `Case.from_pandapower`, `Case.from_matpower` |

Two runnable examples are in the repository under `examples/nygrid/`. They need no data, and the
tests run both, so they stay current with the code.

## A design family

The design stage asks every family one thing: the episode of a window
(`EpisodeFamily.designed` in `src/fdia_graph/engine/attacks/family.py`). Subclass
`AttackFamily`, which answers that for you, and write `design`:

```python
import numpy as np

from fdia_graph.engine.attacks.family import AttackFamily
from fdia_graph.models.plan import Episode


class MyFamily(AttackFamily):
    def design(self, window, rng):
        onset, vm, va = window
        ...
        return Episode(...)  # or None
```

**What `design` receives.** `window` is `(onset, vm, va)`: the pool frame the window starts at,
and the true voltage magnitudes (p.u.) and angles (degrees) over it, each `[L, N]`. `rng` is the random
generator for this design.

**What it returns.** An `Episode` (`models/plan.py`), or `None` where the window holds nothing
to attack.

| Field of `Episode` | Meaning |
|---|---|
| `family` | the family's name; it must be a name in `fg.FAMILIES`, whose code the file stores |
| `onset`, `length` | the window |
| `a` | the attack vector `[L, M]`, added to each scan |
| `buses` | the buses it attacks |
| `clients` | the owners of the meters it moves |
| `target_branch` | the branch it is anchored on; -1 for none |

**Saying why there is no attack.** When `design` returns `None`, set
`self.last_miss = "no_target"` if the window held nothing to attack. `design_explained` then
reports `no_target`. A `None` left unexplained reads as `infeasible`.

**Limiting the onsets.** Override `onset_mask(vm, va, length)` to hold the family to the frames
it may start at. It receives the states of the whole pool, `[T, N]`, and returns one flag per
frame. The default allows any frame.

**Keeping it stealthy.** An attack the residual test should not see has to change the readings
as `h(x_false) - h(x_true)` for a false state, as the package's `Attacker` does
([The attack families](../concepts/attacks.md#the-local-false-state)). A family that does not,
like the example's slow bias, is the naive attack the residual test is meant to catch. It is
useful as a baseline.

**Where it plugs in today.** `NewYorkSource` takes its families as a dictionary, name to
family, beside the package's own. `source.generate(path)` then places, designs and writes its
episodes like theirs. The dictionary key may be any name, but `Episode.family` must be a name
in `fg.FAMILIES`. A `SearchSource` builds its two families itself and takes none from the
caller.

Example:
[custom_attack_family.py](https://github.com/myersben9/fdia-graph/blob/main/examples/nygrid/custom_attack_family.py).

## An alarm rule

A rule turns a statistic per scan into the first scan it alarms on, or -1. The statistic can be
a residual sum, a gap between two meters, or any score. `AlarmRule` is a protocol in the
analysis toolkit, in `se/detect/rules.py` under `analysis/src/fdia_analysis/`. Any object with
this method is a rule:

```python
import numpy as np

from fdia_analysis.se.detect.rules import AlarmRule, PersistenceRule


class FirstOver:
    """Alarm at the first scan above a fixed level."""

    def __init__(self, level: float) -> None:
        self.level = level

    def first_alarm(self, stat: np.ndarray) -> int:
        hit = np.flatnonzero(np.asarray(stat) > self.level)
        return int(hit[0]) if len(hit) else -1


rule = FirstOver(3.0)
print(isinstance(rule, AlarmRule))          # True
print(rule.first_alarm([1.0, 2.0, 4.0]))    # 2
```

The package ships one rule, `PersistenceRule`: it alarms when `k` of the last `n` statistics
exceed a threshold (2 of 3 by default), which a single noisy scan cannot trip.

```python
benign = np.random.default_rng(0).normal(size=2000)
rule = PersistenceRule.fit(benign, q=99.9)
print(rule.first_alarm(benign[:300]))  # -1
```

- Calibrate a rule on benign scans only. `PersistenceRule.fit` takes a percentile of benign
  statistics.
- A rule fitted on no statistic at all raises `NoBenignRecords`, since it would never alarm.
- Report a rule's false alarms on benign data beside its detection rate.

The package ships the rule and the protocol. The code that computes the statistic and asks the
rule is yours. `se/detect/__init__.py` exports no name: import from `rules.py`.

Example:
[custom_alarm_rule.py](https://github.com/myersben9/fdia-graph/blob/main/examples/nygrid/custom_alarm_rule.py).

## A test system

A test system is a `Case` ([Test systems](../concepts/test_systems.md)). There are two ways to
make one of your own.

| Call | Takes | Note |
|---|---|---|
| `Case.from_pandapower(net)` | a pandapower net with its power flow solved | the general route |
| `Case.from_matpower(path)` | a MATPOWER case file | needs the `[ny]` extra; the file must follow the New York file's conventions |

`Case.from_matpower` reads a file laid out like `case_nygrid.m`: area 1 on every bus, and the
zone codes of that file's legend (1 to 11 the NYISO load zones, 12 to 15 the neighbors). A file
whose parts disagree is refused with a `DataConditionError` that names the file and what is
wrong.

A `SearchSource` generates a timeline on any `Network`:

```python
import numpy as np
import pandapower as pp
import pandapower.networks as pn

import fdia_graph as fg
from fdia_graph.data.profiles import RawSeries
from fdia_graph.generation.search import SearchSource
from fdia_graph.models.config import TimelineSettings
from fdia_graph.systems.case import Case
from fdia_graph.systems.network import Network

net = pn.case14()
pp.runpp(net)
grid = Network(Case.from_pandapower(net))

# a pool for this grid: [T, N, 4], one solved state per frame
hours = np.arange(300) / 60.0
profile = fg.load_profile(RawSeries(100 + 10 * np.sin(hours)))
pool = fg.generate_states("ieee14", profile)

settings = TimelineSettings.of(
    families=("At",), ramp_len=20, min_tamper=False
)
source = SearchSource(grid, pool, settings, seed=1)
path = source.generate("own_grid.h5")
print(fg.FdiaGraph(path).summary())
```

The example borrows the IEEE-14 pool builder, because `pn.case14()` is that grid. For a grid of
your own you bring the pool. Two things hold for it today:

- `fg.generate_states` builds pools for the eight IEEE cases only.
- The search-based designs read a scan's load and generation from its state by the rule of the
  IEEE pools: each bus's load and generation scaled by one factor, load positive. A pool built
  another way does not fit them.

One source generates one timeline. A second `generate` on the same `SearchSource` is refused;
build a new source.

To change the New York grid itself, edit its case file
([Test systems](../concepts/test_systems.md#the-new-york-case-file)). Edit rows in place, or add
and delete whole rows. A variant in code is a copy, never a change in place:
`hvdc_links(case)`, `externals_at_base(case, base)`, `case.with_voltage_margin(0.02)`.

## What an extension should keep to

- **Named errors.** Raise the errors of `fdia_graph.errors`: `ConfigError` for a wrong
  argument, a `DataConditionError` for data that does not meet a condition, `SolverUnavailable`
  for an environment that cannot run the solver.
- **Settings are arguments.** A setting is an argument of the function or constructor that uses
  it. A variant is a copy, such as `InterfaceFdia.with_overshoot(mw)`.

Changing the package itself is covered in
[Contributing](https://github.com/myersben9/fdia-graph/blob/main/CONTRIBUTING.md).
