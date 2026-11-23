# The attack families

A timeline holds attacks so that a detector can be trained and scored on them. This page says
what an attack in a timeline is, which families the generator makes, and which class produces
each. It describes what the dataset contains. The equations are in
[Formulas](../reference/FORMULAS.md), each with its source.

Paths are under `src/fdia_graph/`.

## An episode

An attack is an episode: a run of consecutive frames under one design. An episode has a family,
an onset, a length and the buses it attacks. Episodes never overlap, so a frame has at most one
active family. Every frame of an episode carries the attack, except a frame whose scan cannot
be built: it is stored benign and counted in `fallback_benign`.

## The local false state

Both families the generator makes are built on one idea, the local false state of [WU26].

The attacker wants the operator to believe a state of the grid that is not the true one. A
detector that checks the readings against the grid's equations would catch readings that fit no
state. So the attacker picks a false state that the equations allow:

1. Choose a small area of the grid: the buses within a few branches of the targets. A ramp's
   targets are one neighbourhood: a load, and the eligible loads within the same few branches of it.
2. Change the loads inside the area (and, for an overload, the generator outputs).
3. Solve the AC equations of the area again, with every voltage outside it held true.
4. Add the difference of the readings to the true scan.

The attack vector is `a = h(x_false) - h(x_true)`, where `h` is the measurement function
([Meters and noise](meters.md)). It is zero on every meter the area does not reach. Each meter
keeps its usual noise. So the observed scan looks like a normal noisy scan of the false state,
and a residual test finds nothing wrong.

A false state must also be plausible. Inside the buses it re-solves it stays within the bus
voltage limits and the generator limits of the case [WU26, eqs. 21-23]. The buses one branch
outside take up the change in their reported injection, so a generator there is held to its limits
too: both families check the generators of the reach, the buses re-solved and the buses one branch
outside. A state that shows a generator outside its limits anywhere on the reach is rejected.

A bus reports one net injection, its load less its generation, and the model has no separate
telemetry of a unit's output. A change of a unit's implied output inside the injection meter's
error is therefore not visible. A limit counts as passed only when the implied output is outside it
by more than that meter can resolve: the meter's accuracy at the true reading, as the package
simulates it [ASP14], and never less than 0.001 MW. The tolerance follows the bus's reading, not the
unit's size. The slack and a static generator on a load bus have no limits in the case, so they are
never refused.

[WU26] leaves tampering below the noise out of its tamper count (Sec. V-A). Applying that idea to a
limit is this package's step, not the paper's.

The package has one noise, and four things read it: the readings of a file carry it, the ramp's
stealth bound is measured in it, the tamper count of both families is counted against it, and the
limit tolerance is taken from it. It is the meter's simulated accuracy at its true reading
([Meters and noise](meters.md#accuracy-classes)). The noise of [WU26]'s own case studies is flat:
0.03 p.u. on SCADA channels and 0.01 p.u. on PMU channels. Counting with the meter's class is this
package's reading of the paper's count. `MeterSettings.power_floor_pu` puts the paper's SCADA figure
under the power meters of a data set, and everything above follows it.

A bus with no load and no producing unit injects nothing, whether or not it has a shunt: a shunt is
part of the grid's admittances, not an injection. No attack shows an injection at such a bus. An
area that would have one on its boundary grows to take it in.

## The two families

| Family | Code | What the attacker does |
|---|---|---|
| `At`, the slow ramp | 5 | a false load that moves away from the true load a little each frame (up or down), holds, and returns to one step away from it |
| `Am`, the overload | 7 | the reported flow of a line rises to its rating |

**`At`** is hard to see because each step is small, so a detector that looks for jumps sees
none. Its ramp profile is described in [DAT26], and the published attack it follows is in
[the citation list](https://github.com/myersben9/fdia-graph#citation).

**`Am`** is the multi-snapshot overload attack of [WU26, eqs. 24-25]. Over the window the
reported flow of the target moves from its true value to its limit. The operator sees a line
about to overload that is not.

## The designs

A design decides what an episode changes on each frame. The package's designs are search-based,
and they run on every test system: the IEEE cases and the New York case.

| | `At` | `Am` |
|---|---|---|
| Family class | `SearchRamp` | `SearchOverload` |
| Targets | up to 5 neighbouring loads | one or two lines, at their ratings |
| False state from | a local AC solve in numpy | a local AC solve in numpy |
| Designed | as the walk reaches it | before its split is walked |

Every family answers the same question for the timeline: the episode of a window
(`EpisodeFamily.designed` in `engine/attacks/family.py`). That is why one path generates every
timeline ([Generate a timeline](../guides/generation.md)).

The designs read a scan's load and generation from its state. How a bus's injection divides
into the two is the setting `loads`
([How a pool's load is read](../guides/generation.md#how-a-pools-load-is-read)).

### The search-based designs

The designs are methods of `EpisodeDesigns`, the generator's `episode_designs`: `ramp_design` and
`am_overload_design` (`engine/attacks/episodes.py`). Settings are in `TimelineSettings`
([the tables](../guides/generation.md#timelinesettings)).

**The fewest-tamper search.** [WU26, eq. 12] asks for the attack that tampers the fewest
devices. The search tries supports, the sets of buses the false state moves: the whole area first, then
smaller ones, smallest first. It keeps the one whose attack changes the fewest devices beyond noise
(`engine/attacks/minimize.py`). It solves at most `min_budget` supports per search. A change a meter
cannot tell from its noise is not tampering, so a ramp's support that moves no device beyond noise
costs nothing and is the cheapest there is; the search records it with zero devices. An overload
must move at least one device, since a flow that stays inside the noise shows no overload.

**`At`.**

- The targets are up to 5 loads (`RampSettings.targets`) among the stealthy loads: attackable loads
  that are not on a generator bus [BOY22]. One is drawn at random and the others from the stealthy
  loads within `hops` branches of it, fewer when fewer are in reach, so a ramp acts in one
  neighbourhood (`draw_ramp`). An attackable load has nonzero active power, is not on the slack bus
  and is at most `max_load_mw`.
- One load factor moves all of them: it rises `ramp_rate` per frame, holds, and returns at the
  same rate. The rise lasts 20 % to 45 % of the episode and the hold 0 % to 25 % (`draw_ramp`).
  The factor never falls below one `ramp_rate`, so every frame of the episode carries an attack:
  after the return the episode stays one step away from the true load until it ends, and a ramp
  with a long rise can end before its return is complete. A ramp goes up or down with equal chance.
- A ramp is kept only on a support on which no metered channel moves by more than its rated
  accuracy times `stealth_scale` between frames, the first frame counted from the frame before the
  episode. With `min_tamper` (the default) that support is the one the search finds. With it off, it
  is the whole area, tested the same way.
- When no support keeps the bound, other targets are drawn, for up to 8 designs
  (`RAMP_SUPPORT_TRIES`). If none keeps it, the episode does not begin at that onset, and the walk
  moves it later or drops it as it does any episode that cannot begin. A ramp is never run past its
  own bound.
- A step with no local solution is halved, up to 6 times (`STEP_HALVINGS`).
- A design is tested on every frame it will occupy before it is accepted, up to 40 draws
  (`ONSET_DRAWS`).

**`Am`.**

- The targets are lines that are rated, flow-metered, below their rating at every snapshot
  of the window, and near a load or a generator the attacker can move (`eligible_lines`). Each
  episode overloads `n_lines` lines, two by default as in
  the case studies of [WU26].
- The ratings are 1.25 times each branch's peak true flow over the pool
  (`rating_source="pool"`). `rating_source="pglib"` uses the PGLib-OPF ratings instead, which
  exist for IEEE-14, 118 and 300; another case raises `NoLineRatings`. `rating_source="delta"`
  rates each line from its own window: its true flow at the window's last snapshot plus
  `rating_delta` (per unit of the system base).
- The goal of each snapshot is a linear share of the way from the true flow to the rating, and
  the last snapshot reaches the rating (`overload_goal`).
- The loads and generators of the support are free inside their limits. The load a bus shows
  changes by at most `load_cap` times its true load [YUA11].
- A device counts as tampered when a channel of it moves beyond its meter's noise, the count the
  ramp uses too. An overload is kept only when it moves at least one.
- Nothing bounds how far `Am` moves a channel between snapshots, as in [WU26].
- `support_method="rref"` picks the support by the row reduction of [WU26] instead of the
  search (`engine/attacks/rref.py`).

The overload episodes these settings build, measured with `families=("Am",)` and seed 1
(experiment `generation.am_overload`). Every line of every built episode reaches its rating.

<!-- results: gen.am -->
| system | recipe | frames | episodes | fallen back | devices (mean) | channels (mean) | largest change, pu (median / max episode) | s per episode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IEEE-14 | one line, without the load and generator bounds | 3000 | 25 | 0 | 5.3 | 15.0 | 0.18 / 5.74 | 15 |
| IEEE-14 | one line, with the load and generator bounds | 3000 | 25 | 0 | 3.9 | 9.9 | 0.14 / 0.82 | 30 |
| IEEE-14 | two lines, with the load and generator bounds (default) | 3000 | 25 | 0 | 6.8 | 19.3 | 0.17 / 1.59 | 39 |
| IEEE-30 | one line, without the load and generator bounds | 600 | 5 | 0 | 3.8 | 8.8 | 0.09 / 0.29 | 20 |
| IEEE-118 | one line, without the load and generator bounds | 2000 | 17 | 0 | 6.8 | 21.4 | 1.05 / 44.94 | 43 |
| IEEE-118 | one line, with the load and generator bounds | 2000 | 17 | 0 | 9.6 | 30.4 | 0.40 / 3.29 | 71 |
| IEEE-118 | two lines, with the load and generator bounds (default) | 2000 | 17 | 0 | 17.8 | 72.0 | 1.33 / 3.39 | 78 |
<!-- /results -->

### A family of your own

A source takes design families of your own, each in place of the search family of one name
([Generate a timeline](../guides/generation.md#a-design-family-of-your-own)). The simplest is a
subclass of `AttackFamily` (`engine/attacks/family.py`). It returns an `Episode`
(`models/plan.py`) from `design(window, rng)`, or `None` when the window holds no attack. The
window is the true state over the frames the episode would cover. The episode holds one attack
vector: a value for every meter on every frame. The walk adds each row to the true scan of its
frame (`VectorEpisode`).

A design that finds no attack can say why (`design_explained`): a reason the family sets in
`last_miss`, `solver_error` when the family counted a failed solve, else `infeasible`.
[Add a design family, an alarm rule or a test system](../guides/extending.md) has the steps.

## Labels and the tamper masks

| In the file | Shape | Meaning |
|---|---|---|
| `data/y` | `[T, N]` | 1 at each attacked bus |
| `data/family` | `[T]` | the family code; 0 is benign |
| `data/seq_id` | `[T]` | the episode's row; -1 is benign |
| `attack/node_tamper`, `attack/edge_tamper` | as the readings | 1 at a tampered meter |
| `attack/pmu_i_tamper` | as the currents | 1 at a tampered PMU current |
| `episodes/` | one row per episode | onset, length, family, buses |

What a 1 in a tamper mask means depends on the family that designed the episode.

| Family | A meter is tampered | A bus is attacked |
|---|---|---|
| `SearchRamp`, `SearchOverload` | where the attack changed the stored reading | at the buses the design names |
| a family of your own that designs an attack vector | where the vector moves the reading by more than the meter's accuracy class | where a tampered meter sits at the bus |

The file's attribute `tamper`, on `attack/`, states the rule of the package's own families, and
both rules when the source was given a family of your own.

In every file the observed scan minus the benign scan is the attack, exactly
([Layers and labels](layers.md)).

## What the released files additionally contain

The released files of data release v0.8.3 hold seven families. The current generator makes
two.

| Code | Family | In released files | Generated today |
|---|---|---|---|
| 1 | `Aq`, stealthy load scaling | yes | no |
| 2 | `Ad`, meter corruption | yes | no |
| 3 | `As`, meter scaling | yes | no |
| 4 | `Ar`, replay | yes | no |
| 5 | `At`, slow ramp | yes | yes |
| 6 | `Al`, load redistribution | yes | no |
| 7 | `Am` | yes, as a load redistribution | yes, as the overload |

The older families are read-only. Their codes stay in the label table `fg.FAMILIES`, and the
files that carry them load, filter and score by name:

```python
import fdia_graph as fg

old = fg.load("ieee14", release="v0.8.3", families=["Aq"])
```

The generator refuses them. In those files `Aq`, `Al`, `Ad`, `As` and `Ar` are one frame each,
and the `Am` episodes drive no line to its limit. Those files also have an angle reading at
every voltmeter bus and no PMU currents ([Meters and noise](meters.md)).
`fg.STEALTHY_FAMILIES` is the set of codes that are local false states: `Aq`, `At`, `Al` and
`Am`.

## Where the code is

| File | Holds |
|---|---|
| `engine/attacks/family.py` | `EpisodeFamily`, `AttackFamily`, `DesignedEpisode`, `VectorEpisode`: what every family answers |
| `engine/attacks/search.py` | `SearchRamp`, `SearchOverload` and their episodes |
| `engine/attacks/episodes.py` | `ramp_design`, `am_overload_design` |
| `engine/attacks/area.py` | the attacker's area and its boundary |
| `engine/attacks/false_state.py` | the local false state: the area re-solved |
| `engine/attacks/frames.py` | the attack vector of a false state, written onto the true scan |
| `engine/attacks/ramp_frames.py` | one scan of `At`: a false state added to the true scan |
| `engine/attacks/overload.py` | the overload goal, the ratings, the eligible lines |
| `engine/attacks/minimize.py` | the fewest-tamper search |
| `engine/attacks/blas.py` | one BLAS thread for the length of a search |
| `engine/attacks/window.py` | one attack window, and what a support costs over it |
| `engine/attacks/rref.py` | the support by row reduction |

To add a family of your own, see
[Add a design family, an alarm rule or a test system](../guides/extending.md).
