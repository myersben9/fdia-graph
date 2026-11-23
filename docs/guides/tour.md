# Follow one attack through the code

This page follows one overload (`Am`) episode on IEEE-14 from the pool of operating points to the
frame in the file. Each step names the class and the method that does it, the file they are in,
and what they produced for this episode. Read it with the code open beside it.

The terms area, support, boundary, reach and snapshot are used as the
[Glossary](../reference/GLOSSARY.md) defines them. Paths are under `src/fdia_graph`.

## Run it

Everything below is printed by one script, `examples/tour.py`. It takes about a minute.

```bash
pip install -e ".[generate]"
python examples/tour.py
```

It generates a small timeline by the public call, then rebuilds the file's one episode by hand. The
timeline is registered as `tour` in the local cache and replaces a dataset of that name:

```python
fg.generate("ieee14", name="tour", frames=600, seed=123, families=("Am",), attacked_frac=0.2)
```

This is what it prints. Buses and branches are positions, counted from zero. On IEEE-14 the bus
number is the position plus one.

<!-- tour:start -->
```text
1. wrote a timeline of 600 frames, registered as 'tour'
2. case: 14 buses, 20 branches, slack at position 0
   pool: (600, 14, 4) = frames, buses, (|V|, P_inj, Q_inj, theta)
   meters: 89, PMUs at [3, 11], a flow meter on 18 of 20 branches
   free injections: [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13], of which generators [1, 2, 5, 7]
   zero-injection buses: [6]
3. episode 0: onset 92, length 60, split code 0
   target branches [10, 11], between buses [[8, 9], [8, 13]]
4. eligible branches: 18 of 20
   branch 10: true flow 6.43 to 6.21 MVA, goal 6.48 to 9.13 MVA, rating 9.13 MVA
   branch 11: true flow 9.00 to 9.13 MVA, goal 9.05 to 12.14 MVA, rating 12.14 MVA
5. area: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], boundary [0]
   the whole area costs SupportCost(devices=5, channels=8, size=13)
   support found: [9, 13], SupportCost(devices=3, channels=3, size=2)
   candidates solved: 256
   boundary of the support: [8, 10, 12], reach [8, 9, 10, 12, 13]
   the file recorded support [9, 13], devices, channels, solved [3, 3, 256]
6. snapshot 0: converged True, voltages moved at [9, 13]
   flow reached [6.48, 9.05] MVA, pretended load change in MW {9: 0.05, 13: 0.08}
6. snapshot 59: converged True, voltages moved at [9, 13]
   flow reached [9.13, 12.14] MVA, pretended load change in MW {9: 3.2, 13: 4.86}
7. last frame: labelled buses [9, 13], in the file [9, 13]
   node channels written: 12, at buses [8, 9, 10, 12, 13]
   flow channels written: 8, on branches [10, 11, 12, 14]
   PMU current channels written: 0
   equal to frame 151 of the file: True
8. loaded: Summary(system=14, N=14, E=20, n=600, families={'benign': 540, 'Am': 60})
   episode buses: [9, 13]
```
<!-- tour:end -->

The sections below take the numbered lines in order. `tests/test_tour.py` runs the script and
compares its output with the block above. That test runs only with `FDIA_SLOW=1`, and no scheduled
workflow runs it, so run it after a change to the attack code.

## 1. The source

`fg.generate` (`generation/__init__.py`) reads the pool, keeps its first 600 frames and calls
`timeline.generate_timeline`, which builds an `IeeeSource` (`generation/ieee.py`). `IeeeSource` is
a `SearchSource` (`generation/search.py`) on an IEEE case by its name. The one driver of every
timeline is `TimelineSource.generate` (`generation/source.py`): open the file, cut the splits, then
split by split place the episodes, design them and walk the frames.
[Generate a timeline](generation.md#the-steps) describes each of those steps.

The script builds the same source a second time by hand, with the same pool, settings and seed, so
that it can call the parts one by one.

## 2. The case, the pool and the meters

- **The case.** `Network.from_ieee(14)` (`systems/network.py`) holds the `Case` and its admittance
  matrices. `FdiaGenerator` (`engine/core.py`) reads it: 14 buses, 20 branches, the slack at
  position 0.
- **The pool.** `generation._load_states` returns the operating points as an array
  `[T, N, 4]`: voltage magnitude, active injection, reactive injection and angle per bus. Each
  frame is one solved AC state. Injections are positive for load (`NodeChannel`,
  `constants/channels.py`).
- **The meter plan.** `MeterPlan.from_coverage` (`sensing/meters.py`) draws the meters from the
  seed, at the coverage of `MeterSettings`, and places them with `MeterPlan.from_placement`. Here
  that is 89 meters: PMUs at positions 3 and 11, and
  a flow meter on 18 of the 20 branches. Only a PMU reads an angle [WU26 eqs. 17-20].
  [Meters and noise](../concepts/meters.md) has the rules.
- **Who can be moved.** `FalseState.free_injection_buses` (`engine/attacks/false_state.py`) lists
  the buses whose injection an overload attacker may change: the attackable loads, and the
  generators of `FalseState.generator_buses` [WU26 eqs. 13-14]. Position 6 injects nothing, so it
  is held at zero.

`SearchSource.__init__` also sets the line ratings before any frame is made
(`Overload.use_line_ratings`, `engine/attacks/overload.py`). By default a branch's rating is
`rating_margin` times its largest true flow over the pool the source was given, here the 600 frames.

## 3. The window and the targets

- **The split.** `TimelineSource.cut_splits` cuts the frames into train, validation and test in time
  order. Frame 92 is in the train split, code 0.
- **The onset.** `TimelineSource.place` calls `place_split` (`generation/plan.py`), which draws each
  episode's onset from the generator's random stream. With `attacked_frac=0.2` the train split gets
  one 60-frame episode and the two shorter splits none. The onset drawn is frame 92, so
  the window is frames 92 to 151.
- **The targets.** `SearchDesigner.task` (`generation/design.py`) builds the episode's task, and
  `Designer._design` hands the window's 60 states to `SearchOverload.designed`
  (`engine/attacks/search.py`), which calls
  `EpisodeDesigns.am_overload_design` (`engine/attacks/episodes.py`). It does three things:
    1. `Overload.eligible_lines` keeps the branches that are rated, flow metered, below their rating
       at every snapshot, have both ends at the grid's own buses, and have a free injection in the
       area around their ends. Here 18 qualify.
    2. It puts them in a random order, drawn from a stream keyed by this episode, and
       `Overload.target_sets` forms pairs: a branch with the first other branch whose two ends lie
       inside its area.
    3. It tries each pair in turn, at most `AM_LINE_TRIES` of them, and keeps the first whose search
       (step 5) moves at least one device beyond noise.

The pair kept is branches 10 and 11, which join position 8 to positions 9 and 13. The file records
it in `episodes/am_line`, and the script reads it from there.

## 4. The goal

`Overload.overload_goal` builds a `FlowGoal` (`models/frames.py`): the apparent flow each target
branch must show at each snapshot [WU26 eqs. 24-25]. The schedule is `_schedule` in
`engine/attacks/overload.py` [WU26 eq. 25]. It is the true flow at that snapshot, plus a share of the gap between
the last true flow and the rating. The share grows by one sixtieth per snapshot, so the last
snapshot reaches the rating.

The printed lines show it for both branches. On branch 10 the true flow is lower at the end of the
window than at its start, while the goal climbs to the rating.

## 5. The area, the support and the boundary

`Minimizer.min_tamper` (`engine/attacks/minimize.py`) chooses where the attacker acts
[WU26 eq. 12].

- **The area.** `Minimizer.window_area` asks `AttackArea.attack_area` (`engine/attacks/area.py`),
  which under the default `area_rule` is `AttackArea.hop_area`: the buses within `hops` branches of
  the target branches' ends, the slack left out. With two hops
  on a grid this small, the area is every bus but the slack, and its boundary is the slack alone.
- **The candidates.** The whole area is the first candidate. `Minimizer._supports` then yields
  smaller ones, smallest first: each starts from one end of each target branch and grows by one
  adjacent area bus at a time. A zero-injection bus on a candidate's boundary joins it, so a
  candidate can grow by more than one bus. A candidate must hold a free injection.
- **The cost.** `AttackWindow.cost` (`engine/attacks/window.py`) solves the false state of a
  candidate at all 60 snapshots and returns a `SupportCost`: the devices and the channels moved
  beyond [WU26]'s noise at some snapshot [WU26 eq. 28], and the candidate's size. A candidate has
  no cost when its solve fails at any snapshot, when it moves no device beyond noise, or when it
  cannot beat the best found so far.
- **The result.** The search solves at most `min_budget` candidates, 256 by default, and it used
  all of them here. The cheapest it found is the support of positions 9 and 13: 3 devices, against
  5 for the whole area. The search makes no claim that this is the minimum.

The boundary of that support is positions 8, 10 and 12: the buses one branch outside it. Their
voltages stay true. The reach is the support with its boundary, the five buses whose reported
injection the attack can change. The file records the support and the search's counts in the
`min_*` columns of `episodes/`, and they match.

## 6. The false state

For each frame, `FalseState.goal_state` (`engine/attacks/false_state.py`) solves the false state of
that snapshot's goal on the held support. For a flow goal it calls `FalseState.flow_goal_state`,
which calls `FalseState.solve_flow_local` and gets a `FlowSolve` (`models/frames.py`):

- **Unknowns.** The voltage magnitude and angle of the support's buses. Every other voltage is held
  at its true value. The printed line shows the voltage magnitude moving at positions 9 and 13
  only.
- **What must hold.** Each target branch shows its goal flow, by `formulas.network.local_flow_solve`.
  The free injections of the support change to make that so. Here both are loads, and
  `FlowSolve.load_change` is the load each one pretends.
- **Bounds.** A load shows a change of at most `load_cap` times its true load [YUA11]. A generator
  stays inside its limits and a voltage inside its limit [WU26 eqs. 21-23]. A value that would pass
  a bound is pinned there and the solve repeated.
- **The last check.** `flow_goal_state` then rejects a state that leaves the limits anywhere on
  the reach (`formulas.attacks.within_limits`).

The pretended change is a few hundredths of a megawatt at the first snapshot and a few megawatts at
the last, because the goal moves a little further from the truth at each snapshot.

## 7. The frame and the meters it moves

During the walk, `Walk.episode` (`generation/emit.py`) asks the episode for each frame.
`OverloadEpisode.frame` calls `Overload.overload_step`, which solves the false state as in step 6
and hands it to `AttackFrames.frame_from_state` (`engine/attacks/frames.py`):

1. `Emitter.emit_from_state` (`engine/measurement.py`) emits the true scan, with each meter's bias
   and this frame's noise. The noise is keyed by the frame's timestep, so the script sets
   `g.scan_key` to get the same draw.
2. `AttackFrames.attack_vector` is the noiseless reading of the false state minus that of the true
   state [WU26 eqs. 13-18].
3. The vector is added to the true scan on the node and flow meters it moves.
4. The labels `y` are set at the free injections of the support, positions 9 and 13.
5. `AttackFrames.currents_with_attack` does the same for the PMU currents, and the masks of every
   meter moved are returned as a `MeterLayers` (`models/frames.py`): node, flow and PMU current.

Two counts appear on this page and they measure different things. The search's 3 devices are those
moved beyond [WU26]'s noise somewhere in the window. The masks mark every stored reading the attack
changed in this frame, however little: 12 node channels across the reach, and both flow channels of
the four metered branches with an end in the support. No PMU current is written, because neither
PMU sits at an end of those branches.

The frame built this way equals frame 151 of the file, value for value.

## 8. The file

`TimelineWriter` (`generation/write.py`) stores each frame as the walk produces it, and after the
walk writes the tables. For this episode:

| In the file | Holds |
|---|---|
| `data/node_x`, `data/edge_x`, `data/y` | the observed readings and the labels, one row per frame |
| `benign/`, `clean/` | the same frames with the attack removed, and the noiseless truth |
| `attack/node_tamper`, `attack/edge_tamper`, `attack/pmu_i_tamper` | the masks of step 7 |
| `episodes/onset`, `length`, `family`, `bus_idx` | the episode's row |
| `episodes/am_line`, `am_rating_mva`, `am_target_mva`, `am_reached_mva`, `am_emitted_mva` | one row per target branch |
| `episodes/min_devices`, `min_channels`, `min_evaluated`, `min_support_idx` | what the search chose |

[The data dictionary](../reference/DATA_DICTIONARY.md#the-timeline-file) describes the datasets
every timeline holds, and [its episode tables](../reference/DATA_DICTIONARY.md#the-episode-tables) give the shape, type
and meaning of each `am_` and `min_` column.

`fg.load("tour", order="time")` reads the file back. `ds.episodes` is the episode table, and its
`buses` are the buses the episode labelled: positions 9 and 13, the support's free injections.

## Where next

- [The attack families](../concepts/attacks.md): what the overload and the ramp are, and why.
- [Formulas](../reference/FORMULAS.md): each [WU26] equation the code cites, and where.
- [Check the package yourself](verify.md): the tests behind the claims made here.
- The reading order of the attack code is at the top of `engine/attacks/__init__.py`.
