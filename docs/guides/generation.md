# Generate a timeline

A timeline is the meter readings of one test system, frame by frame, with attack episodes placed
in it. This guide generates one. It is the same guide for every test system: one method makes
every timeline, and what differs between systems is settings and data.

Terms are in the [glossary](../reference/GLOSSARY.md). Paths are under `src/fdia_graph/`.

## What you need

- **A test system.** One of the eight IEEE cases, or the New York case
  ([Test systems](../concepts/test_systems.md)).
- **A pool of operating points.** One solved AC state per frame, in time order
  ([Build the operating points](operating_points.md)).

For an IEEE case `fg.generate` downloads the released pool, so you need nothing else. For the
New York case you build the pool first.

## The steps

`TimelineSource.generate(out)` in `generation/source.py` makes every timeline. Its body is short,
and it runs these steps in this order.

![Eight boxes, top to bottom. A test system and its pool: one AC state per frame. 1. Open: the file, under a .part name. 2. Cut the splits: blocks in time order, or whole weeks. 3. Place: a split's episodes, whole, at random onsets. 4. Design: each episode. 5. Walk: the frames, in time order; an arrow labelled next split goes back to Place. 6. Finish: the episode table. Last, one timeline file: observed, benign, clean.](../figures/diagrams/generation_flow.png)

A source is a subclass of `TimelineSource`. It builds the generator, the pool and the designer
of its attack families, and it says what its files record. There are two.

| Source | File | Built on |
|---|---|---|
| `SearchSource` | `generation/search.py` | a `Network` and a pool array |
| `NewYorkSource` | `generation/newyork.py` | the New York pool file |

`IeeeSource` (`generation/ieee.py`) is a `SearchSource` for an IEEE case by its name.

### 1. Open

Owner: `TimelineSource.open`, with `TimelineWriter` (`generation/write.py`).

The file is started beside `out` under a `.part` name. It takes the name `out` only when it is
whole. A run that fails leaves no file, and a file already at `out` is left as it was.
The writer stores the static graph first: the branches and what gives the admittance matrix.

### 2. Cut the splits

Owner: `TimelineSource.cut_splits`, with `split_bounds` and `split_weeks` in
`generation/plan.py`.

The train, validation and test splits are cut before any episode is placed, so no episode is cut
by a split. The setting `split_by` chooses how.

- `"time"`: three blocks in time order.
- `"week"`: whole weeks, drawn into the three splits. This needs each frame's time. The weeks
  are ranked by how many allowed onsets they hold and drawn in strata of 20 weeks
  (`split_weeks`). Where no family limits its onsets, as on a `SearchSource`, the order is
  random.

Just before the cut the source runs `prepare`. It does nothing on a `SearchSource`. A
`NewYorkSource` finds there the onsets its families may take (see
[the New York settings](#the-new-york-settings)).

### 3. Place

Owner: `TimelineSource.place`, with `place_split` in `generation/plan.py`.

Each split gets whole episodes at random onsets.

- **How many.** The whole number of episodes whose frames come closest to `attacked_frac` of the
  split. The families share them so that each family gets about the same number of attacked
  frames (`whole_episode_counts`, `largest_remainder`).
- **Where.** Longest first, each at an onset drawn uniformly among the positions where it fits.
  Episodes never overlap and none is cut (`uniform_onset`).
- **When it jams.** A placement that cannot fit is retried whole 10 times (`PLACE_TRIES`). Then
  one episode of the family that jammed is dropped, recorded, and reported with a
  `RuntimeWarning`.

Adjacent episodes and long quiet stretches are outcomes of the uniform draw, not of a rule.

### 4. Design

Owner: `TimelineSource.design`, with `SplitDesign` and the source's `Designer`
(`generation/design.py`).

The designer asks each family for the episode of a window: what the attacker changes on each
frame ([The attack families](../concepts/attacks.md)). An episode with no design at its onset
moves to another free onset of its split, up to 20 times (`REDRAWS`). After that it is given up,
recorded, and reported with a `RuntimeWarning`.

On a `SearchSource` the ramp `At` is the one family designed as the walk reaches it, because
its design reads the frame before its onset. A ramp with no design therefore moves to a later
onset only (`Walk.relocate_later`).

The designs of a split can run in several processes (`workers`). The file does not depend on the
number of workers. The processes are spawned, so a script that asks for more than one worker
must start under `if __name__ == "__main__":`.

### 5. Walk

Owner: `TimelineSource.walk`, with `Walk.run` in `generation/emit.py`.

Every frame is emitted in time order. A frame outside an episode is a benign scan: the readings
of the pool state plus meter noise ([Meters and noise](../concepts/meters.md)). A frame inside
an episode is that scan plus the attack. A frame of an episode whose own scan cannot be built is
stored benign, and the file counts it in the attribute `fallback_benign`.

The walk writes a frame only after its split is placed and designed (`planned_until` in
`generation/plan.py`). With `split_by="time"` each split is one block, so the walk writes that
block next. With `split_by="week"` the weeks of the three splits alternate, so the walk stops
at the first frame whose split is not designed yet.

### 6. Finish

Owner: `TimelineSource.finish`, with `TimelineWriter.finish`.

After the last frame the writer stores what the file carries besides its frames: the per-frame
columns, the episode table, the attributes, and the temporal features where the file has them.
The attributes record the settings as JSON (`settings`) with their hash (`settings_hash`), and
per split and family the episodes requested, built, moved, dropped and short.
[Layers and labels](../concepts/layers.md) says what each layer means, and
[the data dictionary](../reference/DATA_DICTIONARY.md) lists every array.

## The settings

Settings arrive as models from `models/config.py`. Each model checks its values when it is
built, and a value it does not allow raises `ConfigError`.

### `TimelineSettings`

The settings of a `SearchSource`, so of every IEEE timeline.

| Field | Default | Meaning |
|---|---|---|
| `attacked_frac` | 0.5 | share of each split's frames under an episode |
| `families` | `("At", "Am")` | the families placed |
| `am_len` | `None` | `Am` episode length in frames; `None` is the ramp's |
| `max_load_mw` | 2000.0 | a larger load is never a target; `None` is off |
| `workers` | 1 | processes that design the `Am` episodes |
| `split_by` | `"time"` | how the splits are cut |
| `am_attack` | `"overload"` | what an `Am` episode is; the one kind |
| `ramp` | `RampSettings()` | the slow ramp `At` |
| `search` | `SearchSettings()` | the attacker's area and the search |
| `overload` | `OverloadSettings()` | the overload `Am` |
| `meters` | `MeterSettings()` | the meter plan |
| `split` | `SplitSettings()` | the three fractions |

The sections:

| Model | Field | Default | Meaning |
|---|---|---|---|
| `RampSettings` | `rate` | 0.002 | `At` growth per frame |
| `RampSettings` | `length` | 60 | `At` episode length in frames |
| `RampSettings` | `stealth_scale` | 1.0 | multiplier on `At`'s stealth bound |
| `SearchSettings` | `hops` | 2 | the attacker's reach in branches |
| `SearchSettings` | `min_tamper` | `True` | hold each `At` episode on the fewest devices |
| `SearchSettings` | `min_budget` | 256 | candidate supports solved per search |
| `SearchSettings` | `area_rule` | `"hops"` | how the attacker's area is chosen |
| `OverloadSettings` | `rating_source` | `"pool"` | `"pool"`, `"pglib"` or `"delta"` |
| `OverloadSettings` | `rating_margin` | 1.25 | rating over the pool's peak flow |
| `OverloadSettings` | `load_cap` | 0.5 | largest shown load change, as a share |
| `OverloadSettings` | `n_lines` | 2 | lines one episode overloads |
| `OverloadSettings` | `support_method` | `"search"` | `"search"` or `"rref"` |
| `OverloadSettings` | `rating_delta` | 0.10 | with `"delta"`: p.u. added to the target's true flow |
| `MeterSettings` | `vbus_frac` | 0.6 | share of buses with a voltmeter |
| `MeterSettings` | `pmu_frac` | 0.2 | share of buses with a PMU |
| `MeterSettings` | `flow_frac` | 0.9 | share of branches with a flow meter |
| `MeterSettings` | `meter_model` | `"hybrid"` | what the meters read |
| `SplitSettings` | `fractions` | `(0.6, 0.2, 0.2)` | train, validation, test |

`fg.generate` and `generate_timeline` take a `TimelineSettings`, and also flat keywords that set
one field each. A flat keyword is applied over the settings.

| Flat keyword | Sets |
|---|---|
| `attacked_frac`, `families`, `am_len`, `max_load_mw`, `workers`, `split_by` | the field of that name |
| `ramp_rate`, `ramp_len`, `stealth_scale` | `ramp.rate`, `ramp.length`, `ramp.stealth_scale` |
| `hops`, `min_tamper`, `min_budget` | the `search` field of that name |
| `redundancy` | `meters` (a `MeterSettings`) |
| `split` | `split` (three fractions, or a `SplitSettings`) |
| `am_attack` | `"overload"` sets `am_attack`; an `OverloadSettings` sets `overload` |

`seed` (default 123) is an argument beside the settings. It fixes the meter plan, the meter
biases and every attack draw.

```python
from fdia_graph.models.config import RampSettings, TimelineSettings

settings = TimelineSettings(
    attacked_frac=0.3,
    ramp=RampSettings(length=30),
)
print(settings.lengths_for(("At", "Am")))  # {'At': 30, 'Am': 30}
```

### `split_by` and `time=`

`split_by` is a setting of every source.

| Value | The splits | Needs |
|---|---|---|
| `"time"` | three blocks in time order | nothing |
| `"week"` | whole weeks, drawn into the three splits | each frame's time |

The frames' times are the argument `time=`: numpy `datetime64` values, one per frame, strictly
increasing. Anything else is refused when the source is built. An IEEE pool stores no time, so
an IEEE timeline split by weeks needs `time=` passed in. A New York pool stores its times.

With times, the file stores `data/time`, and no episode is placed across a gap in time. Without
them, `split_by="week"` is refused with a `ConfigError` that says the pool carries no time.

### The New York settings

`NewYorkSettings` holds the settings of a `NewYorkSource`.

| Field | Default | Meaning |
|---|---|---|
| `length` | 16 | frames per episode, or `{family: frames}` |
| `attacked_frac` | 0.2 | share of each split's frames under an episode |
| `split` | `SplitSettings((0.7, 0.15, 0.15))` | train, validation, test |
| `split_by` | `"week"` | how the splits are cut |
| `workers` | 1 | processes that design the episodes |
| `verbose` | `False` | print progress |
| `survey` | 0 | frames between surveyed onsets; 0 is off |
| `survey_cache` | `None` | folder that keeps the survey by chunk |

The overload family of the New York case waits for a loaded grid: it may start only where an
interface is near its limit. `prepare` asks each family for its `onset_mask`. With `survey` on,
it also designs every allowed onset on that grid of frames first, and then places episodes only
at onsets where an attack was found. A survey saved under `survey_cache` lets a stopped build
resume.

`NewYorkRelease` holds what `NewYorkSource.production` builds its source from.

| Field | Default | Meaning |
|---|---|---|
| `families` | `("Am", "At")` | the families built |
| `interfaces` | `("SPR/DUN-SOUTH",)` | the interfaces `Am` targets |
| `workers` | 12 | processes that design the episodes |
| `survey` | 12 | the survey's grid in frames; 0 is off |
| `survey_cache` | `None` | folder that keeps the survey by chunk |
| `root` | `"."` | folder whose `data/` holds the inputs |
| `hold_external` | `False` | keep every bus outside New York true |
| `v_margin` | 0.0 | widens the attacker's voltage limits (p.u.) |
| `gen_v_tol` | `None` | holds generator voltages near schedule (p.u.) |
| `load_cap` | `None` | cap on a shown load change; `None` keeps 0.1 |
| `band` | `None` | basepoint band's share; `None` keeps 0.03 |

`production` sets the episode lengths itself: 32 frames for `Am` and 16 for `At`.

## What differs between the IEEE cases and the New York case today

Every row is a difference in settings or in data. The steps above are the same.

| Item | IEEE cases | New York case |
|---|---|---|
| Case | `Case.from_ieee(n)`, the case pandapower ships | `nygrid_case()`, a MATPOWER file in the package |
| Load data | one system load profile scales every bus | a load per bus, from `LoadModel` |
| Pool solve | one power flow per step, in pandapower | one OPF per frame in Ipopt, then one power flow |
| A step that does not solve | dropped from the pool | kept in place, with an empty state |
| Frame times | not stored in the pool | stored in the pool |
| Meter placement | `MeterPlan.from_coverage`: random shares | `plan_meters`: every bus and branch end a utility owns |
| Design family | `SearchRamp`, `SearchOverload` | `Ramp`, `InterfaceFdia` |
| Episode lengths | 60 frames, both families | 16 frames; `NewYorkSource.production` uses 32 for `Am` |
| Attacked share | 0.5 | 0.2 |
| Split default | `"time"`, 0.6 / 0.2 / 0.2 | `"week"`, 0.7 / 0.15 / 0.15 |
| Allowed onsets | any onset of the split | `Am` only where an interface is near its limit |
| Temporal features | `temporal_delta` and `swing` are written | not written |
| Entry point | `fg.generate` | `pipelines/newyork/build_timeline.py` |
| Released data | a pool and a timeline per system | none; you build both |

Two of these rows are tied together today. The search-based designs read the meters that
`MeterPlan.from_coverage` places, and the New York designs read the meters that `plan_meters`
places. So the meter placement and the design family both follow the source. Neither is a
setting of its own. [The attack families](../concepts/attacks.md) says what each design does.

## Entry point: an IEEE case

`fg.generate` loads the pool, generates the timeline and registers the file under a name.

```python
import fdia_graph as fg

# needs fdia-graph[generate]
# downloads the IEEE-14 pool (about 25 MB) once
# about five minutes on a laptop
fg.generate("ieee14", name="my_run", frames=2000)
mine = fg.load("my_run", split="train")
```

| Argument | Default | Meaning |
|---|---|---|
| `system` | | `"ieee14"` to `"ieee300"`, or the bus count |
| `name` | | the name `fg.load` finds the file under |
| `frames` | `None` | walk only the first frames of the pool |
| `states` | `None` | your own pool: an array or a path |
| `out` | `None` | the file's path; `None` is `<name>.h5` in the cache |
| `seed` | 123 | the seed |
| `settings` | `None` | a `TimelineSettings` |
| `time` | `None` | each frame's time |

Without `states`, the pool is the path in `$FDIA_GRAPH_INIT` when that variable is set, else the
released pool of the system, downloaded once.

`fg.generate` calls `generate_timeline` in `timeline.py`, which builds an `IeeeSource` and calls
its `generate`. To generate on a grid that is not an IEEE case, build a `SearchSource` on its
`Network` ([Add a design family, an alarm rule or a test system](extending.md)).

Two measured builds on IEEE-14 follow. The first is the three release recipes at 72,000 frames
(seed 123, experiment `generation.split_first`; episodes and redraws per family, `At / Am`).

<!-- results: gen.split_first -->
| dataset | split | frames | attacked | episodes asked | built | redraws | shortfall |
|---|---|---:|---:|---:|---:|---:|---:|
| At | train | 43200 | 0.500 | 360 | 360 | 0 | 0 |
| At | val | 14400 | 0.500 | 120 | 120 | 0 | 0 |
| At | test | 14400 | 0.500 | 120 | 120 | 0 | 0 |
| At | build |  |  |  |  |  | 10 min |
| Am | train | 43200 | 0.500 | 360 | 360 | 88 | 0 |
| Am | val | 14400 | 0.496 | 120 | 119 | 32 | 1 |
| Am | test | 14400 | 0.500 | 120 | 120 | 26 | 0 |
| Am | build |  |  |  |  |  | 413 min |
| At+Am | train | 43200 | 0.500 | 180 / 180 | 180 / 180 | 0 / 32 | 0 / 0 |
| At+Am | val | 14400 | 0.500 | 60 / 60 | 60 / 60 | 0 / 18 | 0 / 0 |
| At+Am | test | 14400 | 0.500 | 60 / 60 | 60 / 60 | 0 / 15 | 0 / 0 |
| At+Am | build |  |  |  |  |  | 234 min |
<!-- /results -->

The overload `Am` takes most of the time. A split's `Am` episodes are independent, so
`workers=N` designs them in N processes and the file is the same for any N. An `Am`-only IEEE-14
timeline (experiment `generation.parallel_design`):

<!-- results: gen.workers -->
| workers (3000 frames) | seconds | speed-up | Am episodes | same file |
|---|---:|---:|---:|---:|
| 1 | 754 | 1.0x | 25 | yes |
| 16 | 253 | 3.0x | 25 | yes |
<!-- /results -->

## Entry point: the New York case

The build scripts are in the repository's `pipelines/newyork/`, not in the wheel, so work from a
checkout. You need the `[ny]` extra, Ipopt, the inputs and the pool
([Build the operating points](operating_points.md)).

Run from the work folder whose `data/` holds the inputs:

```bash
python $FG/pipelines/newyork/build_timeline.py \
    out/timeline_2019.h5 --pool out/pool_2019_v3.h5 --workers 4
```

| Option | Default | Meaning |
|---|---|---|
| `--pool` | `out/pool_2019_v3.h5` | the pool file |
| `--families` | `Am,At` | the families built |
| `--interface` | `SPR/DUN-SOUTH` | the interfaces `Am` targets |
| `--workers` | 12 | processes that design the episodes |
| `--survey` | 12 | the survey's grid in frames; 0 is off |
| `--survey-cache` | none | folder that keeps the survey by chunk |
| `--fetch` | | download the inputs into the work folder and stop |

The script writes a summary beside the file, as JSON. In Python the same build is:

```python
from fdia_graph.generation.newyork import NewYorkSource
from fdia_graph.models.config import NewYorkRelease

if __name__ == "__main__":
    release = NewYorkRelease(workers=4)
    pool = "out/pool_2019_v3.h5"
    source = NewYorkSource.production(pool, release)
    source.generate("out/timeline_2019.h5")
    print(source.summary()["built"])
```

`production` reads the case, assigns each New York bus to its utility, rates each unrated branch
at its peak flow over the pool, places the meters and builds the two families. `summary()` then
counts, per split and family, the slots placed, the moves, the episodes built and the slots given
up. [Troubleshooting](troubleshooting.md#the-timeline) explains its miss reasons.

### One episode alone

`pipelines/newyork/build_single_episodes.py` writes the whole day of one onset with a single
`Am` episode in it:

```bash
python $FG/pipelines/newyork/build_single_episodes.py OUT_DIR \
    --pool out/pool_2019_v3.h5 --onset 2019-07-05T10:30:00
```

It uses `SingleEpisodeSource` (`generation/newyork.py`), a `NewYorkSource` with one episode
placed by hand:

```python
import numpy as np

from fdia_graph.generation.newyork import SingleEpisodeSource

# source: the NewYorkSource built above, with workers=1
when = np.datetime64("2019-07-05T10:30:00")
times = source.pool.time.astype("datetime64[s]")
onset = int(np.flatnonzero(times == when)[0])
one = SingleEpisodeSource(source, onset, family="Am")
one.generate("one.h5")
```

`onset` is a frame of the pool. The snippet finds the frame of the same time the command above
names.

`lo` and `hi` set the pool frames the file covers. `using=` designs the episode with a variant
of the family, such as `InterfaceFdia.with_overshoot(mw)`. `lag=` designs it on the window that
many frames earlier.

### Reading the file by utility

A New York timeline opens with `fg.FdiaGraph(path)` like any timeline. `Dataset`
(`dataset/clients.py`) is a second view of it, by meter and by utility:

```python
from fdia_graph.dataset.clients import Dataset

ds = Dataset("out/timeline_2019.h5")
con_ed = ds.client("ConEd", split="train")
x = con_ed.node_features()      # [T, N_c, 4]
op = ds.operator(split="test")  # every meter
```

A client sees only its own meters and the labels of its own buses. The operator view is the full
scan. `ds.clients` lists the utility names.

| On a `ClientView` | Shape | Holds |
|---|---|---|
| `z` | `[T, M_c]` | its observed scans, one column per meter it owns |
| `labels` | `[T, N_c]` | the attack labels of its buses |
| `meters` | | its rows of the meter table |
| `buses` | `[N_c]` | the positions of the buses it owns |
| `frames` | `[T]` | the frames selected; a week split has gaps |
| `node_features()` | `[T, N_c, 4]` | voltage, angle, P, Q per bus; NaN where not metered |
| `temporal()` | `[T, M_c]` | the change from the frame before |

| On the `Dataset` | Holds |
|---|---|
| `ds.plan` | the meter table |
| `ds.z(frames)` | the flat reading vector of that table |
| `ds.z(frames, benign=True)` | the same scans without the attack |
| `ds.tampered(frames)` | the tampered meters |

`Dataset` refuses a file that is not a New York timeline with `NotATimeline`.

## Where the code is

| File | Holds |
|---|---|
| `generation/source.py` | `TimelineSource`: the steps, written once |
| `generation/search.py` | `SearchSource`: coverage meters, search-based designs |
| `generation/ieee.py` | `IeeeSource`: a `SearchSource` for an IEEE case by name |
| `generation/newyork.py` | `NewYorkSource`, `SingleEpisodeSource`, `Survey` |
| `generation/plan.py` | the splits, the episode counts, the placement |
| `generation/design.py` | `Designer`, `SplitDesign`, the design processes |
| `generation/emit.py` | `Walk`: the frames, in time order |
| `generation/write.py` | `TimelineWriter`: the one writer of the file |

Every source emits its scans through one generator, `FdiaGenerator`, in `engine/`:

| File | Holds |
|---|---|
| `engine/core.py` | `FdiaGenerator`: a `Network`, its meter plan and noise |
| `engine/base.py` | `GridBase`: the attributes the generator's parts share |
| `engine/measurement.py` | `MeasurementMixin`: a scan from a stored state |
| `engine/physics.py` | a scan's load and generation, read from its state |

The attack designs are under `engine/attacks/`
([The attack families](../concepts/attacks.md#where-the-code-is)).
