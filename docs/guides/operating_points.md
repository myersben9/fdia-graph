# Build the operating points

A timeline starts from a pool: one solved AC operating point per frame, in time order. This guide
gets or builds the pool of a test system. [Solving](../concepts/solving.md) explains the two
kinds of solve and the setting that chooses between them.

Paths are under `src/fdia_graph/`.

## What a pool is

| Test system | Pool | Columns per bus |
|---|---|---|
| IEEE case | an array `[T, N, 4]` | voltage (p.u.), P (MW), Q (MVAr), angle (deg) |
| New York case | a `Pool` record, in a file | `vm`, `va`, `p`, `q`, with each frame's `time` |

A pool holds the truth only. Meters, noise and attacks come later
([Generate a timeline](generation.md)).

## An IEEE case: use the released pool

Each of the eight IEEE systems has a released pool of 72,000 operating points. `fg.generate`
downloads it once and keeps it in the cache
([Install and data versions](install.md#the-cache)). You do not build it.

```python
import fdia_graph as fg

# downloads pool_ieee14.h5 (about 25 MB) on first use
fg.generate("ieee14", name="my_run", frames=2000)
```

The file is `pool_ieee{N}.h5` with one dataset, `X`. The pool stores no timestamps. Frame `t`
of a timeline is step `t` of its pool.

## An IEEE case: build your own pool

Three functions build a pool. All three are on the top-level package.

| Function | Does |
|---|---|
| `fg.fetch_profile` | downloads an operator's system load |
| `fg.load_profile` | reads a load series you already have |
| `fg.generate_states` | scales the case along the profile and solves each step |

A profile is one number per step: the load series standardized to zero mean and unit variance.

```python
import numpy as np

import fdia_graph as fg
from fdia_graph.data.profiles import RawSeries

# ten hours of a daily load curve, one value per minute
hours = np.arange(600) / 60.0
load_mw = 100 + 10 * np.sin(2 * np.pi * hours / 24)
profile = fg.load_profile(RawSeries(load_mw))

# needs fdia-graph[generate]
pool = fg.generate_states("ieee14", profile)
print(pool.shape)  # (600, 14, 4)

fg.generate("ieee14", name="own_pool", states=pool)
```

`fg.load_profile` takes one of three sources from `data/profiles.py`: `RawSeries(values)`,
`CsvColumn(path, column)` or `IsoFolder(iso, directory)`.

`fg.fetch_profile` downloads a real load series instead:

```python
profile = fg.fetch_profile(
    "nyiso", "2024-01-01", "2024-01-07", resample_min=1
)
```

| `iso` | Source | Needs |
|---|---|---|
| `"nyiso"` | NYISO's public monthly archives | nothing extra |
| `"caiso"`, `"ercot"` | the gridstatus package | `fdia-graph[iso]` |

`resample_min` interpolates the series in time to that many minutes per step.

### What `generate_states` does at each step

| Item | Value | Where |
|---|---|---|
| Scale of a bus | `clip(1 + k S_t + j_t, 0.7, 1.3)` | `_ar1_scale` |
| `k` | 0.1 | `K_DEFAULT` |
| Jitter `j_t` | AR(1) per bus, coefficient 0.98, standard deviation 0.03 | `LOAD_JITTER_RHO`, `SIGMA_DEFAULT` |
| What is scaled | each load's P and Q and each generator's P | `_solve_states_chunk` |
| Solve | AC power flow in pandapower | `_solve_states_chunk` |
| A step that fails | dropped | `_solve_states_chunk` |

`S_t` is the profile at step `t`. The AR(1) jitter makes each bus drift smoothly from step to
step and not jump. The clip holds every bus inside a plausible load band.

A shunt's draw is subtracted from the stored injections. The estimator models a shunt in the
admittance matrix, not as an injection (`_remove_shunt_injections`).

`generate_states` takes `workers=N` to solve in N processes. A script that passes it must start
under `if __name__ == "__main__":`.

### The solve setting

`generate_states` solves a power flow unless you pass `solve=`:

```python
from fdia_graph.models.config import SolveSettings

by_opf = fg.generate_states(
    "ieee14",
    profile[:20],
    solve=SolveSettings(method="opf", engine="pandapower"),
)
```

With `method="opf"` each step scales the loads only and chooses the generator outputs by a
least-cost AC OPF. `engine="ipopt"` runs the same problem in Ipopt.
[Solving](../concepts/solving.md#the-solve-setting) lists every choice and which are refused.

### Your own pool file

`states=` of `fg.generate` takes an array or a path.

| `states` | Read as |
|---|---|
| an array `[T, N, 4]` | the pool itself |
| a `.npz` file | its array `X` |
| a `.h5` file | its dataset `X` |
| a folder | files `X_<t>.npy`, one per step; more than 8,000 are read at an even stride |

The column order is detected from the data: voltage is the only column whose every value is
near 1.0. A pool in neither known order raises `UnknownColumnOrder`.

## The New York case: build the pool

No New York pool is released. You build it from public inputs. The build scripts are in the
repository's `pipelines/newyork/`, not in the wheel, so work from a checkout.

### What it needs

```bash
git clone https://github.com/myersben9/fdia-graph
pip install -e "./fdia-graph[ny]"
conda install -c conda-forge ipopt
export FG=$PWD/fdia-graph
```

- The `[ny]` extra: pandapower, pandas, scipy, Pyomo, openpyxl and matpowercaseframes.
- Ipopt, a solver that pip does not install
  ([Solving](../concepts/solving.md#what-to-install)).
- A work folder for the inputs and the outputs. The scripts read `data/` and write `out/`
  relative to the folder you run them from.

### 1. Fetch the inputs

```bash
mkdir nygrid && cd nygrid
python $FG/pipelines/newyork/build_timeline.py --fetch
```

This calls `fetch_inputs` in `data/inputs.py`, which writes `data/`:

| Input | Source |
|---|---|
| NYgrid's `Data/` folder | the NYgrid repository at one pinned commit |
| five-minute zone loads of 2019 | NYISO's public archives |
| utility service territories | New York State Department of Public Service |
| the location of each New York bus | shipped with the package |

A file that is already there is kept. The grid itself needs no download: it is the case file in
the package ([Test systems](../concepts/test_systems.md#the-new-york-case-file)).

The load model downloads its own sources the first time a pool is built: building-stock load
profiles, utility sales and industrial plant data. They are downloaded once and cached.

### 2. Run the pool

```bash
python $FG/pipelines/newyork/run_pool.py \
    2019-01-01 2020-01-01 out/pool_2019_v3.h5 --workers 4
```

| Option | Default | Meaning |
|---|---|---|
| `START`, `END` | | local-clock dates; `END` is not included |
| `OUT` | | the pool file to write |
| `--workers` | 1 | processes that solve the chunks |
| `--chunk-days` | 7 | days per chunk |
| `--frames` | 0 | stop after this many frames; 0 is all |
| `--seam-frames` | 288 | most frames solved again at a seam |
| `--loads` | `out/loads_2019_v2.npz` | the cached bus loads; built when missing |
| `--compare-hours` | off | also print the dispatch cost against the hourly AC points |

The year is 105,120 five-minute frames and takes several hours to solve. Try a week first:

```bash
python $FG/pipelines/newyork/run_pool.py \
    2019-01-07 2019-01-14 out/pool_week.h5 --workers 4
```

### What the script does

1. **Loads.** `LoadModel` (`data/loads.py`) builds a load for each New York bus at every frame.
2. **Hourly dispatch.** `read_nygrid` (`data/inputs.py`) reads NYgrid's hourly data, and
   `hourly_dispatch` (`data/dispatch.py`) sets each hour's dispatch from it.
3. **Chunks.** The window is cut into chunks, a week each by default, solved in parallel.
   `PoolBuilder` (`data/pool.py`) solves each chunk frame by frame: one OPF in Ipopt
   (`IpoptSolver`, `data/ipopt_solver.py`), then one power flow at its outputs
   (`PandapowerSolver`, `data/pandapower_solver.py`).
4. **Seams.** The first frames of each chunk (at most `--seam-frames`) are solved again from
   the last state of the chunk before. They replace the stored frames up to the first frame
   where every thermal unit is within 0.5 MW of the stored dispatch. A seam that does not merge
   has its whole chunk solved again.
5. **The file.** `write_pool` stores the pool.

`PoolBuilder.from_settings` takes a `SolveSettings` like `generate_states`. It builds by an OPF
in Ipopt and refuses the other choices with the reason
([Solving](../concepts/solving.md#why-a-choice-is-refused)).

### Reading the pool back

```python
from fdia_graph.constants.newyork import SolveKind
from fdia_graph.data.pool import read_pool

pool = read_pool("out/pool_2019_v3.h5")
solved = pool.solve_kind == SolveKind.SOLVED
print(solved.mean())
```

A frame that does not solve keeps its place with a NaN state, and only solved frames are used
later.
[Troubleshooting](troubleshooting.md#the-operating-points) lists the solve kinds.

| Field of `Pool` | Shape | Holds |
|---|---|---|
| `time` | `[T]` | each frame's time, NYISO's local clock |
| `vm`, `va` | `[T, N]` | voltage magnitude (p.u.) and angle (deg) |
| `p`, `q` | `[T, N]` | net injection (MW, MVAr) |
| `solve_kind` | `[T]` | how the frame was solved (`SolveKind`) |
| `unit_p` | `[T, G]` | each unit's dispatched output (MW) |

### How the bus loads are built

`LoadModel` (`data/loads.py`) builds each New York load bus's profile bottom-up, by the method
of [LI21]: prototype buildings and industrial plants, mixed by its utility's sector sales. The
zone's real five-minute load is split over its buses in proportion to these profiles. Each bus
then gets two terms of its own. Their constants are in `constants/newyork.py`.

**Fluctuation** (`FLUCT_C`, `FLUCT_BETA`, `FLUCT_PHI`). An AR(1) factor on the bus's load,
`p (1 + e)`.

- `e` has standard deviation `FLUCT_C size^-FLUCT_BETA`, where size is the bus's mean load in
  MW. `FLUCT_PHI` is its AR(1) coefficient.
- The term is the regulation component of ORNL/TM-2001/13.
- The constants were fitted on the NYISO zones of 2018, with zone H left out.

**Industrial switching** (`SWITCH_*`). Pulses on a bus's industrial load.

- The constants were fitted on zone D's five-minute steps of 2019 above 5 % of its mean load.
- The number of pulses is a Poisson draw at `SWITCH_RATE` per MW of the bus's mean industrial
  load and hour.
- A pulse's size, relative to the bus's mean industrial load, is the threshold `SWITCH_U` plus
  a generalized Pareto excess (shape `SWITCH_XI`, scale `SWITCH_SCALE`). A size above
  `SWITCH_MAX` is drawn again.
- Its duration is lognormal (`SWITCH_DUR_MU`, `SWITCH_DUR_SIGMA`, log minutes).
- A share `SWITCH_OFF` of pulses are drops. A drop is clipped so the load stays at or above
  zero.

Last, each zone's buses are rescaled to sum exactly to the zone's real load.

The fit functions are not part of the package.

## Where the code is

| File | Holds |
|---|---|
| `data/profiles.py` | `fetch_profile`, `load_profile` and the load sources |
| `data/ieee_pool.py` | `generate_states`: the pool of an IEEE case |
| `data/inputs.py` | `fetch_inputs` and the readers of the New York inputs |
| `data/load_sources.py` | the building, sales and plant data, downloaded once |
| `data/loads.py` | `LoadModel`, `LoadSeries`: the New York bus loads |
| `data/dispatch.py` | `Dispatch`, `hourly_dispatch` |
| `data/config.py` | `Settings` of the New York build, and its random streams |
| `data/solver.py` | `Solver`: the interface of the two engines |
| `data/ipopt_solver.py` | `IpoptSolver`: the OPFs, in Pyomo with Ipopt |
| `data/pandapower_solver.py` | `PandapowerSolver`: the power flow, and pandapower's OPF |
| `data/ipopt.py` | `ipopt`: finds Ipopt and checks that it starts |
| `data/pool.py` | `Pool`, `PoolBuilder`, `read_pool`, `write_pool`, `peak_ratings` |
