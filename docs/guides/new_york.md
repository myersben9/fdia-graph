# The New York grid: building the timeline

fdia-graph builds a federated FDIA timeline on the New York grid: NYgrid's modified NPCC-140 case in
AC, a 2019 five-minute operating-point pool from a ramp-constrained AC dispatch, one client per utility
service territory, a hybrid SCADA and PMU meter plan, and attacks against NYISO interface limits. The
modules are in
[the architecture](../concepts/new_york_grid.md), one run end to end in [the walkthrough](../getting-started/new_york_walkthrough.md), how to
add an attack or an alarm rule in [the extending guide](extending_new_york.md), and what each stage reports
when it fails in [the troubleshooting page](troubleshooting_new_york.md). No built pool or timeline is published yet, so the steps
below build them from the public inputs.

## Install

The build tools live in the repository's `pipelines/newyork/`, not in the wheel, so work from a checkout:

```bash
git clone https://github.com/myersben9/fdia-graph && pip install -e "./fdia-graph[ny]"
conda install -c conda-forge ipopt   # the AC dispatch solves through Pyomo with Ipopt
export FG=$PWD/fdia-graph
```

The steps below run in a separate work folder that holds the inputs and the outputs.

## 1. Inputs

```bash
mkdir nygrid && cd nygrid
python $FG/pipelines/newyork/build_timeline.py --fetch
```

The grid itself needs no download: it is the MATPOWER case file shipped with the package (`nygrid_case()`).
`fetch_inputs` writes `data/`: NYgrid's `Data/` folder at commit 47698b6 (its hourly 2019 data), NYISO's 2019
five-minute zonal load, the NYS DPS utility territories, and the bus locations that place each New York bus in a territory
(shipped with the package). The building-stock load profiles, EIA-861 and the industrial assessment
database are downloaded by the load model the first time the pool is built (about 200 MB, 15 minutes).
Every step keeps a file that is already there.

## 2. The operating-point pool

```bash
python $FG/pipelines/newyork/run_pool.py 2019-01-01 2020-01-01 out/pool_2019_v3.h5 --workers 8
```

The year is dispatched in weekly chunks in parallel, then each seam between chunks is solved again until
the dispatch is continuous. The pool holds the true state and injections of every five-minute frame.

### How the within-hour load terms are fitted

The load model (`data/loads.py`) gives each bus two five-minute terms of its own, so a
bus's steps are not its zone's. Their constants are in `constants/newyork.py`.

- **Fluctuation** (`FLUCT_C`, `FLUCT_BETA`, `FLUCT_PHI`). An AR(1) factor on the bus's load,
  `p (1 + e)`, with `e` of standard deviation `FLUCT_C size^-FLUCT_BETA` (size is the bus's mean load,
  MW) and lag-1 coefficient `FLUCT_PHI`. The constants are fitted on the real NYISO zones of 2018, so
  2019 is not fitted to itself. A zone's fluctuation is its regulation component in the sense of Kirby
  and Hirst (ORNL/TM-2001/13): the 5-minute load less its 30-minute rolling mean of L(t-3) .. L(t+2),
  relative to that mean. Its spread is the robust standard deviation (1.4826 MAD), so large industrial
  steps do not inflate it. The spread is fitted against zone mean load as `c size^-beta` across the
  zones. Zone H (Millwood, 320 MW) is left out: its residual is four times what the other ten zones'
  fit predicts at its size, and its near-white steps suggest a metering artifact. One `phi` is the
  median lag-1 autocorrelation of the zones' residuals, read through the filter: white noise already
  gives -0.23 after the rolling mean, so the residual's autocorrelation is mapped back to the AR(1)
  coefficient that produces it, and `c` is scaled to the AR(1)'s own spread the same way. With beta
  near 0.5, independent bus terms add up to the zone's own fluctuation, so the final rescaling leaves
  each bus its own spread and the zone its real load.
- **Industrial switching** (`SWITCH_*`). Zone D's real five-minute steps above 5% of its mean load
  (2,799 in 2019, up to 314 MW) are taken as large plants switching. Same-sign steps in consecutive
  frames are merged into one event. A pulse starts at an event and ends at the next event of opposite
  sign. Pulses arrive as a Poisson process at `SWITCH_RATE` per MW of the bus's mean industrial load
  and hour (zone D's pulses, half its events, over the model's industrial load of zone D). A pulse's
  size relative to the bus's mean industrial load is the threshold `SWITCH_U` (5% of zone D's mean
  load, which is 0.22 of its mean industrial load) plus a generalized Pareto excess (shape
  `SWITCH_XI`, scale `SWITCH_SCALE`), truncated at `SWITCH_MAX`, the largest step seen. Its duration is
  lognormal (`SWITCH_DUR_MU`, `SWITCH_DUR_SIGMA`, log minutes), from the times between opposite-sign
  events. A share `SWITCH_OFF` of pulses start with a drop, and a drop never takes more than the bus's
  industrial load at that frame.

Last, each zone's buses are rescaled to sum exactly to the zone's real 5-minute load.

The constants are the results of two fits. The fit functions are not part of the package.

## 3. The timeline

```bash
python $FG/pipelines/newyork/build_timeline.py out/timeline_2019.h5 --pool out/pool_2019_v3.h5 --workers 8
```

The script builds the release's source and generates it: `NewYorkSource.production(pool_path,
NewYorkRelease(...)).generate(path)`. `generate` is the one path every timeline takes, the IEEE ones too
([generation.md](generation.md#one-path-two-grids)). The file is written as `timeline_2019.h5.part` and takes
its name when it is whole, so a build that stops leaves no file to mistake for a timeline. The design
processes are spawned: a script of your own that asks for more than one worker must start under
`if __name__ == "__main__":`. `NewYorkSource.production` puts every decided
bound in one place: line ratings from the pool's peak flows, the
overload attack (Am) against the Sprain Brook/Dunwoodie South interface, the slow ramp (At), 20% of
each split's frames attacked, and onsets surveyed among the hours an attack can reach. The weeks are split
into train, validation and test, stratified on those hours.

For one attack studied by hand, `pipelines/newyork/build_single_episodes.py OUT_DIR --pool out/pool_2019_v3.h5
--onset 2019-07-05T10:30:00` writes the whole day of that onset with a single Am episode in it.

## 4. Reading it as clients

```python
from fdia_graph.dataset.clients import Dataset

ds = Dataset("out/timeline_2019.h5")
con_ed = ds.client("ConEd", split="train")  # only this utility's meters, buses and labels
x = con_ed.node_features()  # [T, N_c, 4] metered |V|, angle, P, Q per owned bus
op = ds.operator(split="test")  # every meter, for state estimation
```

A client sees only its own meters and the labels of its own buses. The operator view is the full scan.

The file has the layout of every timeline (`fdia_graph.schema`; the table is in
[new_york_grid.md](../concepts/new_york_grid.md#the-timeline-file)), so `fg.FdiaGraph("out/timeline_2019.h5")`
opens it too: the static graph, the admittance matrices and the layers per frame. `Dataset` is the view by
meter and by client: `ds.plan` is the meter table, `ds.z(frames)` the flat reading vector of that table,
`ds.z(frames, benign=True)` the same scans without the attack and `ds.tampered(frames)` the tampered meters.
A file with any other layout is refused.

## Tests

`tests/nygrid` reads the inputs and the pool under `NYGRID_ROOT` (the folder of steps 1 to 3) and skips a
test whose input is missing:

```bash
NYGRID_ROOT=/path/to/nygrid pytest $FG/tests/nygrid
```

Most tests need only the inputs of step 1, with the pool frames of two days checked in under
`tests/nygrid/data/days`; the tests that rate lines or survey onsets over the year also need the pool of step 2.
`tests/nygrid/inputs.py` says which input each test needs. CI runs the folder on the inputs of step 1
(`.github/workflows/nygrid.yml`).
