# The New York grid: building the timeline

`fdia_graph.generation.nygrid` builds a federated FDIA timeline on the New York grid: NYgrid's modified NPCC-140 case in
AC, a 2019 five-minute operating-point pool from a ramp-constrained AC dispatch, one client per utility
service territory, a hybrid SCADA and PMU meter plan, and attacks against NYISO interface limits. The
modules are in
[the architecture](../nygrid/ARCHITECTURE.md), one run end to end in [the walkthrough](../nygrid/WALKTHROUGH.md), how to
add an attack, a detector or an operator in [the extending guide](../nygrid/EXTENDING.md), and what each stage reports
when it fails in [the troubleshooting page](../nygrid/TROUBLESHOOTING.md). No built pool or timeline is published yet, so the steps
below build them from the public inputs.

## Install

The build tools live in the repository's `tools/nygrid/`, not in the wheel, so work from a checkout:

```bash
git clone https://github.com/myersben9/fdia-graph && pip install -e "./fdia-graph[ny]"
conda install -c conda-forge ipopt   # the AC dispatch solves through Pyomo with Ipopt
export FG=$PWD/fdia-graph
```

The steps below run in a separate work folder that holds the inputs and the outputs.

## 1. Inputs

```bash
mkdir nygrid && cd nygrid
python $FG/tools/nygrid/build_timeline.py --fetch
```

`fetch_inputs` writes `data/`: NYgrid's `Data/` folder at commit 47698b6, NYISO's 2019 five-minute zonal
load, the NYS DPS utility territories, and the bus locations that place each New York bus in a territory
(shipped with the package). The building-stock load profiles, EIA-861 and the industrial assessment
database are downloaded by the load model the first time the pool is built (about 200 MB, 15 minutes).
Every step keeps a file that is already there.

## 2. The operating-point pool

```bash
python $FG/tools/nygrid/run_pool.py 2019-01-01 2020-01-01 out/pool_2019.h5 --workers 8
```

The year is dispatched in weekly chunks in parallel, then each seam between chunks is solved again until
the dispatch is continuous. The pool holds the true state and injections of every five-minute frame.

## 3. The timeline

```bash
python $FG/tools/nygrid/build_timeline.py out/timeline_2019.h5 --pool out/pool_2019.h5 --workers 8
```

`production_builder` puts every decided bound in one place: line ratings from the pool's peak flows, the
overload attack (Am) against the Sprain Brook/Dunwoodie South interface, the slow ramp (At), 20% of
each split's frames attacked, and onsets surveyed among the hours an attack can reach. The weeks are split
into train, validation and test, stratified on those hours.

For one attack studied by hand, `tools/nygrid/build_single_episodes.py OUT_DIR --pool out/pool_2019.h5
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

## Tests

`tests/nygrid` reads the inputs and the pool under `NYGRID_ROOT` (the folder of steps 1 to 3) and skips a
test whose input is missing:

```bash
NYGRID_ROOT=/path/to/nygrid pytest $FG/tests/nygrid
```

Most tests need only the inputs of step 1, with the attack days and their pool frames checked in under
`tests/nygrid/data/days`; the tests that rate lines or survey onsets over the year also need the pool of step 2.
`tests/nygrid/nygrid_inputs.py` says which input each test needs. CI runs the folder on the inputs of step 1
(`.github/workflows/nygrid.yml`).
