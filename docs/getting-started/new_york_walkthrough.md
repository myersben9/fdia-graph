# The New York grid: one run, end to end

This page follows one attack through every stage of `fdia_graph.generation.nygrid`, from the network to the clients' view,
with the call that does each step. [The architecture](../concepts/new_york_grid.md) says how the modules fit together,
[the troubleshooting page](../guides/troubleshooting_new_york.md) what each stage reports when it fails, and
[the extending guide](../guides/extending_new_york.md) how to add an attack or an alarm rule of your own.

![The New York build in seven stages, top to bottom: the inputs (fetch_inputs: NYgrid tables, NYISO 2019 load, utility territories); the network (Case, Network, Ownership: a client per utility); the operating points (LoadModel, hourly_dispatch, PoolBuilder: the Pool); the meters (plan_meters: the MeterPlan, Measurement h(x), NoiseModel); the attacker (Attacker: a local false state, AttackFamily.design: an Episode); the timeline (NewYorkSource.generate); the clients' view (Dataset, ClientView)](../figures/diagrams/nygrid_pipeline.png)

The snippets run from a work folder that holds the inputs (`python pipelines/newyork/build_timeline.py --fetch`, see
[the guide](../guides/new_york.md)) and the 2019 pool (`pipelines/newyork/run_pool.py`, several hours).

## 1. The network and the clients

The grid is one MATPOWER case file shipped with the package (NYgrid's case with the changes that make it
solvable in AC). `nygrid_case()` reads it into a `Case`, a frozen record of arrays, and `Network` holds its
admittances. Each New York bus belongs to one utility, a client of the federated dataset.

```python
from fdia_graph.systems.case import nygrid_case
from fdia_graph.systems.network import Network

case = nygrid_case()
net = Network(case)
```

A change to the grid is an edit to the case file ([the architecture](../concepts/new_york_grid.md) lists its fields); a variant in code is a copy (`hvdc_links(case)`, `case.with_voltage_margin(0.02)`), never a change in place.

## 2. The operating points

`fdia_graph.generation.nygrid` gives the true AC state of every five-minute frame of 2019: the loads by bus (`LoadModel`), NYgrid's
hourly dispatch, and a ramp-limited least-cost AC dispatch solved frame by frame (`PoolBuilder`). The year takes
hours, so it is built once and read back:

```python
from fdia_graph.constants.newyork import SolveKind
from fdia_graph.generation.nygrid.io import read_pool

pool = read_pool("out/pool_2019_v3.h5")
# 0.998: the frames that solved; only these are used downstream
print((pool.solve_kind == SolveKind.SOLVED).mean())
```

## 3. The meters, the attacker and the timeline

`NewYorkSource.production` assembles the release's source: the clients, the meter plan, the attacker and its
families, over the pool. A source's `generate(path)` writes its timeline, by the same path the IEEE timelines
take. One attack window can be written on its own, which is how the studies of the paper are made:

```python
from fdia_graph.generation.nygrid.source import NewYorkSource, SingleEpisodeSource
from fdia_graph.models.config import NewYorkRelease

release = NewYorkRelease(families=("Am",), workers=1, survey=0)
source = NewYorkSource.production("out/pool_2019_v3.h5", release)
one = SingleEpisodeSource(source, onset=48220, family="Am")  # the interface attack at the limit, on its day
one.generate("one.h5")
```

The Am family (`InterfaceFdia`) raises the reported flow of the Sprain Brook and Dunwoodie South interface to its
limit while keeping every reading consistent with a false state, so the operator's residual test cannot see it.
`source.generate(path)` writes the whole year instead, with episodes placed at random;
`source.summary()` then counts what it placed, built and, by reason, missed.

## 4. The clients' view

A client reads only its own meters, and the labels of its own buses:

```python
from fdia_graph.dataset.clients import Dataset

ds = Dataset("one.h5")
view = ds.client(ds.clients[0])
print(view.z.shape, view.labels.shape)  # scans x this client's meters, scans x its buses
every = view.frames  # all frames of the file
attack = ds.z(every) - ds.z(every, benign=True)  # scans x every meter: the attack vector
```
