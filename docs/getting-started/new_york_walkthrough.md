# The New York grid: one run, end to end

This page follows one attack through every stage of `fdia_graph.generation.nygrid`, from the network to the clients' view,
with the call that does each step. [The architecture](../concepts/new_york_grid.md) says how the modules fit together,
[the troubleshooting page](../guides/troubleshooting_new_york.md) what each stage reports when it fails, and
[the extending guide](../guides/extending_new_york.md) how to add an attack or an alarm rule of your own.

![The stages of fdia_graph.generation.nygrid](../figures/diagrams/nygrid_pipeline.png)

The snippets run from a work folder that holds the inputs (`python pipelines/newyork/build_timeline.py --fetch`, see
[the guide](../guides/new_york.md)) and the 2019 pool (`pipelines/newyork/run_pool.py`, several hours).

## 1. The network and the clients

The grid is one MATPOWER case file shipped with the package (NYgrid's case with the changes that make it
solvable in AC). `nygrid_case()` reads it into a `Case`, a frozen record of arrays, and `Network` holds its
admittances. Each New York bus belongs to one utility, a client of the federated dataset.

```python
from fdia_graph.generation.nygrid import Network, nygrid_case

case = nygrid_case()
net = Network(case)
```

A change to the grid is an edit to the case file ([the architecture](../concepts/new_york_grid.md) lists its fields); a variant in code is a copy (`hvdc_links(case)`, `case.with_voltage_margin(0.02)`), never a change in place.

## 2. The operating points

`scenario` gives the true AC state of every five-minute frame of 2019: the loads by bus (`LoadModel`), NYgrid's
hourly dispatch, and a ramp-limited least-cost AC dispatch solved frame by frame (`PoolBuilder`). The year takes
hours, so it is built once and read back:

```python
from fdia_graph.generation.nygrid import read_pool
from fdia_graph.generation.nygrid import SOLVED

pool = read_pool("out/pool_2019_v3.h5")
print((pool.solve_kind == SOLVED).mean())  # 0.998: the frames that solved; only these are used downstream
```

## 3. The meters, the attacker and the timeline

`production_builder` assembles the release build: the clients, the meter plan, the attacker and its families, over
the pool. One attack window can be built on its own, which is how the studies of the paper are made:

```python
from fdia_graph.generation.nygrid import production_builder

b = production_builder("out/pool_2019_v3.h5", ("Am",), workers=1, survey=0)
b.build_episode("one.h5", onset=48220, family="Am")  # the interface attack at the limit, on its day
```

The Am family (`InterfaceFdia`) raises the reported flow of the Sprain Brook and Dunwoodie South interface to its
limit while keeping every reading consistent with a false state, so the operator's residual test cannot see it.
`b.build(path)` writes the whole year instead, with episodes placed at random; its summary counts what it placed,
built and, by reason, missed.

## 4. The clients' view

A client reads only its own meters, and the labels of its own buses:

```python
from fdia_graph.generation.nygrid import Dataset

ds = Dataset("one.h5")
view = ds.client(ds.clients[0])
print(view.z.shape, view.labels.shape)  # scans x this client's meters, scans x its buses
```
