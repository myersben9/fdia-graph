# fdia-graph

A Python package that builds attacked measurement timelines on power-grid test systems, with a
second toolkit to analyse them. Download a ready timeline or generate your own.

A timeline is the meter readings of one grid, frame by frame, with false data injection attacks
(FDIA) placed in it and labelled per bus.

## Two toolkits

| Toolkit | Package | What it does |
|---|---|---|
| `fdia-graph` | `fdia_graph` | generates a timeline and loads it |
| `fdia-analysis` | `fdia_analysis` | runs estimators and localizers on a timeline |

`fdia-analysis` holds state estimation, attack localization, federated training, trusted-meter
selection and a results store. It depends on `fdia-graph`. `fdia-graph` never imports it.

## Test systems

| System | Buses | Data |
|---|---|---|
| IEEE and other public cases | 14, 30, 57, 89, 118, 145, 200, 300 | released datasets, downloaded by `fg.load` |
| New York grid | 140 | case file in the package; you build the dataset |

Each released dataset is one timeline of 72,000 frames. The New York grid is a MATPOWER case file
with 140 buses and 227 branches; [the New York guide](docs/guides/new_york.md) builds its dataset.

The generator makes two attack families: `At`, a slow load ramp, and `Am`, a line overload. The
released files hold seven families: five that the generator does not make (`Aq`, `Ad`, `As`, `Ar`,
`Al`), and an `Am` that drives no line to its limit.
[The generation guide](docs/guides/generation.md) has the details.

## Three steps

**1. Load** a released dataset.

```python
import fdia_graph as fg

# downloads the IEEE-118 timeline once, then caches it
train = fg.load("ieee118", split="train")
arrays = train.export()
print(arrays["node_x"].shape, arrays["y"].shape)
```

**2. Generate** your own, then load it by name.

```python
# needs fdia-graph[generate]
# downloads the IEEE-14 pool of solved operating points
# about five minutes on a laptop
fg.generate("ieee14", name="my_run", frames=2000)
mine = fg.load("my_run", split="train")
```

**3. Analyse** it with an estimator.

```python
# needs fdia-analysis[se]
from fdia_analysis.se.methods import WLS

test = fg.load("ieee118", split="test")
est = WLS().fit(train)
print(est.score(test))
```

## How it flows

<img src="https://raw.githubusercontent.com/myersben9/fdia-graph/main/docs/figures/diagrams/flow.png" width="450" alt="Six stages, top to bottom. 1. Test system: Case, Network. 2. Operating points: generate_states for the IEEE cases, PoolBuilder for New York. 3. Meters and attacks: MeterPlan, NoiseModel, the attack designs. 4. Timeline file: TimelineSource.generate, TimelineWriter. 5. Load: fg.load returns FdiaGraph. 6. Analyse: fdia_analysis.">

## Install

| Command | Gives |
|---|---|
| `pip install fdia-graph` | the loader: opens a timeline, exports splits as numpy arrays |
| `pip install "fdia-graph[torch]"` | adds PyTorch: single records, batches and the DataLoader |
| `pip install "fdia-graph[generate]"` | adds pandapower, to generate timelines |
| `pip install "fdia-analysis[se]"` | the analysis toolkit with state estimation |

Every extra of both toolkits is listed in [Install and data versions](docs/guides/install.md).

## Where next

- [How the package flows](docs/concepts/flow.md), [the test systems](docs/concepts/test_systems.md)
  and [how they are solved](docs/concepts/solving.md)
- [How a timeline is generated](docs/guides/generation.md)
- [Building the New York dataset](docs/guides/new_york.md)
- [Data dictionary: what every array means](docs/reference/DATA_DICTIONARY.md)
- [Loading options and runnable baselines](docs/reference/EXAMPLES.md)
- Analysis pages: [state estimation](docs/analysis/estimation/README.md),
  [localization](docs/analysis/localization/README.md),
  [trusted meters](docs/analysis/trust/README.md),
  [federated training](docs/analysis/federated/README.md)
- [Contributing: rules, pull requests, releases](CONTRIBUTING.md)

## Citation

- `Al` (released files) and the load bound of `Am`: Yuan, Li & Ren, *Modeling load redistribution
  attacks in power systems*, IEEE T-SG 2(2), 2011.
- `At`: Haghshenas, Hasnat & Naeini, *A Temporal GNN for Cyber Attack Detection and Localization
  in Smart Grids*, IEEE ISGT 2023.
- Measurement model: Zaman & Lin, *PING: Physics-Informed GNNs to Generalize FDIA Localization*,
  NAPS 2025.
- Meter noise: Asprou, Kyriakides & Albu, *Variable Weights in a WLS State Estimator*, IEEE T-IM
  63, 2014.
- Protocol: Boyaci et al., *Joint Detection and Localization of Stealth FDIA*, IEEE T-SG, 2022.
- Local false states, `Am`, trusted meters: Wu, Wang, Hu, Ye & Tang, *Dynamic PMU configuration
  for stealthy multi-snapshot FDIA mitigation*, IEEE T-SG 17(1), 2026.

## License

Data under CC BY 4.0, code under MIT (see `LICENSE`). Synthetic, from public IEEE cases. Not for
operational decisions.
