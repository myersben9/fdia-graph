# fdia-graph

Stealthy FDIA localization datasets for power grids, PyTorch-ready in one line. Eight IEEE systems
(14 / 30 / 57 / 89 / 118 / 145 / 200 / 300 buses), 72,000 records each.

```python
import fdia_graph as fg

ds = fg.load("ieee118", split="train")     # auto-downloads + caches
loader = ds.loader(batch_size=64)
for batch in loader:
    batch["node_x"], batch["edge_x"], batch["edge_index"], batch["y"], batch["family"]
```

**New here?**

| Read | To learn |
|---|---|
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | which file does what |
| [`docs/reference/DATA_DICTIONARY.md`](docs/reference/DATA_DICTIONARY.md) | what every array means |
| [`docs/reference/CONCEPTS_TO_CODE.md`](docs/reference/CONCEPTS_TO_CODE.md) | paper equations → functions |
| [`docs/reference/EXAMPLES.md`](docs/reference/EXAMPLES.md) | runnable baselines, streams, dataset stats |
| [`docs/se/`](docs/se/README.md) · [`docs/localization/`](docs/localization/README.md) | the two analysis modules, with real results |

## Install

```bash
pip install fdia-graph              # loader
pip install "fdia-graph[torch]"     # + PyTorch DataLoader
pip install "fdia-graph[pyg]"       # + torch_geometric
pip install "fdia-graph[se]"        # + state estimation / residual localization (torch + pandapower)
pip install "fdia-graph[generate]"  # + pandapower, to generate custom data
```

Data is pinned per SDK version and cached in `~/.cache/fdia_graph`. Pin a data version with
`fg.load(..., release="v0.7.2")`; `pip install --upgrade fdia-graph` moves it forward.

## Load

```python
fg.load("ieee300", split="train")                              # 60/20/20 chronological split
fg.load("ieee118", split="test", families=["Aq","At","Al"])    # family subset
fg.load("ieee118", units="pu")                                 # per-unit + radians (default is physical)
```

- Whole split at once: `ds.to_numpy()` / `.to_torch()` / `.to_pandas()`.
- Custom data: `fg.generate(system, name, per_family=..., attack_intensity=..., ...)` then `fg.load(name)`.
- Continuous timeline for LSTM/TGN: `fg.load_stream(system)`.

Details for all three in [`docs/reference/EXAMPLES.md`](docs/reference/EXAMPLES.md).

## State estimation

```python
from fdia_graph.se import WLS, SubspacePrior   # pip install "fdia-graph[se]"

train, test = fg.load("ieee118", split="train"), fg.load("ieee118", split="test")
est = SubspacePrior(rank_frac=0.5, reweight="huber", c=2.5).fit(train)
xhat = est.estimate(test)          # [n, 2N-1] = [theta rad (non-slack) | V pu (all buses)]
print(est.score(test))             # per-family angle/voltage MAE vs the clean truth
```

`WLS`, `AdaptiveWeighting`, `ResidualRemoval`, `SubspacePrior` share one solver and differ only in
state space and weights. Results and a walkthrough: [`docs/se/`](docs/se/README.md).

## Localization

```python
from fdia_graph.localization import SwingThreshold   # numpy only

loc = SwingThreshold(fa_target=0.01).fit(train)     # per-bus thresholds from benign records only
flag = loc.localize(test)                           # [n, N] bool: which buses are called attacked
print(loc.score(test))                              # per-family node-F1, strict accuracy, DR next to FA
```

`SwingThreshold`, `DeltaThreshold`, `ResidualLocalizer` share one calibration and metric protocol and
differ only in the per-bus score. `BusCNN` and `BusMLP` are the papers' learned localizers on the
14-dim per-bus feature vector (needs `[torch]`):

```python
from fdia_graph.localization import BusCNN

zs = dict(families=[0, 1, 2])                        # papers' zero-shot protocol: As/Ar unseen in train
cnn = BusCNN().fit(fg.load("ieee118", split="train", **zs), val=fg.load("ieee118", split="val", **zs))
print(cnn.score(fg.load("ieee118", split="test", families=[0, 1, 2, 3, 4]))["all"]["macro_f1"])
```

Results: [`docs/localization/`](docs/localization/README.md).

## Data

Each record is a sparse measurement graph with **N buses** (nodes) and **E branches** (edges).
Read a shape as "values per item": `[N,4]` = 4 numbers per bus, `[E,8]` = an 8-dim vector per branch.

| field | shape | columns | meaning |
|---|---|---|---|
| `node_x` | `[N,4]` | <code>&#124;V&#124;</code>, `P_inj`, `Q_inj`, `theta` | bus meters (`node_m` `[N,4]` is the mask) |
| `edge_x` | `[E,2]` | `P_from`, `Q_from` | branch flows (`edge_m` `[E,2]` is the mask) |
| `edge_index` | `[2,E]` | `from_bus`; `to_bus` | connectivity |
| `edge_attr` | `[E,8]` | `r`, `x`, `b`, `g`, `gs`, `bs`, `tap`, `shift` | static line physics (`ds.edge_attr`) |
| `y` | `[N]` | | 1 attacked, 0 clean. Which buses |
| `family` | scalar | | 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At, 6 Al. Which attack (`fg.FAMILIES`) |
| `temporal_delta`, `swing` | `[N,2]` | `ΔP`, `ΔQ` | temporal features |
| `clean` | `[N,4]` | same as `node_x` | noiseless truth, all buses. SE target (v0.7.2+) |
| `edge_clean` | `[E,2]` | same as `edge_x` | noiseless true flows, unmetered branches zeroed |
| `slack` | scalar | | index of the reference bus (`ds.slack`, `Data.slack` in PyG) |

In PyG format (`format="pyg"` and `fg.pyg_stream`) the flows are `Data.edge_attr` (alias `Data.edge_x`),
the mask is `edge_mask`, and the static line physics are `Data.edge_phys`.
Full reference: [`docs/reference/DATA_DICTIONARY.md`](docs/reference/DATA_DICTIONARY.md).

## Attacks

Three **stealthy** families that evade classical bad-data detection (BDD), plus three **detectable**
ones as a contrast set.

| family | attack | classical BDD |
|--------|--------|---------------|
| `Aq` | stealthy load rescale + AC re-solve | evades |
| `At` | slow temporal load ramp | evades |
| `Al` | targeted load redistribution (hides overloads) | evades |
| `Ad` / `As` / `Ar` | meter corruption / scaling / replay | caught |

![BDD statistic per family: the three stealthy families sit below the alarm line with benign, the three tampering families sit far above it](docs/figures/fig_bdd.png)

*Bad-data statistic J relative to the alarm threshold, per family (Aq is labeled A_o here). Green
families are indistinguishable from benign; red ones trip the alarm.*

- Every per-bus change stays in a plausibility band: 2% noise floor to 20% cap.
- Meter error follows an accuracy-class model (per-meter bias plus per-scan jitter).
- Report **per-family node-F1 with the false-alarm rate**, not accuracy: clean buses dominate.
- A lightweight per-bus MLP reaches ~0.92 localization macro-F1 (see the examples page).

## Citation

Cite the attack- and measurement-model sources:

- Yuan, Li & Ren, *Modeling load redistribution attacks in power systems*, IEEE T-SG 2(2), 2011. *(LRA)*
- Haghshenas, Hasnat & Naeini, *A Temporal GNN for Cyber Attack Detection and Localization in Smart Grids*, IEEE ISGT 2023. *(ramp)*
- Zaman & Lin, *PING: Physics-Informed GNNs to Generalize FDIA Localization*, NAPS 2025. *(measurement model)*
- Asprou, Kyriakides & Albu, *Variable Weights in a WLS State Estimator*, IEEE T-IM 63, 2014. *(meter noise)*
- Boyaci et al., *Joint Detection and Localization of Stealth FDIA*, IEEE T-SG, 2022. *(protocol)*

## License

Data under CC BY 4.0, code under MIT (see `LICENSE`). Synthetic, from public IEEE cases. Not for
operational decisions.
