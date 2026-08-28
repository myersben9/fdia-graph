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

**New here?** [`docs/ROADMAP.md`](docs/ROADMAP.md) — file map + how it all connects · [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) — what every array means · [`docs/CONCEPTS_TO_CODE.md`](docs/CONCEPTS_TO_CODE.md) — paper equations → functions · [`docs/EXAMPLES.md`](docs/EXAMPLES.md) — runnable baselines, streams, stats.

## Install

```bash
pip install fdia-graph              # loader
pip install "fdia-graph[torch]"     # + PyTorch DataLoader
pip install "fdia-graph[pyg]"       # + torch_geometric
pip install "fdia-graph[generate]"  # + pandapower, to generate custom data
```

Data is pinned per SDK version and cached in `~/.cache/fdia_graph`. Pin a version with
`fg.load(..., release="v0.7.1")`; `pip install --upgrade fdia-graph` moves it forward.

## Load

```python
fg.load("ieee300", split="train")                              # 60/20/20 chronological split
fg.load("ieee118", split="test", families=["Aq","At","Al"])    # family subset
fg.load("ieee118", units="pu")                                 # per-unit + radians (default is physical)
```

Whole split at once: `ds.to_numpy()` / `.to_torch()` / `.to_pandas()`. Custom data:
`fg.generate(system, name, per_family=..., attack_intensity=..., ...)` then `fg.load(name)`.
Continuous timeline for LSTM/TGN: `fg.load_stream(system)`. Both in [`docs/EXAMPLES.md`](docs/EXAMPLES.md).

## Data

Each record is a sparse measurement graph with **N buses** (nodes) and **E branches** (edges). Read a shape as "values per item": `[N,4]` = 4 numbers per bus, `[E,8]` = an 8-dim vector per branch, `[2,E]` = 2 rows × E branches. Full reference in [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

```
node_x [N,4] = [ |V|, P_inj, Q_inj, theta ]      bus meters      node_m [N,4] = mask
edge_x [E,2] = [ P_from, Q_from ]                branch flow     edge_m [E,2] = mask
edge_index [2,E] = [ from_bus; to_bus ]          connectivity
edge_attr  [E,8] = [ r,x,b,g,gs,bs,tap,shift ]   static line physics  (ds.edge_attr)
y [N] = attacked (1) / clean (0)                 localization target
swing [N,2], temporal_delta [N,2]                temporal features
```

## Attacks

Three **stealthy** families that evade classical bad-data detection, plus three **detectable** ones as a
contrast set. Every per-bus change stays in a plausibility band (≈2% noise floor to 20% cap); meter error
follows an accuracy-class model.

| family | attack | classical BDD |
|--------|--------|---------------|
| `Aq` | stealthy load rescale + AC re-solve | evades |
| `At` | slow temporal load ramp | evades |
| `Al` | targeted load redistribution (hides overloads) | evades |
| `Ad` / `As` / `Ar` | meter corruption / scaling / replay | caught |

Report **per-family node-F1** with the false-alarm rate, not accuracy (clean buses dominate). A lightweight
per-bus MLP reaches ~0.92 localization macro-F1; see [`docs/EXAMPLES.md`](docs/EXAMPLES.md).

## Citation

Cite the attack- and measurement-model sources:

- Yuan, Li & Ren, *Modeling load redistribution attacks in power systems*, IEEE T-SG 2(2), 2011. *(LRA)*
- Haghshenas, Hasnat & Naeini, *A Temporal GNN for Cyber Attack Detection and Localization in Smart Grids*, IEEE ISGT 2023. *(ramp)*
- Zaman & Lin, *PING: Physics-Informed GNNs to Generalize FDIA Localization*, NAPS 2025. *(measurement model)*
- Asprou, Kyriakides & Albu, *Variable Weights in a WLS State Estimator*, IEEE T-IM 63, 2014. *(meter noise)*
- Boyaci et al., *Joint Detection and Localization of Stealth FDIA*, IEEE T-SG, 2022. *(protocol)*

## License

Data under CC BY 4.0, code under MIT (see `LICENSE`). Synthetic, from public IEEE cases — not for
operational decisions.
