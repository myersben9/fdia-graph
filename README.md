# fdia-graph

**Load and generate ML-only *dangerous* FDIA localization datasets for power grids — PyTorch-ready, zero data plumbing.**

`fdia-graph` gives power-systems / ML researchers one-line access to a benchmark of **false-data-injection attacks that evade every classical detector but are localizable only by a model** — on realistic sparse SCADA/PMU measurement graphs for IEEE-14 / 118 / 300. Install it, call `load(...)`, and train.

```python
import fdia_graph as fg

ds = fg.load("ieee118", split="train")     # auto-downloads + caches the newest release
loader = ds.loader(batch_size=64)          # ready-to-train PyTorch DataLoader
for batch in loader:
    batch["node_x"], batch["edge_x"], batch["edge_index"], batch["y"], batch["family"], ...
```

---

## Why this dataset

Most FDIA benchmarks contain attacks a bad-data detector (BDD) catches, so "ML beats BDD" is unsurprising. This dataset is built around the opposite: **three stealthy families that provably evade classical detection** — and are genuinely dangerous — alongside three detectable families as a contrast set.

| Family | Type | Classical BDD |
|--------|------|---------------|
| `Ao`   | state-consistent load redistribution | **evades (stealthy)** |
| `ramp` | slow temporal creeping redistribution, multi-timestep (Haghshenas et al., IEEE ISGT 2023) | **evades (stealthy)** |
| `LRA`  | targeted masked-overload (Yuan, Li & Ren, IEEE T-SG 2011) | **evades (stealthy)** |
| `Ad`   | random meter corruption | caught |
| `As`   | meter scaling | caught |
| `Ar`   | replay | mostly caught |

Each record is a **PING-style measurement graph** (branch flows as edge features, metered injections + |V| + sparse PMU angles as node features, with availability masks) — what a real EMS actually sees (redundancy ≈ 2–3), not the full-injection idealization.

## Install

```bash
pip install fdia-graph              # loader (numpy + h5py)
pip install "fdia-graph[torch]"     # + PyTorch Dataset/DataLoader
pip install "fdia-graph[pyg]"       # + torch_geometric graph format
pip install "fdia-graph[generate]"  # + pandapower, to generate custom datasets
pip install "fdia-graph[all]"       # everything
```

## Load

```python
ds  = fg.load("ieee300", split="train")                 # 60/20/20 chronological split
val = fg.load("ieee300", split="val")
tst = fg.load("ieee300", split="test")

# family subsets and the unseen-attack generalization protocol
stealthy = fg.load("ieee118", split="test", families=["Ao", "ramp", "LRA"])
heldout  = fg.load("ieee118", split="train", heldout=True)   # As/Ar excluded from train (Boyaci et al. 2022)
```

### Any framework you like

The `.loader()` streams records for training; these pull the whole split at once for analysis:

```python
ds = fg.load("ieee14", split="test")
arrays = ds.to_numpy()      # dict of numpy arrays
tensors = ds.to_torch()     # dict of torch tensors  (float32 features, int64 labels)
tf_t   = ds.to_tf()         # dict of tf.Tensors      (needs tensorflow)
df     = ds.to_pandas()     # flat pandas DataFrame, one row per record
```

### Dataset versioning

Datasets are GitHub **releases**, so your group version-controls them:

```python
fg.load("ieee118")                      # newest release (default — everyone stays current)
fg.load("ieee118", release="v0.3.0")    # pin an exact version for a reproducible experiment
```

## Generate a custom dataset (research knobs)

Turn any research knob and load the result by name — no data plumbing:

```python
fg.generate("ieee118", name="high_intensity",
            per_family=5000,             # samples per attack family
            families=["Ao", "ramp", "LRA"],
            attack_intensity=0.25,       # per-bus load-shift magnitude (± fraction)
            ramp_rate=0.003, ramp_len=80,
            n_benign=30000,
            redundancy={"pmu_frac": 0.3, "flow_frac": 0.95},
            split=(0.7, 0.15, 0.15),
            seed=7)

ds = fg.load("high_intensity", split="train")   # your custom dataset, ready to train
```

Generation ships with compact operating-point **pools** (a few MB/system), so you never need the raw simulation data. Benign records are emitted *exactly* from the stored operating state (0-error AC flows); only attacks re-solve a power flow.

## Schema

One HDF5 file per system (`ml_only_ieee{14,118,300}.h5`), with `N` = buses and `E` = branches (lines + transformers). The **static graph** is stored once; everything else is **per record** (`T` records total). A record is one realistic measurement snapshot — benign or attacked.

**Static graph** (read once, shared by every record):

| Field | Shape | Dtype | Meaning |
|-------|-------|-------|---------|
| `edge_index`     | `[2, E]` | int64   | `[from_bus; to_bus]` for each branch (lines first, then transformers) |
| `edge_reactance` | `[E]`    | float32 | per-branch reactance (p.u.) — handy for physics-informed models |

**Per-record measurement graph** (indexed `0 … T-1`; access one via `ds[i]`, or a whole split via `ds.to_numpy()`):

| Field | Shape | Dtype | Meaning |
|-------|-------|-------|---------|
| `node_x`         | `[N, 4]` | float32 | node features `[ \|V\| (p.u.),  P_inj (MW),  Q_inj (MVAr),  θ (deg) ]` |
| `node_m`         | `[N, 4]` | float32 | node **availability mask** (1 = that meter exists at that bus, else 0; masked entries are zeroed) |
| `edge_x`         | `[E, 2]` | float32 | branch-flow features `[ P_from (MW),  Q_from (MVAr) ]` |
| `edge_m`         | `[E, 2]` | float32 | edge availability mask |
| `y`              | `[N]`    | float32 | **localization target** — per-bus label, `1` = bus is attacked, `0` = clean |
| `temporal_delta` | `[N, 2]` | float32 | *(v0.3+)* current-minus-previous-scan injection `[ΔP_inj, ΔQ_inj]` — the temporal feature for replay/ramp |
| `family`         | scalar   | int     | `0` benign · `1` Ao · `2` Ad · `3` As · `4` Ar · `5` ramp · `6` LRA |
| `stealthy`       | scalar   | int     | `1` if the attack evades classical bad-data detection (Ao/ramp/LRA), else `0` |
| `split`          | scalar   | int     | `0` train · `1` val · `2` test (60/20/20 chronological, sequence-boundary safe) |
| `seq_id`         | scalar   | int     | ramp-sequence id (`≥0` groups the scans of one multi-timestep ramp); `-1` otherwise |
| `timestep`       | scalar   | int     | source operating-point index (the benign snapshot the record was built from) |
| `gap`            | scalar   | int     | `1` if this is a physics non-convergence NA row (`≈0%` in the shipped data) |

Sparsity is real: `node_m`/`edge_m` encode a redundancy of ≈ 2–3 (a realistic EMS regime), so a model must consume the masks — not every bus is metered, and PMU angles (`θ`) are sparse.

## Project structure

```
fdia-graph/
├── pyproject.toml              # package metadata, deps, optional extras ([torch], [pyg], [generate], [all])
├── README.md                   # this file
├── src/fdia_graph/
│   ├── __init__.py             # public API: load() and generate()  ← start here
│   ├── registry.py             # dataset version control: name → GitHub release + file; latest vs pinned; cache dir
│   ├── download.py             # fetches release-asset .h5 shards (private-repo token auth) → ~/.cache/fdia_graph
│   ├── dataset.py              # FdiaGraph: torch Dataset over one .h5 (lazy slicing, split/family filters, exporters)
│   ├── generate.py             # generate(): tunable-knob dataset creation → new .h5, registered by name
│   └── _core.py                # generation engine: physics (Ybus/PTDF) + the 7 attack families
└── examples/
    ├── quickstart.py           # smallest load → train loop
    ├── train_gnn.py            # baseline GNN localizer
    ├── train_arma.py           # ARMA + physics-biased attention localizer (the strong model)
    ├── train_tgnn.py           # temporal-graph localizer
    ├── hpo_arma.py             # Optuna hyperparameter search (Boyaci-style space)
    └── reproduce_report.ipynb  # end-to-end notebook: generate → figures → train/test
```

Read order to understand the codebase: `__init__.py` (the two entry points) → `registry.py`/`download.py` (how a name becomes a local `.h5`) → `dataset.py` (how that `.h5` becomes tensors) → `generate.py`/`_core.py` (how new datasets are made). Every module is commented line-by-line.

## Evaluation protocol

60/20/20 **chronological** split cut on sequence boundaries (ramp sequences never straddle a split — a random shuffle would leak them). Equal count per attack family. **Report per-attack-type node-F1**, not accuracy — the stealthy families (`Ao`/`ramp`/`LRA`) are the hard ones and accuracy hides them.

## Citation

If you use this dataset, please cite the attack model and measurement-model sources:

- Yuan, Li & Ren, *Modeling load redistribution attacks in power systems*, IEEE Trans. Smart Grid 2(2), 2011. *(LRA attack)*
- Haghshenas, Hasnat & Naeini, *A Temporal Graph Neural Network for Cyber Attack Detection and Localization in Smart Grids*, IEEE ISGT 2023. *(ramp attack)*
- Zaman & Lin, *PING: Physics-Informed GNNs to Generalize FDIA Localization*, NAPS 2025. *(measurement model)*
- Boyaci et al., *Joint Detection and Localization of Stealth FDIA*, IEEE Trans. Smart Grid, 2022. *(protocol)*

## License

MIT.
