# fdia-graph

**Load and generate stealthy FDIA localization datasets for power grids — PyTorch-ready, one line, zero data plumbing.**

A benchmark of false-data-injection attacks that evade classical bad-data detection but are localizable by a model, on realistic sparse SCADA/PMU measurement graphs for IEEE-14 / 118 / 300.

```python
import fdia_graph as fg

ds = fg.load("ieee118", split="train")     # auto-downloads + caches the newest release
loader = ds.loader(batch_size=64)          # ready-to-train PyTorch DataLoader
for batch in loader:
    batch["node_x"], batch["edge_x"], batch["edge_index"], batch["y"], batch["family"], ...
```

![Attack traces over time](docs/fig_traces.png)

*One IEEE-118 bus under each attack (dashed = benign load). Only the slow ramp `At` perturbs many scans.*

![Per-bus attack magnitude](docs/fig_band.png)

*Attack magnitude per family, inside the 2%–20% plausibility band (dotted floor, dashed cap).*

![Swing vs attack move](docs/fig_swing.png)

*Attacked-bus swing rises with attack size; below ~2% it falls into the benign floor (dotted).*

![Chi-square J vs threshold](docs/fig_bdd.png)

*Bad-data statistic per family. Stealthy families stay below the alarm line (1×), detectable ones sit well above.*

## Why this dataset

Most FDIA benchmarks contain attacks a bad-data detector (BDD) catches, so "ML beats BDD" is unsurprising. This one is built around the opposite: three stealthy families that provably evade classical detection and are genuinely dangerous, alongside three detectable families as a contrast set.

| Family | Type | Classical BDD |
|--------|------|---------------|
| `Aq`   | stealthy load scaling: bounded per-bus rescale + AC re-solve — **our contribution** (cf. Boyaci `Ao`, Liu FDIA) | **evades (stealthy)** |
| `At`   | temporal load surge, ramps up then back down (Haghshenas et al., IEEE ISGT 2023) | **evades (stealthy)** |
| `Al`   | targeted masked-overload / load redistribution (Yuan, Li & Ren, IEEE T-SG 2011) | **evades (stealthy)** |
| `Ad`   | random meter corruption | caught |
| `As`   | meter scaling | caught |
| `Ar`   | replay | mostly caught |

Two things keep it honest (v0.6.0): every attack's per-bus magnitude sits in a **plausibility band** — above a ~2% meter-noise floor (can't hide in noise) and below a 20% cap (stays realistic), with the slow ramp `At` exempt from the floor by design — and meter error follows an **accuracy-class model** (Asprou class-0.2, a fixed per-meter bias plus a per-scan jitter) rather than a single made-up variance. Each record is a PING-style measurement graph (branch flows as edges, metered injections + |V| + sparse PMU angles as nodes, with availability masks) at a realistic redundancy of ≈ 2–3.

## Install

```bash
pip install fdia-graph              # loader (numpy + h5py)
pip install "fdia-graph[torch]"     # + PyTorch Dataset/DataLoader
pip install "fdia-graph[pyg]"       # + torch_geometric graph format
pip install "fdia-graph[generate]"  # + pandapower, to generate custom datasets
pip install "fdia-graph[all]"       # everything (adds gridstatus for ISO load download)
```

Datasets, operating-point pools, and the whole simulation pipeline all ship with the package and its GitHub releases. There is no separate data drop to sync.

## Load

```python
ds  = fg.load("ieee300", split="train")                             # 60/20/20 chronological split
stealthy = fg.load("ieee118", split="test", families=["Aq","At","Al"])   # family subset ("Ao"/"ramp"/"LRA" alias)
heldout  = fg.load("ieee118", split="train", heldout=True)          # As/Ar excluded from train (Boyaci et al. 2022)

fg.load("ieee118", units="physical")   # default: |V| p.u., P/Q in MW/MVAr, θ in degrees
fg.load("ieee118", units="pu")         # everything per-unit on baseMVA, θ in radians (for ML/physics models)
fg.load("ieee118", release="v0.6.0")   # pin a version for a reproducible experiment (default is newest)
```

Pull a whole split at once for analysis with `ds.to_numpy()`, `ds.to_torch()`, `ds.to_tf()`, or `ds.to_pandas()`.

## Generate

Turn any research knob and load the result by name:

```python
fg.generate("ieee118", name="custom",
            per_family=5000, families=["Aq","At","Al"],
            attack_intensity=0.25,          # top of the plausibility band (floor is the ~2% noise level)
            n_benign=30000, seed=7,
            targeting="centrality",         # "uniform" (default) | "centrality" — bias toward critical buses
            targeting_strength=1.5)         # exponential tilt; 0 == uniform, larger = more concentrated

ds = fg.load("custom", split="train")
```

`targeting="centrality"` draws attacked buses toward the structurally critical ones (degree + closeness + betweenness centrality; Doostinia et al., IEEE T-IA 2025), modeling a smart attacker instead of a uniform random one. Each generated `.h5` also gets a `<out>.mag.npz` sidecar recording the designed per-bus magnitude and swing plus the band's floor/cap, so you can verify the attacks really landed inside the band.

Also supported: **N-1 line outages** (`outage=` builds a shard under a post-contingency topology) and **your own load profiles** (`fg.fetch_profile(...)` for NYISO/CAISO/ERCOT or bring your own array → `fg.generate_states(...)` → `fg.generate(...)`). See docstrings in `profiles.py` and `_core.py`.

## Schema

One HDF5 file per system (`ml_only_ieee{14,118,300}.h5`), `N` buses and `E` branches. The static graph is stored once; everything else is per record (`T` total). Access one record via `ds[i]`, a whole split via `ds.to_numpy()`.

**Static graph:** `edge_index [2,E]` (`[from_bus; to_bus]`, lines then transformers), `edge_reactance [E]` (p.u.).

**Per record:**

| Field | Shape | Dtype | Meaning |
|-------|-------|-------|---------|
| `node_x`         | `[N, 4]` | float32 | node features `[ \|V\|,  P_inj,  Q_inj,  θ ]` (units per `units=`) |
| `node_m`         | `[N, 4]` | float32 | node **availability mask** (1 = meter exists; masked entries zeroed) |
| `edge_x`         | `[E, 2]` | float32 | branch-flow features `[ P_from,  Q_from ]` |
| `edge_m`         | `[E, 2]` | float32 | edge availability mask |
| `y`              | `[N]`    | float32 | **localization target** — per-bus label, `1` = attacked |
| `temporal_delta` | `[N, 2]` | float32 | current-minus-previous-scan injection `[ΔP, ΔQ]` |
| `swing`          | `[N, 2]` | float32 | *(v0.6+)* windowed per-bus swing — reading minus recent-window mean, over recent-window std (spikes big, ramps small) |
| `family`         | scalar   | int     | `0` benign · `1` Aq · `2` Ad · `3` As · `4` Ar · `5` At · `6` Al |
| `stealthy`       | scalar   | int     | `1` if the attack evades classical BDD |
| `split`          | scalar   | int     | `0` train · `1` val · `2` test (60/20/20 chronological, sequence-safe) |
| `seq_id`         | scalar   | int     | ramp-sequence id (`≥0` groups a ramp's scans; `-1` otherwise) |
| `timestep`       | scalar   | int     | source operating-point index |
| `gap`            | scalar   | int     | `1` for a physics non-convergence NA row (`≈0%` shipped) |

Sparsity is real: `node_m`/`edge_m` encode redundancy ≈ 2–3, so a model must consume the masks.

## Evaluation protocol

60/20/20 **chronological** split cut on sequence boundaries (ramps never straddle a split). Equal count per family. **Report per-attack-type node-F1**, not accuracy — the stealthy families are the hard ones and accuracy hides them.

## Citation

If you use this dataset, please cite the attack- and measurement-model sources:

- Yuan, Li & Ren, *Modeling load redistribution attacks in power systems*, IEEE Trans. Smart Grid 2(2), 2011. *(LRA attack)*
- Haghshenas, Hasnat & Naeini, *A Temporal Graph Neural Network for Cyber Attack Detection and Localization in Smart Grids*, IEEE ISGT 2023. *(ramp attack)*
- Zaman & Lin, *PING: Physics-Informed GNNs to Generalize FDIA Localization*, NAPS 2025. *(measurement model)*
- Asprou, Kyriakides & Albu, *The Effect of Variable Weights in a WLS State Estimator Considering Instrument Transformer Uncertainties*, IEEE Trans. Instrum. Meas. 63, 2014. *(accuracy-class meter noise)*
- Doostinia, Falabretti, Verticale & Bolouki, *A Novel Centrality-Driven ML Approach for Clustering Critical Nodes in Cyber-Physical Power Systems*, IEEE Trans. Ind. Appl., 2025. *(centrality-guided targeting)*
- Boyaci et al., *Joint Detection and Localization of Stealth FDIA*, IEEE Trans. Smart Grid, 2022. *(protocol)*

## License

Dataset (HDF5 shards and operating-point pools) under CC BY 4.0; source code in `src/` and `examples/` under MIT. See `LICENSE`. The data is synthetic, generated from public IEEE test cases, and must not be used for operational decisions.
