# Load a dataset

This tutorial loads a released dataset and looks at it four ways: one record, one batch, one
whole split and one window. It takes a few minutes, most of it the download.

A dataset is one timeline: the meter readings of one test system, frame by frame, with attacks
placed in it and labelled per bus.

## Install

```bash
pip install "fdia-graph[torch]"
```

The plain `pip install fdia-graph` is enough for `summary()`, `export`, `windows` and
`episodes`. One record (step 2) and the batches (step 3) need PyTorch, and
`export(format="pandas")` needs pandas.

## 1. Load

```python
import fdia_graph as fg

train = fg.load("ieee14", split="train")
print(train.summary())
```

The first call downloads the IEEE-14 timeline, a file of about 87 MB, and keeps it in
`~/.cache/fdia_graph`. Later calls read the cached file. The eight released systems are
`"ieee14"` to `"ieee300"`; the IEEE-118 file is about 730 MB.

`summary()` gives the system, the number of buses `N` and branches `E`, the number of frames in
this view and how many belong to each attack family.

`fg.load` reads the static graph once and slices frames on request, so the whole file is never
in memory.

## 2. One record

A record is one frame.

```python
record = train[0]
print(record["node_x"].shape)   # [N, 4]
print(record["edge_x"].shape)   # [E, 2]
print(record["y"].shape)        # [N]
print(int(record["family"]))    # 0 is benign
```

| Field | Shape | Holds |
|---|---|---|
| `node_x` | `[N, 4]` | bus readings: voltage, P, Q, angle |
| `node_m` | `[N, 4]` | 1 where a meter exists |
| `edge_x` | `[E, 2]` | branch flow readings: P, Q at the from end |
| `y` | `[N]` | 1 at each attacked bus |
| `family` | one number | which attack; `fg.FAMILIES` names the codes |

A reading with no meter is stored as zero, so read the mask `node_m` with `node_x`.
The static graph is on the dataset, not on each record: `train.edge_index` (`[2, E]`),
`train.edge_attr` (`[E, 8]`, the branch properties) and `train.ybus` (the admittance matrix).
[The data dictionary](../reference/DATA_DICTIONARY.md) lists every field.

## 3. One batch

```python
loader = train.loader(batch_size=64)
batch = next(iter(loader))
print(batch["node_x"].shape)  # [64, N, 4]
print(batch["y"].shape)       # [64, N]
print(batch["edge_index"].shape)  # [2, E]
```

`loader` returns a PyTorch `DataLoader`. Each batch holds the per-frame fields stacked, and the
static graph (`edge_index`, `edge_attr`) once.

## 4. One export

`export` returns a whole split at once.

```python
arrays = train.export(format="numpy")
print(arrays["node_x"].shape)  # [T, N, 4]
print(arrays["y"].shape)       # [T, N]
print(arrays["family"].shape)  # [T]
```

| Call | Gives |
|---|---|
| `train.export()` | numpy arrays (the default) |
| `train.export(format="torch")` | torch tensors |
| `train.export(format="pandas")` | one data frame |
| `train.export(["node_x", "y"])` | only those fields |

Every frame comes in three layers. `node_x` is what the operator saw. `benign` is the same scan
with the attack removed. `clean` is the true state with no noise.

```python
metered = arrays["node_m"] == 1
attack = (arrays["node_x"] - arrays["benign"])[metered]
noise = (arrays["benign"] - arrays["clean"])[metered]
print(abs(attack).max(), abs(noise).max())
```

The two differences hold where a meter exists, which is why the snippet reads the mask.

[Layers and labels](../concepts/layers.md) explains the layers and which of them a detector may
read.

## 5. One window

A timeline is in time order, so it can be cut into windows for a sequence model.

```python
X, y = train.windows(W=24, stride=12)
print(X.shape)  # [n, 24, N, 4]
print(y.shape)  # [n, N]
```

Each window is 24 consecutive frames of `node_x`, and `y` marks a bus attacked at any frame
of the window.

| Argument | Default | Meaning |
|---|---|---|
| `W` | | the window length in frames |
| `stride` | 1 | the step between windows |
| `label` | `"any"` | `"any"`, `"last"` (the last frame's label) or `"frame"` (`[n, W, N]`) |
| `layer` | `"node_x"` | `"node_x"`, `"benign"` or `"clean"` |
| `per_bus` | `False` | one sequence per bus: `[n*N, W, 4]` |

The episodes of the view are a table:

```python
episodes = train.episodes
print(len(episodes.onset))
print(episodes.onset[:3], episodes.length[:3])
print(episodes.family[:3])
```

## Options of `fg.load`

```python
# one split: "train", "val" or "test"; None is every frame
fg.load("ieee14", split="test")
# a subset of the families
fg.load("ieee14", split="test", families=["At"])
# per unit and radians; the default is physical units
fg.load("ieee14", units="pu")
# the same frames as a shuffled record table
fg.load("ieee14", order="random", seed=0)
```

| Option | Default | Meaning |
|---|---|---|
| `split` | `None` | `"train"`, `"val"`, `"test"`, or every frame |
| `families` | `None` | keep only these families, by name or code |
| `units` | `"physical"` | `"physical"` (p.u., MW, MVAr, degrees) or `"pu"` (per unit, radians) |
| `order` | `"time"` | `"time"`, or `"random"`: a permutation fixed by `seed` |
| `format` | `"torch"` | `"torch"`, or `"pyg"` for torch_geometric records |
| `release` | `None` | a data release such as `"v0.8.1"`; `None` is the pinned one |
| `preload` | `False` | read the whole split into memory once |
| `heldout` | `False` | keep `As` and `Ar` out of train and validation [BOY22] |
| `seed` | 0 | the seed of the `"random"` order |

`windows` needs `order="time"` and raises `MissingCapability` otherwise.

## What the released files hold

The released files hold seven attack families on 72,000 frames per system, about half of the
frames under attack. The current generator
makes two of them, `At` and `Am`, with a different `Am`
([The attack families](../concepts/attacks.md#what-the-released-files-additionally-contain)).
To get a file with the current families, generate one: [Generate a timeline](generate.md).

## Where next

- [Generate a timeline](generate.md): make your own dataset.
- [Analyse a timeline](analyse.md): run an estimator and a localizer on one.
- [Examples](../reference/EXAMPLES.md): three runnable baselines.
