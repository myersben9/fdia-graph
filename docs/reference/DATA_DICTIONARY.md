# Data dictionary

What every array from `fg.load()` / `fg.load_stream()` holds.

**N** = number of buses (nodes), **E** = number of branches (edges). A shape is "values per item":
`[N,4]` = 4 numbers per bus, `[E,8]` = an 8-dim vector per branch, `[2,E]` = 2 rows × E branches.

## Cheat sheet

| field | shape | columns | meaning |
|---|---|---|---|
| `node_x` | `[N,4]` | <code>&#124;V&#124;</code>, `P_inj`, `Q_inj`, `theta` | bus meter readings |
| `node_m` | `[N,4]` | same columns | 1 metered, 0 not (value zero-filled) |
| `edge_x` | `[E,2]` | `P_from`, `Q_from` | power leaving the branch (sign = direction) |
| `edge_m` | `[E,2]` | same columns | 1 metered, 0 not |
| `edge_index` | `[2,E]` | row 0 `from_bus`, row 1 `to_bus` | connectivity |
| `edge_attr` | `[E,8]` | `r`, `x`, `b`, `g`, `gs`, `bs`, `tap`, `shift` | static branch electrical properties |
| `y` | `[N]` | | 1 attacked, 0 clean. Which buses |
| `family` | scalar | | 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At, 6 Al. Which attack |
| `temporal_delta` | `[N,2]` | `ΔP`, `ΔQ` | scan-to-scan injection change |
| `swing` | `[N,2]` | `ΔP`, `ΔQ` | `temporal_delta` as a z-score of recent volatility |
| `clean` | `[N,4]` | same as `node_x` | noiseless truth, all buses. The SE target (v0.7.2+) |
| `edge_clean` | `[E,2]` | same as `edge_x` | noiseless true flows, unmetered branches zeroed |

## Labels: `y` says which buses, `family` says which attack

Two fields, two questions. `y` `[N]` is binary per bus because a bus is either tampered with or not.
`family` is one code per record because the generator injects one attack family per record, on one
or more buses at once; it never mixes families within a record, and stream episodes never overlap,
so each frame has one active family too. That is a property of how these shards were built, not of
the schema: concurrent attacks of different families would need a per-bus family map (`family[N]`,
0 on clean buses, `y = family > 0`), which would come with a new data release.

| | shape | values | in a batch |
|---|---|---|---|
| `y` | `[N]` | 0 clean, 1 attacked | `batch["y"]` `[B,N]` (PyG: `batch.y` `[B*N]`) |
| `family` | scalar | 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At, 6 Al | `batch["family"]` `[B]` (PyG: `batch.family` `[B]`) |

- Names: `fg.FAMILIES[code]`. Stealthy subset: `fg.STEALTHY_FAMILIES` = `{1, 5, 6}`.
- Per-bus family label, if a model needs one: `y * family[:, None]` gives `[B,N]` with 0 on clean buses.
- Every attacked record flags at least one bus. Benign records flag none.
- Per-family evaluation: filter by `family` and score `y` inside each group. `fg.load(..., families=[...])`
  does the filtering at load time.

## `node_x` `[N,4]`: bus measurements (voltage first)

| col | name | physical units | pu units |
|-----|------|----------------|----------|
| 0 | <code>&#124;V&#124;</code> | per-unit | per-unit |
| 1 | `P_inj` | MW | pu |
| 2 | `Q_inj` | MVAr | pu |
| 3 | `theta` | degrees | radians |

Sign of `P_inj`/`Q_inj`: `+` = net consumption (load), `−` = net injection (gen). This is pandapower's
`res_bus` convention. Example (case14): the slack bus reads ≈ −235 MW, a 94 MW load bus reads ≈ +93 MW.

## `edge_attr` `[E,8]`: static branch properties (per-unit, never change)

| col | name | meaning |
|-----|------|---------|
| 0 | `r` | series **impedance**, real part (resistance). `Z = r + jx` |
| 1 | `x` | series impedance, imag part (reactance) |
| 2 | `b` | line-charging susceptance (shunt) |
| 3 | `g` | shunt conductance (usually 0) |
| 4 | `gs` | series **admittance**, real part (conductance). `Y = 1/(r+jx) = gs + j·bs` |
| 5 | `bs` | series admittance, imag part (susceptance) |
| 6 | `tap` | transformer tap ratio (1.0 = plain line) |
| 7 | `shift` | transformer phase shift, degrees (0 = plain line) |

- `r,x` and `gs,bs` are the same branch inverted (`Y = 1/Z`). Both ship so you never compute one from the other.
- In PyG format (`fg.load(..., format="pyg")` and `fg.pyg_stream`), `Data.edge_attr` is the `[E,2]` flows
  (also exposed as `Data.edge_x`) and this `[E,8]` table is `Data.edge_phys`. On the dataset object it is `ds.edge_attr`.

## Where the branch flows live, per format

| you have | flows `[E,2]` | flow mask `[E,2]` | static line physics `[E,8]` |
|---|---|---|---|
| dict record `ds[i]` or a `ds.loader()` batch | `["edge_x"]` | `["edge_m"]` | `["edge_attr"]` |
| PyG `Data` or `DataBatch` (`format="pyg"`, `fg.pyg_stream`) | `.edge_attr` or `.edge_x` | `.edge_mask` | `.edge_phys` |
| `ds.to_numpy()` | `["edge_x"]` `[n,E,2]` | `["edge_m"]` | `ds.edge_attr` |
| stream dict `fg.load_stream()` | `["edge_x"]` `[T,E,2]` | `["edge_m"]` | `["edge_attr"]` |

The `[E,8]` physics keep the name `edge_attr` on dicts and the dataset, and become `edge_phys` on PyG
objects because PyG's `edge_attr` slot is its conventional home for per-edge model input, which
here is the flows.

Flows are per record, so they only exist on records and batches. The static per-unit branch
physics live on the dataset as `ds.edge_attr` `[E,8]` and one column each as `ds.branch_r`,
`ds.branch_x`, `ds.branch_b`, `ds.branch_g`, `ds.branch_gs`, `ds.branch_bs`, `ds.branch_tap`,
`ds.branch_shift`. The older `ds.edge_r` … `ds.edge_shift` spellings of all eight still work but raise
a `DeprecationWarning`; the rename exists because `ds.edge_x` (the reactance) collided with the flows' name.

## The rest

| field | shape | meaning |
|---|---|---|
| `edge_x` | `[E,2]` | `[P_from, Q_from]`, power leaving the from-end (sign = direction). MW/MVAr or pu. |
| `node_m`, `edge_m` | `[N,4]`, `[E,2]` | `1` metered, `0` not. Metering is sparse: read the mask. |
| `edge_index` | `[2,E]` | row 0 from-bus, row 1 to-bus |
| `y`, `family` | `[N]`, scalar | see Labels above (Aq = paper `A_o`) |
| `slack` | dataset attribute | index of the reference (slack) bus, `ds.slack`. Derived from the clean layer, so v0.7.2+ only. Also `Data.slack` in PyG format |
| `ybus` | dataset attribute | full nodal admittance matrix `[N,N]`, complex per-unit, in `node_x` bus order: `ds.ybus` (torch complex128) or `ds.ybus_np`. Built from `edge_attr` and the bus shunts with pandapower's branch model, equal to the engine's matrix. Static per shard, so it lives on the dataset, not on each record |
| `stealthy` | scalar | 1 for the re-solve families `Aq`/`At`/`Al` (BDD-evading by construction), 0 for benign and `Ad`/`As`/`Ar` |
| `split` | scalar | 0/1/2 = train/val/test |
| `timestep` | scalar | position in the source load profile |

### `clean` and `edge_clean` (v0.7.2+)

- The noiseless, attack-free truth at the record's timestep, in `node_x` column order.
- `clean` covers every bus with no mask, whatever the meter placement.
- `edge_clean` zeroes unmetered branches, like `edge_x`.
- On benign records `node_x − clean` is the meter error.
- This is the state-estimation target.

## Streams (`fg.load_stream`)

Leading time axis `T`, three aligned layers each for node and edge:

| layer | meaning |
|-------|---------|
| `node_x` / `edge_x` | observed (attacked + noise). The model input. |
| `benign` / `edge_benign` | attack removed, noise kept |
| `clean` / `edge_clean` | noiseless true state. The SE target. |

- `benign − clean` = noise.
- `observed − benign` = the attack. Exact for Ad/As/Ar. For Aq/At/Al it also carries a noise term, so
  use `clean` as the SE target there.
