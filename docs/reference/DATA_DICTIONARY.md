# Data dictionary

What every array from `fg.load()` / `fg.load_stream()` holds.

**N** = number of buses (nodes), **E** = number of branches (edges). A shape is "values per item":
`[N,4]` = 4 numbers per bus, `[E,8]` = an 8-dim vector per branch, `[2,E]` = 2 rows × E branches.

## Cheat sheet

```
node_x [N,4]    = [ |V| , P_inj , Q_inj , theta ]     bus meter readings
node_m [N,4]    = 1 metered / 0 not (value zero-filled)
edge_x [E,2]    = [ P_from , Q_from ]                 power leaving branch (sign = direction)
edge_m [E,2]    = 1 metered / 0 not
edge_index [2,E]= [ from_bus ; to_bus ]               connectivity
edge_attr  [E,8]= [ r, x, b, g, gs, bs, tap, shift ]  static branch electrical properties
y [N]           = 1 attacked / 0 clean                localization target
temporal_delta [N,2] = scan-to-scan [ΔP, ΔQ]
swing [N,2]     = temporal_delta as a z-score of recent volatility
clean [N,4]     = [ |V| , P_inj , Q_inj , theta ]     noiseless truth, ALL buses (SE target, v0.7.2+)
edge_clean [E,2]= [ P_from , Q_from ]                 noiseless true flows (unmetered branches zeroed)
```

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
- In `format="pyg"`, `Data.edge_attr` is the `[E,2]` flows (also exposed as `Data.edge_x`). The `[E,8]` table is `ds.edge_attr`.

## Where the branch flows live, per format

| you have | flows `[E,2]` | flow mask `[E,2]` |
|---|---|---|
| dict record `ds[i]` or a `ds.loader()` batch | `["edge_x"]` | `["edge_m"]` |
| PyG `Data` or `DataBatch` (`format="pyg"`) | `.edge_attr` or `.edge_x` | `.edge_mask` |
| `ds.to_numpy()` | `["edge_x"]` `[n,E,2]` | `["edge_m"]` |

`ds.edge_x` on the dataset object itself is **not** the flows. It is the static per-unit series
reactance `[E]` (column 1 of `edge_attr`). Flows are per record, so they only exist on records and batches.

## The rest

| field | shape | meaning |
|---|---|---|
| `edge_x` | `[E,2]` | `[P_from, Q_from]`, power leaving the from-end (sign = direction). MW/MVAr or pu. |
| `node_m`, `edge_m` | `[N,4]`, `[E,2]` | `1` metered, `0` not. Metering is sparse: read the mask. |
| `edge_index` | `[2,E]` | row 0 from-bus, row 1 to-bus |
| `y` | `[N]` | per-bus attack label |
| `family` | scalar | 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At, 6 Al (Aq = paper `A_o`) |
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
