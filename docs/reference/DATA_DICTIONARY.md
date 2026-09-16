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
| `edge_clean_full` | `[E,2]` | same as `edge_x` | noiseless true flows on every branch, metered or not (computed from `clean` through `yf` on load; equals `edge_clean` where a flow meter exists) |

## Labels: `y` says which buses, `family` says which attack

Two fields, two questions. `y` `[N]` is binary per bus because a bus is either tampered with or not.
`family` is one code per record because the generator injects one attack family per record, on one
or more buses at once; it never mixes families within a record, and stream episodes never overlap,
so each frame has one active family too. That is a property of how these shards were built, not of
the schema: concurrent attacks of different families would need a separate per-bus family map, say
`bus_family` `[N]` with 0 on clean buses (so `y = bus_family > 0`), added next to the scalar `family`
in a new data release.

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
| 2 | `b` | total line-charging susceptance. The pi model puts `b/2` at each end; it is stored whole, not halved |
| 3 | `g` | total charging conductance (transformer iron losses, 0 on lines), also stored whole |
| 4 | `gs` | series **admittance**, real part (conductance). `Y = 1/(r+jx) = gs + j·bs` |
| 5 | `bs` | series admittance, imag part (susceptance) |
| 6 | `tap` | transformer tap ratio (1.0 = plain line) |
| 7 | `shift` | transformer phase shift, degrees (0 = plain line) |

- `r,x` and `gs,bs` are the same branch inverted (`Y = 1/Z`). Both ship so you never compute one from the other.
- Bus shunts are not branch properties and are not in this table: `ds.bus_shunt_g` / `ds.bus_shunt_b` in
  MW / MVAr at 1 pu (divide by `ds.baseMVA` for per-unit admittance). `ds.ybus` already includes them on
  the diagonal, together with the halved charging and the tap / shift handling.
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
here are the flows.

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
| `yf`, `yt` | dataset attributes | from-end and to-end branch admittance matrices `[E,N]` (rows follow `edge_index`, columns `node_x` bus order): `ds.yf` / `ds.yt` (torch complex128) or `ds.yf_np` / `ds.yt_np`. For a complex bus voltage vector `V`, `V[from] * conj(Yf @ V)` is the from-end branch flow in per-unit (times `baseMVA` gives `edge_x` / `edge_clean` units). Same construction as makeYbus |
| `stealthy` | scalar | 1 for the re-solve families `Aq`/`At`/`Al` (BDD-evading by construction), 0 for benign and `Ad`/`As`/`Ar` |
| `split` | scalar | 0/1/2 = train/val/test |
| `timestep` | scalar | position in the source load profile |

### `clean`, `edge_clean` and `edge_clean_full` (v0.7.2+, `edge_clean_full` v0.15.0+)

- The noiseless, attack-free truth at the record's timestep, in `node_x` column order.
- `clean` covers every bus with no mask, whatever the meter placement.
- `edge_clean` zeroes unmetered branches, like `edge_x`.
- `edge_clean_full` is the same true flow on every branch, metered or not, computed from `clean`
  through `yf` when the shard is loaded (`V[from] * conj(Yf @ V)`, times `baseMVA`). It equals
  `edge_clean` wherever a flow meter exists, so a graph model can supervise every edge.
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

<!-- models:begin -->
## Models

Every dict the package returns is a typed bundle (`fdia_graph.models.Bundle`): a frozen
dataclass that is also the `dict` it always was, so `out["node_x"]`, `**out` and iteration
keep working and `out.node_x` is new. Fields set to None are absent from the dict. Generated
by `tools/models_doc.py` from the dataclasses.

### `RecordBundle` (`fdia_graph.dataset`)

One record as `FdiaGraph[i]` returns it (format="torch"): tensors in self.units, the static graph shared by every record, the label and provenance, and the optional layers the file carries. A dict as well, so DataLoaders, `**item` and `item["node_x"]` keep working.

| field | dict key | type | meaning |
|---|---|---|---|
| `edge_index` | `edge_index` | Any | [2, E] long, the same tensor for every record |
| `node_x` | `node_x` | Any | [N, 4] &#124;V&#124;, P_inj, Q_inj, theta |
| `node_m` | `node_m` | Any | [N, 4] meter mask |
| `edge_x` | `edge_x` | Any | [E, 2] P_from, Q_from |
| `edge_m` | `edge_m` | Any | [E, 2] flow-meter mask |
| `y` | `y` | Any | [N] per-bus attack label |
| `family` | `family` | int | 0 benign, 1 Aq, 2 Ad, 3 As, 4 Ar, 5 At, 6 Al |
| `stealthy` | `stealthy` | int | 1 for the re-solve families Aq, At, Al |
| `seq_id` | `seq_id` | int | source sequence of the record |
| `timestep` | `timestep` | int | position in the source load profile |
| `edge_attr` | `edge_attr` | Any (optional) | [E, 8] per-unit line physics (v0.5.0+ shards) |
| `temporal_delta` | `temporal_delta` | Any (optional) | [N, 2] injection change vs the previous pool scan (v0.3+) |
| `swing` | `swing` | Any (optional) | [N, 2] that change as a z-score of the bus's typical recent change (v0.4.1+) |
| `clean` | `clean` | Any (optional) | [N, 4] noiseless attack-free truth at the record's timestep (v0.7.2+) |
| `edge_clean` | `edge_clean` | Any (optional) | [E, 2] exact true flows on metered branches |
| `edge_clean_full` | `edge_clean_full` | Any (optional) | [E, 2] exact true flows on every branch |

### `BatchBundle` (`fdia_graph.dataset`)

A batch of records as `FdiaGraph.collate` builds it: per-record tensors stacked along a leading batch axis B, the static graph once, scalar metadata as long tensors.

| field | dict key | type | meaning |
|---|---|---|---|
| `node_x` | `node_x` | Any | [B, N, 4] |
| `node_m` | `node_m` | Any | [B, N, 4] |
| `edge_x` | `edge_x` | Any | [B, E, 2] |
| `edge_m` | `edge_m` | Any | [B, E, 2] |
| `y` | `y` | Any | [B, N] |
| `temporal_delta` | `temporal_delta` | Any (optional) | [B, N, 2] |
| `swing` | `swing` | Any (optional) | [B, N, 2] |
| `clean` | `clean` | Any (optional) | [B, N, 4] |
| `edge_clean` | `edge_clean` | Any (optional) | [B, E, 2] |
| `edge_clean_full` | `edge_clean_full` | Any (optional) | [B, E, 2] |
| `edge_index` | `edge_index` | Any (optional) | [2, E], the first record's (the same for every record) |
| `edge_attr` | `edge_attr` | Any (optional) | [E, 8], the first record's |
| `family` | `family` | Any (optional) | [B] long |
| `stealthy` | `stealthy` | Any (optional) | [B] long |
| `seq_id` | `seq_id` | Any (optional) | [B] long |
| `timestep` | `timestep` | Any (optional) | [B] long |

### `ArraysBundle` (`fdia_graph.dataset`)

A whole split of n records as `to_numpy` (arrays), `to_torch` (tensors) or `to_tf` return it: every per-record field that was requested and the file carries, plus the static graph. Fields not requested are absent from the dict view.

| field | dict key | type | meaning |
|---|---|---|---|
| `edge_index` | `edge_index` | Any (optional) | [2, E] |
| `edge_reactance` | `edge_reactance` | Any (optional) | [E], deprecated units, kept for old callers |
| `node_x` | `node_x` | Any (optional) | [n, N, 4] |
| `node_m` | `node_m` | Any (optional) | [n, N, 4] |
| `edge_x` | `edge_x` | Any (optional) | [n, E, 2] |
| `edge_m` | `edge_m` | Any (optional) | [n, E, 2] |
| `y` | `y` | Any (optional) | [n, N] |
| `temporal_delta` | `temporal_delta` | Any (optional) | [n, N, 2] |
| `swing` | `swing` | Any (optional) | [n, N, 2] |
| `clean` | `clean` | Any (optional) | [n, N, 4] |
| `edge_clean` | `edge_clean` | Any (optional) | [n, E, 2] |
| `edge_clean_full` | `edge_clean_full` | Any (optional) | [n, E, 2] |
| `family` | `family` | Any (optional) | [n] |
| `stealthy` | `stealthy` | Any (optional) | [n] |
| `seq_id` | `seq_id` | Any (optional) | [n] |
| `timestep` | `timestep` | Any (optional) | [n] |

### `Summary` (`fdia_graph.dataset`)

`FdiaGraph.summary()`: the system, its size, the number of records in the view, and the record count per family present.

| field | dict key | type | meaning |
|---|---|---|---|
| `system` | `system` | int | bus count of the system (14, 118, 300, ...) |
| `N` | `N` | int | buses |
| `E` | `E` | int | branches |
| `n` | `n` | int | records in this view |
| `families` | `families` | Dict[str, int] | record count per family name present |

### `Stream` (`fdia_graph.streams`)

A continuous attacked time series as `generate_stream` and `load_stream` return it: three aligned measurement layers per frame, the same three for branch flows, the static graph and meter masks, labels, the two temporal features, and the episode list. A dict as well, so `windows`, `pyg_stream` and every `s["node_x"]` keep working.

| field | dict key | type | meaning |
|---|---|---|---|
| `node_x` | `node_x` | array | [T, N, 4] observed |
| `benign` | `benign` | array | [T, N, 4] attack removed, noise kept |
| `clean` | `clean` | array | [T, N, 4] noiseless truth |
| `edge_x` | `edge_x` | array | [T, E, 2] observed flows |
| `edge_benign` | `edge_benign` | array | [T, E, 2] attack removed, noise kept |
| `edge_clean` | `edge_clean` | array | [T, E, 2] noiseless true flows |
| `edge_index` | `edge_index` | array | [2, E] |
| `edge_attr` | `edge_attr` | array | [E, 8] |
| `node_m` | `node_m` | array | [N, 4] |
| `edge_m` | `edge_m` | array | [E, 2] |
| `y` | `y` | array | [T, N] |
| `family` | `family` | array | [T] |
| `temporal_delta` | `temporal_delta` | array | [T, N, 2] |
| `swing` | `swing` | array | [T, N, 2] |
| `timestep` | `timestep` | array | [T] |
| `episodes` | `episodes` | Any | list of {onset, length, family, buses} |
| `system` | `system` |  (optional) | bus count, set by generate_stream |
| `attacked_frac` | `attacked_frac` |  (optional) | fraction of frames with an attacked bus, set by generate_stream |

### `EstimatorScores` (`fdia_graph.se.base`)

`SEBase.score`: the error pair of every record class present and their geometric mean (`geo`, the estimation paper's table cell). Indexable by family name as before, `geo` last.

| field | dict key | type | meaning |
|---|---|---|---|
| `geo` | `geo` | ErrorPair | geometric mean over the classes present |
| `benign` | `benign` |  (optional) | attack-free records |
| `Aq` | `Aq` |  (optional) | stealthy re-solve attack |
| `Ad` | `Ad` |  (optional) | additive bias |
| `As` | `As` |  (optional) | scaling |
| `Ar` | `Ar` |  (optional) | replay |
| `At` | `At` |  (optional) | slow ramp |
| `Al` | `Al` |  (optional) | load redistribution |

### `ErrorPair` (`fdia_graph.se.base`)

Mean absolute error of one record class: angles in degrees, voltage magnitudes per unit.

| field | dict key | type | meaning |
|---|---|---|---|
| `angle_mae_deg` | `angle_mae_deg` | float | mean &#124;theta_hat - theta&#124; over buses and records, degrees |
| `voltage_mae_pu` | `voltage_mae_pu` | float | mean &#124;&#124;V&#124;_hat - &#124;V&#124;&#124; over buses and records, per unit |

### `LocalizerScores` (`fdia_graph.localization.base`)

`LocalizerBase.score`: `all` (pooled), `benign`, and one entry per attacked family present. Indexable by family name as before.

| field | dict key | type | meaning |
|---|---|---|---|
| `all` | `all` | OverallMetrics | pooled over every record |
| `benign` | `benign` |  (optional) | attack-free records |
| `Aq` | `Aq` |  (optional) | stealthy re-solve attack |
| `Ad` | `Ad` |  (optional) | additive bias |
| `As` | `As` |  (optional) | scaling |
| `Ar` | `Ar` |  (optional) | replay |
| `At` | `At` |  (optional) | slow ramp |
| `Al` | `Al` |  (optional) | load redistribution |

### `OverallMetrics` (`fdia_graph.localization.base`)

Pooled over every record, benign included: the papers' per-bus macro F1, detection rate and false-positive rate over the attackable buses, and the micro node F1.

| field | dict key | type | meaning |
|---|---|---|---|
| `macro_f1` | `macro_f1` | float | mean per-bus F1 over the attackable buses |
| `macro_dr` | `macro_dr` | float | mean per-bus detection rate over the attackable buses |
| `macro_fr` | `macro_fr` | float | mean per-bus false-positive rate on benign records |
| `node_f1` | `node_f1` | float | micro F1 over every bus call |

### `BenignMetrics` (`fdia_graph.localization.base`)

On benign records: the record-level false-alarm rate and the mean per-bus alarm rate.

| field | dict key | type | meaning |
|---|---|---|---|
| `false_alarm_rate` | `false_alarm_rate` | float | benign records with any bus flagged |
| `bus_alarm_rate` | `bus_alarm_rate` | float | mean per-bus flag rate on benign records (calibrated to fa_target) |

### `FamilyMetrics` (`fdia_graph.localization.base`)

On one attacked family: strict localization accuracy, micro node precision/recall/F1, per-bus macro F1 over the buses the family attacks, per-sample F1, and the record-level detection rate.

| field | dict key | type | meaning |
|---|---|---|---|
| `strict_acc` | `strict_acc` | float | records whose flagged set equals the attacked set |
| `node_precision` | `node_precision` | float | micro precision over bus calls |
| `node_recall` | `node_recall` | float | micro recall over bus calls |
| `node_f1` | `node_f1` | float | micro F1 over bus calls |
| `macro_f1` | `macro_f1` | float | mean per-bus F1 over the buses the family attacks |
| `sample_f1` | `sample_f1` | float | mean per-record F1 |
| `detection_rate` | `detection_rate` | float | records with any bus flagged |

### `JacobianOutputs` (`fdia_graph.se.jacobian`)

`JacobianFeatures.transform`: the per-bus block [n, N, 8], the global features [n, 4] (under the dict key "global"), the implied state change [n, SD] and the unexplained residual [n, m].

| field | dict key | type | meaning |
|---|---|---|---|
| `bus` | `bus` | array | [n, N, 8] per-bus features |
| `global_` | `global` | array | [n, 4] per-record features, dict key "global" |
| `dx_hat` | `dx_hat` | array | [n, 2N-1] implied state change (H^T W H)^-1 H^T W dz |
| `r_perp` | `r_perp` | array | [n, m] residual the Jacobian cannot explain, (I - P) dz |

### `LineCandidate` (`fdia_graph.engine.core`)

One line of `line_outage_candidates`: its pandapower index and branch position, terminals, name, intact-case active flow, and, when rejected, the reason.

| field | dict key | type | meaning |
|---|---|---|---|
| `line` | `line` | int | pandapower line index, the generate(..., outage=) argument |
| `pos` | `pos` | int | branch position in edge_index |
| `from_bus` | `from_bus` | int | from-end bus |
| `to_bus` | `to_bus` | int | to-end bus |
| `name` | `name` | str | line name from the case, or line<idx> |
| `base_flow_mw` | `base_flow_mw` | float | active flow in the intact case, MW |
| `reason` | `reason` |  (optional) | why the line was rejected (islands the grid), rejected list only |
<!-- models:end -->
