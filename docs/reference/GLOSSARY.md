# Glossary

One definition per term and where it lives in `src/fdia_graph`. The generation pipeline behind
the data terms is in [`../guides/generation.md`](../guides/generation.md); citation keys are in
[`REFERENCES.md`](REFERENCES.md).

## Data

| term | definition | in code |
|---|---|---|
| scan | One emission of every meter in the plan from one grid state: readings and masks. | `models.frames.Scan`, `engine.measurement.MeasurementMixin.emit_from_state` |
| frame | One row of a timeline: the scan at one pool timestep with the attack of its episode applied, or none. | `models.frames.Frame`, `timeline._TimelineBuffers.store` |
| record | One row as the loader serves it, `ds[i]`: a frame of a timeline, or a row of a record shard. | `dataset/records.py`, `models.data.RecordBundle` |
| timestep | The index of the pool state a frame was emitted from; on a timeline frame `t` is timestep `t`. | `data/timestep`, `schema.TIMESTEP` |
| episode | A run of consecutive frames under one attack of one family on one design; episodes never overlap. | `timeline._Episode`, `episodes/` group, `data/seq_id` |
| onset | The first frame of an episode, drawn uniformly among the positions where the episode fits. | `timeline._uniform_onset`, `episodes/onset` |
| pool | The operating-point pool: `[T, N, 4]` converged AC states in <code>&#124;V&#124;</code>, `P_inj`, `Q_inj`, `theta` order, one per minute of the load profile. | `profiles.generate_states`, `generation._load_states`, `pool_ieee{C}.h5` |
| timeline | One HDF5 file per system: every pool timestep scanned in time order, with attack episodes placed on it; the file format of every data release from v0.8.0. | `timeline.generate_timeline`, `schema.KIND_TIMELINE` |
| record shard (legacy) | The pre-v0.8.0 file layout, `ml_only_ieee{C}.h5`: independent records with gap rows, no `benign/`, `episodes/` or `attack/` groups; still readable with `release="v0.7.2"`. | `registry.dataset_file`, `schema.GAP` |
| release | A tagged set of data assets (`data-v<x.y.z>` from v0.8.0), pinned by sha256; `registry._RELEASE` is the default. | `registry.py`, `tools/upload_assets.py` |
| split | The chronological train, val and test partition (0, 1, 2), boundaries moved so no episode is cut. | `timeline._frame_split`, `data/split`, `schema.SPLIT_CODE` |
| family | The attack type of a frame: 0 benign, 1 `Aq`, 2 `Ad`, 3 `As`, 4 `Ar`, 5 `At`, 6 `Al`, 7 `Am`. | `schema.FAMILIES`, `engine.FAM_ID`, `data/family` |
| stealthy | 1 for the families built as local false states (`Aq`, `At`, `Al`, `Am`), which pass the residual test. | `schema.STEALTHY_FAMILIES`, `data/stealthy` |
| local false state | A state in which only a subnetwork around the targets is re-solved under false loads, boundary voltages held true, and the attack vector `h(x_false) - h(x_true)` is added to the true scan [WU26]. | `engine.records._stealthy_frame`, `engine.physics.PhysicsMixin.solve_local`, `local_region` |
| theta_ref | The one slack angle every estimate is referenced to, fixed at fit time from the training split. | `se.base.SEBase._fit_reference`, `ref_angles` |
| active buses | The buses attacked somewhere in the records being scored; macro scores average over them. | `models.scores.OverallMetrics`, `formulas.metrics.tau_from_counts` |
| attackable buses | Buses whose load an attack may target: nonzero active load, not the slack, at most `max_load_mw`; the stealthy families also skip generator buses [BOY22]. A learned localizer's `_attackable` is the buses labelled attacked in its training split. | `engine.core.FdiaGenerator._load_tables` (`attackable_pos`, `stealthy_pos`), `localization.learned.LearnedLocalizer` |

## Features

| term | definition | in code |
|---|---|---|
| temporal_delta | The one-step change of each bus's observed `[P_inj, Q_inj]` against the previous frame [FED26]. | `formulas.temporal.temporal_delta`, `data/temporal_delta` |
| swing | `temporal_delta` divided by the bus's recent-change scale, the std of its observed changes over the last `SWING_WINDOW = 60` frames [FED26]. | `formulas.temporal.swing_zscore`, `recent_change_scale`, `data/swing` |
| KCL residual | Per-bus partial power balance: metered incident inflow minus the bus injection; a true balance only where every incident branch is metered. | `localization.learned.kcl_residual` |
| Jacobian block | Eight per-bus features from the measurement Jacobian applied to the change `z_t - h(x_hat_{t-1})` against the previous frame's estimate [JAC26]. | `se.jacobian.JacobianFeatures` |
| previous-frame fields (`prev_*`) | `prev_node_x`, `prev_edge_x`, `prev_timestep`, `prev_swing`: the readings of the frame one file row earlier, served on request from a timeline only. | `dataset.export`, `models.fields` |

## Estimation

| term | definition | in code |
|---|---|---|
| WLS | Weighted least squares state estimation, each meter weighted by the inverse square of its calibrated error [SCH70]. | `se.methods.WLS`, `formulas.estimation.wls_step` |
| chord-Newton | The Gauss-Newton iteration with the Jacobian fixed at the benign mean state instead of recomputed per iteration. | `se.base.SEBase`, `formulas.estimation` |
| Huber | Iterative reweighting that scales a meter's weight by `min(1, c / abs(r_n))`, `r_n` its normalized residual [HUB64]. | `se.methods.AdaptiveWeighting`, `formulas.estimation.huber_weights` |
| BDD / residual test | Bad-data detection: an alarm when the largest normalized residual of a WLS solve exceeds a level calibrated on benign training records. | `trust.base.TrustSelector._alarm_level`, `_max_residual` |
| LNR (largest normalized residual) | The residual divided by its standard deviation, largest per record or per bus; removed one meter at a time in `ResidualRemoval` [HAN75]. | `formulas.estimation.normalized_residual`, `se.methods.ResidualRemoval`, `localization.methods.ResidualLocalizer` |
| prior (SubspacePrior) | The estimate restricted to `mean + VK c`, a low-rank basis of whitened benign training states, optionally composed with Huber [EST26]. | `se.methods.SubspacePrior`, `formulas.estimation.whitened_svd_basis` |
| gate / oracle gate | A fitted localizer's flagged buses down-weight their meters before the solve; the oracle gate uses the true labels `y` and gives the ceiling. | `se.methods.GatedPrior`, `formulas.estimation.gate_weights` |
| geo | The geometric mean of the per-class errors over the record classes present (benign and each family). | `se.base.SEBase.score`, `models.scores.EstimatorScores.geo` |

## Localization and metrics

| term | definition | in code |
|---|---|---|
| FA budget / `fa_target` | The per-bus benign alarm rate each threshold is set to: the `(1 - fa_target)` quantile of the bus's benign training score; default 0.01. | `localization.base.LocalizerBase.fit` |
| validation tau | One global probability threshold on a 0.05 to 0.95 grid that maximizes mean per-bus F1 over the active buses of the validation split. | `localization.learned.LearnedLocalizer.tune_threshold`, `formulas.metrics.tau_from_counts` |
| zero-shot protocol | Train and val hold benign, `Aq` and `Ad`; test adds `As` and `Ar`, never seen in training; learned arms use the validation tau. | `localization/learned.py` docstring, `docs/localization/run_localization.py` |
| common protocol | Every family in train and test; every method calibrated at the same `fa_target`. | `docs/localization/run_localization.py` |
| macro-F1 | The mean per-bus F1 over the active buses (per family: over the buses the family attacks). | `models.scores.OverallMetrics.macro_f1`, `FamilyMetrics.macro_f1` |
| node-F1 | The micro F1 over every bus call. | `models.scores.OverallMetrics.node_f1` |
| DR | Detection rate: the mean per-bus recall over the active buses; per family, `detection_rate` is the share of records with any bus flagged. | `OverallMetrics.macro_dr`, `FamilyMetrics.detection_rate`, `PerBusMetrics.dr` |
| FR | False-alarm rate: the mean per-bus alarm rate on benign records, or over every record with `fr_over="all"`. | `OverallMetrics.macro_fr`, `LocalizerBase.score_perbus` |
| AUPRC | Per-bus average precision of the score, the step area under the precision-recall curve [DG06]. | `formulas.metrics.average_precision`, `PerBusMetrics.auprc` |

## Federated and trust

| term | definition | in code |
|---|---|---|
| client | One utility in federated training: it owns a set of buses and never shares its records. | `federated.localizer._Client` |
| partition | The split of buses into `K` clients by spectral clustering of the bus adjacency, with each client's interior and boundary buses. | `federated.partition.spectral_partition`, `models.federated.Partition` |
| FedAvg | Every round the server averages the clients' model weights and sends the average back [MCM17]; the localizers weight every client equally, since each holds every training record. | `formulas.federated.fedavg`, `federated.aggregate.fedavg_state`, `FederatedLocalizer` |
| local KCL | `kcl="local"`: a client's power-balance channel counts only the flows metered at buses it owns (the from-bus end). | `federated.localizer.FederatedLocalizer` |
| halo | Other clients' buses within `halo` hops of a client, whose own meters it reads as read-only context when `halo > 0`. | `formulas.federated.halo_nodes`, `federated.partition.compute_nodes` |
| trusted meter | A meter chosen for securing, in order of the attack-cost gain, by greedy row reduction or a deep Q-network [WU26]. | `trust.TrustedMeters`, `trust.TrustedMetersDQN`, `formulas.trust` |
| secured meter | A meter the attacker cannot write: it reads its benign value on every frame. | `trust.secured_copy`, `TrustSelector.score`, `GatedPrior(secured=)` |
