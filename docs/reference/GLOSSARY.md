# Glossary

One definition per term and where it lives in the code: under `src/fdia_graph`, or under
`analysis/src/fdia_analysis` for `se`, `localization`, `trust` and `federated`. The generation pipeline behind
the data terms is in [`../guides/generation.md`](../guides/generation.md), the meters in
[`../concepts/meters.md`](../concepts/meters.md), the attacks in
[`../concepts/attacks.md`](../concepts/attacks.md) and the layers in
[`../concepts/layers.md`](../concepts/layers.md); citation keys are in
[`REFERENCES.md`](REFERENCES.md).

## Data

| term | definition | in code |
|---|---|---|
| scan | One emission of every meter in the plan from one grid state: readings and masks. | `models.frames.Scan`, `engine.measurement.Emitter.emit_from_state` |
| frame | One row of a timeline: the scan at one pool timestep with the attack of its episode applied, or none. | `models.frames.Frame`, `generation.write.TimelineWriter.store` |
| record | One row as the loader serves it, `ds[i]`: a frame of a timeline. | `dataset/records.py`, `models.data.RecordBundle` |
| timestep | The index of the pool state a frame was emitted from; on a timeline frame `t` is timestep `t`. | `data/timestep`, `schema.TIMESTEP` |
| episode | A run of consecutive frames under one attack of one family on one design; episodes never overlap. | `models.plan.Episode`, `episodes/` group, `data/seq_id` |
| onset | The first frame of an episode, drawn uniformly among the positions of its split where the episode fits. | `generation.plan.uniform_onset`, `episodes/onset` |
| pool | The operating-point pool: one converged AC state per frame, in time order. An IEEE pool is an array `[T, N, 4]` in <code>&#124;V&#124;</code>, `P_inj`, `Q_inj`, `theta` order, one state per minute of the load profile. The New York pool is a `Pool` record, one state per five-minute frame; a frame that did not solve is kept empty. | `data.ieee_pool.generate_states`, `generation._load_states`, `pool_ieee{C}.h5`; `data.pool.Pool`, `data.pool.PoolBuilder` |
| timeline | One HDF5 file per system: every pool timestep scanned in time order, with attack episodes placed on it; the file format of every data release from v0.8.0. | `timeline.generate_timeline`, `schema.KIND_TIMELINE` |
| release | A tagged set of data assets (`data-v<x.y.z>` from v0.8.0), pinned by sha256; `registry._RELEASE` is the default. | `registry.py`, `tools/upload_assets.py` |
| split | The train, val and test partition (0, 1, 2), cut before any episode is placed, so every episode lies inside one split: three blocks in time order by default, or whole weeks (the setting `split_by`). | `generation.plan.split_bounds`, `generation.plan.split_weeks`, `generation.source.TimelineSource.cut_splits`, `data/split`, `models.choices.Split`, `schema.SPLIT_CODE` |
| family | The attack type of a frame: 0 benign, 1 `Aq`, 2 `Ad`, 3 `As`, 4 `Ar`, 5 `At`, 6 `Al`, 7 `Am`. | `constants.families.FAMILIES`, `constants.families.FAMILY_CODE`, `data/family` |
| stealthy | 1 for the families built as local false states (`Aq`, `At`, `Al`, `Am`), which pass the residual test. | `constants.families.STEALTHY_FAMILIES`, `data/stealthy` |
| area | The buses an attacker may act in: by default the buses within `hops` branches of the goal's buses, the slack left out, grown over the zero-injection buses of its boundary, with `hops` reduced until a boundary is left. A support is chosen inside it. | `engine.attacks.area.AttackArea.hop_area`, `attack_area` |
| support | The buses whose voltages a local false state moves; only meters that depend on them can change. The fewest-tamper search chooses it; without the search it is the whole area. | `engine.attacks.minimize.Minimizer.min_tamper` |
| boundary | The buses one branch outside a support (or an area). Their voltages stay true; their reported injections take up the change. | `engine.attacks.area.AttackArea.boundary` |
| reach | A support with its boundary: every bus whose reported injection a false state on that support can change. | `engine.attacks.false_state.FalseState.reach` |
| snapshot | One state of an attack window, counted from the window's first frame. | `engine.attacks.window.AttackWindow` |
| device | One SCADA terminal (a bus's injection meters, the flows metered at that bus, a voltage meter without PMU) or one PMU (a bus's voltage magnitude and angle, and under the hybrid meter model the branch currents it reads at the bus), the unit [WU26] counts. | `formulas.attacks.tampered_devices` |
| meter model | What the meters read: "hybrid", new generation's, where a SCADA voltmeter reads `|V|` only and each PMU reads `|V|`, the angle and the current phasor of every in-service branch at its bus; or "v083", the released files', with an angle at every voltmeter bus and no currents. | `models.choices.MeterModel`, `models.config.MeterSettings`, `generate_timeline(..., redundancy={"meter_model": ...})` |
| PMU branch current | The current phasor a PMU reads at its end of a branch, `I_f = Y_f V` or `I_t = Y_t V`, stored as real and imaginary part per end in `pmu_i` [WU26, eqs. 19-20]; noise at IEEE C37.118.1's 1% total vector error [C37118]. | `formulas.network.branch_currents`, `formulas.noise.current_sigma`, the generator's `masks.current` |
| overload attack | `Am` in new generation: the apparent flow of two lines at once by default (`OverloadSettings.n_lines`; `FlowGoal.more`), as the tampered measurements carry it before noise, driven snapshot by snapshot to the line's rating over the window, on the support that tampers the fewest devices, its attackable loads and generators free inside the limits [WU26, eqs. 12-25]. | `engine.attacks.overload.Overload` |
| zero-injection bus | A bus with no load and no producing unit, whether or not it has a shunt: a shunt is in the admittance matrix and a pool stores a bus's injection without the shunt's draw, so the bus's true injection is zero. No attack shows an injection there: an area grows over one on its boundary, and a candidate support that would leave one on its boundary takes it in. | `engine.core.FdiaGenerator` (`zero_inj`), `engine.attacks.area.AttackArea.grow_over_zero_injection` |
| free injection | A bus whose injection a flow-goal attacker may change: an attackable load or a generator (the latter inside its limits to within what its bus's injection meter resolves, [WU26, eqs. 22-23]); a zero-injection bus is held at zero. | `engine.attacks.false_state.FalseState.free_injection_buses` |
| load cap | The overload attack's load-plausibility bound: every load bus the attack moves, in its support or on the support's boundary (a bus with a base load, no generator, not the slack), shows an active change of at most `load_cap` times its true load [YUA11]; 0.5 by default. | `models.config.OverloadSettings.load_cap`, `generate_timeline(..., am_attack={"load_cap": ...})`, `engine.attacks.false_state.FalseState.solve_flow_local` |
| line rating | A branch's thermal limit S_max in MVA: by default `rating_margin` (1.25) times its peak true apparent flow over the operating pool, or on request the PGLib-OPF `rate_a` matched by end buses (`generate_timeline(..., am_attack={"rating_source": "pglib"})`, the margin as `am_attack={"rating_margin": ...}`, both through `models.config.OverloadSettings`); the IEEE cases pandapower ships carry only a 9,900 MVA placeholder. | `engine.attacks.overload.Overload.use_line_ratings`, `models.config.OverloadSettings`, `fdia_graph.ratings`, `formulas.attacks.branch_ratings` |
| stealth scale | A multiplier on `At`'s stealth bound, whose unit is the meters' rated accuracy; 1 by default. `Am` has no stealth bound, as in [WU26]. | `generate_timeline(stealth_scale=...)` |
| paper noise | The noise of [WU26]'s case studies: flat, 0.03 pu on SCADA and 0.01 pu on PMU channels. The package counts with the meters' own class; `MeterSettings.power_floor_pu` at 0.03 puts the SCADA figure under the power meters of a data set. | `constants.meters.WU26_SCADA_NOISE_PU`, `WU26_WAMS_NOISE_PU`, `models.config.MeterSettings` |
| tamper cost | The devices with a channel moved beyond its meter's noise at some snapshot of an attack window, the objective of [WU26, eq. 12]. The noise is the accuracy-class sigma at the true reading, the noise the file's readings carry, for both families. A ramp's cost may be zero; an overload's is at least one device. | `formulas.attacks.tampered_channels`, `tampered_devices`, `engine.attacks.window.AttackWindow.cost` |
| accuracy-class sigma | A meter's rated accuracy at a reading (relative for power channels plus a floor, absolute for voltage and angle): the sigma the measured calibration uses, the threshold of the fewest-tamper search for both families, the ramp's stealth bound and, by default, the slack on a generator limit. | `NoiseModel.sigmas`, `formulas.noise.accuracy_class_sigma` |
| jitter sigma | The per-scan random part of a meter's error, a quarter of its accuracy class; the emitter draws each reading's noise with it. Emission only. | `NoiseModel.sample` |
| local false state | A state in which only the support is re-solved under false loads, boundary voltages held true, and the attack vector `h(x_false) - h(x_true)` is added to the true scan; it meets constraints (13)-(18) and (21)-(23) of [WU26], and in new generation the branch-current phasors (19)-(20); new generation also holds it on the fewest-device support of eq. (12). | `engine.attacks.ramp_frames.RampFrames._stealthy_frame`, `engine.attacks.false_state.FalseState.solve_local`, `engine.attacks.area.AttackArea.hop_area` |
| theta_ref | The one slack angle every estimate is referenced to, fixed at fit time from the training split. | `se.base.SEBase._fit_reference`, `ref_angles` |
| active buses | The buses attacked somewhere in the records being scored; macro scores average over them. | `models.scores.OverallMetrics`, `formulas.metrics.tau_from_counts` |
| attackable buses | Buses whose load an attack may target: nonzero active load, not the slack, at most `max_load_mw`; the stealthy families also skip generator buses [BOY22]. A learned localizer's `_attackable` is the buses labelled attacked in its training split. | `engine.core.FdiaGenerator._load_tables` (`attackable_pos`, `stealthy_pos`), `localization.learned.LearnedLocalizer` |

## Features

| term | definition | in code |
|---|---|---|
| temporal_delta | The one-step change of each bus's observed `[P_inj, Q_inj]` against the previous frame [FED26]. | `formulas.temporal.temporal_delta`, `data/temporal_delta` |
| swing | `temporal_delta` divided by the bus's recent-change scale, the std of its observed changes over the last `SWING_WINDOW_SCANS = 60` frames [FED26]. | `schema.SWING_WINDOW_SCANS`, `formulas.temporal.swing_zscore`, `recent_change_scale`, `data/swing` |
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
| PMU pseudo-measurement | The voltage phasor at the far end of a PMU-metered branch computed from the PMU's voltage and current, `V_far = (I_near - y_nn V_near) / y_nf` [WU26, eq. (3)], averaged over the branches reaching the bus; the estimator option `pmu_pseudo` places it in the `|V|` and angle slots no meter reads. | `formulas.estimation.pmu_pseudo_voltages`, `se.base.SEBase(solve=SolveConfig(pmu_pseudo=True))` |
| geo | The geometric mean of the per-class errors over the record classes present (benign and each family). | `se.base.SEBase.score`, `models.scores.EstimatorScores.geo` |

## Localization and metrics

| term | definition | in code |
|---|---|---|
| FA budget / `fa_target` | The per-bus benign alarm rate each threshold is set to: the `(1 - fa_target)` quantile of the bus's benign training score; default 0.01. | `localization.base.LocalizerBase.fit` |
| validation tau | One global probability threshold on a 0.05 to 0.95 grid that maximizes mean per-bus F1 over the active buses of the validation split. | `localization.learned.LearnedLocalizer.tune_threshold`, `formulas.metrics.tau_from_counts` |
| zero-shot protocol | Train and val hold benign, `Aq` and `Ad`; test adds `As` and `Ar`, never seen in training; learned arms use the validation tau. | `localization/learned.py` docstring |
| common protocol | Every family in train and test; every method calibrated at the same `fa_target`. | `docs/analysis/localization/README.md` |
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
| trusted meter | A meter chosen for securing, in order of the attack-cost gain, by greedy row reduction or a deep Q-network [WU26]. | `trust.base.TrustedMeters`, `trust.meter_dqn.TrustedMetersDQN`, `formulas.trust` |
| secured meter | A meter the attacker cannot write: it reads its benign value on every frame. | `trust.secured.secured_copy`, `TrustSelector.score`, `GatedPrior(secured=)` |
