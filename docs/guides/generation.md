# How a timeline is generated

One timeline per system is the dataset. This page follows it from the load profile to the file,
and states which layers a detector may read. Every value below is the code's default; the source of
each is named beside it. Terms are defined in [`../reference/GLOSSARY.md`](../reference/GLOSSARY.md).

| stage | code | output |
|---|---|---|
| 1. operating-point pool | `profiles.fetch_profile`, `profiles.generate_states`, the v0.7.1 pool build | `[T, N, 4]` AC states, one per minute |
| 2. meter plan and noise | `engine/core.py` (`FdiaGenerator.__init__`), `engine/measurement.py` | which channels are metered, one noisy scan per state |
| 3. attack families | `engine/attacks/` (the `AttackMixin`, the family designers and support strategies), `engine/records.py` | the attacked scan of each family |
| 4. plan: split and episode placement | `generation/plan.py` (`split_bounds`, `place_split`) | the split, onsets, lengths |
| 5. design: the overload Am of each split | `generation/design.py` (`AmDesigner`, in parallel over `workers` processes) | each Am's design, or its move |
| 6. emit and write: the per-frame layers | `generation/emit.py` (`walk_split`, `TimelineBuffers`), `generation/write.py` (`write_temporal_layers`) | the HDF5 file |

`timeline.generate_timeline` orchestrates the stages; each hands the next a typed value (the slots, the
designs, the buffers).

![The generation pipeline: an ISO load profile is resampled to one minute with per-bus AR(1) jitter and solved into 72,000 AC states; the frames are split chronologically first, then each split gets the whole episodes closest to its attacked fraction, shared between the families and placed at uniform random onsets inside it; the walk emits one noisy scan per state, its jitter keyed by the seed and the timestep, and, inside an episode, solves a local false state (At, Am); after the walk the temporal features are written from the observed frames into one HDF5 file with observed, benign and clean layers](../figures/diagrams/generation_flow.png)

The published default release is v0.8.3 (`registry._RELEASE`), built by fdia-graph 0.20 with the
seven families of that release (the single-snapshot `Aq`, `Al`, `Ad`, `As`, `Ar` among them) and the
v0.8.3 meters. This version reads every release and generates the multi-snapshot families `At` and
`Am` on the hybrid meters, the recipe of the next data release. Releases up to v0.8.1 also hold
`Aq` for 15 to 44 frames and `Al` for 10 to 29, and compute the swing scale from the noiseless pool.

## 1. Operating-point pools

| item | value | source |
|---|---|---|
| load profile | NYISO system load, 11 zones summed, native 5-minute cadence | `profiles.NyisoArchive` (the NYISO entry of `profiles._FEEDS`) |
| window | 2024-01-01 to 2024-03-16 | the v0.7.1 pool build (a release script kept outside the repository; `WINDOW`) |
| resampling | time-interpolated to a 1-minute grid, then standardized to zero mean and unit variance | `profiles.fetch_profile(resample_min=1)` |
| per-bus scale | `clip(1 + k * S_t + j_t, 0.7, 1.3)`, `k = 0.1` | `profiles._ar1_scale`, `K_DEFAULT`, `CLIP_DEFAULT` |
| jitter `j_t` | per-bus AR(1), `j_t = 0.98 j_{t-1} + sqrt(1 - 0.98^2) * 0.03 * eps`, stationary std 0.03 | `JITTER_RHO`, `SIGMA_DEFAULT` |
| what is scaled | every load's P and Q and every generator's P at the bus, by the same factor | `profiles._solve_states_chunk` |
| solve | pandapower AC power flow, flat start, 50 iterations, 1e-6 MVA tolerance | `profiles._solve_states_chunk` |
| states kept | 82,000 requested, the first 72,000 converged kept per system | the v0.7.1 pool build (`REQUEST`, `TARGET`) |
| column order | <code>&#124;V&#124;</code> pu, `P_inj` MW, `Q_inj` MVAr, `theta` deg, voltage first | `generation.as_v_first`, pool attribute `columns` |
| file | `pool_ieee{C}.h5`, dataset `X` | `registry.pool_spec` |

The AR(1) jitter makes each bus drift smoothly from minute to minute rather than jump. The clip holds
every bus inside a plausible load band. Shunt draws are subtracted from the stored injections,
because the estimator models a shunt in the admittance matrix, not as an injection
(`profiles._remove_shunt_injections`).

Consecutive pool timesteps are one minute apart on the profile grid. A timestep whose power flow
does not converge is skipped, not filled, and the pool stores no timestamps. Frame `t` of a
timeline is pool timestep `t` (`data/timestep`).

## 2. Meter plan and noise

The meter plan is drawn once per generator from the seed and is the same on every frame
(`FdiaGenerator._meter_plan`, `emit_from_state`).

| channel | metered at | default |
|---|---|---|
| <code>&#124;V&#124;</code> | the union of the voltage buses and the PMU buses | `vbus_frac = 0.6` (`int(0.6 N)` buses), `pmu_frac = 0.2` (`max(1, int(0.2 N))` buses) |
| `theta` | the PMU buses (hybrid meters); every <code>&#124;V&#124;</code> bus (v0.8.3 meters) | as above |
| branch current, `pmu_i` | both ends' real and imaginary part of every in-service branch with a PMU at that end (hybrid meters only) | as above |
| `P_inj`, `Q_inj` | every injection bus and every zero-injection bus | always |
| `P_from`, `Q_from` | each branch independently with probability `flow_frac` | `flow_frac = 0.9` |

The meter model decides what a meter reads (`meter_model` in the meter plan, the plan's D10). The
default, `"hybrid"`, is a SCADA and PMU plan as in [WU26]: a SCADA voltmeter reads `|V|` only, a PMU
reads `|V|`, the angle and the current phasor of every in-service branch at its bus [WU26, eqs.
17-20]. The released files up to v0.8.3 wrote an angle at every voltmeter bus and read no
currents; the loader reads them, the generator no longer writes them (`models.MeterSettings`, the
plan's D12).

Injection buses are those with a generator, a load, the external grid, a shunt or a producing static
generator; every other bus is zero-injection (`FdiaGenerator._load_tables`). The two sets cover every
bus, so every bus carries P and Q injection meters. On IEEE-14 the plan meters `|V|` and `theta` at
9 buses, P and Q at 14 buses, and 18 of 20 branch flows.

Each reading is the true value plus a constant per-meter bias plus a fresh per-scan jitter
(`emit_from_state`). The accuracy-class standard deviations follow [ASP14]
(`FdiaGenerator.__init__`, `self.SD`):

| channel | total std | kind |
|---|---|---|
| `P_inj`, `Q_inj`, `P_from`, `Q_from` | 0.017 | relative to the reading |
| <code>&#124;V&#124;</code> | 0.0012 pu | absolute |
| `theta` | 0.00168 rad | absolute |

A branch current (hybrid meters) carries the PMU class of IEEE C37.118.1, a total vector error of
at most 1% taken as three standard deviations, relative to the current's magnitude on the real and
the imaginary part alike, plus a `1e-5` pu floor (`formulas.noise.current_sigma`,
`PMU_CURRENT_CLASS`) [C37118]; its bias is drawn once per channel after the other meters' biases.

`formulas.noise.bias_jitter_split` splits each total into jitter `0.25 SD` and bias
`sqrt(1 - 0.25^2) SD`, so the two add to the class total in quadrature. The bias is drawn once per
meter (`_meter_bias`). Power and flow jitter also carry an absolute floor of `1e-3` MW or MVAr, so a
near-zero reading still has noise. Treating the whole class error as per-scan noise would
over-jitter one-minute traces and hide the temporal signal (`bias_jitter_split` docstring).

## 3. The attack families

The two families are local false states: the attacker changes loads (and, for `Am`, generator
outputs) inside a subnetwork, solves that subnetwork with the boundary voltages held true, and adds
`a = h(x_false) - h(x_true)` to the true scan (`engine/attacks/stealthy._stealthy_frame`,
`engine/attacks/overload.py`). Every meter keeps its own noise draw, so the residual test sees noise
only. Every false state must also stay within the case's bus voltage limits and the generator P and
Q limits widened to the range the pool used (`engine/attacks/false_state._within_limits`,
`operating_limits`). They satisfy equations (13)-(25) of [WU26] (the SCADA measurements, the PMU
voltage magnitudes, angles and branch currents, the operating limits and the line-overload goal) and
its objective, eq. (12): each episode is held on the support that tampers the fewest devices
(`docs/plans/WU_MSFDIA_PLAN.md`).

| family | code | construction | magnitude per bus | episode length | stealthy | target set |
|---|---|---|---|---|---|---|
| `At` | 5 | slow ramp: one load factor on a fixed bus set, rise, hold, return, each frame a local false state on the fewest-tamper support | 0.2% per frame (`ramp_rate`), peak 2.4% to 5.2% | 60 frames (`ramp_len`) | yes | 5 stealthy loads (`draw_ramp`) |
| `Am` | 7 | the overload attack of [WU26]: one or two lines' reported flow driven to their ratings over the window, on the fewest-tamper support | the least-norm false state that meets the goal | 60 frames (`am_len` defaults to `ramp_len`) | yes | the rated, metered lines of the window (`eligible_lines`) |

The codes of the single-snapshot families of older releases (1 `Aq`, 2 `Ad`, 3 `As`, 4 `Ar`, 6 `Al`)
stay in the read-only label table `fg.FAMILIES`; fdia-graph 0.20 describes and generates them.

Both families run on the fewest-tamper search. `Am` is the overload attack of [WU26, eqs. 24-25]
(`engine/attacks/overload.py`): at onset, the eligible branches (rated, flow metered, true flow
below the rating at every snapshot of the window) are tried in a random order, and the first whose
fewest-tamper support meets the goal at every snapshot is the episode's target. Snapshot t's goal is
the true flow plus a linear share of what separates the window's last true flow from the rating,
`S_true_t + (t - kappa)/T (S_max - S_true_{kappa+T})` (the plan's D9), on the noiseless reading of
the false state, reaching the rating `S_max` at the last snapshot; the attackable loads of the
support are free, every other bus keeps its injection, and the false state is the least-norm voltage
change that meets the flow. The ratings are by default 1.25 times each branch's peak true apparent flow
over the operating pool (`am_attack=OverloadSettings(rating_margin=...)`, the plan's D15), which works on every system; the
PGLib-OPF ratings are the alternative, `am_attack=OverloadSettings(rating_source="pglib")` (IEEE-14, 118 and 300;
other cases raise `NoLineRatings`). The generators of the support are free inside their limits, like its attackable loads, and the
flow solve enforces the voltage and generator limits by an active set (the plan's D14); the limits
bind every generator the attack moves, on the support's edge too, and every load bus it moves shows
at most `load_cap` (0.5, `OverloadSettings(load_cap=...)`) times its true load [YUA11] (D16). Each
episode overloads two lines at once, as [WU26]'s case studies do, drawn so one held support reaches
both (`OverloadSettings(n_lines=1)` for one; D17); `episodes/am_*` holds one row per target line. `OverloadSettings(support_method="rref")` picks the support by [WU26]'s own method instead of the search: row reduction of the transposed attack-area Jacobian with column exchanges (p. 655 and Sec. IV-D1, after [YAN17]; `engine/attacks/rref.py`), the sparsest attack that moves each target line at every snapshot joined over the window, grown by the next sparsest rows when the AC solve cannot reach the goal on it (ours). It is much faster on IEEE-14 and tampers more devices than the search (experiment `wu26.method_compare`, docs/plans/WU_DEFENSE_PLAN.md, section 3). As in
[WU26], nothing bounds how far `Am` moves a channel between snapshots: a
device counts as tampered beyond [WU26]'s own case-study noise (0.03 pu SCADA, 0.01 pu PMU, the
current channels included; `formulas.noise.paper_sigma`, the plan's D8 and D11), and the attack is
stealthy because every snapshot is one AC state. `At` keeps its bound, each step within the meters'
rated accuracy (D7). Before D11 `Am` carried a between-snapshot bound as well, under which no overload window was
stealthy: moving one line's flow moves the injections and flows around its ends by several times
that change, beyond the rated accuracy of the small loads there. A window with no stealthy
overload moves the episode to another free onset of its split, earlier or later, up to 20 times
(`generation.design.AmDesigner`: the Am of a split are designed before it is walked); past that it is
given up and counted in `episode_shortfall`. With the hybrid meters the attacker also
writes the PMU branch currents its false state moves, which counts in the PMU of the bus at that end.
Only `At`'s channels are bounded between snapshots, the currents included (the PMU class, D7); `Am`'s
current channels are only counted when they move by more than 0.01 pu (D8, D11).

The overload episodes these recipes build, measured on the hybrid meters with `families=("Am",)`,
seed 1, the pool ratings computed over the frames walked (IEEE-14 and IEEE-118 timelines of 3,000
and 2,000 frames; IEEE-30 a smoke run), before the search speed-up of #163 (experiment
`generation.am_overload`; every line of every built episode reaches its rating):

<!-- results: gen.am -->
| system | recipe | frames | episodes | fallen back | devices (mean) | channels (mean) | largest change, pu (median / max episode) | proven | s per episode |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IEEE-14 | one line, before the D16 bounds | 3000 | 25 | 0 | 5.3 | 15.0 | 0.18 / 5.74 | 36% | 15 |
| IEEE-14 | one line, D16 bounds | 3000 | 25 | 0 | 3.9 | 9.9 | 0.14 / 0.82 | 8% | 30 |
| IEEE-14 | two lines, D16 bounds (default) | 3000 | 25 | 0 | 6.8 | 19.3 | 0.17 / 1.59 | 0% | 39 |
| IEEE-30 | one line, before the D16 bounds | 600 | 5 | 0 | 3.8 | 8.8 | 0.09 / 0.29 | 0% | 20 |
| IEEE-118 | one line, before the D16 bounds | 2000 | 17 | 0 | 6.8 | 21.4 | 1.05 / 44.94 | 24% | 43 |
| IEEE-118 | one line, D16 bounds | 2000 | 17 | 0 | 9.6 | 30.4 | 0.40 / 3.29 | 0% | 71 |
| IEEE-118 | two lines, D16 bounds (default) | 2000 | 17 | 0 | 17.8 | 72.0 | 1.33 / 3.39 | 0% | 78 |
<!-- /results -->

Notes on the table:

- A stealthy target is an attackable load that is not on a generator bus, zero-MW condensers
  included [BOY22] (`_load_tables`, `stealthy_pos`). An attackable load has nonzero active power,
  is not on the slack bus and is at most `max_load_mw = 2000` MW (`attackable_pos`).
- The `At` rise lasts 20% to 45% of the episode and the hold 0% to 25% (`draw_ramp`). Every frame
  carries at least one `ramp_rate` of change, so every frame labelled `At` is attacked
  (`ramp_dev`).
- A ramp step with no local solution is halved, up to 6 times (`STEP_HALVINGS`). `At` designs are
  tested on every frame they will occupy before they are accepted, up to 40 draws (`ONSET_DRAWS`).
- An overload `Am` window's snapshots are kappa+1 ... kappa+T after the untouched reference kappa
  [WU26, eq. 25], so its first snapshot already carries 1/T of the way to the rating and every frame
  labelled `Am` is attacked (the plan's D9). Before this fix the first snapshot's goal was the true
  flow and that frame tampered nothing.

## 4. Episode placement

| step | rule | source |
|---|---|---|
| split | chronological, cut before anything is placed: train `round(0.6 T)` frames, val `round(0.2 T)`, test the rest (`split=`, validated by `SplitSettings`) | `plan.split_bounds` |
| count | per split, the whole number of episodes whose frames come closest to `attacked_frac` of its frames (whole episodes, so a short split can be off by one or two), shared between the families by largest remainder over the weights `1 / length`, so every family gets about the same share of attacked frames | `plan.episode_counts`, `plan.largest_remainder` |
| place | per split, longest first, each at an onset drawn uniformly among the positions where it fits inside the split, never overlapping and never cut; a placement that jams is retried whole 10 times, then one episode of the family with the most is dropped and recorded (`RuntimeWarning`) | `plan.place_split`, `plan.uniform_onset` |
| walk | every frame emitted in time order, its meter jitter from its own stream keyed by the seed and the timestep: benign runs between episodes, each episode where it was placed | `emit.walk_split`, `MeasurementMixin._jitter_stream` |
| move | an episode with no feasible design at its onset moves, up to 20 times, then is given up and recorded (`RuntimeWarning`): an overload Am to an onset drawn uniformly among every free one of its split, earlier or later, since the design stage designs a split's Am before any of its frames is written and an Am design reads no earlier frame (no stealth bound, D11); a ramp At to a later free onset, since its design reads the frame before its onset and the walk has written the frames before the failed one. The moves draw from streams keyed by the seed, the split, the episode and the move, so they do not depend on the workers | `design.AmDesigner`, `emit.relocate_later` |
| fallback | a frame inside a built episode whose own scan cannot be built is emitted benign; the count is the `fallback_benign` attribute | `write._timeline_attrs` |
| record | the split fractions and sizes, the attacked fraction each split reached, and per split and family the episodes requested, built, moved, dropped and short | `write.placement_attrs` |

Adjacent episodes and long quiet stretches are outcomes of the uniform draw, not of a rule. Placing
the longest episodes first keeps a long episode from being squeezed out by shorter ones. The
default `attacked_frac = 0.5` gives a balanced file: on a 72,000-frame timeline with 60-frame
episodes every split is half attacked, up to an overload window that stays infeasible after its 20
moves. The full IEEE-14 builds of the three release datasets (seed 123, experiment
`generation.split_first`; episodes and redraws per family, `At / Am`):

<!-- results: gen.split_first -->
| dataset | split | frames | attacked | episodes asked | built | redraws | shortfall |
|---|---|---:|---:|---:|---:|---:|---:|
| At | train | 43200 | 0.500 | 360 | 360 | 0 | 0 |
| At | val | 14400 | 0.500 | 120 | 120 | 0 | 0 |
| At | test | 14400 | 0.500 | 120 | 120 | 0 | 0 |
| At | build |  |  |  |  |  | 10 min |
| Am | train | 43200 | 0.500 | 360 | 360 | 88 | 0 |
| Am | val | 14400 | 0.496 | 120 | 119 | 32 | 1 |
| Am | test | 14400 | 0.500 | 120 | 120 | 26 | 0 |
| Am | build |  |  |  |  |  | 413 min |
| At+Am | train | 43200 | 0.500 | 180 / 180 | 180 / 180 | 0 / 32 | 0 / 0 |
| At+Am | val | 14400 | 0.500 | 60 / 60 | 60 / 60 | 0 / 18 | 0 / 0 |
| At+Am | test | 14400 | 0.500 | 60 / 60 | 60 / 60 | 0 / 15 | 0 / 0 |
| At+Am | build |  |  |  |  |  | 234 min |
<!-- /results -->

Because
the jitter of frame t depends only on the seed and t, the files of one seed and pool that place
different families (`families=("Am",)`, `("At", "Am")`, `("At",)`) carry byte-identical benign
frames; only the attacks differ.

### Designing in parallel

The overload Am dominates generation time: each episode runs the fewest-tamper search on up to
`AM_LINE_TRIES` line pairs. A split's Am episodes are independent (a flow goal reads only its own
window, and each design draws its line order from a stream keyed by the seed, the split, the episode
and the move), so `workers=N` designs them in N processes and the file is byte-identical for any N.
On Windows a script that passes `workers` above 1 must start under `if __name__ == "__main__":`, as
any program using `multiprocessing` must. An Am-only IEEE-14 timeline (experiment
`generation.parallel_design`, `tools/experiments/design_speed.py`):

<!-- results: gen.workers -->
| workers (3000 frames) | seconds | speed-up | Am episodes | same file |
|---|---:|---:|---:|---:|
| 1 | 754 | 1.0x | 25 | yes |
| 16 | 253 | 3.0x | 25 | yes |
<!-- /results -->

## 5. The per-frame layers

| group | datasets | meaning | written by |
|---|---|---|---|
| `data/` | `node_x [T, N, 4]`, `edge_x [T, E, 2]` | observed readings: benign plus noise, plus the attack where attacked | `_TimelineBuffers.store` |
| `data/` | `node_m`, `edge_m` | the meter plan per frame | `_write_masks` |
| `data/` | `y [T, N]` | 1 on each attacked bus | `store` |
| `data/` | `family`, `stealthy`, `seq_id`, `timestep`, `split [T]` | family code, 1 for a stealthy family (`At`, `Am`; `Aq` and `Al` in older releases), episode index (-1 benign), pool timestep, 0/1/2 | `_finish_timeline` |
| `data/` | `temporal_delta`, `swing [T, N, 2]` | temporal features from the observed frames | `write_temporal_layers` |
| `benign/` | `node_benign`, `edge_benign` | the same scan with the attack removed and the same noise draw | `store` |
| `clean/` | `node_clean`, `edge_clean` | the noiseless pool state and the exact flows on metered branches | `_clean_slice` |
| `attack/` | `node_tamper`, `edge_tamper` | 1 where the attacker wrote the meter | `_store_attack` |
| `data/`, `benign/`, `attack/` | `pmu_i`, `pmu_i_m`, `pmu_i_benign`, `pmu_i_tamper [T, E, 4]` | hybrid meters only: the PMU branch currents (Re and Im of `I_from`, then of `I_to`, per unit), their mask, the attack-removed twin, and 1 where the attacker wrote the channel | `_store_currents`, `_write_masks` |
| `attack/` | `mag_ptr`, `mag_bus`, `mag` | designed change per attacked bus, ragged by frame | `_write_episodes` |
| `episodes/` | `onset`, `length`, `family`, `bus_ptr`, `bus_idx` | one row per episode | `_write_episodes` |

`node_x - node_benign` is the attack exactly, for every family. The loader derives
`edge_clean_full`, the clean flow on every branch, at read time (`dataset/__init__.py`).

The temporal features are written after the walk from `data/node_x` alone
(`timeline.write_temporal_layers`, `formulas.temporal`):

| feature | definition |
|---|---|
| `temporal_delta` | `[P_inj, Q_inj](t) - [P_inj, Q_inj](t-1)` of the observed frames; zero on frame 0 |
| `swing` | `temporal_delta` divided by the bus's recent-change scale |
| scale | std of the observed scan-to-scan absolute change over the last `SWING_WINDOW = 60` changes before `t`, plus `1e-3`; `1e-3` alone with fewer than three changes (`recent_change_scale`) |

A spike reads as a large swing at its first frame. The slow ramp `At` changes by less than the typical
recent change on every frame, so its swing stays near the benign level (`formulas/temporal.py`).

## 6. The rule for features

At test time a feature may use only measurements and quantities derived from them. The `clean` and
`benign` layers exist for evaluation and for fitting, never as feature inputs. `y`, `family`,
`stealthy`, `seq_id`, the tamper masks and the magnitudes are labels.

| SDK path | reads | layer role |
|---|---|---|
| `SEBase.estimate` | `node_x`, `edge_x` | measurements |
| `SEBase.fit(calibrate="truth")` | `node_x`, `edge_x`, `family`, `clean` | fitting for the estimation benchmark: meter sigmas, benign mean state, `theta_ref` |
| `SEBase.fit(calibrate="measured")` | `node_x`, `edge_x`, `family` | fitting from the meters' accuracy classes and the estimated benign states |
| `SEBase.score` | `node_x`, `edge_x` (through `estimate`), then `family`, `clean` | the estimate from measurements, then evaluation against the truth |
| `JacobianFeatures`, `JacobianWeighting` | `node_x`, `edge_x`, `prev_node_x`, `prev_edge_x`, `prev_timestep` | measurements |
| `GatedPrior(gate="oracle")` | `y` | evaluation ceiling only |
| `SwingThreshold`, `DeltaThreshold` | `swing`, `temporal_delta` | measurements |
| `ResidualLocalizer` | `node_x`, `edge_x` | measurements |
| `BusCNN`, `BusMLP`, `FedBusCNN`, `FedBusMLP` | `node_x`, `node_m`, `edge_x`, `temporal_delta`, `swing`, `prev_swing`, `prev_*`; `y` as the training label | measurements |
| `TrustSelector.fit` | `node_x`, `edge_x`, `family` | measurements |
| `TrustSelector.score` | `benign`, `edge_benign` | evaluation: what a secured meter reads |
| `trust.secured_copy` | `benign`, the tamper masks | builds a counterfactual file, then rewrites the temporal features from its observed frames |

Sources: `se/base.py`, `se/jacobian.py`, `se/methods.py`, `localization/methods.py`,
`localization/learned.py` (`_fields`), `trust/base.py`, `trust/secured.py`. Every path whose
output feeds a detector (`ResidualLocalizer`, the Jacobian feature sets, `TrustSelector`) fits its
estimator with `calibrate="measured"`, so no clean layer is read at fit or test time. The state
estimators themselves default to `calibrate="truth"`, the estimation benchmark's calibration.

## 7. Building your own

```python
import fdia_graph as fg

fg.generate("ieee118", name="my_run", frames=10_000, attacked_frac=0.5)  # needs [generate]
ds = fg.load("my_run", order="time")
```

`generation.generate(system, name, frames=None, states=None, out=None, seed=123, settings=None, **knobs)` loads the
pool, keeps the first `frames` timesteps, calls `timeline.generate_timeline` and registers the file
under `name`. Without `states`, it reads `$FDIA_GRAPH_INIT` or downloads the system's pool.

| knob | default | meaning |
|---|---|---|
| `attacked_frac` | 0.5 | fraction of frames under an episode |
| `families` | ("At", "Am") | the families in rotation; the single-snapshot families of older releases are refused |
| `min_tamper` | True | hold each episode on the support that tampers the fewest devices [WU26, eq. 12] |
| `am_attack` | "overload" | `Am` as the overload attack of [WU26], or an `OverloadSettings` (a dict still works and warns; dicts go in 0.22) |
| `stealth_scale` | 1.0 | a multiplier on At's stealth bound, in units of the rated accuracy (Am has none, D11) |
| `redundancy` | `MeterSettings()` | the meter plan, a `MeterSettings` (a dict warns, as above): coverage `vbus_frac` 0.6, `pmu_frac` 0.2, `flow_frac` 0.9, and `meter_model` `"hybrid"` (D10, D12) |
| `ramp_rate` | 0.002 | `At` growth per frame |
| `ramp_len` | 60 | `At` episode length |
| `am_len` | `ramp_len` | `Am` episode length |
| `hops` | 2 | the attacker's subnetwork reach in branches |
| `split` | (0.6, 0.2, 0.2) | chronological train/val/test fractions |
| `max_load_mw` | 2000.0 | a larger load is never a target; None disables |
| `seed` | 123 | the meter plan, the biases and every attack draw |
| `workers` | 1 | processes the overload Am designs run on; the file does not depend on it |

Every knob is also a field of `TimelineSettings`, nested by subject (`ramp`, `search`, `overload`,
`meters`, `split`): `generate(..., settings=TimelineSettings(ramp=RampSettings(length=30)))`, with any
flat knob applied over it. The file records the settings it ran as JSON (attribute `settings`) with
their hash (`settings_hash`).

A data release (the build scripts of v0.8.0 to v0.8.3 are in the git history at `aa77d0b`):

1. Per system, copy the unchanged pool and run `fg.generate(C, ..., states=pool, seed=123, frames=72000)`
   into `timeline_ieee{C}.h5`; each system writes a manifest fragment with sha256 and size.
2. Run once more with `MERGE=1` to combine the fragments into `manifest.json`.
3. Upload with `python tools/upload_assets.py vX.Y.Z notes.md <files>`; it creates the `data-vX.Y.Z`
   tag and never replaces an asset already on the release.
4. Add the release's sha256 values to `registry._SHA256`, then bump `registry._RELEASE` last.
