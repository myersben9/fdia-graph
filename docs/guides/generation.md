# How a timeline is generated

One timeline per system is the dataset. This page follows it from the load profile to the file,
and states which layers a detector may read. Every value below is the code's default; the source of
each is named beside it. Terms are defined in [`../reference/GLOSSARY.md`](../reference/GLOSSARY.md).

| stage | code | output |
|---|---|---|
| 1. operating-point pool | `profiles.fetch_profile`, `profiles.generate_states`, the v0.7.1 pool build | `[T, N, 4]` AC states, one per minute |
| 2. meter plan and noise | `engine/core.py` (`FdiaGenerator.__init__`), `engine/measurement.py` | which channels are metered, one noisy scan per state |
| 3. attack families | `engine/attacks/` (the `AttackMixin`), `engine/records.py`, `timeline.py` | the attacked scan of each family |
| 4. episode placement | `timeline._place_episodes`, `timeline._frame_split` | onsets, lengths, the split |
| 5. per-frame layers | `timeline._TimelineBuffers`, `timeline.write_temporal_layers` | the HDF5 file |

![The generation pipeline: an ISO load profile is resampled to one minute with per-bus AR(1) jitter and solved into 72,000 AC states; episodes are placed at uniform random onsets with an equal share of attacked frames per family; the walk emits one noisy scan per state and, inside an episode, either solves a local false state (Aq, At, Al, Am) or tampers the readings in place (Ad, As, Ar); after the walk the temporal features are written from the observed frames and the frames are split chronologically into one HDF5 file with observed, benign and clean layers](../figures/diagrams/generation_flow.png)

The published default release is v0.8.3 (`registry._RELEASE`), built by this code. Releases up to
v0.8.1 differ in two ways: they hold `Aq` for 15 to 44 frames and `Al` for 10 to 29, and they
compute the swing scale from the noiseless pool instead of the observed frames.

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
17-20]. The released files' `"v083"` model wrote an angle at every voltmeter bus and read no
currents; the v0.8.3 recipe asks for it explicitly, `generate_timeline(redundancy={"meter_model":
"v083"})` (`models.MeterSettings`, the plan's D12).

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
`PMU_CURRENT_CLASS`) [C37118]; its bias is drawn once per channel after the other meters' biases, so
the v0.8.3 draws are unchanged.

`formulas.noise.bias_jitter_split` splits each total into jitter `0.25 SD` and bias
`sqrt(1 - 0.25^2) SD`, so the two add to the class total in quadrature. The bias is drawn once per
meter (`_meter_bias`). Power and flow jitter also carry an absolute floor of `1e-3` MW or MVAr, so a
near-zero reading still has noise. Treating the whole class error as per-scan noise would
over-jitter one-minute traces and hide the temporal signal (`bias_jitter_split` docstring).

`NOISE_FLOOR = 0.02` (`generation.py`) is the lower edge of the attack plausibility band: a change
below 2% of the reading sits inside meter error and resolves to noise.

## 3. The attack families

`attack_intensity = 0.20` is the upper edge of the band for Aq, Al, Ad and As, and bounds the
redistribution Am draws at onset (`generate_timeline`, `lra_delta`). The At ramp is set by
`ramp_rate` and `ramp_len`, Am's steps by `am_rate` and `am_len`, and Ar records the realized change
of its replay without bounding it. The stealthy families are local false states: the attacker
changes loads inside a subnetwork within `hops = 2` branches, solves that subnetwork with the
boundary voltages held true, and adds `a = h(x_false) - h(x_true)` to the true scan
(`engine/attacks/stealthy._stealthy_frame`). Every meter keeps its own noise draw, so the residual
test sees noise only. Every false state must also stay within the case's bus voltage limits and the
generator P and Q limits widened to the range the pool used
(`engine/attacks/false_state._within_limits`, `operating_limits`). They satisfy equations (13)-(18)
and (21)-(23) of [WU26] (the SCADA measurements, the PMU voltage magnitudes and angles, and the
operating limits; and, in new generation, whose PMUs read branch currents, the current phasors (19)-(20) as well). New generation also
solves its objective, eq. (12): each `At` and `Am` episode is held on the support that tampers the
fewest devices, and `Am` drives a line's reported flow to its rating (eqs. 24-25; by default 1.25
times the line's peak flow over the pool, the plan's D15).
The released files' stealthy families drew their targets at random and drove no line to its limit;
`LEGACY_FAMILIES` with `am_attack="redistribution"`, `min_tamper=False` and `redundancy={"meter_model": "v083"}` reproduces them
(`docs/plans/WU_MSFDIA_PLAN.md`).

| family | code | construction | magnitude per bus | episode length | stealthy | target set |
|---|---|---|---|---|---|---|
| `Aq` | 1 | local false state: each target load scaled by its own factor, rise or drop | 5% to 20% (`single_shot_design`) | 1 frame (`_ONE_FRAME`) | yes | 1 to 6 stealthy loads (`pick_targets`) |
| `Ad` | 2 | in-place bias: additive shift on P and Q, a `N(0, 0.02)` pu shift on <code>&#124;V&#124;</code>, a shift on each incident flow | 2% to 20% per channel, random sign (`_corrupt_bias`) | `corrupt_len = 1` | no | 4 attackable loads |
| `As` | 3 | in-place scaling: one gain on P and Q, one gain per incident flow | gain 1.02 to 1.20 (`_corrupt_scaling`) | `corrupt_len = 1` | no | 4 attackable loads |
| `Ar` | 4 | in-place replay: the bus's four node channels copied from an earlier benign scan, at least 20 benign scans back once that many are buffered | not bounded; the realized change is recorded (`_corrupt_replay`) | `corrupt_len = 1` | no | 4 attackable loads |
| `At` | 5 | slow ramp: one load factor on a fixed bus set, rise, hold, return, each frame a local false state | 0.2% per frame (`ramp_rate`), peak 2.4% to 5.2% | 60 frames (`ramp_len`) | yes | 5 stealthy loads (`draw_ramp`) |
| `Al` | 6 | redistribution: load-conserving shift across the two PTDF sides of a target line, lowering its apparent flow | 2% to 20% (`_lra_for_line`) | 1 frame (`_ONE_FRAME`) | yes | up to 6 stealthy loads per PTDF side, inside the line's subnetwork |
| `Am` | 7 | v0.8.3: multi-snapshot, an `Al` redistribution drawn at onset, reached in steps (new generation: the overload attack below) | per-frame step at most `am_rate * NOISE_FLOOR` = 1.8%, peak 2% to 20% (`_AmShape.under_floor`) | 60 frames (`am_len` defaults to `ramp_len`) | yes | as `Al` |

New generation (from 0.21) makes the multi-snapshot families only, `At` and `Am`, both on the
fewest-tamper search. `Am` is the overload attack of [WU26, eqs. 24-25]
(`engine/attacks/overload.py`): at onset, the eligible branches (rated, flow metered, true flow
below the rating at every snapshot of the window) are tried in a random order, and the first whose
fewest-tamper support meets the goal at every snapshot is the episode's target. Snapshot t's goal is
the true flow plus a linear share of what separates the window's last true flow from the rating,
`S_true_t + (t - kappa)/T (S_max - S_true_{kappa+T})` (the plan's D9), on the noiseless reading of
the false state, reaching the rating `S_max` at the last snapshot; the attackable loads of the
support are free, every other bus keeps its injection, and the false state is the least-norm voltage
change that meets the flow. The ratings are by default 1.25 times each branch's peak true apparent flow
over the operating pool (`am_attack={"rating_margin": ...}`, the plan's D15), which works on every system; the
PGLib-OPF ratings are the alternative, `am_attack={"rating_source": "pglib"}` (IEEE-14, 118 and 300;
other cases raise `NoLineRatings`). Measured with new generation's defaults (hybrid meters, `families=("Am",)`, seed 1, the pool ratings computed over the frames walked, the bounds of D16): IEEE-14 (3000 frames) 25 episodes built and 0 fallen back to benign, 3.9 devices and 9.9 channels on average, the largest change on a channel 0.14 pu at the median episode and 0.82 pu at most, 8% of the searches proven, 30 s of generation per episode; IEEE-118 (2000 frames) 17 episodes built and 0 fallen back to benign, 9.6 devices and 30.4 channels on average, the largest change on a channel 0.40 pu at the median episode and 3.29 pu at most, 0% of the searches proven, 71 s of generation per episode; every episode's noiseless flow reaches its rating. Before D16 bounded the edge of the support the same runs gave IEEE-14 (3000 frames) 25 episodes built and 0 fallen back to benign, 5.3 devices and 15.0 channels on average, the largest change on a channel 0.18 pu at the median episode and 5.74 pu at most, 36% of the searches proven, 15 s of generation per episode; IEEE-118 (2000 frames) 17 episodes built and 0 fallen back to benign, 6.8 devices and 21.4 channels on average, the largest change on a channel 1.05 pu at the median episode and 44.94 pu at most, 24% of the searches proven, 43 s of generation per episode, and IEEE-30 (600 frames, a smoke run) 5 episodes of 3.8 devices. The generators of the support are free inside their limits, like its attackable loads, and the
flow solve enforces the voltage and generator limits by an active set (the plan's D14); the limits
bind every generator the attack moves, on the support's edge too, and every load bus it moves shows
at most `load_cap` (0.5, `am_attack={"load_cap": ...}`) times its true load [YUA11] (D16). Each
episode overloads two lines at once, as [WU26]'s case studies do, drawn so one held support reaches
both (`am_attack={"n_lines": 1}` for one; D17); `episodes/am_*` holds one row per target line. Measured with new generation's defaults (hybrid meters, `families=("Am",)`, seed 1, pool ratings, the D16 bounds, two lines): IEEE-14 (3000 frames) 25 two-line episodes built and 0 fallen back to benign, 6.8 devices and 19.3 channels on average, the largest change on a channel 0.17 pu at the median episode and 1.59 pu at most, 0% of the searches proven, 39 s of generation per episode; IEEE-118 (2000 frames) 17 two-line episodes built and 0 fallen back to benign, 17.8 devices and 72.0 channels on average, the largest change on a channel 1.33 pu at the median episode and 3.39 pu at most, 0% of the searches proven, 78 s of generation per episode; every line of every episode reaches its rating. As in
[WU26], nothing bounds how far `Am` moves a channel between snapshots: a
device counts as tampered beyond [WU26]'s own case-study noise (0.03 pu SCADA, 0.01 pu PMU, the
current channels included; `formulas.noise.paper_sigma`, the plan's D8 and D11), and the attack is
stealthy because every snapshot is one AC state. `At` keeps its bound, each step within the meters'
rated accuracy (D7). Before D11 `Am` carried a between-snapshot bound as well: under the meters'
rated accuracy (D7) no overload window was stealthy: 0 of 10 IEEE-14 and 0 of 5 IEEE-118
60-snapshot windows, since moving one line's flow moves the injections and flows around its ends by
several times that change, beyond the rated accuracy of the small loads there. Measured under D8 and
D9 on 60-snapshot windows at the 5-minute pool cadence: IEEE-14 2 of 10 windows (8 devices, the
search not proven within its budget), IEEE-118 4 of 5 (3 to 11 devices, median 7, 2 proven, 0.22 s
per snapshot); the reported noiseless flow reaches the rating exactly. A window with no stealthy
overload stays benign and is counted in `fallback_benign`. With the hybrid meters the attacker also
writes the PMU branch currents its false state moves, which counts in the PMU of the bus at that end.
Only `At`'s channels are bounded between snapshots, the currents included (the PMU class, D7); `Am`'s
current channels are only counted when they move by more than 0.01 pu (D8, D11). Measured with `WLS` on the test split, same seed and pool, only the meter model changed (IEEE-14 3,000 frames, IEEE-118 2,000): benign angle MAE with an angle at every voltmeter 0.0097 and 0.0105 degrees, with angles at the PMUs only 0.0113 and 0.0118 (17% and 12% higher), and with the eq. (3) pseudo-measurements added 0.0113 and 0.0096 (level with PMUs only on IEEE-14, 18% lower on IEEE-118); on `At` records 0.063, 0.079 and 0.075 degrees on IEEE-14 and 0.0103, 0.0125 and 0.0120 on IEEE-118. The two meter models draw different noise and different episodes, so the attacked rows compare different attacks. Measured with the default recipe (`At` and `Am`, seed 1; IEEE-14 3,000 frames, IEEE-118 2,000), IEEE-14: `Am` 12 stealthy overload episodes on the hybrid meters (6.7 devices, 23.6 channels on average) and 14 on the v0.8.3 meters (5.6 devices), against 0 under the bound; 0 frames fell back to benign instead of 720; `At`, still bounded, 10.3 devices on the hybrid meters and 8.5 on the v0.8.3 ones; IEEE-118: `Am` 7 stealthy overload episodes on the hybrid meters (7.3 devices, 23.7 channels on average) and 8 on the v0.8.3 meters (4.4 devices), against 3 under the bound; 0 frames fell back to benign instead of 220; `At`, still bounded, 11.7 devices on the hybrid meters and 8.8 on the v0.8.3 ones. The released files are v0.8.3's recipe:
`families=LEGACY_FAMILIES, am_attack="redistribution", min_tamper=False,
redundancy={"meter_model": "v083"}`.

Notes on the table:

- A stealthy target is an attackable load that is not on a generator bus, zero-MW condensers
  included [BOY22] (`_load_tables`, `stealthy_pos`). An attackable load has nonzero active power,
  is not on the slack bus and is at most `max_load_mw = 2000` MW (`attackable_pos`). `Ad`, `As` and
  `Ar` draw from the attackable set.
- The `At` rise lasts 20% to 45% of the episode and the hold 0% to 25% (`draw_ramp`). Every frame
  carries at least one `ramp_rate` of change, so every frame labelled `At` is attacked
  (`ramp_dev`).
- `Al` picks its target line at random from the 15 lines with the largest achievable flow change
  (`_pick_lra_target`). `Am` flips the sign per episode with `am_direction = "both"`: "mask" makes a
  loaded line read lighter, "induce" makes a safe line read loaded (`am_sign`).
- A stealthy step with no local solution is halved: up to 3 times and never below the noise floor
  for `Aq` and `Al` (`AQ_HALVINGS`), up to 6 times for a ramp frame (`STEP_HALVINGS`). `Aq`, `At`
  and `Am` designs are tested on every frame they will occupy before they are accepted, up to 40
  draws (`ONSET_DRAWS`).
- `Ad`, `As` and `Ar` do not re-solve the grid, so bad-data detection can see them [DAT26].
  Unmetered channels stay zero after corruption (`_corrupt_frame`).

## 4. Episode placement

| step | rule | source |
|---|---|---|
| weights | each family is drawn with weight `1 / expected length`, so every family gets about the same share of attacked frames | `_Schedule.build` |
| draw | episodes are drawn until their frames sum to exactly `round(attacked_frac * T)`; the last one is clipped to the remainder | `_draw_episodes` |
| place | longest first, each at an onset drawn uniformly among the positions where it fits, never overlapping | `_place_episodes`, `_uniform_onset` |
| walk | every frame emitted in time order: benign runs between episodes, each episode where it was placed | `_walk` |
| fallback | a frame whose attack cannot be built is emitted benign; the count is the `fallback_benign` attribute | `_timeline_attrs` |
| split | chronological `split = (0.6, 0.2, 0.2)`; each boundary moves to the end of an episode it would cut | `_frame_split` |

Adjacent episodes and long quiet stretches are outcomes of the uniform draw, not of a rule. Placing
the longest episodes first keeps a 60-frame episode from being squeezed out by one-frame ones. The
default `attacked_frac = 0.5` gives a balanced file. The v0.8.1 and v0.8.3 files report `fallback_benign = 0`
on every system (`generate_timeline` docstring).

## 5. The per-frame layers

| group | datasets | meaning | written by |
|---|---|---|---|
| `data/` | `node_x [T, N, 4]`, `edge_x [T, E, 2]` | observed readings: benign plus noise, plus the attack where attacked | `_TimelineBuffers.store` |
| `data/` | `node_m`, `edge_m` | the meter plan per frame | `_write_masks` |
| `data/` | `y [T, N]` | 1 on each attacked bus | `store` |
| `data/` | `family`, `stealthy`, `seq_id`, `timestep`, `split [T]` | family code, 1 for `Aq/At/Al/Am`, episode index (-1 benign), pool timestep, 0/1/2 | `_finish_timeline` |
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
fg.generate("ieee118", name="my_run", frames=10_000, attacked_frac=0.5)   # needs [generate]
ds = fg.load("my_run", order="time")
```

`generation.generate(system, name, frames=None, states=None, out=None, seed=123, **knobs)` loads the
pool, keeps the first `frames` timesteps, calls `timeline.generate_timeline` and registers the file
under `name`. Without `states`, it reads `$FDIA_GRAPH_INIT` or downloads the system's pool.

| knob | default | meaning |
|---|---|---|
| `attacked_frac` | 0.5 | fraction of frames under an episode |
| `families` | ("At", "Am") | the families in rotation; the single-snapshot families are deprecated for generation |
| `min_tamper` | True | hold each episode on the support that tampers the fewest devices [WU26, eq. 12] |
| `am_attack` | "overload" | `Am` as the overload attack of [WU26]; "redistribution" is v0.8.3's `Am` |
| `stealth_scale` | 1.0 | a multiplier on At's stealth bound, in units of the rated accuracy (Am has none, D11) |
| `redundancy` | `{}` | the meter plan (`MeterSettings`): coverage `vbus_frac` 0.6, `pmu_frac` 0.2, `flow_frac` 0.9, and `meter_model` `"hybrid"` (D10, D12); the v0.8.3 recipe passes `{"meter_model": "v083"}` |
| `attack_intensity` | 0.20 | upper edge of the band of Aq, Al, Ad and As |
| `ramp_rate` | 0.002 | `At` growth per frame |
| `ramp_len` | 60 | `At` episode length |
| `am_len` | `ramp_len` | `Am` episode length |
| `am_rate` | 0.9 | `Am` largest per-frame step as a fraction of the noise floor |
| `am_direction` | "both" | "mask", "induce" or one drawn per episode |
| `hops` | 2 | the attacker's subnetwork reach in branches |
| `corrupt_len` | 1 | `Ad`/`As`/`Ar` episode length; None draws 5 to 24 frames |
| `replay_tau` | None | fixed `Ar` lag in frames; None draws a lag of at least 20 |
| `redundancy` | None | `{vbus_frac, pmu_frac, flow_frac}`, default 0.6/0.2/0.9 |
| `split` | (0.6, 0.2, 0.2) | chronological train/val/test fractions |
| `max_load_mw` | 2000.0 | a larger load is never a target; None disables |
| `seed` | 123 | the meter plan, the biases and every attack draw |

A data release follows the build script of the latest release (`examples/_build_timelines_v0*.py`):

1. Per system, copy the unchanged pool and run `fg.generate(C, ..., states=pool, seed=123, frames=72000)`
   into `timeline_ieee{C}.h5`; each system writes a manifest fragment with sha256 and size.
2. Run once more with `MERGE=1` to combine the fragments into `manifest.json`.
3. Upload with `python tools/upload_assets.py vX.Y.Z notes.md <files>`; it creates the `data-vX.Y.Z`
   tag and never replaces an asset already on the release.
4. Add the release's sha256 values to `registry._SHA256`, then bump `registry._RELEASE` last.
