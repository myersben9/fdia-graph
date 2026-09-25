# How a timeline is generated

One timeline per system is the dataset. This page follows it from the load profile to the file,
and states which layers a detector may read. Every value below is the code's default; the source of
each is named beside it. Terms are defined in [`../reference/GLOSSARY.md`](../reference/GLOSSARY.md).

| stage | code | output |
|---|---|---|
| 1. operating-point pool | `profiles.fetch_profile`, `profiles.generate_states`, the v0.7.1 pool build | `[T, N, 4]` AC states, one per minute |
| 2. meter plan and noise | `engine/core.py` (`FdiaGenerator.__init__`), `engine/measurement.py` | which channels are metered, one noisy scan per state |
| 3. attack families | `timeline.py`, `engine/records.py`, `engine/attacks.py` | the attacked scan of each family |
| 4. episode placement | `timeline._place_episodes`, `timeline._frame_split` | onsets, lengths, the split |
| 5. per-frame layers | `timeline._TimelineBuffers`, `timeline.write_temporal_layers` | the HDF5 file |

![The generation pipeline: an ISO load profile is resampled to one minute with per-bus AR(1) jitter and solved into 72,000 AC states; episodes are placed at uniform random onsets with an equal share of attacked frames per family; the walk emits one noisy scan per state and, inside an episode, either solves a local false state (Aq, At, Al, Am) or tampers the readings in place (Ad, As, Ar); after the walk the temporal features are written from the observed frames and the frames are split chronologically into one HDF5 file with observed, benign and clean layers](../figures/diagrams/generation_flow.png)

The published default release is v0.8.1 (`registry._RELEASE`). It was built before three changes
in this code: `Aq` and `Al` episodes are now one frame (v0.8.1 holds `Aq` for 15 to 44 frames), and
the swing scale now comes from the observed frames instead of the noiseless pool. Those changes
reach the data with the next release (`CHANGELOG.md`, Unreleased).

## 1. Operating-point pools

| item | value | source |
|---|---|---|
| load profile | NYISO system load, 11 zones summed, native 5-minute cadence | `profiles._fetch_nyiso` |
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
| <code>&#124;V&#124;</code> and `theta` | the union of the voltage buses and the PMU buses | `vbus_frac = 0.6` (`int(0.6 N)` buses), `pmu_frac = 0.2` (`max(1, int(0.2 N))` buses) |
| `P_inj`, `Q_inj` | every injection bus and every zero-injection bus | always |
| `P_from`, `Q_from` | each branch independently with probability `flow_frac` | `flow_frac = 0.9` |

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

`formulas.noise.bias_jitter_split` splits each total into jitter `0.25 SD` and bias
`sqrt(1 - 0.25^2) SD`, so the two add to the class total in quadrature. The bias is drawn once per
meter (`_meter_bias`). Power and flow jitter also carry an absolute floor of `1e-3` MW or MVAr, so a
near-zero reading still has noise. Treating the whole class error as per-scan noise would
over-jitter one-minute traces and hide the temporal signal (`bias_jitter_split` docstring).

`NOISE_FLOOR = 0.02` (`generation.py`) is the lower edge of the attack plausibility band: a change
below 2% of the reading sits inside meter error and resolves to noise.

## 3. The attack families

`attack_intensity = 0.20` is the upper edge of the band for Aq, Al, Ad and As (`generate_timeline`). The ramp families At and Am are set by `ramp_rate` and `ramp_len`, and Ar records the realized change of its replay without bounding it.
The stealthy families are local false states [WU26]: the attacker changes loads inside a subnetwork
within `hops = 2` branches, solves that subnetwork with the boundary voltages held true, and adds
`a = h(x_false) - h(x_true)` to the true scan (`engine/records._stealthy_frame`). Every meter keeps
its own noise draw, so the residual test sees noise only. Every false state must also stay within
the case's bus voltage limits and the generator P and Q limits widened to the range the pool used
(`records._within_limits`, `FdiaGenerator.operating_limits`).

| family | code | construction | magnitude per bus | episode length | stealthy | target set |
|---|---|---|---|---|---|---|
| `Aq` | 1 | local false state: each target load scaled by its own factor, rise or drop | 5% to 20% (`_draw_single_shot`) | 1 frame (`_ONE_FRAME`) | yes | 1 to 6 stealthy loads (`_pick_targets`) |
| `Ad` | 2 | in-place bias: additive shift on P and Q, a `N(0, 0.02)` pu shift on <code>&#124;V&#124;</code>, a shift on each incident flow | 2% to 20% per channel, random sign (`_corrupt_bias`) | `corrupt_len = 1` | no | 4 attackable loads |
| `As` | 3 | in-place scaling: one gain on P and Q, one gain per incident flow | gain 1.02 to 1.20 (`_corrupt_scaling`) | `corrupt_len = 1` | no | 4 attackable loads |
| `Ar` | 4 | in-place replay: the bus's four node channels copied from an earlier benign scan, at least 20 benign scans back once that many are buffered | not bounded; the realized change is recorded (`_corrupt_replay`) | `corrupt_len = 1` | no | 4 attackable loads |
| `At` | 5 | slow ramp: one load factor on a fixed bus set, rise, hold, return, each frame a local false state | 0.2% per frame (`ramp_rate`), peak 2.4% to 5.2% | 60 frames (`ramp_len`) | yes | 5 stealthy loads (`_draw_ramp`) |
| `Al` | 6 | redistribution: load-conserving shift across the two PTDF sides of a target line, lowering its apparent flow | 2% to 20% (`_lra_for_line`) | 1 frame (`_ONE_FRAME`) | yes | up to 6 stealthy loads per PTDF side, inside the line's subnetwork |
| `Am` | 7 | multi-snapshot [WU26]: an `Al` redistribution drawn at onset, reached in steps | per-frame step at most `am_rate * NOISE_FLOOR` = 1.8%, peak 2% to 20% (`_AmShape.under_floor`) | 60 frames (`am_len` defaults to `ramp_len`) | yes | as `Al` |

Notes on the table:

- A stealthy target is an attackable load that is not on a generator bus, zero-MW condensers
  included [BOY22] (`_load_tables`, `stealthy_pos`). An attackable load has nonzero active power,
  is not on the slack bus and is at most `max_load_mw = 2000` MW (`attackable_pos`). `Ad`, `As` and
  `Ar` draw from the attackable set.
- The `At` rise lasts 20% to 45% of the episode and the hold 0% to 25% (`_draw_ramp`). Every frame
  carries at least one `ramp_rate` of change, so every frame labelled `At` is attacked
  (`_ramp_dev`).
- `Al` picks its target line at random from the 15 lines with the largest achievable flow change
  (`_pick_lra_target`). `Am` flips the sign per episode with `am_direction = "both"`: "mask" makes a
  loaded line read lighter, "induce" makes a safe line read loaded (`_am_sign`).
- A stealthy step with no local solution is halved: up to 3 times and never below the noise floor
  for `Aq` and `Al` (`AQ_HALVINGS`), up to 6 times for a ramp frame (`STEP_HALVINGS`). `Aq`, `At`
  and `Am` designs are tested on every frame they will occupy before they are accepted, up to 40
  draws (`_ONSET_DRAWS`).
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
default `attacked_frac = 0.5` gives a balanced file. The v0.8.1 files report `fallback_benign = 0`
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
| `SEBase.fit` | `node_x`, `edge_x`, `family`, `clean` | fitting: meter sigmas, benign mean state, `theta_ref` |
| `SEBase.score` | `family`, `clean` | evaluation |
| `JacobianFeatures`, `JacobianWeighting` | `node_x`, `edge_x`, `prev_node_x`, `prev_edge_x`, `prev_timestep` | measurements |
| `GatedPrior(gate="oracle")` | `y` | evaluation ceiling only |
| `SwingThreshold`, `DeltaThreshold` | `swing`, `temporal_delta` | measurements |
| `ResidualLocalizer` | `node_x`, `edge_x` | measurements |
| `BusCNN`, `BusMLP`, `FedBusCNN`, `FedBusMLP` | `node_x`, `node_m`, `edge_x`, `temporal_delta`, `swing`, `prev_swing`, `prev_*`; `y` as the training label | measurements |
| `TrustSelector.fit` | `node_x`, `edge_x`, `family` | measurements |
| `TrustSelector.score` | `benign`, `edge_benign` | evaluation: what a secured meter reads |
| `trust.secured_copy` | `benign`, the tamper masks | builds a counterfactual file, then rewrites the temporal features from its observed frames |

Sources: `se/base.py`, `se/jacobian.py`, `se/methods.py`, `localization/methods.py`,
`localization/learned.py` (`_fields`), `trust/base.py`, `trust/secured.py`. Every path that fits an
estimator (`ResidualLocalizer`, the Jacobian feature sets, `TrustSelector`) reads `clean` at fit
time through `SEBase.fit`, and only there.

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
| `families` | all seven | the families in rotation |
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
