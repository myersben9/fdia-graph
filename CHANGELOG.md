# Changelog

Every release lists what a user of the package can see change. "No user-visible change" means
the public API, the generated files and the numbers are the same as the previous release.

## Unreleased

- **Removed: single-snapshot generation, the v0.8.3 recipe and the frozen suite.** The generator
  makes the multi-snapshot families `At` and `Am` only, on the hybrid meters; every published data
  release (v0.7.1 to v0.8.3) still loads read-only, with its family names (`Aq`, `Ad`, `As`, `Ar`,
  `Al`), aliases, the `heldout=` protocol and per-family scores. The removed code is at commit
  `aa77d0b`; fdia-graph 0.20 is the last version that generates the removed families.
  - Generation: `families=` accepts `At` and `Am` (new `models.inputs.GeneratedFamilies`; an older
    family raises before any work). Gone: the in-place `Ad`/`As`/`Ar` (`engine/attacks/corrupt.py`),
    the load redistribution behind `Al` and the v0.8.3 `Am` (engine/attacks/redistribution.py),
    `single_shot_design`, `am_design` / `am_step`, the `Aq`/`Al` frames, and the knobs
    `attack_intensity`, `am_rate`, `am_direction`, `corrupt_len`, `replay_tau` and
    `am_attack="redistribution"` (with `AmDirection`, `LEGACY_FAMILIES`, `DEPRECATED_FOR_GENERATION`,
    `ONE_FRAME_FAMILIES`, `generation.NOISE_FLOOR` and the models `Band`, `TamperTarget`,
    `Redistribution`, `AmDesign`). `FrameKnobs` keeps the fields new generation uses, all with
    defaults. `generate_stream` keeps its entry point with new generation's families and defaults.
  - The v0.8.3 meter model: `meter_model` accepts `"hybrid"` only, now also the engine's default.
    Files written with the old meters load unchanged.
  - Dead or retired code: the full pandapower re-solve (`PhysicsMixin.resolve_states`, `solve`,
    `_pin_generation`, `MeasurementMixin.emit`, `state_from_net`, `ResolvedPool`),
    `FdiaGenerator.centrality_probs`, `nl` and `load_genP`, the N-1 line-outage screen
    (`fg.line_outage_candidates`, `FdiaGenerator(outage=)`, `LineCandidate`, `OutageRef`,
    `GridIslanded`), `fg.pyg_stream` and `fg.torch_windows` (torch_data.py; `ds.windows` and
    `fg.load(..., format="pyg")` replace them), the exporters `ds.to_numpy` / `to_torch` / `to_tf` /
    `to_pandas` (`ds.export(format=...)`), `dataset.check_split` / `check_units` / `check_order` and
    `federated.check_partition` (the models check these), `SplitFractions`, `StreamSystem`, and the
    private `SEBase._normal_matrices` and `_cond`. `load_stream` and `streams.windows` stay: they
    read the v0.7 stream files. The `ds.edge_*` aliases stay (`edge_gs` and `edge_bs` are requested
    names).
  - Tests: the frozen suite (tests/test_frozen.py, tests/frozen/, tools/freeze_reference.py, the
    strict mode of `tools/prereview.py`) is gone; the pinned search answers
    (`tests/test_search_speed.py`), [WU26]'s scenarios and the unit tests guard the generated data.
    The shared test timeline is new generation's (`At` and `Am`); old releases are tested on
    checked-in files (`tests/test_old_releases.py`, `tests/data/README.md`).
  - Docs: the finished plans (data models, field groups, one dataset, readability, reference study,
    restructure, validation), FUTURE_DATASETS.md, the v0.8.0 to v0.8.3 release build scripts and
    five unlinked figures.
  - Fixed: `load_stream` at v0.7.1 and v0.7.2 reads the stream files again. Those files carry the
    frames only, and the release's graph sidecar (`graph_ieee{N}.npz`: `edge_index`, `edge_attr`,
    `node_m`, `edge_m`) that 0.18 stopped attaching is attached again; the tests read a slice of the
    published IEEE-118 stream with its sidecar, and one timeline of each of v0.8.0, v0.8.1 and
    v0.8.3 written by the SDK that built it.
  - `Stream` carries the PMU branch currents of a hybrid-meter file (`pmu_i`, `pmu_i_benign`, the
    static mask `pmu_i_m`), so `generate_stream` and `stream_of` no longer drop them. The benchmark
    table records the fixture recipe per row and `tools/bench.py --check` compares only rows of the
    same machine and recipe.
  - Known behaviour made visible by the new test timeline, unchanged here: an overload `Am` whose
    window has no stealthy design falls back to benign for the whole episode (`fallback_benign`), and
    its first snapshot can tamper nothing, its drift-free goal there being the true flow (D9).

- [WU26]'s trusted-PMU configuration as a Markov decision process (docs/plans/WU_DEFENSE_PLAN.md, PR B):
  `trust.WuDefenseEnv(g, states, goal, k, WuDefenseConfig(pmus, slots))` on one overload window, with
  `reset`, `step` and `valid`. An action trusts one more PMU at the next configuration step (Sec. IV-D2,
  "one PMU per step"); the reward is the rise in the fewest-tamper search's cost under the schedule so
  far (eq. 33 per step, eqs. 28-32), in measurements by default or devices (`WuDefenseConfig.unit`,
  new choice `CostUnit`, the plan's E3); the state is what Fig. 1 lists, the node and flow readings at
  the step's snapshot, the target lines' load rate and the trusted mask. Every schedule's answer is
  cached (`solves` counts the searches), since a training run revisits the same schedules. Ours, where
  the paper is silent: the readings are those of the attack under the schedule so far without noise, an
  attack the schedule makes infeasible costs every attackable channel or device, and Algorithm 1's line
  8 (a defended cost below the undefended one ends the episode without a reward) is kept and counted in
  `breaks`. The search is the cost oracle rather than the row reduction (`support_method="rref"`), which
  matched Table II and Fig. 12 worse (the plan's section 3). The search minimizes the unit the reward
  counts: new `FrameKnobs.objective` ("devices", the default and what generation uses, or "channels",
  which puts the tampered measurements first as eqs. 28 and 33 count them; the device bound then no
  longer prunes). The environment refuses a step outside the window (`WindowSlots`) and an action that
  is not a PMU still on offer (`ChosenAction`); its state reads only metered channels (the load rate
  from the full flows), and the infeasible cost counts the plan's metered channels or the devices that
  hold one. New validated models `WuDefenseConfig`, `WindowSlots` and `ChosenAction`.

- [WU26]'s own attack construction as the overload attack's support method (`OverloadSettings.support_method`, "search" by default or "rref"; new choice `SupportMethod`, `FrameKnobs.support_method`, file attribute `support_method`). The paper states it twice: the attacker "can utilize the transpose of the measurement matrix h, applying multiple elementary row transformations and column exchanges to find the optimal attack vector solution [25]" (p. 655, [YAN17]), and Solution 1's steps 1-5 (p. 657). New `formulas.trust.sparsest_rows` (the reduction, the column exchanges with their tracking order, the rows mapped back) and `engine.attacks.rref.RrefMixin.rref_support`: at each snapshot the Jacobian of the attack area's attackable channels (scaled by [WU26]'s noise, a trusted PMU's bus left out, eqs. 27 and 29) with respect to |V| and angle, for each target line the sparsest row whose state change moves its flow, the window's support the union over the snapshots (eq. 28), the magnitudes from the AC local flow solve so the goal, the limits and the D16 bounds hold. Ours: the chase starts from the transpose's own rows (the single-variable attacks), it chases the sparsest goal-moving row rather than the sparsest outright, and a support that cannot reach the goal grows by the next sparsest rows. On [WU26]'s scenarios the search matches Table II better: at k = 1.1 the trust schedule raises the search's devices 20% and 29% (the paper 24% to 35%) and the reduction's -14% and +13%; on IEEE-118 the search's devices rise 20% (the paper 16.4% mean) and the reduction's fall. The reduction is 0.5 to 0.8 s per 20-snapshot IEEE-14 window against the exhaustive search's 16 to 33 s, and 2.4 to 3.2 s per 10-snapshot IEEE-118 window against 2.8 to 9.6 s at budget 256. [WU26]'s reference now lists the paper's authors and pages (S. Wu, Q. Wang, J. Hu, Y. Ye, Y. Tang; pp. 651-665).

- [WU26]'s trusted-PMU defense as a constraint on the fewest-tamper search (docs/plans/WU_DEFENSE_PLAN.md,
  PR A): `min_tamper(..., trust=TrustSchedule(buses, slots))`. A PMU trusted at slot s keeps its bus's
  |V| and angle true at every snapshot t >= s (eqs. 26-32: the deviation is zero on its two secure rows,
  and trust accumulates, eqs. 30-31); its branch currents stay untrusted, as eq. (27) leaves them in the
  nonsecure set. With `per_slot` (the default) the attack's support may change at each slot, since eq.
  (28) takes each snapshot's deviation on its own (the plan's E13): a search one segment at a time from
  the held support, with the other segments fixed, and `MinimizerResult.plan` carries the support of each
  segment. New model `TrustSchedule` (validated: one slot per bus, each PMU once, non-negative indices).
  Without a schedule the search and its answers are unchanged. Measured on [WU26]'s IEEE-14 scenarios,
  20 pool frames, the PMUs trusted at the paper's snapshots 2, 4, 6 and 8, ratings k times each target
  line's peak flow: at k = 1.1 the attack survives with 5 to 6 devices and 8 to 11 channels (lines 3-4
  and 6-11) and 7 to 9 devices and 23 to 28 channels (lines 1-2 and 4-5), +20% and +29% in devices and
  +38% and +22% in channels against the paper's Table II 24% to 35%; at k = 1.2 no candidate reaches
  both ratings with the four PMUs trusted, held or with a support per segment. When no held support is
  feasible, the per-slot search seeds a plan with each segment's own cheapest support before calling
  the window infeasible (`_segment_seed`); on these scenarios no segment after the last slot has one.
  Segments are seeded in order, each from the attack vector its predecessor leaves, so an `At` window's
  stealth bound measures a segment's first step from where the previous one ended; the seed is greedy
  (a None from it is not a proof), and every plan the per-slot search returns is checked over the whole
  window.
  Per slot and held give the same answer, and so do the paper's trust order and the swapped one.

- Development only, no user-visible change: the test suite runs in parallel. pytest-xdist joins the
  `test` and `dev` extras, CI runs the three suites with `-n auto`, and `tools/prereview.py` does so
  (at most 8 workers) when the interpreter has it. The test extra's pip cache is kept between CI runs,
  and a pull request that changes only Markdown or `docs/` (not the data dictionary, which a test
  checks) skips the suites' install and run steps, the jobs still reporting success. The malformed-input
  tests' ids no longer carry memory addresses, which differed between workers.

- A certifier for the fewest-tamper search, optional and never on the generation path
  (docs/plans/RELAX_CERTIFIER_PLAN.md): `engine.attacks.certify.certify(g, states, goal, k)` runs the
  search and bounds its device count from below by a mixed-integer second-order-cone relaxation of
  [WU26] eq. (12) over the same area: device binaries with big-M links, Jabr's W with cuts that tie
  it to the voltages the PMUs pin, the goal (every line of a multi-line goal), the voltage limits,
  the zero injections, the search's support rule, and for `Am` the D14 and D16 injection bounds
  (generator limits (22)-(23) and `load_cap`, over the area and its edge). It returns a
  `Certificate` (`models.frames`): both bounds, whether they meet, the solve time and how far the
  relaxed point is from an AC state. `CertifyOptions` (`models.config`, checked on construction)
  sets SCIP's time limits (the relaxation, and 5 s per bound-tightening solve), the kept snapshots
  (non-empty, inside the window) and the cut families, new choice `CutFamily`, of
  `engine/attacks/relax_cuts.py` (bound tightening of each bus's voltage move, the QC relaxation,
  bus angles closing every cycle). A certificate is claimed only clear of SCIP's tolerances
  (docs/plans/RELAX_CERTIFIER_PLAN.md, section 4.1): a bound b proves ceil(b - `bound_margin`)
  devices, a would-be certificate (an infeasibility at one device fewer, or an optimum that reaches
  the search's count) stands only when a re-solve with numerics/feastol loosened to `robust_feastol`
  certifies it too, and a cut relaxation whose bound falls below the cone
  relaxation's is a contradiction; `Certificate.verdict` (new choice `CertifyVerdict`: "certified",
  "gap", "uncertain"), `reason` and `cone_lower` report it, and `models.frames.BoundClaim` is what
  one solve proves. Solved with SCIP through cvxpy, in the new
  optional extra `[certify]`, also part of `[all]`. Every noise threshold the relaxation reproduces
  is widened by the float32 roundoff of the search's own classification
  (`formulas.relax.roundoff_slack`; At's step bound by the float32 roundoff of attack values
  as large as the box allows, `step_roundoff_slack`), so the relaxation never cuts the search's
  attack. After bound tightening a would-be certificate is re-solved on the untightened
  relaxation, and the search's forced-device bound is no longer a floor. Every claim comes from a solve's proven
  bound or infeasibility: an infeasibility at cutoff c proves c + 1 devices, one without a cutoff
  proves nothing (0, and "uncertain" when the search found an attack). The certifier refuses
  knobs without finite voltage limits with the new named error `NoOperatingLimits` (input model
  `models.inputs.CertifiableLimits`): its voltage box and big-M constants come from the search's own
  limits, never from a default box. The bound is valid but loose: on IEEE-14 with new generation's
  defaults it certifies 0 of 10 two-line `Am` episodes (gaps of 3 to 11 devices, median 4) and 0 of
  [WU26]'s two scenarios (gaps 7 and 6) at every cut level, and 0 of 4 `At` episodes: the one
  the relaxation met without the guard (the cone relaxation's optimum at 9, the search's count) bounds
  5 to 6 at the loosened tolerance and is "uncertain". By default `tests/test_certify.py` checks the cone
  relaxation's validity on an IEEE-14 two-line `Am` window and on an `At` window whose stealth bound
  starts from a non-zero previous attack vector; the cut families' cases and the full solve run with
  `FDIA_SLOW=1`.

- The fewest-tamper search (`MinimizeMixin.min_tamper`) holds BLAS to one thread while it runs and
  restores the caller's setting after, through threadpoolctl (added to the `generate` and `all`
  extras; without it the search runs on whatever BLAS is set to). Its products are too small for
  threads to pay: on a many-core machine starting them cost more than the arithmetic. The limit is process-wide:
  while any search runs, other BLAS work in the process is on one thread too, and overlapping searches
  share one limit, so the caller's setting comes back when the last of them leaves. No knob is added. Measured on the same fixed episodes as the entry below
  (seed 1, hybrid meters, pool ratings, the D16 bounds with `load_cap` 0.5, two-line `Am`, budget
  256, 20-snapshot windows), before and after side by side on the same shared machine. Median seconds
  per episode, before to after:

  | System | Default threads, Am | Default threads, At | `OPENBLAS_NUM_THREADS=1`, Am | `OPENBLAS_NUM_THREADS=1`, At |
  |---|---|---|---|---|
  | IEEE-14 (10 Am, 4 At) | 3.1 to 3.1 | 0.58 to 0.28 | 3.0 to 2.7 | 0.31 to 0.31 |
  | IEEE-30 (5 Am, 3 At) | 5.5 to 5.6 | 0.81 to 0.33 | 5.5 to 5.1 | 0.32 to 0.26 |
  | IEEE-118 (4 Am, 3 At) | 7.2 to 3.1 | 0.89 to 0.44 | 3.3 to 3.3 | 0.38 to 0.51 |

  With the default threads the search now runs as fast as it did with the whole process on one
  thread; with the process already on one thread nothing changes but timing noise. Every one of the
  29 episodes, in all four runs, returns the same target lines, support, device and channel counts,
  proof, candidates solved and failed solves, and the pinned searches of tests/test_search_speed.py
  are unchanged.

- The fewest-tamper search is faster and finds the same supports, bit for bit. The flow solve keeps
  the full Ybus products of the released solve, since a search's answer can turn on the last bit of a
  solve near its convergence limit (a product over the region's rows alone changed the answer of 4 of
  30 pinned IEEE-14 searches), and does less around them.
  - Each voltage vector tried costs one product Ybus V, which the residual and, for an accepted
    vector, the next iteration's Jacobian share (`local_flow_solve`, `local_ac_solve`); the solve's
    slices of Ybus and Yf are cut once per support and reused by the active set's re-solves
    (`local_flow_solve(blocks=...)`, `formulas.network._flow_blocks`).
  - The bus adjacency is cached (`AreaMixin._boundary`) rather than rebuilt from the edge list for
    every candidate, and the generator and free-injection buses are built once.
  - Each snapshot's true flows and currents are computed once per window, and the attack vector is
    evaluated on the support's branches only; every other branch reads the same on both sides.
  - The tamper count is a union of boolean masks instead of Python sets.

  Measured on a fixed set of episodes (seed 1, hybrid meters, pool ratings, the D16 bounds with
  `load_cap` 0.5, two-line `Am`, budget 256, 20-snapshot windows), before and after side by side on
  the same shared machine. Median seconds per episode, before to after:

  | System | Default threads, Am | Default threads, At | One thread, Am | One thread, At |
  |---|---|---|---|---|
  | IEEE-14 (10 Am, 4 At) | 4.5 to 3.0 | 0.84 to 0.66 | 4.5 to 2.9 | 0.36 to 0.21 |
  | IEEE-30 (5 Am, 3 At) | 8.5 to 4.8 | 1.1 to 1.0 | 8.4 to 5.1 | 0.54 to 0.41 |
  | IEEE-118 (4 Am, 3 At) | 14.5 to 9.5 | 1.8 to 1.0 | 5.2 to 3.3 | 1.1 to 0.4 |

  Every one of the 29 episodes, in both thread settings, returns the same target lines, support,
  device and channel counts, proof, candidates solved and failed solves as before, and 160 flow
  solves on IEEE-14 and 118 (with and without the load cap) return the same false states bit for bit.
  On IEEE-118 most of what remains is the full Ybus product's BLAS thread start-up: the same search
  runs about three times faster with BLAS held to one thread (`OPENBLAS_NUM_THREADS=1`), with the same
  answers.
  Exhaustive searches on fixed IEEE-14 windows, with and without the load cap and with two-line goals,
  are pinned to their earlier answers (tests/test_search_speed.py).

- `SEBase.estimate` builds the measurement vectors, the `pmu_pseudo` slots included, one `chunk` at a
  time, and reads the pseudo values alone (new `formulas.estimation.pmu_pseudo_phasors`); the
  propagated covariances are computed only for the measured calibration's scans in `fit`. On a
  72k-frame IEEE-300 timeline this drops several whole-split float64 tensors (the [n, N, 2, 2]
  covariance alone about 691 MB). The estimates are unchanged, bit for bit.
- An overload episode drives two lines at once by default, as [WU26]'s case studies do
  (docs/plans/WU_MSFDIA_PLAN.md, D17): `OverloadSettings.n_lines` (2, or 1), passed as
  `am_attack={"n_lines": 1}`; the pair is drawn inside one attack area so one held support reaches
  both, at most `AM_LINE_TRIES` pairs are tried and an episode with none stays benign. The file
  records `n_lines`, and `episodes/am_*` holds one row per target line with the new
  `am_target_mva` (the goal at the window's end); `AmOverloadDesign.rating` became `ratings`, and
  `overload_step` returns the flow reached on every target line. New tables `WU26_SCENARIOS`,
  `WU26_PMUS` (IEEE-118's from the paper's Fig. 9) and `WU26_ATTACK_AREA`, with
  `OverloadMixin.wu26_branch` and `wu26_buses`; new tests `tests/test_wu_scenarios.py`. Measured with new generation's defaults (hybrid meters, `families=("Am",)`, seed 1, pool ratings, the D16 bounds, two lines): IEEE-14 (3000 frames) 25 two-line episodes built and 0 fallen back to benign, 6.8 devices and 19.3 channels on average, the largest change on a channel 0.17 pu at the median episode and 1.59 pu at most, 0% of the searches proven, 39 s of generation per episode; IEEE-118 (2000 frames) 17 two-line episodes built and 0 fallen back to benign, 17.8 devices and 72.0 channels on average, the largest change on a channel 1.33 pu at the median episode and 3.39 pu at most, 0% of the searches proven, 78 s of generation per episode; every line of every episode reaches its rating.

- The overload attack bounds the edge of its support (docs/plans/WU_MSFDIA_PLAN.md, D16): the
  generator limits (22)-(23) apply to every generator whose reported output the attack changes, the
  support's edge included (pinned in the solve, P and Q apart, and checked by `within_limits` over the
  support and its edge), and every load bus it moves, in the support or on its edge, shows an active
  change of at most `load_cap` times its true load [YUA11] (new `OverloadSettings.load_cap`, 0.5 by
  default, in (0, 1], passed as `am_attack={"load_cap": ...}` and recorded as the file attribute
  `load_cap`). `local_flow_solve` holds injections at edge buses as well
  (`formulas.network._injection_jacobian`); `FalseStateMixin.touched_buses`; `FrameKnobs.load_cap`.

- The overload attack's line ratings default to the operating pool (docs/plans/WU_MSFDIA_PLAN.md,
  D15): S_max of each branch is 1.25 times its peak true apparent flow over the pool the timeline
  walks, so `Am` runs on every system of the ladder. The PGLib-OPF ratings stay available,
  `generate_timeline(am_attack={"rating_source": "pglib"})` (IEEE-14, 118 and 300; `NoLineRatings`
  elsewhere). New model `OverloadSettings` (`rating_source` "pool" or "pglib", `rating_margin` > 1),
  passed as a dict through `am_attack`; new choice `RatingSource`;
  `OverloadMixin.use_line_ratings`, `MeasurementMixin.all_flows_from_states`; new file attributes
  `rating_source` and `rating_margin` when the overload attack runs. Measured with new generation's defaults (hybrid meters, `families=("Am",)`, seed 1, the pool ratings computed over the frames walked, the bounds of D16): IEEE-14 (3000 frames) 25 episodes built and 0 fallen back to benign, 3.9 devices and 9.9 channels on average, the largest change on a channel 0.14 pu at the median episode and 0.82 pu at most, 8% of the searches proven, 30 s of generation per episode; IEEE-118 (2000 frames) 17 episodes built and 0 fallen back to benign, 9.6 devices and 30.4 channels on average, the largest change on a channel 0.40 pu at the median episode and 3.29 pu at most, 0% of the searches proven, 71 s of generation per episode; every episode's noiseless flow reaches its rating. Before D16 bounded the edge of the support the same runs gave IEEE-14 (3000 frames) 25 episodes built and 0 fallen back to benign, 5.3 devices and 15.0 channels on average, the largest change on a channel 0.18 pu at the median episode and 5.74 pu at most, 36% of the searches proven, 15 s of generation per episode; IEEE-118 (2000 frames) 17 episodes built and 0 fallen back to benign, 6.8 devices and 21.4 channels on average, the largest change on a channel 1.05 pu at the median episode and 44.94 pu at most, 24% of the searches proven, 43 s of generation per episode, and IEEE-30 (600 frames, a smoke run) 5 episodes of 3.8 devices.

- Review fixes to the hybrid meters and the overload attack: a generator pinned at one limit keeps
  its other component free (the active set pins P and Q apart); an overload frame's magnitude at a
  generator bus is its apparent change against the true generator output (at a load bus, the
  active change against the true load; `OverloadMixin.pretended_change`); a PMU current channel's
  systematic bias scales with its end phasor's magnitude, the scale of `current_sigma`
  (`formulas.noise.biased_current`, `current_magnitude`), which changes hybrid emission only.

- `Am` follows [WU26]'s model, eqs. (12)-(25), exactly (docs/plans/WU_MSFDIA_PLAN.md, D11): no
  bound on how far the attack moves between snapshots, which the paper does not have; its noise
  (0.03 pu SCADA, 0.01 pu PMU, currents included) only decides which changes the tamper count
  ignores. The bound was what made the overload attack infeasible on IEEE-14. `At` keeps its
  rated-accuracy bound, and `stealth_scale` is now `At`'s alone. Measured with the recipe of the time (one line per episode; `At` and `Am`, seed 1; IEEE-14 3,000 frames, IEEE-118 2,000), IEEE-14: `Am` 12 stealthy overload episodes on the hybrid meters (6.7 devices, 23.6 channels on average) and 14 on the v0.8.3 meters (5.6 devices), against 0 under the bound; 0 frames fell back to benign instead of 720; `At`, still bounded, 10.3 devices on the hybrid meters and 8.5 on the v0.8.3 ones; IEEE-118: `Am` 7 stealthy overload episodes on the hybrid meters (7.3 devices, 23.7 channels on average) and 8 on the v0.8.3 meters (4.4 devices), against 3 under the bound; 0 frames fell back to benign instead of 220; `At`, still bounded, 11.7 devices on the hybrid meters and 8.8 on the v0.8.3 ones. The rest of the `Am` path follows the paper too (D14, below).
- `Am`'s generators are free (D14): [WU26, eqs. 13-14] let every injection be tampered and (22)-(23)
  bound only the generator output, so a generator bus in the support has free P and Q injection
  inside its limits, like an attackable load (`FalseStateMixin.free_injection_buses`,
  `generator_buses`); a support needs an end bus of each goal line and one free injection. The flow
  solve enforces the limits (21)-(23) by an active set (a generator pinned at the limit it would
  pass, a bus voltage held at its limit, the least-norm solve repeated;
  `formulas.network.local_flow_solve(vm_fixed=...)`). `FlowGoal` takes one or more lines
  (`FlowGoal.more`, `lines`, `targets_at`; `overload_goal(window, line, *more)`,
  `solve_flow_local(line=[...])`): one held support, each line to its own rating. Generated episodes
  drive two lines by default since D17 (`OverloadSettings.n_lines`, one line when set to 1). A zero-injection bus stays held at zero, a rule of ours. The labels of an
  overload frame are the free-injection buses of the support. Measured on the hybrid meters with the recipe of the time (one line per episode; `At` and `Am`, seed 1; IEEE-14 3,000 frames, IEEE-118 2,000): `Am` 12 stealthy overload episodes on IEEE-14 (6.5 devices, 23.9 channels on average) and 7 on IEEE-118 (7.0 devices, 27.3 channels), no frame falling back to benign; `At` 10.3 and 12.5 devices (the At episodes differ from the previous run because the Am designs draw from the same random stream). On [WU26]'s IEEE-14 metering and two-line scenarios with the PGLib-OPF ratings (the paper
  does not state its limits) the attack is feasible and tampers more devices than the paper's, with
  changes of several pu, since the ratings are 3 to 19 times the true flows (the plan's D14 table).
- `meter_model` is its own knob (D12), a field of the new meter-plan model `MeterSettings` that
  `generate_timeline(redundancy=...)` takes beside the coverage fractions (no new parameter on the
  entry point): `"hybrid"` by default, and the v0.8.3 recipe (the frozen test
  timeline, the v0.8.3 build script, `generate_stream`) pins `"v083"`.
- New timeline field `prev_pmu_i` (D13): the previous emitted frame's PMU branch currents, offered
  by `export` on a hybrid-meter file beside `prev_node_x` and `prev_edge_x`. `JacobianWeighting`
  takes `pmu_pseudo=True` and reads it.

- The hybrid meter model (docs/plans/WU_MSFDIA_PLAN.md, decision D10), what new generation's meters
  read: a SCADA voltmeter reads `|V|` only and the angle is a PMU channel, and every PMU reads the
  current phasor of each in-service branch at its bus [WU26, eqs. 19-20], the real and imaginary
  part at each end, per unit on the base current (`formulas.network.branch_currents`). Their noise
  is IEEE C37.118.1's 1% total vector error taken as three standard deviations of the current's
  magnitude plus a 1e-5 pu floor (`formulas.noise.current_sigma`, `PMU_CURRENT_CLASS`), one rule
  the emitter (with the jitter part) and the fewest-tamper search (with the whole class) share
  [C37118]. `"hybrid"` is new generation's default; the v0.8.3 recipe passes
  `redundancy={"meter_model": "v083"}` (D12); `FdiaGenerator`'s own default stays `"v083"`. A hybrid file gains `data/pmu_i`, `data/pmu_i_m`,
  `benign/pmu_i_benign` and `attack/pmu_i_tamper` and the attributes `meter_model`,
  `current_feat` and `current_units`; a v0.8.3-meter file has none of them and loads unchanged.
  Records, batches and `export` carry `pmu_i`, `pmu_i_m` and `pmu_i_benign` when the file has them,
  in per unit on every view (new capability `pmu_currents`). The stealthy families write
  `h(x^a) - h(x)` on the current channels too; the fewest-tamper search counts a tampered current
  in the PMU of the bus at its end, and bounds its step by the PMU accuracy class for `At` (D7); for `Am`
  it counts a current beyond 0.01 pu (D8, D11, `formulas.noise.paper_current_sigma`). Under either meter model the
  search charges an angle channel only at a PMU bus. New option `pmu_pseudo` (off by default) on `WLS`,
  `AdaptiveWeighting`, `ResidualRemoval`, `SubspacePrior`, `GatedPrior` and `JacobianWeighting`: the pseudo voltage
  phasors of [WU26, eq. (3)] at the far end of every PMU-metered branch fill the `|V|` and angle
  slots no meter reads, weighted in the measured calibration by their first-order propagated
  variance (`formulas.estimation.pmu_pseudo_links`, `pmu_pseudo_voltages`). New models
  `MeterModel`, `CurrentColumns`, `CurrentIndex`, `CURRENT`, `PmuCurrentFields`, `PseudoLinks` and
  `PseudoVoltages`; `AttackVector` is a named tuple with an optional `current`. Measured with `WLS` on the test split, same seed and pool, only the meter model changed (IEEE-14 3,000 frames, IEEE-118 2,000): benign angle MAE with an angle at every voltmeter 0.0097 and 0.0105 degrees, with angles at the PMUs only 0.0113 and 0.0118 (17% and 12% higher), and with the eq. (3) pseudo-measurements added 0.0113 and 0.0096 (level with PMUs only on IEEE-14, 18% lower on IEEE-118); on `At` records 0.063, 0.079 and 0.075 degrees on IEEE-14 and 0.0103, 0.0125 and 0.0120 on IEEE-118. The two meter models draw different noise and different episodes, so the attacked rows compare different attacks. Measured with the recipe of the time (one line per episode; `At` and `Am`, seed 1; IEEE-14 3,000 frames, IEEE-118 2,000), IEEE-14: `Am` 12 stealthy overload episodes on the hybrid meters (6.7 devices, 23.6 channels on average) and 14 on the v0.8.3 meters (5.6 devices), against 0 under the bound; 0 frames fell back to benign instead of 720; `At`, still bounded, 10.3 devices on the hybrid meters and 8.5 on the v0.8.3 ones; IEEE-118: `Am` 7 stealthy overload episodes on the hybrid meters (7.3 devices, 23.7 channels on average) and 8 on the v0.8.3 meters (4.4 devices), against 3 under the bound; 0 frames fell back to benign instead of 220; `At`, still bounded, 11.7 devices on the hybrid meters and 8.8 on the v0.8.3 ones. The v0.8.3 recipe and the strict frozen suite are bit-exact.
- The documentation states what the stealthy families take from [WU26]: its constraints (13)-(18)
  and (21)-(23) (the SCADA measurements, the PMU voltage magnitudes and angles, and the operating
  limits; and, in new generation, whose PMUs read branch currents, the current phasors (19)-(20) as well). New generation also solves
  its objective, eq. (12), for `At` and `Am`, and `Am` is its overload attack (below); the released
  files' `Aq`, `Al` and `Am` drew their targets at random and drove no line to its limit.
- New generation makes the multi-snapshot families of [WU26], `At` and `Am`, both on the
  fewest-tamper search (docs/plans/WU_MSFDIA_PLAN.md, decisions D2 and D6): `generate_timeline` and
  `fg.generate` default to `families=("At", "Am")`, `min_tamper=True` and `am_attack="overload"`.
  `Am` is the overload attack of [WU26, eqs. 24-25]: a rated branch whose flow is metered and whose
  true flow stays below its rating over the window is driven, snapshot by snapshot, until the flow
  the tampered measurements carry before noise reaches the rating, on the support that tampers the
  fewest devices. An episode tries at most eight eligible branches in a random order (each try is a
  full search), a bounded heuristic: when none of them admits an attack the episode stays benign and
  is counted. Each snapshot's false state frees the attackable loads of the support and holds
  every other bus's injection, the least-norm voltage change that meets the flow
  (`formulas.network.local_flow_solve`, `FalseStateMixin.solve_flow_local`). The ratings are the
  `rate_a` of the PGLib-OPF v23.07 versions of IEEE-14, 118 and 300 (CC BY 4.0), stored in
  `fdia_graph.ratings` and matched to the branches by their end buses
  (`formulas.attacks.branch_ratings`, `OverloadMixin.line_ratings`); pandapower rates every branch
  9,900 MVA, the "no limit" placeholder. Other cases refuse `Am` with the new named error
  `NoLineRatings`. Each overload episode writes its branch, rating, reached noiseless flow and
  emitted flow under `episodes/` (`am_*`); new models `FlowGoal` and `AmOverloadDesign`; new knob
  `stealth_scale` (a multiplier on the stealth bound, 1 by default; `At`'s alone since D11). `Am`'s
  tamper count (and, until D11, its stealth bound) use [WU26]'s own case-study noise, 0.03 pu on SCADA channels and 0.01 pu on PMU channels in
  the stored units (`formulas.noise.WU26_NOISE`, `paper_sigma`; the plan's D8), while `At` keeps the
  meters' rated accuracy (D7). Snapshot t's goal is the true flow plus a linear share of what
  separates the window's last true flow from the rating (D9), so the attack rides on the load's own
  drift instead of cancelling it and reaches the rating at the last snapshot. Generating `Aq`, `Ad`,
  `As`, `Ar` or `Al` warns (`DeprecationWarning`) and is refused from 0.22; released files holding
  them load unchanged. `timeline.LEGACY_FAMILIES` with `am_attack="redistribution"` and
  `min_tamper=False` reproduces data release v0.8.3 (its build script and the frozen test timeline
  use it; the strict frozen suite is bit-exact). Under the meters' rated accuracy (D7) no overload
  window was stealthy: 0 of 10 IEEE-14 and 0 of 5 IEEE-118 60-snapshot windows, since moving one
  line's flow moves the injections and flows around its ends by several times that change, beyond
  the rated accuracy of the small loads there. Measured under D8 and D9 on 60-snapshot windows at
  the 5-minute pool cadence: IEEE-14 2 of 10 windows (8 devices, the search not proven within its
  budget), IEEE-118 4 of 5 (3 to 11 devices, median 7, 2 proven, 0.22 s per snapshot); the reported
  noiseless flow reaches the rating exactly.
- The fewest-tamper search of [WU26, eq. 12], behind `generate_timeline(min_tamper=True)` (on for
  new generation; `min_tamper=False` walks the released files' recipe). Each At episode is held on the support (the buses
  its false state moves) that tampers the fewest devices over the episode, a device being one SCADA
  terminal or one PMU per bus and a change under a meter's accuracy-class sigma not counted. A
  support is held for the whole episode and must solve at every snapshot inside the operating
  limits and move no metered channel by more than its accuracy-class sigma from one snapshot to the
  next, the first snapshot measured from the frame before the episode (its attack vector when that
  frame was attacked, since episodes may be adjacent). The search solves candidate supports in the
  attacker's area smallest first, each closed over the zero-injection buses of its boundary,
  starting from the region the episode was accepted on; a local solve that does not converge proves
  nothing, so those candidates are counted (`min_unsolved`) and the optimum over held supports is
  claimed only among converged solves, when it exhausts the area or reaches the devices the goal
  forces above their accuracy-class sigma, settles after `min_budget` candidates otherwise, and
  treats a support moving no device beyond its sigma as no attack, and records `min_devices = -1`
  when no held support meets every constraint (the episode then runs on its region). Each search is
  written under `episodes/` (`min_support_*`, `min_devices`, `min_unsolved`, `min_channels`,
  `min_proven`, `min_evaluated`, `min_lower_bound`). New formulas `formulas.noise.jitter_sigma`
  (the emitter's per-scan noise rule, emission only) and `accuracy_sigma` (a meter's rated accuracy
  at a reading, the measured calibration's rule through `accuracy_class_sigma`, what the search
  counts against), `formulas.attacks.tampered_channels` and `tampered_devices`; new models
  `LoadGoal` and `MinimizerResult`; `MeasurementMixin.meter_masks` gives the masks without a random
  draw.
- Every attack is built in one place, `engine.attacks`, a package whose modules compose the
  generator's `AttackMixin`: `area` (the attacker's area), `false_state` (the local solve, the
  operating limits, the attack vector and the tamper set), `redistribution` (the `Al` and `Am`
  redistribution), `stealthy` (the `Aq`/`At`/`Al`/`Am` frames), `episodes` (what an episode
  attacks, drawn at onset, typed as `RampDesign` and `AmDesign`) and `corrupt` (`Ad`/`As`/`Ar` and
  the replay buffer). This code was spread over `engine/records.py`, `engine/physics.py`,
  `engine/attacks.py` and `timeline.py`; `records` now emits a benign scan and hands an attacked
  family to the mixin, and `timeline` decides when and where each episode runs. No user-visible
  change: the generated files are bit for bit the same. The public names keep their old import
  paths; a moved private name (such as `engine.records._stealthy_frame` or `timeline._AmShape`)
  still imports from its old module with a `DeprecationWarning` naming its new home, until 0.22.
- Every annotation carries its real type. The engine's pandapower network and its internal case are
  `engine.pp_types.PandapowerNet` and `PpcTables`; torch modules and tensors, HDF5 files and
  datasets, sparse admittance matrices and the stream mappings are named; the column views are
  `NodeColumns`/`EdgeColumns`/`BranchColumns` of arrays and the index constants `NODE`/`EDGE`/`BRANCH`
  are `NodeIndex`/`EdgeIndex`/`BranchIndex` of ints (the same fields, still tuples); a stream's
  episodes are `models.data.EpisodeRow` and its summary `StreamSummary`. `Any` stays only where the
  value can be anything (the validation engine's raw input, keyword pass-throughs, mixed-value
  staging dicts), each place listed in `ANY_ALLOWED` in `tools/readability.py` with its reason, and
  the readability gate refuses an unlisted one. No behaviour changes.

- `load_profile` takes a load source that reads itself: `IsoFolder(iso, directory)` (an operator's
  CSV export; `models.config.IsoExport` checks the operator and refuses one with no known export
  format, ERCOT, when the folder is described), `CsvColumn(path, column)` or `RawSeries(values)`, or
  any object with a `loads()` method (`profiles.LoadSource`). The old form, `load_profile(source,
  path=, column=)` with a string or an array, still works for one minor version and raises a
  `DeprecationWarning`.
- `fetch_profile` reads from the operator's feed in `profiles._FEEDS`: `NyisoArchive` (built in, no
  dependency) for NYISO, `GridstatusFeed` for CAISO and ERCOT, which imports `gridstatus` itself and
  names the install when it is missing; `models.config.ProfileFetch` checks the operator and
  `resample_min`. The downloaded data and the returned vector are unchanged.
- Every input is checked in one place (the validation plan, in the git history at `aa77d0b`). Each consumer's settings
  are one model in `fdia_graph.models.config` (`LoadOptions`, `WindowSpec`, `HuberConfig`,
  `PriorConfig`, `LearnedConfig`, `FederatedSettings`, `TimelineKnobs` and the rest), whose fields
  declare their rules (`Annotated[float, Positive()]`, `OneOf(Units)`); one engine,
  `models.validation`, checks them when the model is built. The public signatures are unchanged:
  the keyword arguments build the model. Every fixed set of values is a `Choice` enum in
  `models.choices`, still importable where it was used. A dataset view is checked through one table,
  `ds.require(...)` over `dataset.base.CAPABILITIES`. A condition only the data reveals raises a
  named error from `fdia_graph.errors` (`NoBenignRecords`, `NoAttackedRecords`, `GridIslanded`, ...).
  The formulas and the parsers keep their signatures and build their input model
  (`models.inputs`: `ClientUpdates`, `ClientGraph`, `LabelGrids`, `FamilySelection`, `SystemRef`,
  `ReleaseName`, `OutageRef`, `StatePool`, ...), so no function body checks its arguments.
  Every error is still a `ValueError`; the messages now have one shape, "<Model>.<field> <rule>, got
  <value>". The loader's record `format` ("torch" or "pyg") is now checked too: a typo such as
  "pygg" used to fall through to the torch records silently. `check_split`, `check_units` and
  `check_order` still work for one minor version and raise a `DeprecationWarning`. The private
  helpers the models replace are gone (`_check_knobs`, `_check_settings`, `_check_frac`,
  `_check_graph`, `_check_blocks`, `_index_array`, `_is_int`, `_FORMATS`, `_ISOS`, `_FIELD_FLAG`);
  `dataset._FAMILY_ALIAS` and `profiles.system_id` still resolve.
  Integer settings (`npass`, `iters`, `n_calib`, `layers`, `hidden`, `k`, a window's `T`, the
  federated `partition_clients` and `epochs`, the episode lengths, `max_test`) now refuse a float
  at construction; before, a value such as `npass=1.5` passed and failed later with a raw
  `TypeError`. A malformed value of any setting (a string where a number belongs) is a
  `ConfigError` with the same message as an out-of-range one.
  The deprecated `pyg_stream` and `torch_windows` refuse an unknown `layer` with a `ConfigError`
  before the stream loads; before, it was a raw `KeyError` after loading. `fetch_profile`'s
  `resample_min` must be a whole number of minutes: `1.5` used to be truncated to a 1-minute
  cadence without a word, and is now a `ConfigError`. `partition_from_assignment` checks the
  assignment and the branch list before sizing the grid from them, so a scalar assignment or a
  non-integer `edge_index` is a `ConfigError`, not a raw `TypeError`, and so is a negative bus
  count.
  A profile date is a 'YYYY-MM-DD' string, a date or a datetime on every Python version (from
  3.11 '20240131' and the integer 20240131 used to parse too). `load_profile` refuses a source
  that is not a `LoadSource`, an operator name, a CSV path or a 1-d series of loads (a scalar
  used to pass as a one-point profile); `generate`'s `states` must be an array or a path; and
  `export(fields=...)` takes a list or tuple of names, so a lone string, a mapping or a set is a
  `ConfigError`. `RawSeries` takes a non-empty 1-d series: a scalar used to pass as a one-point
  profile and a 2-d array was flattened without a word.
  `CsvColumn` checks its path and column when built (a missing column used to surface as a
  pandas `KeyError` on read), and a dataset name that is neither a name nor a bus count, such
  as `fg.load(["ieee14"])`, is a `ConfigError` instead of a raw `TypeError`.
  A profile date span that ends before it starts, and an `IsoFolder` directory that is not a
  path, are refused when given; `generate(states=pathlib.Path(...))` now reads the file instead
  of failing with an `AttributeError`.
  Arguments reach their models unconverted, so a malformed one is a `ConfigError`, not the raw
  error of a conversion done first: `families=3`, `pool_moments(None)`, `fedavg(None, ...)`,
  `block_diagonal_basis(3, d)`, `score(ds, scores="bad")`, `score(ds, xhat="bad")` and a
  non-numeric in-memory state pool. A test refuses `Model(tuple(arg))`-style calls in the package.
  A Huber `c`, a removal threshold, `am_rate` and the learned localizer's `lr` and `pos_weight`
  must be finite as well as positive, and a solver `tol` finite and non-negative. The learned
  localizer checks its whole training setup (`dropout`, `lr`, `weight_decay`, `batch_size`,
  `epochs`, `pos_weight`, `seed`) on `LearnedConfig` when it is built. `generate_timeline` checks
  `families` on `FamilySelection` before it builds the case, and the admissible-target check is a
  model, `models.inputs.AdmissibleTargets`, which raises `NoAdmissibleTarget` as before (a model
  names its failure type in `Validated.error`; the named data errors now live in `models.errors`
  and `fdia_graph.errors` re-exports them). `TimelineKnobs.am_len` defaults to None, resolved to
  `ramp_len` by the model. `check_partition` is deprecated in favour of building
  `PartitionOnGrid`, and the internal `check_window_args` and `check_targets` are gone
  (`WindowSpec`, `AdmissibleTargets`). A value no model can even read (a number past the float
  range, a mapping where a sequence belongs) is a `ConfigError`, never a raw `OverflowError`,
  `KeyError` or `AttributeError`; `tests/test_malformed_inputs.py` sweeps every config and input
  model with values of the wrong kind and the public entry points with one bad argument each.
  Loose input is read by a parser model, so no function outside `models/` dispatches on a type:
  `ProfileSource` (what `load_profile` was handed), `DateSpan` (`fetch_profile`'s dates; a date it
  cannot read is a `ConfigError` before any download), `StateSource` (an in-memory pool or a path)
  and `DatasetName` (the registry's lookup key). `tools/readability.py` now refuses
  `isinstance(...)` outside `fdia_graph.models` as it refuses a bare `raise ValueError`.
- The estimator solve path no longer takes the slack angle. `_solve(z, w)`, `_w_solve(z, w)`,
  `_nres(x, z)` and the rest solve at the fitted reference (`ref_angles`, through the new
  `_h_ref(x)`); a custom `SEBase` subclass that overrides `_solve` drops its `thsl` argument.
  `JacobianFeatures.previous_estimate` returns the state alone. The numbers are unchanged.
- Detectors calibrate from measurements only. `SEBase.fit(ds, calibrate="measured")` sets every
  meter's sigma from its accuracy class (`formulas.estimation.accuracy_class_sigma`, the classes now
  one definition, `engine.base.ACCURACY_CLASS`, shared with the noise model), takes the angle
  reference from the case and the linearization point from the mean estimate of the benign training
  scans, and reads no clean layer. The Jacobian feature sets, `ResidualLocalizer` and
  `TrustSelector` fit their estimator this way; the state estimators keep `calibrate="truth"` as the
  benchmark default. Residual-based sigmas were tried first and rejected: a meter's constant bias
  is absorbed into the state, so they shrink on the biased meters and collapse the fit. On the
  v0.8.3 test splits plain WLS with the class calibration scores 0.086, 0.028 and 0.034 degrees
  against 0.091, 0.020 and 0.027 truth-calibrated on IEEE-14, 118 and 300. The frozen
  localization reference moves for the residual arm.
## 0.20.0

- Data release v0.8.3 is the default (`fg.load("ieee118")`; v0.8.1 stays readable with
  `release="v0.8.1"`). The eight timelines are regenerated from the unchanged pools (byte-identical)
  with seed 123 and the v0.8.1 knobs, built by `examples/_build_timelines_v083.py`: Aq and Al
  episodes are one frame, and the stored `temporal_delta` and `swing` are functions of the observed
  frames alone. Each file holds about 24,600 episodes instead of about 16,100, keeps 72,000 frames,
  exactly half attacked, no frame falling back to benign, and every family present.
  `data-v0.8.2`, an intermediate build with one-frame Aq only, is superseded.
- Engine: parameter groups that travelled through every attack function are data models
  (`models.frames`): `Band(floor, cap)`, the plausibility band of an in-place tamper
  (`FrameKnobs.band`); `TamperTarget(buses, branches)`, where an in-place attack writes; and
  `AttackDesign(targets, mult, interior)`, a stealthy attack's design on one scan.
  `FdiaGenerator.corrupt(scan, buses, kind, replay, band)` tampers a `Scan` in place and returns
  `(weak, mags)`; `attack_frame`, `is_feasible` and `stealthy_state` take an `AttackDesign`, and
  `with_region` fills in a design's interior. The random-draw order is unchanged: every frozen
  reference matches bit for bit.
- Features are computed from measurements only. The swing feature's scale is now the recent change
  of the observed frames, not of the noiseless pool: `timeline.write_temporal_layers` writes
  `temporal_delta` and `swing` after the walk from the observed injections alone, and
  `trust.secured_copy` recomputes them the same way. Every estimate takes one angle reference,
  `SEBase.theta_ref`, fixed at fit time from the training split's slack angle (the case's reference
  angle on the released pools; a slack angle that varies is refused), so estimating, the Jacobian transform, the residual localizer's scores and the
  trust scoring read no clean layer. Two uses of the truth remain, by design: fitting an estimator
  calibrates the meter weights, the benign mean state and the angle reference on the training
  split's clean layer, and
  `SEBase.score` compares estimates with the truth to report the error.
  `write_temporal_layers` runs in blocks with bounded memory. Takes effect in the data with the next
  release; scores on existing files are unchanged.
- Al is a single-snapshot attack: every Al episode of a generated timeline is one frame, as Aq.
- N-1 timeline generation is disabled: `fg.generate` and `generate_timeline` take no outage;
  `line_outage_candidates` remains a screening aid and `FdiaGenerator(outage=)` stays for engine use.
- The tiny test timeline moves to seed 1, the first seed that puts every family in both its train
  and test split under the new episode lengths; the frozen references are rewritten from it.
- Localization: `features="full14+prev"` (16 channels) and `"full14+prev+jac"` (24) append the
  previous frame's swing to the papers' 14, for `BusCNN`, `BusMLP` and the federated localizers
  (each bus's own reading, so client-local). A timeline export offers it as `prev_swing` (file
  row - 1). On a timeline the frame after a one-frame attack carries the same jump back with the
  opposite sign, labelled benign (31% of IEEE-118's benign training frames); a one-frame model
  cannot tell the two apart and stops trusting the swing. With the previous swing, the zero-shot
  BusCNN on v0.8.1 goes 0.78 -> 0.84, 0.55 -> 0.83 and 0.49 -> 0.81 macro-F1 on IEEE-14, 118 and 300
  (seeds 123 to 125), As 0.56 -> 0.95 and Ar 0.32 -> 0.83 on 118, at a lower false-alarm rate.
- Aq is a single-snapshot attack: every Aq episode of a generated timeline is one frame, as
  `corrupt_len=1` already made Ad, As and Ar. Through v0.8.1 Aq ran as 15 to 44 frame episodes, so
  a change-based feature saw only an episode's first frame (on IEEE-118 the zero-shot CNN recalled
  0.39 of the attacked buses on that frame and 0.00 on the rest). The tiny test timeline moves to
  seed 23, the first seed that puts every family in both its train and its test split, and the
  frozen references are rewritten from it. Takes effect with the next data release.
- The Jacobian features never read the true state. `JacobianFeatures` took each frame's change
  against the previous pool timestep's clean state, truth an operator does not hold; it now takes it
  against the previous frame's estimate, dz = z_t - h(x_hat_{t-1}), the fitted estimator's plain
  solve of the frame emitted just before (the slack angle stays the shared angle reference). A
  timeline export offers that frame's readings on request as `prev_node_x`, `prev_edge_x` and
  `prev_timestep` (`models.PreviousFrameFields`), read from file row - 1 whatever split or family it
  belongs to. `BusCNN` / `BusMLP` / the federated localizers with a Jacobian feature set and
  `JacobianWeighting` follow; a record shard is refused (its rows are not consecutive frames). On
  the tiny timeline `JacobianWeighting` moves in the fourth significant digit; the stealthy re-solve
  now raises the explained energy at an episode's first frame, as the temporal features do. A test
  rewrites the clean layer and checks the features do not change. The guides' Jacobian rows are
  rerun on it: the zero-shot CNN with the Jacobian block reads 0.820, 0.809 and 0.812 on IEEE-14, 118
  and 300 (0.873, 0.896 and 0.840 with the truth reference), its gain now on the unseen `As` and `Ar`
  rather than on `Aq`; the federated guide's tables are marked pending a rerun.
- Docs: the SE, localization and trust guides rerun on the v0.8.1 timelines, every table and quoted
  number regenerated from the new JSON. Directions that changed: the CNN gate now lowers the
  estimator's error on IEEE-14 (0.049 to 0.044) and 300, residual removal beats WLS on 300, and the
  zero-shot CNN with the Jacobian block reads 0.873, 0.896 and 0.840.
- Docs: `docs/federated/`, the federated localization papers' protocol on the v0.8.1 timelines.
  `FedBusCNN` and `FedBusMLP` with `full14+jac`, zero-shot, K = 1 to 3, seeds 123 to 125, in the
  paper's Table IV layout with per-family F1 and FR over every record and over benign records. The
  run script saves each run and skips saved ones; `make_report.py` builds the tables, figures and
  CSV sidecars from the JSON.
- `FedBusCNN` and `FedBusMLP` accept the Jacobian feature sets (`features="full14+jac"`, `"jac"`):
  the 14 channels stay per client (local power balance by default) and the 8-channel Jacobian block
  is the whole system's estimator applied to every meter's change, computed once centrally per pass
  and shared with every client (appended after the 14 channels for `full14+jac`, the whole input for
  `jac`); that block is the one feature a client does not build from its own meters. One client equals the centralized `BusCNN(features="full14+jac")`.
- Localization, the federated paper's tables: `LocalizerBase.score_perbus(ds, buses=, fr_over=)`
  returns per-bus F1, detection rate, false-alarm rate and AUPRC (`models.scores.PerBusScores`,
  `PerBusMetrics`) over every record and per family (that family plus benign), on the active buses
  or, for a learned localizer, the training-attackable ones; `fr_over="all"` is the paper's Table IV
  false-alarm rate, `"benign"` equals `score()["all"].macro_fr`. `LearnedLocalizer.tune_grid_threshold`
  and `score_grid` give record-level detection at a validation-tuned grid threshold
  (`models.scores.GridScores`). New formulas `perbus_rates` and `average_precision` [DG06] (equal
  to scikit-learn's). Existing outputs are unchanged.
- `fdia_graph.federated.RegionalPrior`: the proposed estimator's subspace prior fitted per client
  of a `Partition` (each client's basis from its own non-slack angles and voltages, placed on a block
  diagonal by `formulas.federated.block_diagonal_basis`), so no client's states leave it; one client
  is `SubspacePrior` exactly. With `GatedPrior(gate=FedBusCNN(...))` the federated localizer gates the
  estimator. The meter sigmas and the benign mean were already local; the solve stays central.
- `fdia_graph.federated`: the federated localization paper's localizers trained by FedAvg [MCM17].
  `FedBusMLP` and `FedBusCNN` are `LocalizerBase` methods (`fit(train, val)`, `localize`, `scores`,
  `score` as for `BusMLP`/`BusCNN`) over a K-way partition: each client reads only its own buses,
  builds its power-balance channel from its own flow meters by default (`kcl="local"`; `"global"`
  is the paper's central feature, which reads a neighbour's tie-line meters), trains on its own
  buses (optionally with a read-only halo) and scores them; only pooled moments, model weights and
  per-bus confusion counts cross a client boundary. `history` logs every round's loss and bytes.
  With K = 1 and no clip the fit equals the centralized one weight for weight (tested for both
  encoders). The Jacobian feature sets are supported with one central block (the entry above).
- `fdia_graph.federated`, first part: the split of a system's buses into K clients.
  `spectral_partition(edge_index, N, K)` is the federated localization paper's partition (spectral
  clustering of the bus adjacency, random_state 42 [VLX07]) and matched its cached partitions
  exactly on IEEE 14, 118 and 300 at K = 2 and 3 (the tests pin IEEE 14); `attackable=` biases the cut away from attackable buses. It
  returns a `models.federated.Partition` (client of every bus, interior and boundary buses, cut
  edges); `compute_nodes` gives a client's buses plus an optional halo of other clients' buses.
  `formulas.federated.fedavg` [MCM17], `interior_boundary`, `cut_edge_count`,
  `attackable_affinity` and `halo_nodes` carry the arithmetic. A `federated` extra (torch,
  scikit-learn) and CI installs it.
- Learned localizers, no number changes (the first step of `fdia_graph.federated`): the training
  loop is `localization.learned.LocalTrainer` (optimizer and batch order persist across `run`
  calls, optional owned-bus loss and gradient clipping for a federated client), prediction is
  `predict`, standardization pools channel moments (`formulas.federated.channel_moments`,
  `pool_moments` [CGL79]) and the validation threshold picks from per-bus confusion counts
  (`formulas.metrics.perbus_counts`, `tau_from_counts` [KEC25]), so a federated fit can sum them
  across clients. `models.training.OptimConfig` carries the training knobs. BusMLP and BusCNN
  weights, scores, thresholds and standardization are bit-identical to 0.19.0.
## 0.19.0

- Data release v0.8.1 is the default (`fg.load("ieee118")`; v0.8.0 stays readable with
  `release="v0.8.0"`): the eight timelines regenerated with this release's generator, the same pools
  (byte-identical), seed and knobs, built by `examples/_build_timelines_v081.py`. The stealthy
  families change on every system but IEEE-200, whose file is byte-identical to v0.8.0 (no bus there
  holds both a load and a generator, and its static generators produce nothing). Every file keeps
  72,000 frames, exactly half attacked, no frame falling back to benign, every family present, and
  no stealthy frame labelling a generator bus.
- Generator, the stealthy families change. Aq, At, Al and Am no longer target a load on a generator
  bus (zero-MW condensers included), the local attacker's rule in [BOY22]: a generator bus is too
  risky to falsify. `FdiaGenerator.stealthy_pos` is their target set; Ad, As and Ar keep
  `attackable_pos`. Attackable loads left to the stealthy families: IEEE-14 8 of 11, 30 18 of 20,
  57 35 of 41, 89 29 of 29, 118 54 of 99, 145 11 of 36, 200 108 of 108, 300 156 of 191. The frozen
  timeline is re-frozen. The data dictionary states that `attack/mag` is unsigned (a load rise and
  a drop of the same size record the same value).

- Loading and fitting, no number changes. `fg.load("IEEE118")` works like `"ieee118"` (a locally
  registered name stays case-sensitive). `latest_release` skips a tag it cannot parse instead of
  falling back to the pinned default, and its docstring no longer claims `load(release=None)` calls
  it. Two jobs downloading the same asset at once write separate temp files; the second keeps the
  first's installed copy instead of corrupting it. `fit` refuses a dataset whose slack disagrees
  with the case's. The solve's reduced normal-matrix inverse is built once in `fit` rather than on
  every chunk. The recent-change scale's docstring says "plus 1e-3", what the code does.
- Tests, no package change. `tests/test_public_coverage.py` tests directly what the suite reached only
  through whole pipelines: the replay policy and its buffer, `batched_normal_matrices`,
  `local_ac_solve` on a two-bus grid, `sparse_basis`, `per_bus_sequences`, `read_episodes`,
  `latest_release` online and offline, the `TrustSelector` base, the removal guard, and
  `tune_threshold`.
- Generator, the stealthy families change. The engine recovered a scan's load at a bus holding
  both a load and a generator as the stored injection plus the BASE generation, but the pools scale
  a bus's load and generation by one factor, so the load came out as s L + (1 - s) G instead of
  s L: at s = 0.75 the median load was 1.6x the true one on IEEE-118 (10 of its 99 attackable
  loads) and 2.0x on IEEE-300 (23 of 191), at times negative, so an Aq step there moved the wrong
  amount and a "raise" could lower the load. Every load-recovery site now uses
  `formulas.attacks.bus_load` (the `generator_output` construction the limit check already used),
  through `FdiaGenerator.true_load` and `scan_generation`. Static generators that produce anything
  count as injection buses (IEEE-89 buses 1, 65, 69, 79 and eight on IEEE-300 were treated as
  zero-injection junctions and pulled into attack regions). The attack-redistribution sign: the
  engine's redistribution LOWERS the target line's |flow| in the false state (checked on 77 of 77
  draws), so `am_direction="mask"` now keeps its sign and `"induce"` flips it (they were swapped);
  `"both"`, the default, maps a given draw to the same sign as before. The frozen
  timeline is re-frozen (341 of its 1000 frames change: the corrected stealthy draws spend the
  random stream differently, so later noise shifts too). The released v0.8.0 timelines were
  generated before this fix; regenerating them is a data release.
- CI, no package change. `publish.yml` runs the test suite on the tagged commit and publishes only
  when it passes. The suite also runs on Python 3.9 (the lowest supported) and on Windows, both
  required by `tools/pr.py merge`; the Ubuntu job installs the `[se]` extra by name and reports
  coverage. `ruff format --check` covers `tests` and `tools`. Every action is pinned to a commit
  SHA, kept current by Dependabot; `slow.yml` runs the `FDIA_SLOW` IEEE-118 check weekly and on
  demand.
- Fixes the new CI jobs found. The estimators failed on Python 3.9: its SciPy has no
  `lapack.dtrcon`, which `guarded_inverse` called for its condition estimate; it now falls back to
  Hager's 1-norm estimate (the same value LAPACK returns, a few O(k²) solves) when SciPy lacks it (`[se]` still asks only scipy>=1.8). `tools/readability.py`
  no longer fails on a file on another drive than the repository (Windows CI's temp dir).
- Repository, no package change. `_publish_pypi.sh` is gone: publishing goes through a version tag and
  `publish.yml` only (no manual upload path). Four figures in `docs/` that nothing referenced
  (the README uses the copies in `docs/figures/`) are gone. `.gitignore` loses a duplicated block,
  ignores the pytest and ruff caches, and names `examples/_build_timelines_v080.py`, the v0.8.0
  build script, as tracked on purpose.
- Estimators, numbers change. `ResidualRemoval` is classical largest-normalized-residual removal:
  one meter per record per pass (the largest above the threshold), re-solved, until none exceeds it;
  a removal the observability guard refuses keeps that meter. It used to remove every meter above
  the threshold at once, taking out the honest neighbours a single gross error smears onto. Meter
  sigma (and the trust selectors' alarm level) is calibrated on benign records spread evenly over
  the train split instead of the first ones, which on a timeline are one early load regime. The
  guarded weighted solve also considers its final Newton step. Measured on the IEEE-14 v0.8.0
  timeline (geometric-mean angle error, test split): removal 0.0665 to 0.0552 degrees (replay
  0.0645 to 0.0224), WLS 0.0966 to 0.0961, Huber 0.0613 to 0.0605, prior + Huber 0.0499 to 0.0503,
  Jacobian weighting 0.0826 to 0.0800, gated arms within 0.2 percent. The frozen SE reference moved
  by at most 1.6e-7 relative (its train split has fewer than 600 benign frames).
- Estimators, fixes. Every public path that solves a dataset (`estimate`, `score`,
  `ResidualLocalizer`, `TrustSelector.score`, `JacobianFeatures.fit`, `JacobianWeighting`) refuses a
  `units="pu"` view with a ValueError; `fit` did before, but estimating or scoring a per-unit view
  converted twice and returned errors about ten times too large. `GatedPrior` and
  `JacobianWeighting` supply their per-record weights through one hook (`_record_weights`) instead
  of overriding `estimate`, so `ResidualLocalizer(estimator=GatedPrior(...))` scores the gated
  estimator rather than its ungated parent; the Huber passes live once in `SEBase` and
  `JacobianWeighting` gains `tol` (it ran all 40 passes) and builds its Jacobian features once in
  `fit`. `ResidualLocalizer` no longer refits an estimator that is already fitted
  (`SEBase.is_fitted`). The localizer hook is `_score(d, ds)`.
- `ds.windows(..., copy=False)` returns the windows as a read-only strided view of the frames, no
  memory per window (a 72k-frame IEEE-118 timeline at W=60 was about 8 GB as a copy); the default
  still returns a writable copy, now built from that view. `TrustedMetersDQN` evaluates the attack
  cost once per step instead of twice, keeps its replay buffer in a ring, and defaults to `seed=123`
  like the localizers (pass `seed=0` for the previous default; the trust guide pins it). Windows and
  the seed-0 DQN selection are bit-identical to 0.18.0.
- The trust guide's secured-copy tables for IEEE-118 (`results/secured_ieee118.json`): the DQN set
  opens the residual test on the stealthy families and lifts the learned localizer four points, the
  estimator gains within a percent at 20 meters. No package change.
- The docs site's staging step drops nav entries for pages a checkout lacks, so a backfilled older
  version on the versioned site has no link into a 404. No package change.
- Versioned docs: `.github/workflows/docs.yml` deploys the MkDocs site per package version to GitHub
  Pages with mike (tags as `latest`, main as `dev`) with a version picker on every page; the nav lists
  every page (the trust guide, the class map, the formulas, references, benchmarks and plans,
  contributing and the changelog), and `scripts/site_index.py` stages the three root pages. No
  package change.
- Trusted meters reach the estimators and the localizers: `TrustSelector.secured_copy(ds, out, name)`
  (`trust.secured_copy(selector, ds, out, name)`) writes a copy of a timeline in which the selected meters read their benign
  value on every frame, the attacker locked out of them, with the tamper masks and the stored
  temporal features following, so every estimator and localizer can be scored with a trusted set;
  `GatedPrior(secured=...)` never down-weights a secured meter (on a secured copy every gate was
  worse than no gate until this exemption, and with it the DQN's 20 meters take the proposed
  estimator from 0.050 to 0.020 degrees on IEEE-14); `formulas.projection.meter_positions` maps
  masked measurement indices to the file's layers; `docs/trust/run_secured.py` and the trust guide
  carry the tables. `formulas.temporal.SWING_WINDOW` is the writer's window, imported by
  `generation`.
- The narrative sections of the SE and localization guides (the Jacobian-informed weighting, the
  localization-gated estimation, the digest's ablation, the three readings) are re-derived on the
  v0.8.0 timelines from the results JSONs; their v0.7.2 markers are gone. On the timelines the
  Jacobian block is what makes the localizer's vector work (zero-shot 0.55 to 0.86 on IEEE-118)
  and a gate alone no longer lowers the estimator's error on any system. No package change.
- `docs/reference/CLASS_MAP.md`: one module diagram and six class diagrams of the package, drawn
  from the source by `tools/class_diagrams.py` (inheritance, the classes each class uses, the imports
  between modules) and checked in CI against the code. No package change.
- Every diagram in the docs is a rendered image (`docs/figures/diagrams/<name>.png`, source `<name>.mmd`
  beside it, `tools/render_mermaid.py`), since the GitHub mobile app and PyPI show a Mermaid block
  as code; no Mermaid block remains. No package change.
- `docs/se` and `docs/localization` IEEE-300 columns re-run on the v0.8.0 timeline (tables, figures
  and CSV sidecars); the live result tables of both guides now read the v0.8.0 release, and the
  paper-comparison table stays labeled v0.4.1. No package change.
- The README's pipeline diagram is a pre-rendered image (`docs/figures/diagrams/pipeline.png`, source
  `pipeline.mmd` beside it, `tools/render_mermaid.py`), since PyPI and the GitHub mobile app show a
  Mermaid block as code; README images use absolute URLs so the PyPI page shows them. No package change.
- Docs: the six diagrams that GitHub laid out badly are redrawn (the module map as layers, the
  generate and load pipelines as two rows, the estimator overview as columns, the trust guide with
  fit feeding score, the concepts page on the attack-vector construction with Am); no package change.

## 0.18.0

- Reviews: `tools/pr.py` waits for and requires every installed review bot (Copilot always;
  CodeRabbit and Gemini Code Assist once they have reviewed a pull request) on the head before a
  merge; `.coderabbit.yaml` carries the repository's review instructions (the physics conventions,
  the readability limits, the prose rules); CONTRIBUTING lists the three reviewers and their order,
  with the Claude Code `/code-review ultra` pass first. No package change.
- Names, no behavior: the generator's attributes are words (`_ppc_row`, `_n_ppc_buses`,
  `_from_bus_ppc`, `_base_mva`, `_target_lines`, `_primary_target_line`, `_line_flow_sign`,
  `_ptdf_load_buses`, `_solve_net`, `_injection_buses`, `_bias_sd`, `_draw_noise` and the public
  `n_lines`, whose old name `nl` still answers with a `DeprecationWarning` until 0.19),
  the dataset package's docstring maps each mixin to the questions it answers, and the data
  dictionary and the schema name the static `graph/edge_x` (the series reactance) apart from the
  flow layer of the same name.
- `fdia_graph.schema` is the file protocol: every group, dataset and attribute name of an HDF5
  file, the record-field to path map, the family table and the split codes, defined once; the
  writer, the readers and the tools spell paths through it, and `tools/readability.py` (the CI
  gate) refuses a path-shaped literal ("data/...", "graph/...") in any other module.
  `dataset.base.FAMILIES`, `STEALTHY_FAMILIES` and `timeline.KIND` stay importable. No file or
  number changes.
- One export of a split: `ds.export(fields, format="numpy" | "torch" | "tf" | "pandas", device,
  flatten_features)` replaces `to_numpy`, `to_torch`, `to_tf` and `to_pandas`, which still answer
  with a deprecation notice until 0.19. `ds.windows(..., per_bus=True)` returns one sequence per
  bus (`[n*N, W, 4]`), what `torch_windows` built, and `fg.load(name, split, order="time",
  format="pyg")` gives the graphs `pyg_stream` built with the file's chronological split, so both
  torch helpers are deprecated (retire in 0.19). Eight ways to read a view become three: `export`,
  `ds[i]` and `ds.windows`. No number changes.
- Four findings of a full-branch review: a load on the slack bus is never a target (the slack
  never enters a region, so IEEE-57 frames that scaled its 55 MW load were labelled attacked with
  no attack in the state); the tamper masks are the meters whose stored float32 reading changed
  (the attack vector's 1e-7 tolerance marked 1.8% of entries that the reading could not show); a
  branch out of service is not a hop of the attacker's region on N-1 generators; and the local
  solve builds its interior Jacobian block directly instead of slicing the n x n derivatives,
  about half the cost of a stealthy frame on IEEE-300, with identical states.
- The local power flow of a stealthy frame is a damped Newton: each step is halved until the
  mismatch drops, the full step first, so every frame the plain method solved is bit-identical. A
  step the region still cannot absorb is halved (an Aq step at most three times and never under
  the noise floor, a ramp or Am frame at most six times) so the frame stays attacked at the
  largest step with a solution, an Al frame halves its redistribution and then redraws its line (up to forty),
  and an Aq, At or Am episode tests its design (the full step, the ramp's peak, the held
  redistribution's peak) on every frame it will occupy, each with the step that frame will carry
  (an Aq episode's whole span, every frame of the ramp and of the Am rise, plateau and return),
  and redraws it, up to forty times, when no stealthy state exists on any of them, so an accepted design cannot fall back, and a span with no admissible design stays
  benign and is counted; an Am redistribution is also tried at halved sizes above the floor
  before it is redrawn. A ramp or Am frame never carries a zero step (the profile's first frame
  and return leg floor at one rate), so every frame labelled attacked carries an attack. `fallback_benign` now counts every frame the draw
  meant to attack that is benign, whether or not an episode record exists for it (an Am episode
  with no admissible redistribution left no record and was not counted). An Aq episode now draws its direction like the ramp, a load rise or a load
  drop with the same 5% to 20% scale: on a case that runs below its voltage limits (IEEE-57) a
  rise near the low buses has no admissible state and a drop is the attack that fits. Loads above `max_load_mw` (new knob, default 2000 MW) are never targets: the
  IEEE-145 case lumps whole areas into loads of 4 to 58 GW, and a 5% step on one has no local
  solution, which left 30% of that system's attack frames benign; no other ladder system has a
  load above 1.1 GW, so their files do not change. `fallback_benign` is zero on every released file. Every false state also satisfies the security and operational constraints of Wu et al.
  2026 (`OperatingLimits`, their equations 21 to 23): each bus voltage within the case's own
  limits (a bus the true state already holds outside a limit may not be made worse, IEEE-57 runs
  below its own minimum) and, for the generators of the attacked subnetwork as in the paper (a
  boundary bus is outside it, its voltage held true and its injection whatever balances the
  region), the implied output within its P and Q limits widened per bus to the range the benign
  pool ran it over (the pools scale generation with load and never enforced nameplate), the true
  output recovered exactly from the pool's common load and generation scale, and the pretended
  load change at a target bus not counted against its generator; a false state outside them is rejected and its step halved
  like an unsolvable one. The file records the widest bus limits as `v_lo` and `v_hi`.
- A stealthy frame (Aq, At, Al, Am) is the true scan plus the attack vector a = h(x_false) - h(x_true)
  of its local false state: every meter keeps its own noise draw and the tampered meters are
  shifted by exactly what the false state moves them, so `observed - benign` is the attack for
  every family and a WLS residual test flags the stealthy families at the benign rate. Until now
  the tampered meters were re-emitted from the false state, which drew their noise from the false
  reading; where a boundary bus has a structurally zero injection (the IEEE-14 synchronous
  condenser's P, a zero-injection bus) that noise is several times what the estimator calibrates
  for the meter, and the residual test flagged about a third of the Aq, Al and Am frames on the
  full IEEE-14 timeline. An Al frame whose drawn target line has no feasible redistribution in
  its region redraws the line (up to ten) instead of falling back to a benign frame. Both change
  every generated file; the frozen references are re-frozen.
- `tools/pr.py merge` requires every smoke job green by name on the head (a job that has not
  registered yet is not green), every other check finished without failure, and Copilot's review on
  that head; `wait` waits for the same set. A merge can no longer slip in while CI is still starting.
- `fdia_graph.trust`: which meters to secure so that stealthy attacks stop being stealthy, after
  the trusted-PMU defence of Wu et al. 2026. `TrustedMeters(k)` is the greedy row-reduction
  selection on the WLS Jacobian (secure, on the cheapest open attack, the meter whose protection
  raises the attack cost most), `TrustedMetersDQN(k, episodes)` the same selection learned as a
  Markov decision process with a deep Q-network (state the secured set, reward the attack-cost
  rise; needs torch). Both expose the order and the attack cost after each meter, and `score(test)`
  on a time-ordered timeline pins the secured meters to their benign reading and reports the WLS
  residual detection per family before and after (`TrustScores`). The kernel is
  `formulas.trust` (`attack_subspace`, `sparse_basis`, `attack_cost`, `greedy_trusted_meters`);
  `docs/trust/` has the guide and `run_trust.py`.
- The timeline writer draws its attack episodes so their frames sum to exactly
  `round(attacked_frac * T)`, places every one of them (longest first, each at an onset drawn
  uniformly among the onsets where it fits), and emits benign frames everywhere else. The
  benign-gap band and the rule that started the next episode with no gap are gone, so whether two
  episodes touch or a quiet stretch separates them is a property of the draw; an Am episode redraws
  its redistribution at onset rather than leaving frames benign, and only a non-converging power
  flow falls back to a benign frame (counted in `fallback_benign`). The frozen references are
  re-frozen on the tiny timeline (seed 4).
- Data releases have their own tag namespace: the assets of `v0.8.0` and later live under the
  GitHub tag `data-v0.8.0` (package versions own the bare `v0.x.y` tags, PyPI 0.8.0 is `v0.8.0`),
  while `v0.7.1` and `v0.7.2` keep their bare tags; `registry.release_tag` maps the short name
  users write (`fg.load(name, release="v0.8.0")`, `FDIA_GRAPH_RELEASE`) to the tag, the newest
  data release is picked from the tag list, and `tools/upload_assets.py` refuses a package tag.
- The localization scores in `docs/localization` are re-run on the v0.8.0 timelines and are not
  comparable with the v0.7.2 shard tables: the shard's temporal features compared an attacked
  snapshot with the benign scan before it, so every attacked record spiked, while the timeline's
  compare each frame with the frame emitted one minute earlier, so a sustained episode spikes at
  its onset and the one-frame families also spike on the benign frame after them. The per-record
  numbers are lower for the same detectors and the same data, and the tables say which reference
  they use.
- `fg.load(name)` reads the v0.8.0 timelines by default (`_RELEASE` is `v0.8.0`, the checksums of
  the final files pinned). The test suite's tiny fixture now walks the v0.8.0 pool (72,000 one-minute
  states where the v0.7.2 pool held 36,000), so the frozen references are re-frozen on it.
- The data release `v0.8.0`, step 4 of `docs/plans/ONE_DATASET_PLAN.md`: one timeline file per
  system of the ladder (`timeline_ieee{N}.h5`, 72,000 frames, the seven families, seed 123,
  `attacked_frac` 0.5) and the operating-point pools as HDF5 (`pool_ieee{N}.h5`), built by
  `examples/_build_timelines_v080.py` from the v0.7.1 NYISO pools. `fg.load(name)` reads them by
  default; `fg.load(name, release="v0.7.2")` (or `FDIA_GRAPH_RELEASE=v0.7.2`) still reads the
  record shards, since the registry now maps a release tag to the file layout it shipped
  (`registry.dataset_file`, `pool_spec`, `is_timeline_release`) and knows the sha256 of each pinned
  release. `generate(states=None)` fetches the pool through the same registry. `load_stream`
  (deprecated) returns the timeline through the loader at a timeline release and the v0.7.2
  stream file before it. `tools/upload_assets.py` publishes release assets idempotently.
- One generator path, step 3 of `docs/plans/ONE_DATASET_PLAN.md`: the record-shard writer
  (`generation.generate_shard` and its draw loop), the stream walker and its `.npz` output, the
  magnitude sidecar (`<out>.mag.npz`) and the graph sidecar read of `load_stream` are deleted, with
  the `ShardArrays` and `Record` models and `SINGLE_SHOT_ORDER`. `generate_stream` is now the
  timeline writer followed by a read of the file it wrote (`streams.stream_of` turns a time-ordered
  dataset into the stream dict), so its output is the timeline's and its `out` is the HDF5 path;
  `load_stream` still reads the v0.7.2 stream files (which embed their graph) until the v0.8.0 data
  release. The loader keeps reading the v0.7.2 record shards; `tests/data/tiny_shard_v072.h5` is a
  checked-in file in that layout that the suite loads.
- The frozen references are re-frozen on the tiny timeline the suite builds (`frozen_spec.TIMELINE_KW`,
  1000 IEEE-14 frames, every family): the file's arrays and attributes, and the estimator and
  localizer scores on its test split. The shard and stream references are gone with their writers.
- One loader for both kinds of file, step 2 of `docs/plans/ONE_DATASET_PLAN.md`. `fg.load(name)`
  reads a timeline file (`kind="timeline"`) as well as a v0.7.2 record shard. New arguments:
  `order="time"` (default) keeps the file order, chronological on a timeline; `order="random"` is
  the same records in a permutation fixed by `seed` (default 0), a view index, nothing copied, so
  two people asking for the same seed get the same record table. On a timeline every record,
  batch and export carries `benign` and `edge_benign` (the attack removed, the noise kept, in the
  requested units), `ds.windows(W, stride, label, layer)` slides windows over a time-ordered
  contiguous view (a split or the whole file; it refuses a random order, a family subset and a
  shard), `ds.episodes` is the episode table (`EpisodeTable`: onset, length, family, buses) of the
  view, and `torch_windows(dataset=ds)` / `pyg_stream(dataset=ds)` take the loaded timeline.
  `ds.is_timeline` and `ds.has_benign` say which kind a file is.
- `fg.generate(system, name, frames=None, **knobs)` now writes a timeline (the knobs of
  `timeline.generate_timeline`, plus `frames` to cap the pool timesteps walked) and registers it, so
  `fg.load(name)` and `fg.load(name, order="random")` read it back. The record-shard writer is
  gone (the entry above): the loader still reads the v0.7.2 shards, nothing writes them.
- `generate_stream`, `load_stream` and `windows(stream, ...)` warn with `DeprecationWarning` and
  retire in 0.19: the timeline file replaces the stream, `fg.load(name, order="time")` reads it, and
  `ds.windows` replaces `windows`. They keep working unchanged until then (`load_stream` reads the
  v0.7.2 stream files).
- The timeline writer, step 1 of `docs/plans/ONE_DATASET_PLAN.md`: `fdia_graph.timeline.generate_timeline`
  walks one attacked timeline over a system's operating-point pool and writes one HDF5 file
  (`kind="timeline"`) carrying every layer the streams and shards had between them: the observed,
  benign and clean scans per frame, the full static graph, per-frame `stealthy`, `seq_id` (the
  episode index) and a chronological `split` that never cuts an episode, an `episodes/` table, and
  an `attack/` group with the designed magnitude per attacked bus and the tamper masks (the meters
  the attacker wrote). Families are scheduled by inverse expected episode length so each gets about
  the same share of attacked frames; `corrupt_len=1` makes every Ad/As/Ar frame an independent draw.
  The loader for these files is step 2; `generate` and `load` are unchanged in this release.
- A seventh family, `Am` (code 7, stealthy), the multi-snapshot attack of Wu et al. 2026: the Al
  load redistribution drawn once at episode onset and applied along a ramp whose per-bus per-frame
  step stays under `am_rate` of the noise floor. `fg.FAMILIES[7] == "Am"`, `fg.STEALTHY_FAMILIES`
  now `{1, 5, 6, 7}`; the published shards and streams carry no `Am` frames.
- Every stealthy family (Aq, At, Al, Am) is now a local false state, the attacker model of Wu et al.
  2026: the attacker solves the power flow of the subnetwork within `hops` branches of the attacked
  loads (or the target line) with every other bus voltage held true (`FdiaGenerator.solve_local`,
  `formulas.network.local_ac_solve`), never the whole grid and never the slack, and writes only the
  meters that false state moves (the tamper masks). The measurement vector is exactly consistent
  with an AC state, so a WLS residual test flags these frames at the benign rate; the global
  re-solve with pinned generation is gone from the writer. `hops` (default 2) replaces `am_sigma`.
- `generate_stream` builds its frames through the timeline module's episode primitives; its
  scheduler and RNG order are unchanged and the streams are bit-identical. `generate(states=...)`
  also accepts a pool stored as HDF5 (dataset `X`).
- One definition of the measurement column orders: `fdia_graph.models.NodeColumns` (`v`, `p_inj`,
  `q_inj`, `theta`), `EdgeColumns` (`p_from`, `q_from`) and `BranchColumns` (the eight `edge_attr`
  columns), with `.of(array)` giving named views of any such array and `NODE`, `EDGE`, `BRANCH`
  the indices. Every record, batch, split and stream has `.node()` and `.edge()`; the generator,
  loader and estimator index columns through the definition instead of literals. No behaviour change.
- The shard-shaped bundles (`RecordBundle`, `BatchBundle`, `ArraysBundle`, `Stream`)
  are built from field groups (`fdia_graph.models.fields`: scan, labels, record ids, temporal,
  clean, graph, stream layers), so each shared field is declared once; `Bundle` gains `_order`
  (the dict-key order, unchanged from before) and `_required` (a missing required field raises
  `TypeError` at construction, as a missing argument did). These four bundles are keyword-only
  (a positional call raises `TypeError` instead of binding to the inherited field order); they
  are return types, and no caller in the package or its examples built them positionally.
- `fg.load_stream` now fills `system` and `attacked_frac` (they were None: the stream files carry
  the arrays only and the generator attached the two values after writing). Both are derived from
  the arrays on load, the same way `generate_stream` computes them; `fdia_graph.streams.stream_summary`
  is the shared helper.
- `ruff check` configured in `pyproject.toml` with pyflakes, import order and modern-syntax rules,
  and the package clean under them (`list[int]`-style generics and `collections.abc` imports;
  `Optional`/`Union` stay, since `typing.get_type_hints` on 3.9 cannot evaluate `X | None`). The
  CI `format` job runs it on every pull request. No behaviour change.
- `tools/bench.py` and `docs/reference/BENCHMARKS.md`: per-record timings of shard generation and
  the WLS, Huber and prior+Huber estimators on the tiny shard, appended per run with the machine
  and torch state; `--check` fails when a timing is more than 3x slower than the last row.

- `CONTRIBUTING.md` (the rules, the pull-request flow, releasing), `tools/pr.py` (create, wait,
  comments, reply, merge, with the merge rule enforced) and `tools/release.py` (tag, GitHub
  release, PyPI wait) so a second maintainer can ship; the plan documents move to `docs/plans/`.
- Clear errors at the public edges instead of silent or opaque failures: an unknown family name
  or code in `load(families=...)`, an unknown `split`, an unknown field in `to_numpy` and its
  siblings, and an invalid `label`, window length or stride in `windows` raise `ValueError`
  naming what is allowed (an unknown family used to select nothing; an invalid window label used
  to behave as "any"). `tests/test_edges.py` covers these and the estimators' and localizers'
  argument checks; `FDIA_SLOW=1` additionally runs the IEEE-118 estimator sanity test.

## 0.17.0

State estimation without torch, the formula kernel complete, the loader as concerns, and the
scheduled removals (PRs #74 to #78). What a user sees: `pip install "fdia-graph[se]"` no longer
pulls torch (add `[torch]` for faster per-record inverses); the pre-0.16 generator attribute
names are gone; every helper on the package namespace is the real function. Shards, streams and
the localizer scores are bit-identical to 0.16.0; estimator scores moved by at most 2.7e-6
relative (the closed-form Jacobian, see the entry below). No import path changes.

- State estimation no longer needs torch: `formulas.network.ac_measurement` is the estimator's
  h(x) and `ac_jacobian` its closed-form Jacobian (equal to the automatic-differentiation Jacobian
  to 1e-14), so `fit` takes no gradient; torch, when installed, only speeds up the per-record
  inverses, and the `[se]` extra no longer installs it. The frozen estimator scores moved by at most 2.7e-6 relative (six of 56 cells above
  1e-9, all in the Huber arm, whose passes amplify last-bit differences in h); shards, streams
  and every other score are bit-identical. The torch twin `SEBase._h_t` stays for callers that
  differentiate through it.
- `formulas.estimation` (the WLS step, gain matrix, residual covariance, normalized residual,
  Huber weight, the operating-point prior basis, the localization gate), `formulas.linalg` (the
  guarded inverse, condition number, per-record normal matrices) and `formulas.projection` (the
  weighted pseudo-inverse, explained/unexplained split, leverage, weak directions, meter-to-bus
  aggregation): the estimators and the Jacobian features now call these named functions, each
  with its equation and source key. Identical numbers (bit-identical frozen scores); the
  estimator's private helpers keep their names and delegate.
- Removed, as scheduled one minor version after 0.16: the generator's pre-0.16 attribute names
  (`edge_r` ... `edge_status`, `M`, `flow_meter`, `bias_pi` ... `bias_qf`, `outage`,
  `outage_pos` ... `outage_base_flow_mw`; read `g.branch`, `g.meters`, `g.bias`, `g.contingency`)
  and `fdia_graph.streams._swing_scale` (use `fdia_graph.generation._swing_scale`).
- `fdia_graph.generate`, `generate_stream`, `load_stream`, `windows`, `pyg_stream`, `torch_windows`,
  `load_profile`, `fetch_profile`, `generate_states` and `line_outage_candidates` are the real
  functions, resolved on first use (PEP 562) instead of wrappers that restated their docstrings.
  Same names, same calls, same lazy imports; `help(fg.generate)` now shows the one docstring.
- The loader is the `fdia_graph.dataset` package: `FdiaGraph` is assembled from `graph`,
  `physics`, `records` and `export` mixins over a `base` that declares the shared state, the same
  pattern as the generator. Same import path, same names, same behaviour (docs/plans/READABILITY_PLAN.md,
  step 9).

## 0.16.0

The readability and data-models series (PRs #64 to #72). What a user sees: every dict the package
returns is now a typed bundle with attributes and a docstring, still the same dict with the same
keys; every model lives in `fdia_graph.models`; the generator's old attribute names warn. Shards,
streams and every score are bit-identical to 0.15.0 (the strict frozen tests are the proof), no
data release moves, and no import path changes. The entries below are in the order the steps
merged, newest first within each series.

Data models, steps 1 and 2 (docs/plans/DATA_MODELS_PLAN.md). No user-visible change; one deprecation.

- `fdia_graph.models.Bundle`: the base of typed records, a frozen dataclass that is also a read-only
  mapping, so a bundle works wherever a dict did.
- Internal tuples and dicts are models: `Scan`, `TrueState`, `Redistribution`, `ResolvedPool`,
  `Admittances` (also returned by `formulas.network.branch_admittances`), `AssetSpec` (returned by
  `registry.resolve`, indexable as before), `DownloadTarget`, `ShardArrays`.
- `FdiaGenerator` holds `branch`, `meters`, `bias` and `contingency` models. Deprecated: the previous
  attribute names (`edge_r` ... `edge_status`, `M`, `flow_meter`, `bias_pi` ... `bias_qf`, `outage`,
  `outage_pos` ... `outage_base_flow_mw`) still work as read-only properties and warn; they are
  removed one minor version later.

Data models, step 3: the public dicts are typed bundles. No user-visible change: every one is
still the dict it was (a `dict` subclass with the same keys in the same order), so indexing,
`**`, iteration, JSON dumping of score tables and PyTorch's default collate keep working; each
also has attributes and a docstring naming its fields.

- `dataset.RecordBundle` (`FdiaGraph[i]`), `BatchBundle` (`collate`), `ArraysBundle` (`to_numpy`,
  `to_torch`, `to_tf`), `Summary` (`summary`).
- `se.base.EstimatorScores` of `ErrorPair` rows (`SEBase.score`); `localization.base.LocalizerScores`
  of `OverallMetrics`, `BenignMetrics` and `FamilyMetrics` (`LocalizerBase.score`).
- `se.jacobian.JacobianOutputs` (`JacobianFeatures.transform`; the `"global"` key is the
  `global_` attribute).
- `streams.Stream` (`generate_stream`, `load_stream`) and `engine.core.LineCandidate`
  (`line_outage_candidates`).
- `Bundle` is now a `dict` subclass rather than a `Mapping`, so `isinstance(out, dict)` and
  `json.dump(out)` hold as well; it is read-only on both sides.

Data models, step 5: every data model lives in the `fdia_graph.models` package, grouped by what
the data is (`grid`, `frames`, `data`, `scores`, `assets`), with `fdia_graph.models.PUBLIC` naming
the bundles a user receives. Every previous import path still works (each producer module
re-exports what it used to define), so no user code changes.

Data models, step 4: `docs/reference/DATA_DICTIONARY.md` gains a "Models" section listing every
public bundle and its fields, generated by `tools/models_doc.py` from the dataclasses and kept
current by a test.

Readability series, steps 6 and 7 (docs/plans/READABILITY_PLAN.md): the temporal and ramp formulas in
the kernel, and the rest of the backlog. No user-visible change (bit-identical shards and
streams, identical estimator and localizer scores).

- `formulas.temporal` (`recent_change_scale`, `temporal_delta`, `swing_zscore`) and
  `formulas.attacks.ramp_profile`, each with its equation and source key, used by both generators.
- `AttackMixin.corrupt` decides the family once and runs one short loop per family;
  `PhysicsMixin.solve` pins the generation in `_pin_generation`; `line_outage_candidates` screens a
  line with guard clauses; the loader's `__getitem__` and `collate`, the localizer's `score`, the
  learned model builders, the stream loader and the download are split the same way; the Huber
  passes of the estimators are one method.

Readability series, step 5 (docs/plans/READABILITY_PLAN.md): the generator constructor, the loader
constructor and `to_numpy` split into named steps. No user-visible change (bit-identical shards
and streams, same loader outputs).

- `FdiaGenerator.__init__` reads as a list of what it builds: the case with its contingency, the
  load tables, the meter plan, the edge index, the branch physics, the admittances, the meter
  biases; the accuracy-class split of the meter error is `formulas.noise.bias_jitter_split` and the
  series admittance comes from `formulas.network`.
- `FdiaGraph.__init__` delegates to `_read_header`, `_read_static_graph`, `_read_layers`,
  `_reference_bus`, `_record_mask` and `_preload`; `family_ids` translates family names or codes in
  one place; `to_numpy` gathers the clean layers and converts units through small helpers.

Readability series, step 4 (docs/plans/READABILITY_PLAN.md): one AC measurement function. No
user-visible change (bit-identical shards and streams; `ybus`, `yf`, `yt` and `edge_clean_full`
unchanged).

- `fdia_graph.formulas` (new, public, provisional until 1.0): `formulas.network` holds the AC
  network model as pure functions, `complex_voltages`, `series_admittance`, `branch_admittances`
  (the pi model behind `ybus`/`yf`/`yt`), `bus_injections` and `branch_flows`, each with its
  equation and source key. The generator's measurement emission and the loader's admittance
  and clean-flow computations call them; the estimator's torch measurement function is pinned to
  them by a test, so the package has one AC model.

Readability series, steps 2 and 3 (docs/plans/READABILITY_PLAN.md): one attack frame for shards and
streams, both generators split into named steps. No user-visible change: a seeded shard and a
seeded stream are bit-identical to the previous release (`tests/test_frozen.py`, strict mode).

- `fdia_graph.engine.records`: `attack_frame` builds one scan of any family from a stored
  operating point, for `generate` and `generate_stream` alike; `FrameKnobs` carries the two
  switches that differ between them; `replay_frame` and `remember_benign` name the replay policy.
- `generate` and `generate_stream` are sequences of named draws and episodes; the swing scale,
  the temporal features and the ramp profile are functions with their source keys, shared by both.
- Shard records are a named `Record` tuple and the writer reads them by field name; the writer is
  split into the attributes, graph, data and clean groups.
- No public name changed. The private stream helper `_swing_scale` now lives in `generation`;
  `streams._swing_scale` stays as a deprecated alias that warns, removed one minor version later.

Readability series, step 1 (docs/plans/READABILITY_PLAN.md): the safety net. No user-visible change.

- `tests/test_frozen.py`: a generated shard, a generated stream, and the estimator and localizer
  scores on the test shard are compared with frozen references (`tests/frozen/`): exactly in
  strict mode; otherwise arrays to 1e-7 relative and estimator and localizer scores to 1e-4
  relative, the cross-platform tolerances the test module documents. `tools/freeze_reference.py`
  rewrites the references when a change to the generator or an estimator is intended.
- `tools/readability.py`: the readability measures (complexity, nesting, closures, parameters,
  positional indexing) as a report over the package and as a gate on the functions a change
  touched; runs in CI.
- `docs/reference/FORMULAS.md` and `REFERENCES.md`: the catalogue of formulas and their sources,
  with the docstring template every formula function follows.
- `docs/plans/READABILITY_PLAN.md`, `docs/plans/RESTRUCTURE_PLAN.md`: the plans for the series.

## 0.15.0

- `edge_clean_full`: the exact clean power flow on every branch, computed on load from the clean
  state through the branch admittance matrix; in the record dict, PyG data, `collate`, `to_numpy`
  and the per-unit view.

## 0.14.1

- Per-record normal matrices are built by chunked matrix products instead of one einsum; the
  IEEE-300 estimators run in minutes instead of days. Same numbers.
