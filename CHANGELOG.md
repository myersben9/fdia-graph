# Changelog

Every release lists what a user of the package can see change. "No user-visible change" means
the public API, the generated files and the numbers are the same as the previous release.

## Unreleased

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
  and CSV sidecars); the live result tables of both guides now read the v0.8.0 release, while the
  paper-comparison table (v0.4.1) and the narrative sections (v0.7.2) stay labeled as such. No
  package change.
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
