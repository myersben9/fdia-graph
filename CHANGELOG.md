# Changelog

Every release lists what a user of the package can see change. "No user-visible change" means
the public API, the generated files and the numbers are the same as the previous release.

## Unreleased

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
  `fdia_graph.generation.generate_shard` for one minor version (the frozen references are still
  built from it) and retires in 0.19.
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
- A seventh family, `Am` (code 7, stealthy), the multi-snapshot attack of Wu et al. 2026 built from
  the engine's pieces: the Al load redistribution drawn once at episode onset, applied along a ramp
  whose per-bus per-frame step stays under `am_rate` of the noise floor, every frame re-solved
  with generation pinned, and made sparse by leaving every meter whose designed change is under
  `am_sigma` accuracy-class stds at its un-attacked reading. `fg.FAMILIES[7] == "Am"`,
  `fg.STEALTHY_FAMILIES` now `{1, 5, 6, 7}`; the published shards and streams carry no `Am` frames.
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
