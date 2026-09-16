# Changelog

Every release lists what a user of the package can see change. "No user-visible change" means
the public API, the generated files and the numbers are the same as the previous release.

## Unreleased

Data models, steps 1 and 2 (docs/DATA_MODELS_PLAN.md). No user-visible change; one deprecation.

- `fdia_graph.models.Bundle`: the base of typed records, a frozen dataclass that is also a read-only
  mapping, so a bundle works wherever a dict did.
- Internal tuples and dicts are models: `Scan`, `TrueState`, `Redistribution`, `ResolvedPool`,
  `Admittances` (also returned by `formulas.network.branch_admittances`), `AssetSpec` (returned by
  `registry.resolve`, indexable as before), `DownloadTarget`, `ShardArrays`.
- `FdiaGenerator` holds `branch`, `meters`, `bias` and `contingency` models. Deprecated: the previous
  attribute names (`edge_r` ... `edge_status`, `M`, `flow_meter`, `bias_pi` ... `bias_qf`, `outage`,
  `outage_pos` ... `outage_base_flow_mw`) still work as read-only properties and warn; they are
  removed one minor version later.

Readability series, steps 6 and 7 (docs/READABILITY_PLAN.md): the temporal and ramp formulas in
the kernel, and the rest of the backlog. No user-visible change (bit-identical shards and
streams, identical estimator and localizer scores).

- `formulas.temporal` (`recent_change_scale`, `temporal_delta`, `swing_zscore`) and
  `formulas.attacks.ramp_profile`, each with its equation and source key, used by both generators.
- `AttackMixin.corrupt` decides the family once and runs one short loop per family;
  `PhysicsMixin.solve` pins the generation in `_pin_generation`; `line_outage_candidates` screens a
  line with guard clauses; the loader's `__getitem__` and `collate`, the localizer's `score`, the
  learned model builders, the stream loader and the download are split the same way; the Huber
  passes of the estimators are one method.

Readability series, step 5 (docs/READABILITY_PLAN.md): the generator constructor, the loader
constructor and `to_numpy` split into named steps. No user-visible change (bit-identical shards
and streams, same loader outputs).

- `FdiaGenerator.__init__` reads as a list of what it builds: the case with its contingency, the
  load tables, the meter plan, the edge index, the branch physics, the admittances, the meter
  biases; the accuracy-class split of the meter error is `formulas.noise.bias_jitter_split` and the
  series admittance comes from `formulas.network`.
- `FdiaGraph.__init__` delegates to `_read_header`, `_read_static_graph`, `_read_layers`,
  `_reference_bus`, `_record_mask` and `_preload`; `family_ids` translates family names or codes in
  one place; `to_numpy` gathers the clean layers and converts units through small helpers.

Readability series, step 4 (docs/READABILITY_PLAN.md): one AC measurement function. No
user-visible change (bit-identical shards and streams; `ybus`, `yf`, `yt` and `edge_clean_full`
unchanged).

- `fdia_graph.formulas` (new, public, provisional until 1.0): `formulas.network` holds the AC
  network model as pure functions, `complex_voltages`, `series_admittance`, `branch_admittances`
  (the pi model behind `ybus`/`yf`/`yt`), `bus_injections` and `branch_flows`, each with its
  equation and source key. The generator's measurement emission and the loader's admittance
  and clean-flow computations call them; the estimator's torch measurement function is pinned to
  them by a test, so the package has one AC model.

Readability series, steps 2 and 3 (docs/READABILITY_PLAN.md): one attack frame for shards and
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

Readability series, step 1 (docs/READABILITY_PLAN.md): the safety net. No user-visible change.

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
- `docs/READABILITY_PLAN.md`, `docs/RESTRUCTURE_PLAN.md`: the plans for the series.

## 0.15.0

- `edge_clean_full`: the exact clean power flow on every branch, computed on load from the clean
  state through the branch admittance matrix; in the record dict, PyG data, `collate`, `to_numpy`
  and the per-unit view.

## 0.14.1

- Per-record normal matrices are built by chunked matrix products instead of one einsum; the
  IEEE-300 estimators run in minutes instead of days. Same numbers.
