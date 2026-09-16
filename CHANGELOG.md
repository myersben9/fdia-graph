# Changelog

Every release lists what a user of the package can see change. "No user-visible change" means
the public API, the generated files and the numbers are the same as the previous release.

## Unreleased

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
