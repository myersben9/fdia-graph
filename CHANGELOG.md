# Changelog

Every release lists what a user of the package can see change. "No user-visible change" means
the public API, the generated files and the numbers are the same as the previous release.

## Unreleased

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
