# Plan: what to take from power-grid-model and OGB

Goal: raise the SDK to the standard of the two open-source projects closest to it. **power-grid-model**
(Alliander) is the reference for a grid-calculation library with state estimation and input
validation. **OGB** (Open Graph Benchmark) is the reference for a dataset and benchmark package:
fixed splits, a standalone evaluator and reporting rules. PowerGraph ([NeurIPS 2024](https://arxiv.org/abs/2402.02827)) and the Hugging
Face dataset-card template were read for dataset documentation.

Their code and documentation were read on GitHub on 2026-09-26. Every claim about their side below
cites a file or page, and every claim about ours was checked in the code or on the released files.

## 1. Where the SDK is already ahead

| area | ours | theirs |
|---|---|---|
| input checks | declared on the field and run when the object is built; a function never sees an unchecked argument (`models/validation.py`) | power-grid-model's validator is opt-in and separate from the calls it protects ([data validator](https://power-grid-model.readthedocs.io/en/stable/user_manual/data-validator.html)) |
| dataset preconditions | one capability table, `ds.require(...)` | no equivalent in either |
| downloads | cache keyed by release, atomic `.part` install, sha256 gate, no interactive prompt (`download.py`) | OGB prompts before updating a stale copy ([dataset_pyg.py](https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset_pyg.py)) |
| splits | stored inside the file | OGB ships separate index files, read by `get_idx_split` ([dataset_pyg.py](https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset_pyg.py)) |
| results | per family and per bus, at a calibrated false-alarm budget | OGB reports one scalar per task ([evaluate.py](https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py)) |
| regression safety | bit-for-bit frozen references, readability gate, generated-diagram check, timing benchmark gate | power-grid-model has reference cases ([tests/data](https://github.com/PowerGridModel/power-grid-model/tree/main/tests/data)); OGB has no frozen outputs |

The input-check and dataset-precondition rows came with #147 (`models/validation.py`,
`ds.require(...)`), which this plan builds on.

## 2. Ranked changes

### A. Comparable and trustworthy numbers

**1. A standalone evaluator per task (M, medium risk).**
OGB's `Evaluator(name)` scores a dict of truth and predictions (`y_true`, `y_pred`), reads the
metric and task shape from packaged metadata, checks the input contract and states it in `expected_input_format`
([evaluate.py](https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/nodeproppred/evaluate.py)).

Ours has two problems:

- **Scoring is a method on the model.** `LocalizerBase.score` and `SEBase.score` use the model's own
  threshold and meter mask.
- **The headline averages depend on what is scored.** `macro_f1` averages over the buses attacked in
  the view passed in, and `geo` over the families present. So the zero-shot and common protocols
  compute their headline numbers over different sets.

Add `fdia_graph.evaluate` with `LocalizationEvaluator(system, release, protocol)` and
`EstimationEvaluator(system, release)`. They take predictions (`y_pred`, optional `y_score`;
`x_hat`) and read the truth from the pinned file. They fix the bus set and family set per release
and protocol, state the input contract, and return the existing score models. The contract names
the records scored: predictions come with the record indices of the pinned file they belong to (or
cover the protocol's whole split in file order), and a length or index mismatch is refused before
any score, so a filtered or reordered view cannot be scored against the wrong labels. The `score()` methods
become thin wrappers. A model built outside the SDK can then be scored on the same terms.

Re-baselining the frozen references and the guide tables is part of the change.

**2. Per-record state-estimation status (M, low risk).**
power-grid-model takes `error_tolerance` and `max_iterations` and raises named errors
(`IterationDiverge`, `MaxIterationReached`, `NotObservableError`). In batch mode it reports
`failed_scenarios` and `succeeded_scenarios`
([power_grid_model.py](https://github.com/PowerGridModel/power-grid-model/blob/main/src/power_grid_model/_core/power_grid_model.py),
[errors.py](https://github.com/PowerGridModel/power-grid-model/blob/main/src/power_grid_model/_core/errors.py)).

Ours:

- `_solve_plain` runs a fixed number of iterations with no tolerance.
- `_w_solve` keeps the best iterate and silently replaces non-finite steps.
- So a record that did not converge is invisible, folded into the reported error.

Add an `EstimateResult` (`x_hat`, `converged[n]`, `iterations[n]`, the final objective `J[n]`,
`fallback[n]`, per-meter normalized residuals) returned by an `estimate_detailed` next to
`estimate`. Add `tol` to `SolveConfig`, and `NotObservable` / `SolveDiverged` to `errors.py`, with a
`continue_on_error` choice that returns the failed indices.

The residuals are already computed (`SEBase._nres`).

**3. Provenance and seeds in every result (S + compute, low risk).**
OGB requires mean and unbiased std over 10 seeds, the validation score of the reported model, the
package version, parameter count, hardware and tuning ranges
([leaderboard rules](https://ogb.stanford.edu/docs/leader_rules/)).

Ours:

- The guide results (`docs/*/results/*.json`) record only metrics: no data release, SDK version,
  commit or seed.
- The learned localizers run one seed.

Add a provenance block to every results file. Run the learned arms over 5 to 10 seeds and report
mean ± std, the validation score, and the false-alarm rate at the tuned threshold. Deterministic
arms are marked as such.

### B. Data releases

**4. One metadata table per release (M, low risk).**
OGB's `master.csv` holds each dataset's version, URL, split, metric and task count
([master.csv](https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/master.csv)). Ours is spread
across `registry.py` (hashes) and the docs.

Add one file per release, read by the registry. It lists the systems with bus and meter counts,
records, sha256, tasks and official metrics, the split fractions, and the minimum SDK version. The
loader refuses a file newer than the installed SDK can read, with a clear error.

**5. A dataset card per release (S to M, low risk).**
Sections from the Hugging Face template ([datasetcard_template.md](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md)), fitted to a generated
dataset:

1. summary and the diff from the previous release;
2. systems table;
3. generation process (load profiles and cadence, AC power flow, noise model, each family's
   construction and plausibility caps);
4. labels and the three measurement layers;
5. splits and the zero-shot protocol;
6. official tasks, metrics and the evaluator call;
7. baselines;
8. intended and out-of-scope use (research, not operational decisions);
9. limitations: synthetic data, the generator-bus load issue of v0.8.0 (fixed from v0.8.1), how
   residual-stealthy each family is;
10. license, citation and contact.

Add `CITATION.cff` and a self-citation to the README. Today the README cites other work only.

**6. Pin or retire v0.8.2 (S, low risk).**
Verified: `registry.py` has no sha256 entry for v0.8.2, and `download.py` skips the integrity check
when none is pinned. So `fg.load(..., release="v0.8.2")` downloads unverified. Leaving it out of
the registry does not retire it: `release_name()` accepts any version and `resolve()` builds the
asset for it. So pin its hashes, or have `resolve()` refuse a release listed as retired. It is
superseded as a download; `CHANGELOG.md` and `examples/_build_timelines_v083.py` still name it as
history and as the source of the v0.8.3 build, and those references stay.

**7. A timeline file checker (M, low risk).**
power-grid-model's validator checks whole arrays column by column and reports every failing
component id at once. It stays off the calculation path for speed.

Add `ds.check()` (and `fg.check(path)`). It runs the array rules over a file's contents and reports
failures by frame and bus. The domain checks are:

- the meter masks are the same on every frame. The estimator assumes this (it reads the mask from
  record 0). Verified constant on the v0.8.3 files for IEEE-14, 118 and 300, but nothing enforces it;
- metered slots are finite and unmetered slots zero;
- labels and tamper masks exist together: a frame has a labelled bus exactly when it has a
  tampered meter. This is only an existence check. `data/y` is per bus, the tamper masks are per
  meter channel, and a stealthy family changes meters off its labelled buses, so a bus-level
  agreement rule needs a per-family mapping from buses to meters, which is left to a later step;
- the episode table agrees with the per-frame family and `seq_id` columns.

Run it in the release build before upload.

**8. The next data release (L, high risk: file format).**
Bundle the format changes into one release:

- store the meter plan once instead of per frame (verified constant);
- derive each system's seed from the system, so the eight systems stop sharing one episode draw;
- place episodes per split segment, so `At` and `Am` get their exact share of every split.

### C. The validation engine

**9. Collect every error (M, medium risk).**
power-grid-model returns a list of `ValidationError(component, field, ids)` from one pass. Ours
stops at the first failing field.

Add a collect mode: `ConfigError.errors`, a list of `FieldError(model, field, rule, value)`, with
the first entry's message unchanged so existing callers and tests keep working. Array rules report
the failing indices, not just the fact.

### D. Tests, examples and docs

**10. Tolerances per field in the frozen references (S, low risk).**
power-grid-model gives each reference case a `params.json` with per-attribute tolerances (by
regex) and the methods to run
([an example](https://github.com/PowerGridModel/power-grid-model/blob/main/tests/data/state_estimation/1os2msr-no-angle/params.json), read by
[tests/unit/utils.py](https://github.com/PowerGridModel/power-grid-model/blob/main/tests/unit/utils.py)). Ours are hard-coded in the test. Move them next to each frozen
reference.

**11. One baseline per task in `examples/` (S, low risk).**
OGB ships one runnable, multi-seed baseline per task next to the evaluator. Our `examples/` holds
seven files: a quickstart, three training scripts (`train_arma.py`, `train_gnn.py`, `train_tgnn.py`)
and three release-build scripts (`_build_timelines_v08*.py`), while the guide baselines live in
`docs/*/run_*.py`. Move the release-build scripts under `tools/`, and make each training script the
one baseline of its task, scored by the evaluator over several seeds.

**12. Two guide pages (S, low risk).**
From power-grid-model's manual:

- a "checking data" page covering the validation engine, `ds.check()` and the named errors;
- a "performance and validity" page covering what is checked when, the cost of checks on large
  systems, and the known OpenMP runtime issue with its workaround.

## 3. Open decisions

- **The evaluator's bus set.** Either every attackable bus of the system, or the buses attacked in
  the full test split. The first is fixed per system; the second matches today's numbers more
  closely.
- **The zero-shot test set.** It loads families 0 to 4, so `At`, `Al` and `Am` are never scored
  zero-shot. Keep it and state it in the card, or add them to the test side.
- **Seeds.** 5 or 10 per learned arm; 10 costs about twice the guide compute.
- **Timing of the format release (item 8).** Next minor version, or together with the federated
  rerun.

## 4. Sequence

One pull request each, after #147 (items 2, 7 and 9 build on the validation engine):

| step | items | depends on |
|---|---|---|
| 1 | 6 (v0.8.2), 10 (tolerances), 12 (guide pages) | nothing |
| 2 | 2 (estimation status and named errors) | #147 |
| 3 | 1 (evaluators) and 3 (provenance, seeds) | the bus-set decision |
| 4 | 4 (release metadata), 5 (dataset card), 11 (examples) | step 3, for the evaluator call and baselines |
| 5 | 7 (file checker) and 9 (collect-all errors) | #147 |
| 6 | 8 (format release) | steps 4 and 5; a data release |
