# The results store

Every measured number the docs show comes from here, through `fdia_graph.results`.

| file | holds |
|---|---|
| `<experiment>.csv` | one row per measured value: `run_id`, the keys (`system`, `method`, `family`, `split`, `metric`, `tags`), `value`, `sd`, `ci_lo`, `ci_hi` |
| `runs.csv` | one row per run: SDK version, git commit and dirty flag, data release, seed, settings hash, platform, a note |
| `queries.py` | the queries the docs' results blocks name |

## Writing results

A harness opens a run and adds values; the run writes them with its provenance when it closes, and a
failed run writes nothing.

```python
from fdia_graph.results import Run

with Run("se.estimators", system="ieee14", settings=hp, data_release="v0.9.0", seed=1) as run:
    run.add_tree(est.score(test), method="wls", levels=("family",))  # family -> metric -> value
    run.add("seconds", t, method="wls")
```

A metric must be registered in `fdia_graph.models.results.METRICS` (unit, which direction is
better, display format); a record with an unknown metric, a non-finite value or a malformed tag is
refused. Re-running a run id replaces that run's rows; other runs stay, and a query's `latest`
takes the newest run per measurement.

## Showing results in the docs

No measured number is typed into markdown. A number or a table sits in a results block:

```text
The proposed estimator cuts the angle error <!-- results: red se ieee14 geo angle_mae_deg wls prior+huber --><!-- /results -->%.

<!-- results: se.comparison angle_mae_deg -->
<!-- /results -->
```

`python tools/results_docs.py --write` renders every block from the store; `--check` (a pre-review
gate and a CI step) fails when a block differs from what the store renders; `--lint` lists
decimals and percentages typed outside a block, which is only a report, since settings such as a
noise level are legitimate numbers. Released CHANGELOG sections are history and stay as written;
an Unreleased entry points at the experiment it measured. Figures read the store too, and
`fdia_graph.results.figure_data` writes each figure's CSV sidecar from the same records.
