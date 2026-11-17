# fdia-analysis

The analysis toolkit for [fdia-graph](https://github.com/myersben9/fdia-graph) timelines: state
estimation, attack localization, federated training, trusted-meter selection, and the store the
measured results are kept in.

`fdia-graph` generates a timeline and loads it. `fdia-analysis` is what is measured on one. It reads
the files through `fdia_graph.dataset` and builds on the generation toolkit's formulas and records;
`fdia_graph` never imports it, so the generation toolkit installs and runs without this one.

| package | holds |
|---|---|
| `fdia_analysis.se` | the state estimators (one solver, six estimators), the Jacobian features, and the alarm rules in `fdia_analysis.se.detect` |
| `fdia_analysis.localization` | the per-bus localizers: thresholds on the swing, the temporal change and the residual, and the learned ones |
| `fdia_analysis.federated` | the localizers and the estimator's prior trained over K clients that never pool their records |
| `fdia_analysis.trust` | trusted-meter selection and its scoring |
| `fdia_analysis.results` | the results store: a run's records with their provenance, and the renderers the docs read them through |
| `fdia_analysis.models` | the settings and score tables of the packages above |
| `fdia_analysis.formulas` | their mathematics, as pure functions |

## Install

| Install | Adds |
|---|---|
| `pip install fdia-analysis` | the toolkit and `fdia-graph` (numpy, h5py): the threshold localizers and the results store |
| `pip install "fdia-analysis[se]"` | pandapower and scipy, for `fdia_analysis.se` and the residual localizer |
| `pip install "fdia-analysis[torch]"` | torch, for the learned localizers and faster per-record inverses |
| `pip install "fdia-analysis[federated]"` | torch and scikit-learn, for `fdia_analysis.federated` |
| `pip install "fdia-analysis[all]"` | every extra above |

## One example

```python
import fdia_graph as fg
from fdia_analysis.models import PriorConfig
from fdia_analysis.se import SubspacePrior  # pip install "fdia-analysis[se]"

train, test = fg.load("ieee118", split="train"), fg.load("ieee118", split="test")
est = SubspacePrior(prior=PriorConfig(rank_frac=0.5, reweight="huber", c=2.5)).fit(train)
print(est.score(test))  # angle and voltage error per attack family
```

The documentation is the repository's: the pages under `docs/analysis/` (one per study),
`docs/guides/state_estimation.md`, and `results/README.md` for the results store.

## In this repository

The toolkit lives in `analysis/`: the package under `analysis/src/fdia_analysis`, its tests under
`analysis/tests` (they share the fixtures of the repository root's `conftest.py`, the tiny generated
timeline among them), and its own `pyproject.toml`. The tests run from the repository root, where
pytest is configured: `pytest analysis/tests` runs its suite and
`python tools/prereview.py` runs every gate of both toolkits.
