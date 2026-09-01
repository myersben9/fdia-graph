# State estimation — recovering the true state with `fdia_graph.se`

Given a scan of noisy, possibly attacked measurements, estimate the true bus voltages and angles.
Every shard ships a noiseless attack-free `clean` layer, so the estimate is scored against exact
ground truth. The module is sklearn style: one base class owns the shared machinery (the AC
measurement model, the chord-Newton solve, meter-weight calibration, the classical 2N-1 state
with the slack angle as reference), and each method class changes exactly one thing, so comparing
two methods compares estimators, not implementations.

```python
import fdia_graph as fg
from fdia_graph.se import WLS, AdaptiveWeighting, SubspacePrior

train = fg.load("ieee14", split="train")
test  = fg.load("ieee14", split="test")

est  = SubspacePrior(rank_frac=0.2, reweight="huber", c=1.5).fit(train)
xhat = est.estimate(test)   # [n, 2N-1] = [theta rad (non-slack) | V pu (all buses)]
rep  = est.score(test)      # per-family angle/voltage MAE vs the clean truth
```

Needs the `[se]` extra (`pip install "fdia-graph[se]"`) and a v0.7.2+ shard (the `clean` layer is
the truth). See also the walkthrough in [`../guides/state_estimation.md`](../guides/state_estimation.md).

## The method classes

| Class | What it changes | Knobs |
|---|---|---|
| `WLS` | Nothing — the audited baseline, least squares weighted by accuracy-class meter error | — |
| `AdaptiveWeighting` | Iteratively reweighted least squares (the Huber M-estimator) | `c` |
| `ResidualRemoval` | Largest-normalized-residual removal with an observability guard | `threshold` |
| `SubspacePrior` | Restricts the solve to a low-rank benign operating subspace, optionally composed with Huber | `rank_frac`, `reweight` |

## Results — IEEE-14, test split, validation-selected hyperparameters

![angle MAE per method and family](results/fig_se_ieee14.png)

Angle MAE in degrees per family, `geo` is the geometric mean over families (full metrics including
voltage MAE in [`results/se_ieee14.json`](results/se_ieee14.json), figure data in the CSV sidecar):

| method | benign | Aq | Ad | As | Ar | At | Al | geo |
|---|---|---|---|---|---|---|---|---|
| wls | 0.008 | 0.382 | 0.136 | 0.246 | 0.137 | 0.065 | 0.180 | 0.108 |
| huber | 0.009 | 0.384 | 0.046 | 0.092 | 0.038 | 0.065 | 0.180 | 0.068 |
| prior+huber | 0.012 | 0.402 | **0.027** | **0.080** | **0.025** | 0.064 | **0.154** | **0.059** |

Two readings:

1. **Robustness cleans up what it can see.** The in-place corruption families (`Ad`/`As`/`Ar`)
   leave large residuals, so Huber reweighting cuts their angle error 3–5x and the operating-point
   prior tightens it further (geo 0.108 to 0.059 degrees, and the voltage geo improves 6x, see the
   JSON).
2. **The stealthy families barely move.** `Aq`/`At`/`Al` re-solve the physics, so their
   measurements are consistent and there is nothing for a robust weight or a residual test to
   reject — every method lands within a few percent of WLS there. Recovering the truth under a
   stealthy attack is not a weighting problem; it needs temporal information
   (see [`../localization/README.md`](../localization/README.md) for the detection side of that
   same story).

## Regenerate

```bash
python docs/se/run_se.py                       # IEEE-14, minutes on a laptop
FG_SYSTEM=ieee118 python docs/se/run_se.py     # any system in the ladder
```

Outputs land in `results/`: the metrics JSON, the figure, and a CSV data sidecar so the figure can
be restyled without re-running.
