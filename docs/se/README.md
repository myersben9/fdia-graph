# State estimation with `fdia_graph.se`

Given a scan of noisy, possibly attacked measurements, estimate the true bus voltages and angles.
Every shard ships a noiseless `clean` layer, so the estimate is scored against exact ground truth.

```python
import fdia_graph as fg
from fdia_graph.se import WLS, AdaptiveWeighting, SubspacePrior

train = fg.load("ieee14", split="train")
test  = fg.load("ieee14", split="test")

est  = SubspacePrior(rank_frac=0.2, reweight="huber", c=1.5).fit(train)
xhat = est.estimate(test)   # [n, 2N-1] = [theta rad (non-slack) | V pu (all buses)]
rep  = est.score(test)      # per-family angle/voltage MAE vs the clean truth
```

- Needs the `[se]` extra (`pip install "fdia-graph[se]"`) and a v0.7.2+ shard.
- sklearn style: one base class owns the shared machinery (AC measurement model, chord-Newton solve,
  meter-weight calibration, the 2N-1 state with the slack angle as reference, `ds.slack`). Each
  method class changes exactly one thing, so comparing two methods compares estimators, not
  implementations.
- Walkthrough: [`../guides/state_estimation.md`](../guides/state_estimation.md).

## The method classes

| Class | What it changes | Knobs |
|---|---|---|
| `WLS` | Nothing. The audited baseline: least squares weighted by accuracy-class meter error | |
| `ResidualRemoval` | Largest-normalized-residual removal with an observability guard | `threshold` |
| `AdaptiveWeighting` | Iteratively reweighted least squares (the Huber M-estimator) | `c` |
| `SubspacePrior` | Restricts the solve to a low-rank benign operating subspace, optionally composed with Huber | `rank_frac`, `reweight` |

## Results

Test partition, validation-selected hyperparameters from the estimation paper (IEEE 14: Huber
`c` 1.5, rank fraction 0.20, removal threshold 4.0). Each cell is the mean absolute error
aggregated over the seven record classes as a geometric mean. Full metrics in
`results/se_ieee14.json`. IEEE 118 and 300 columns follow once their runs finish.

**Estimator comparison**

| Estimator | IEEE 14 |
|---|---:|
| *Angle MAE (deg)* | |
| WLS baseline | 0.108 |
| Residual removal | 0.070 |
| Adaptive weighting | 0.068 |
| **Prior + Huber (proposed)** | **0.059** |
| WLS error reduction | 45% |

| Estimator | IEEE 14 |
|---|---:|
| *Voltage MAE (10^-3 pu)* | |
| WLS baseline | 0.606 |
| Residual removal | 0.427 |
| Adaptive weighting | 0.380 |
| **Prior + Huber (proposed)** | **0.147** |
| WLS error reduction | 76% |

**Per-family results of the proposed estimator.** Baseline cells are the WLS error, reduction is
the proposed estimator's percent reduction over that baseline.

| Family | Base angle (deg) 14 | Base volt (10^-3) 14 | Angle red. (%) 14 | Volt red. (%) 14 |
|---|---:|---:|---:|---:|
| Benign | 0.008 | 0.15 | -45 | 58 |
| Bias (Ad) | 0.136 | 3.90 | 80 | 97 |
| Scaling (As) | 0.246 | 1.17 | 67 | 85 |
| Replay (Ar) | 0.137 | 1.12 | 82 | 93 |
| Stealthy re-solve (Aq) | 0.382 | 0.62 | -5 | 30 |
| Slow ramp (At) | 0.065 | 0.19 | 1 | 48 |
| Load redistribution (Al) | 0.180 | 0.34 | 15 | 4 |

Angle MAE per estimator and family (degrees, lower is better, `geo` is the summary column):

![angle MAE per method and family, IEEE 14](results/fig_se_ieee14.png)

Two readings:

1. **Robustness cleans up what it can see.** `Ad`/`As`/`Ar` corrupt meters in place and leave large
   residuals. Removal and Huber each cut their angle error, and the prior on top takes 67 to 82
   percent off the baseline angle error and 85 to 97 percent off the voltage error.
2. **The stealthy families barely move.** `Aq`/`At`/`Al` re-solve the physics, so there is nothing
   for a robust weight or a residual test to reject. Angle reduction sits between -5 and 15 percent
   for every method. Recovering the truth under a stealthy attack needs temporal information, not
   better weighting. The detection side of that story is in
   [`../localization/README.md`](../localization/README.md).

## Regenerate

```bash
FG_SYSTEM=ieee14 python docs/se/run_se.py      # fits, writes results/se_ieee14.json
FG_SYSTEM=ieee118 python docs/se/run_se.py
FG_SYSTEM=ieee300 python docs/se/run_se.py
python docs/se/make_report.py                  # tables (markdown) + figures + CSV from the JSON
```

IEEE-14 takes minutes on a laptop. The larger systems take longer per estimator because every
record is a chord-Newton solve.
