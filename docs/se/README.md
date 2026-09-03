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

Test partition, validation-selected hyperparameters from the estimation paper (Huber `c` 1.5 / 2.5,
rank fraction 0.20 / 0.50, removal threshold 4.0 / 5.0 on IEEE 14 / 118). Each cell is the mean
absolute error aggregated over the seven record classes as a geometric mean. Full metrics in
`results/se_ieee{14,118}.json`. IEEE 300 follows once its run finishes (the robust arms take hours
per system on 300; each estimator's estimates are cached as it completes).

**Estimator comparison**

| Estimator | IEEE 14 | IEEE 118 |
|---|---:|---:|
| *Angle MAE (deg)* | | |
| WLS baseline | 0.108 | 0.054 |
| Residual removal | 0.070 | 0.037 |
| Adaptive weighting | 0.068 | 0.036 |
| **Prior + Huber (proposed)** | **0.059** | **0.030** |
| WLS error reduction | 45% | 45% |

| Estimator | IEEE 14 | IEEE 118 |
|---|---:|---:|
| *Voltage MAE (10^-3 pu)* | | |
| WLS baseline | 0.606 | 0.165 |
| Residual removal | 0.427 | 0.113 |
| Adaptive weighting | 0.380 | 0.111 |
| **Prior + Huber (proposed)** | **0.147** | **0.028** |
| WLS error reduction | 76% | 83% |

The estimation paper reports, on v0.4.1 data, angle 0.164 → 0.068 (14) and 0.075 → 0.033 (118) from
WLS to the proposed estimator, and voltage reductions of 57% and 85%. The v0.7.2 shards carry the
accuracy-class meter model, so absolute errors are lower, and the ordering and the reductions hold.

**Per-family results of the proposed estimator.** Baseline cells are the WLS error, reduction is
the proposed estimator's percent reduction over that baseline.

| Family | Base angle (deg) 14 | Base angle (deg) 118 | Base volt (10^-3) 14 | Base volt (10^-3) 118 | Angle red. (%) 14 | Angle red. (%) 118 | Volt red. (%) 14 | Volt red. (%) 118 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 0.008 | 0.009 | 0.15 | 0.09 | -45 | 25 | 58 | 80 |
| Bias (Ad) | 0.136 | 0.028 | 3.90 | 0.55 | 80 | 66 | 97 | 95 |
| Scaling (As) | 0.246 | 0.031 | 1.17 | 0.33 | 67 | 66 | 85 | 91 |
| Replay (Ar) | 0.137 | 0.043 | 1.12 | 0.17 | 82 | 82 | 93 | 88 |
| Stealthy re-solve (Aq) | 0.382 | 0.168 | 0.62 | 0.10 | -5 | 0 | 30 | 63 |
| Slow ramp (At) | 0.065 | 0.034 | 0.19 | 0.09 | 1 | 4 | 48 | 79 |
| Load redistribution (Al) | 0.180 | 0.766 | 0.34 | 0.14 | 15 | 0 | 4 | 39 |

Angle MAE per estimator and family (degrees, lower is better, `geo` is the summary column):

| IEEE 14 | IEEE 118 |
|---|---|
| ![](results/fig_se_ieee14.png) | ![](results/fig_se_ieee118.png) |

Two readings:

1. **Robustness cleans up what it can see.** `Ad`/`As`/`Ar` corrupt meters in place and leave large
   residuals. Removal and Huber each cut their angle error, and the prior on top takes 66 to 82
   percent off the baseline angle error and 85 to 97 percent off the voltage error, on both systems.
2. **The stealthy families barely move.** `Aq`/`At`/`Al` re-solve the physics, so there is nothing
   for a robust weight or a residual test to reject. Angle reduction sits between -5 and 15 percent
   on 14 and at 0 to 4 percent on 118 for every method. Recovering the truth under a stealthy attack
   needs temporal information, not better weighting. The detection side of that story is in
   [`../localization/README.md`](../localization/README.md).

## Regenerate

```bash
FG_SYSTEM=ieee14 python docs/se/run_se.py      # fits, writes results/se_ieee14.json (estimates cached per arm)
FG_SYSTEM=ieee118 python docs/se/run_se.py
FG_SYSTEM=ieee300 python docs/se/run_se.py
python docs/se/make_report.py                  # tables (markdown) + figures + CSV from the JSON
```

Wall time with the GPU to itself: minutes on IEEE-14; on IEEE-118 WLS 1 min, removal 1.1 h, Huber
5.6 h, prior + Huber 1.9 h. Re-runs score from `results/cache/` in about a minute per arm.
