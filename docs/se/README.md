# State estimation with `fdia_graph.se`

Given a scan of noisy, possibly attacked measurements, estimate the true bus voltages and angles.
Every timeline ships a noiseless `clean` layer, so the estimate is scored against exact ground truth.

```python
import fdia_graph as fg
from fdia_graph.se import WLS, AdaptiveWeighting, SubspacePrior

train = fg.load("ieee14", split="train")
test  = fg.load("ieee14", split="test")

est  = SubspacePrior(rank_frac=0.2, reweight="huber", c=1.5).fit(train)
xhat = est.estimate(test)   # [n, 2N-1] = [theta rad (non-slack) | V pu (all buses)]
rep  = est.score(test)      # per-family angle/voltage MAE vs the clean truth
```

![SEBase shared by every estimator: the measurement function, the chord Jacobian, meter weights from benign residuals, the chord-Newton loop; each estimator changes one thing: WLS nothing, AdaptiveWeighting Huber weights, ResidualRemoval dropping large residuals, SubspacePrior a low-rank basis, JacobianWeighting weights from the unexplained residual, GatedPrior a localizer gating the weights](../figures/diagrams/se_estimators.png)

| needs | `pip install "fdia-graph[se]"`, a timeline (v0.8.1) or a v0.7.2 record shard, `units="physical"` |
|---|---|
| walkthrough | [`../guides/state_estimation.md`](../guides/state_estimation.md) |
| state | 2N-1: every voltage magnitude, every non-slack angle; slack angle pinned per frame (`ds.slack`) |

## The method classes

| Class | What it changes | Knobs |
|---|---|---|
| `WLS` | Nothing. The audited baseline: least squares weighted by accuracy-class meter error | |
| `ResidualRemoval` | Classical largest-normalized-residual removal: the one largest residual above the threshold per pass, re-solved, with an observability guard | `threshold` |
| `AdaptiveWeighting` | Iteratively reweighted least squares (the Huber M-estimator) | `c` |
| `SubspacePrior` | Restricts the solve to a low-rank benign operating subspace, optionally composed with Huber | `rank_frac`, `reweight` |
| `JacobianWeighting` | Huber weights from the physically unexplained part of the scan-to-scan measurement change (the Jacobian-informed digest), one solve; `reweight="huber"` adds the classical passes on top | `c`, `reweight`, `huber_c` |
| `GatedPrior` | The proposed estimator with a localizer gating the weights: meters on flagged buses and their incident flows are down-weighted so the prior fills in the state there; `secured` meters are never down-weighted | `gate`, `gate_factor`, `secured` |

## Results

| protocol | |
|---|---|
| partition | test split, hyperparameters validation-selected in the estimation paper |
| Huber `c` / rank fraction | 1.5 / 0.20 (14), 2.5 / 0.50 (118), 6.0 / 0.50 (300) |
| removal threshold | 4.0 (14), 5.0 (118), 5.0 (300); the 300 column ran it on the v0.8.1 timeline (2.6 hours, the per-record observability guard), where it cuts the WLS angle error 28% |
| cell | geometric mean of the MAE over the eight record classes (benign and the seven families); full metrics in `results/se_ieee{14,118,300}.json` |

**Estimator comparison** (bold marks the proposed rows, not the best cell)

| Estimator | IEEE 14 | IEEE 118 | IEEE 300 |
|---|---:|---:|---:|
| *Angle MAE (deg)* | | | |
| WLS baseline | 0.090 | 0.022 | 0.027 |
| Residual removal | 0.051 | 0.015 | 0.019 |
| Adaptive weighting | 0.056 | 0.015 | 0.020 |
| **Prior + Huber (proposed)** | **0.049** | **0.011** | **0.016** |
| Jacobian weighting | 0.075 | 0.016 | 0.021 |
| **Prior + Huber + CNN gate** | **0.044** | **0.011** | **0.016** |
| Prior + Huber + oracle gate (ceiling) | 0.041 | 0.010 | 0.015 |
| WLS error reduction | 46% | 51% | 40% |

| Estimator | IEEE 14 | IEEE 118 | IEEE 300 |
|---|---:|---:|---:|
| *Voltage MAE (10^-3 pu)* | | | |
| WLS baseline | 0.840 | 0.188 | 0.219 |
| Residual removal | 0.507 | 0.127 | 0.171 |
| Adaptive weighting | 0.536 | 0.129 | 0.177 |
| **Prior + Huber (proposed)** | **0.163** | **0.031** | **0.065** |
| Jacobian weighting | 0.692 | 0.146 | 0.184 |
| **Prior + Huber + CNN gate** | **0.264** | **0.038** | **0.063** |
| Prior + Huber + oracle gate (ceiling) | 0.214 | 0.036 | 0.061 |
| WLS error reduction | 81% | 84% | 70% |

| paper (v0.4.1 data) | 14 | 118 | 300 |
|---|---:|---:|---:|
| angle, WLS → proposed | 0.164 → 0.068 | 0.075 → 0.033 | 0.129 → 0.068 |
| voltage reduction | 57% | 85% | 68% |

The v0.8.1 timelines carry the accuracy-class meter model (since v0.7.2), so absolute errors are
lower than the paper's; the ordering holds and the reductions stay of the same size (angle 46, 51
and 40% here against 59, 56 and 47%).

**Per-family results of the proposed estimator.** Baseline cells are the WLS error, reduction is
the proposed estimator's percent reduction over that baseline.

| Family | Base angle (deg) 14 | Base angle (deg) 118 | Base angle (deg) 300 | Base volt (10^-3) 14 | Base volt (10^-3) 118 | Base volt (10^-3) 300 | Angle red. (%) 14 | Angle red. (%) 118 | Angle red. (%) 300 | Volt red. (%) 14 | Volt red. (%) 118 | Volt red. (%) 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 0.008 | 0.009 | 0.015 | 0.15 | 0.09 | 0.15 | -55 | 27 | 22 | 60 | 81 | 67 |
| Bias (Ad) | 0.126 | 0.028 | 0.025 | 3.60 | 0.55 | 0.33 | 80 | 66 | 41 | 97 | 95 | 81 |
| Scaling (As) | 0.239 | 0.030 | 0.024 | 1.10 | 0.33 | 0.25 | 70 | 66 | 37 | 85 | 92 | 73 |
| Replay (Ar) | 0.230 | 0.052 | 0.123 | 1.19 | 0.21 | 0.42 | 86 | 85 | 89 | 94 | 91 | 86 |
| Stealthy re-solve (Aq) | 0.299 | 0.020 | 0.028 | 2.45 | 0.15 | 0.20 | 17 | 28 | 14 | 73 | 70 | 54 |
| Slow ramp (At) | 0.103 | 0.011 | 0.020 | 0.94 | 0.10 | 0.17 | 26 | 26 | 17 | 76 | 78 | 64 |
| Load redistribution (Al) | 0.051 | 0.021 | 0.023 | 0.40 | 0.17 | 0.18 | 7 | 28 | 13 | 53 | 66 | 59 |
| Multi-snapshot (Am) | 0.047 | 0.025 | 0.020 | 0.38 | 0.20 | 0.17 | -5 | 29 | 15 | 47 | 68 | 62 |

Angle MAE per estimator and family (degrees, lower is better, `geo` is the summary column):

| IEEE 14 | IEEE 118 | IEEE 300 |
|---|---|---|
| ![](results/fig_se_ieee14.png) | ![](results/fig_se_ieee118.png) | ![](results/fig_se_ieee300.png) |

| families | what happens | angle reduction | why |
|---|---|---|---|
| `Ad` `As` `Ar` (in place) | robustness cleans up what it can see | 70 to 86% on 14, 66 to 85% on 118, 37 to 89% on 300; voltage 73 to 97% everywhere | corrupted meters leave large residuals for removal, Huber and the prior to reject |
| `Aq` `At` `Al` `Am` (stealthy) | move part of the way, through the prior alone | -5 to 26% on 14 (`Am` slightly worse), 26 to 29% on 118, 13 to 17% on 300; voltage 47 to 78% | the local false state is a consistent AC state, so no residual exists and Huber sees nothing; it also sits off the benign operating subspace, so the prior pulls the estimate back part of the way; the rest needs temporal information ([`../localization/README.md`](../localization/README.md)) |

## Jacobian-informed weighting

![the scan-to-scan change is split into the part a state change explains and the unexplained rest; Huber weights come from the unexplained part; one weighted solve](../figures/diagrams/se_jacobian_weighting.png)

| result | 14 | 118 | 300 |
|---|---:|---:|---:|
| WLS angle error reduction | 17% | 23% | 21% |
| where it comes from | `Ad` 0.126 → 0.080, `As` 0.239 → 0.164, `Ar` 0.230 → 0.123 | in-place families only | `Ar` 0.123 → 0.029 |
| stealthy families | untouched, as `(I − P_H)a = 0` predicts: `Aq` 0.299, `At` 0.103, `Al` 0.051, `Am` 0.047 on both | same | same |
| against iterated Huber | 0.075 vs 0.056 | 0.016 vs 0.015 | 0.021 vs 0.020 |

The temporal unexplained residual carries what Huber already recovers from the estimate's own
residual, so this route cannot move the proposed estimator. The routes that can are a localizer gate
and, ahead of it, a trusted set of meters.

## Localization-gated estimation

![a localizer flags buses; the weights of every meter of a flagged bus and its branches are scaled by a thousandth; the prior plus Huber solve fills the gap from the benign prior](../figures/diagrams/se_gated_prior.png)

Angle mean absolute error in degrees on the v0.8.1 timelines, the proposed estimator alone, with the
CNN localizer as the gate, and with the true labels as the gate (the ceiling for any gate):

| | IEEE 14 | | | IEEE 118 | | | IEEE 300 | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| angle MAE (deg) | proposed | + CNN gate | + oracle | proposed | + CNN gate | + oracle | proposed | + CNN gate | + oracle |
| Aq stealthy re-solve | 0.248 | 0.203 | 0.231 | 0.015 | 0.015 | 0.014 | 0.024 | 0.026 | 0.024 |
| Ad / As / Ar in place | 0.025 / 0.070 / 0.031 | 0.023 / 0.023 / 0.026 | 0.020 / 0.019 / 0.020 | 0.009 / 0.010 / 0.008 | 0.008 / 0.008 / 0.008 | 0.007 / 0.007 / 0.007 | 0.015 / 0.015 / 0.014 | 0.012 / 0.012 / 0.013 | 0.012 / 0.012 / 0.012 |
| At slow ramp | 0.077 | 0.075 | 0.071 | 0.008 | 0.009 | 0.008 | 0.017 | 0.018 | 0.017 |
| Al redistribution | 0.047 | 0.071 | 0.069 | 0.015 | 0.020 | 0.019 | 0.020 | 0.019 | 0.018 |
| Am multi-snapshot | 0.049 | 0.067 | 0.068 | 0.018 | 0.021 | 0.023 | 0.017 | 0.017 | 0.016 |
| geometric mean | 0.049 | **0.044** | 0.041 | **0.011** | 0.011 | 0.010 | 0.016 | **0.016** | 0.015 |

| families | what the gate does | why |
|---|---|---|
| `Ad` `As` `Ar` (in place) | finishes the job at every size: the CNN gate takes them to 0.023 to 0.026 on 14, 0.008 on 118 and 0.012 to 0.013 on 300, next to the oracle's 0.019 to 0.020, 0.007 and 0.012 | the flagged bus's meters are the corrupted ones, the prior fills a hole that held nothing true |
| `Aq` `At` `Al` `Am` (stealthy) | makes `Al` and `Am` worse on 14 and 118, with the true labels too: `Al` 0.047 → 0.071 on 14, 0.015 → 0.020 on 118; on 300 the gate is neutral to slightly better on them, and `Aq` on 14 improves (0.248 → 0.203) | a local false state is a consistent AC state, so the flagged bus's meters are the evidence the prior was using; pulled out, the prior guesses from the neighbours, which describe the false state |

On the v0.8.1 timelines the CNN gate lowers the geometric mean 10% on IEEE-14 (0.049 to 0.044) and
4% on 300 (0.0162 to 0.0155) and costs 1.5% on 118 (0.0106 to 0.0108): the in-place families (and
`Aq` on 14) gain, and the stealthy-family losses (`Al` and `Am` most) offset that gain on 118. The gate pays most when it has
something true to leave in: with 20 meters secured by the DQN selector of
[`../trust/README.md`](../trust/README.md) and exempt from the gate, IEEE-14 goes from 0.049 degrees
to 0.033 with the secured meters alone and to 0.022 with the same CNN gate added. Recovering a
stealthy false state from one scan otherwise needs the previous scan, the temporal direction.

## Regenerate

```bash
FG_SYSTEM=ieee14 python docs/se/run_se.py      # fits, writes results/se_ieee14.json (estimates cached per arm)
FG_SYSTEM=ieee118 python docs/se/run_se.py
FG_SYSTEM=ieee300 python docs/se/run_se.py
python docs/se/make_report.py                  # tables (markdown) + figures + CSV from the JSON
```

| | |
|---|---|
| skip arms | `FG_SKIP=removal,...` (every published column ran every arm) |
| wall time, CPU, IEEE-300 (v0.8.1 run, sharing the CPU with three other guide jobs) | WLS 30 s, residual removal 2.6 h, Huber 3.4 h, prior + Huber 2.6 h, Jacobian weighting 5 min, each gated arm 53 min |
| re-runs | score from `results/cache/` in about a minute per arm |
