# Localization — per-bus attack localization with `fdia_graph.localization`

Which buses are under attack, per scan. The module mirrors [`fdia_graph.se`](../se/README.md):
one base class owns the calibration protocol and the metrics, and each method class changes
exactly one thing — the per-bus score — so comparing two methods compares detection signals,
not implementations.

```python
import fdia_graph as fg
from fdia_graph.localization import SwingThreshold, DeltaThreshold, ResidualLocalizer

train = fg.load("ieee14", split="train")
test  = fg.load("ieee14", split="test")

loc  = SwingThreshold(fa_target=0.01).fit(train)   # thresholds set on benign records only
flag = loc.localize(test)                          # [n, N] bool — which buses are called attacked
rep  = loc.score(test)                             # per-family metrics + benign false alarms
```

`fit()` calibrates a per-bus threshold at the `(1 - fa_target)` benign quantile, so every method
runs at the same false-alarm budget and no attack data is used to tune. `score()` reports, per
attack family: **strict localization accuracy** (predicted attacked set equals the truth exactly),
node precision/recall/F1, per-sample macro-F1, and the record-level detection rate — always next
to the benign false-alarm rate, because a detection rate on its own is meaningless (a detector
that flags everything has DR 1.0 and FA 1.0).

## The three methods

| Class | Score | Needs |
|---|---|---|
| `SwingThreshold` | The shard's windowed relative-swing feature: each scan's injection change as a z-score of the bus's typical recent change | numpy only |
| `DeltaThreshold` | The raw one-scan change scaled by the bus's benign RMS — the same signal without the windowing, so the gap shows what windowing buys | numpy only |
| `ResidualLocalizer` | Largest normalized residual from a state-estimation solve (any `fdia_graph.se` estimator, default `WLS`), aggregated to each bus's own meters and incident flows — textbook bad-data identification | `[se]` extra |

## Results — IEEE-14, test split, `fa_target=0.01`

![node F1 per method and family](results/fig_loc_ieee14.png)

Node F1 per family (rows carry each method's benign record-level false-alarm rate; full metrics in
[`results/loc_ieee14.json`](results/loc_ieee14.json), figure data in the CSV sidecar):

| method | benign FA | Aq | Ad | As | Ar | At | Al |
|---|---|---|---|---|---|---|---|
| swing | 0.18 | 0.73 | 0.97 | 0.94 | 0.90 | **0.41** | 0.74 |
| delta | 0.16 | 0.70 | 0.98 | 0.96 | 0.95 | **0.35** | 0.73 |
| residual | 0.15 | **0.12** | 0.46 | 0.46 | 0.51 | **0.08** | **0.06** |

Two readings, and they are the dataset's central story reproduced by the SDK's own classes:

1. **The temporal spike catches almost everything.** Any attack edit that exceeds the noise floor
   shows up as a per-bus temporal spike the moment it starts — including the BDD-stealthy re-solve
   families `Aq`/`Al` that the residual arm cannot see. The one family built to defeat it is the
   slow ramp `At`, which stays inside typical per-scan change by construction. `At` is the open
   frontier.
2. **The classical arm misses every stealthy family by construction.** `ResidualLocalizer` detects
   the in-place corruptions (`Ad`/`As`/`Ar` detection rate ≈ 1.0) but localizes them coarsely
   (residuals smear over neighboring buses), and its detection rate on `Aq`/`At`/`Al` (0.18–0.26)
   is indistinguishable from its benign false-alarm rate — those measurements are
   physics-consistent, so there is no residual to find.

## Regenerate

```bash
python docs/localization/run_localization.py            # IEEE-14, minutes on a laptop
FG_SYSTEM=ieee118 python docs/localization/run_localization.py   # any system in the ladder
```

Outputs land in `results/`: the metrics JSON, the figure, and a CSV data sidecar so the figure can
be restyled without re-running.
