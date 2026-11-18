# Analyse a timeline

This tutorial runs two analyses on one timeline: a state estimator, which recovers the state of
the grid from the readings, and a localizer, which says which buses are under attack. Both run
in seconds on a small timeline.

The analyses are in the second toolkit, `fdia_analysis`.

## Install

```bash
pip install "fdia-analysis[se]"
```

This installs `fdia-graph` with it. The `se` extra adds pandapower and scipy, which the
estimators need.

## The data

The tutorial uses `first_run`, the timeline of [Generate a timeline](generate.md). A released
dataset works the same way: replace the name with `"ieee14"`.

```python
import fdia_graph as fg

train = fg.load("first_run", split="train")
test = fg.load("first_run", split="test")
```

Fit on the train split and score on the test split. Pass the train split whole: both analyses
calibrate on its benign frames.

## 1. Estimate the state

`WLS` is weighted least squares, the classical state estimator.

```python
from fdia_analysis.se.methods import WLS

est = WLS().fit(train)
xhat = est.estimate(test)
print(xhat.shape)  # [n, 2N-1]
```

`estimate` reads the measurements only. Each row of `xhat` is one frame's state: the angle of
every bus but the reference bus, in radians, then the voltage magnitude of every bus.

```python
scores = est.score(test)
print(scores.benign)
print(scores.At)
print(scores.Am)
```

`score` compares the estimate with the `clean` layer, the true state, per attack family. The
run measured for this page printed (rounded):

```
ErrorPair(angle_mae_deg=0.017, voltage_mae_pu=0.00012)
ErrorPair(angle_mae_deg=0.023, voltage_mae_pu=0.00017)
ErrorPair(angle_mae_deg=0.159, voltage_mae_pu=0.00183)
```

The error under the overload `Am` is several times the benign error. The readings of an
attacked frame fit a false state, so the estimator returns that false state and reports nothing
unusual. That is what makes these attacks hard to see.

## 2. Locate the attack

`SwingThreshold` is the simplest localizer. It flags a bus when its injection changes by much
more than the bus's recent changes: the `swing` feature of the file.

```python
from fdia_analysis.localization.methods import SwingThreshold

loc = SwingThreshold().fit(train)
flags = loc.localize(test)
print(flags.shape)  # [n, N], True where a bus is flagged
```

`fit` sets one threshold per bus on the benign frames, at a false-alarm target of 1 %.

```python
report = loc.score(test)
print(report.benign.false_alarm_rate)
print(report.At.detection_rate)
print(report.Am.detection_rate)
```

The run measured for this page printed (rounded):

```
0.018
0.017
0.467
```

About 2 % of benign frames raise an alarm. The slow ramp `At` is flagged no more often than a
benign frame: each of its steps is smaller than the load's own changes, so a threshold on the
change cannot see it. The overload `Am` moves faster and is flagged in about half of its frames.

These are the numbers of one 2,000-frame run, shown to explain the output. The measured
results are on the study pages.

## What the two results say

| Analysis | Reads | On these attacks |
|---|---|---|
| `WLS` | the measurements | returns the false state |
| `SwingThreshold` | the change from the frame before | misses the slow ramp |

Both attacks are built to pass the classical checks. The study pages measure what does better,
on the full datasets.

## Where next

- [Estimate the state](../guides/state_estimation.md): an estimator that beats `WLS`.
- [State estimation](../analysis/estimation/README.md) and
  [Localization](../analysis/localization/README.md): every method, with results.
- [Layers and labels](../concepts/layers.md): which layers an analysis may read.
