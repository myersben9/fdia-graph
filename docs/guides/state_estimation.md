# Beating WLS: state estimation with the SDK

Measurements in, state out, better than weighted least squares on the public test cases.
Everything here runs off a shard download. No extra files.

```bash
pip install "fdia-graph[se]"     # torch + pandapower
```

## Baseline first

```python
import fdia_graph as fg
from fdia_graph.se import WLS

train = fg.load("ieee14", split="train")
test  = fg.load("ieee14", split="test")

wls = WLS().fit(train)                 # calibrates meter weights from benign residuals
xhat = wls.estimate(test)              # [n, 2N-1] = [theta rad (non-slack) | V pu (all buses)]
print(wls.score(test)["geo"])          # {'angle_mae_deg': 0.108, 'voltage_mae_pu': 6.06e-4}
```

What `fit()` learns from the train split:

- per-meter error scales (RMS of benign residuals at the shard's `clean` truth)
- the chord Jacobian

What `estimate()` returns:

- the classical 2N-1 state: every voltage magnitude (slack included) plus every non-slack angle
- already in the truth's angle frame. The slack angle is pinned per record to `clean[slack]`, so
  `xhat - truth` needs no alignment step. The reference bus index is `ds.slack`.

## The better estimator

```python
from fdia_graph.se import SubspacePrior

est = SubspacePrior(rank_frac=0.2, reweight="huber", c=1.5).fit(train)
print(est.score(test)["geo"])          # {'angle_mae_deg': 0.059, 'voltage_mae_pu': 1.47e-4}
```

- **Prior**: restricts the estimate to the low-dimensional subspace benign operation occupies (an SVD
  of the training states).
- **Huber**: discards measurements the physics cannot explain.
- **Result on IEEE-14 test**: 45 percent lower angle error and 76 percent lower voltage error than
  WLS (0.108° → 0.059°, 6.06e-4 → 1.47e-4 pu, geometric mean over the seven record classes, v0.7.2 data).
- Most of the voltage gain is the prior learning generator voltage setpoints from benign history.
  WLS re-estimates those from noisy meters at every scan.

Validation-selected hyperparameters per system, from the companion estimation paper:

| system | `rank_frac` | huber `c` | `ResidualRemoval` threshold |
|---|---|---|---|
| ieee14 | 0.20 | 1.5 | 4.0 |
| ieee118 | 0.50 | 2.5 | 5.0 |
| ieee300 | 0.50 | 6.0 | (removal does not help) |

## What to expect per family

`score(test)` breaks the result out per attack family.

| records | behavior | why |
|---|---|---|
| `Ad` bias, `As` scaling, `Ar` replay | improve a lot | they corrupt meters in place, which robust weighting exists to reject |
| `Aq`, `At`, `Al` (stealthy) | near parity for every estimator | a physically valid state inside the learned subspace. No single-scan method can reject it. |
| benign | improves on the larger systems, can lose slightly on ieee14 | the baseline is already at the noise floor there, so subspace truncation bias dominates |

Full per-family table and figure: [`../se/README.md`](../se/README.md).

## Other arms and your own

`AdaptiveWeighting(c=...)` and `ResidualRemoval(threshold=...)` are the classical robust baselines.
They share the same iteration and weights, so comparisons are estimator-vs-estimator.

To add your own method, subclass `SEBase` and override one hook:

| hook | does |
|---|---|
| `_fit_states(x_benign)` | learn anything from the benign training states |
| `_basis()` | return a `[2N-1, K]` basis to restrict the state space, or `None` for full |
| `_solve(z, thsl)` | the per-batch solve. `thsl` is the per-record slack angle reference. |

Inside `_solve`, `self._w_solve(...)` is the divergence-guarded weighted iteration and
`self._nres(...)` gives normalized residuals.

The classes are verified per-record equivalent to the estimation paper's solver, so results slot
directly into its protocol.
