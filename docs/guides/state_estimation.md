# Beating WLS: state estimation with the SDK

Measurements in, state out, better than weighted least squares on the public test cases.
Everything here runs off a shard download — no extra files.

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

`fit()` learns everything from the train split: per-meter error scales (rms of benign residuals at
the shard's `clean` truth) and the chord Jacobian. `estimate()` returns states already in the truth's
angle frame — the state is the classical 2N-1 vector (only the slack ANGLE is fixed, pinned per
record to `clean[slack]`; every voltage magnitude including the slack is estimated, matching
production practice and pandapower's estimator), so `xhat - truth` needs no alignment step.

## The better estimator

```python
from fdia_graph.se import SubspacePrior

est = SubspacePrior(rank_frac=0.2, reweight="huber", c=1.5).fit(train)
print(est.score(test)["geo"])          # {'angle_mae_deg': 0.059, 'voltage_mae_pu': 1.47e-4}
```

The prior restricts the estimate to the low-dimensional subspace benign operation actually occupies
(an SVD of the training states), and the Huber reweighting discards measurements the physics cannot
explain. On the IEEE 14-bus test partition that is **45 percent lower angle error and 76 percent
lower voltage error** than the audited WLS baseline (0.108° → 0.059° and 6.06e-4 → 1.47e-4 pu,
geometric mean over the seven record classes, v0.7.2 data). Much of the voltage gain comes from the
prior learning the generator voltage setpoints from benign history, structure WLS re-estimates from
noisy meters at every scan.

Validation-selected hyperparameters per system, from the companion estimation paper:

| system | `rank_frac` | huber `c` | `ResidualRemoval` threshold |
|---|---|---|---|
| ieee14 | 0.20 | 1.5 | 4.0 |
| ieee118 | 0.50 | 2.5 | 5.0 |
| ieee300 | 0.50 | 6.0 | (removal does not help) |

## What to expect per family

`score(test)` breaks the result out per attack family, and the split matters:

- **Tampering families** (`Ad` bias, `As` scaling, `Ar` replay) improve dramatically — they corrupt
  meters in place, which is what robust weighting exists to reject.
- **Stealthy re-solved families** (`Aq`, `At`, `Al`) sit near parity for every estimator. They
  present a physically valid state inside the learned subspace, so no single-scan method can reject
  them. This is a property of the attacks, not a tuning failure — see the companion paper.
- **Benign** records improve on the larger systems and can lose slightly on ieee14, where the
  baseline is already near the noise floor and the subspace truncation bias dominates.

## Other arms and your own

`AdaptiveWeighting(c=...)` and `ResidualRemoval(threshold=...)` are the classical robust baselines,
sharing the same iteration and weights so comparisons are estimator-vs-estimator. To add your own
method, subclass `SEBase` and override one hook:

- `_fit_states(x_benign)` — learn anything from the benign training states
- `_basis()` — return a `[2N-1, K]` basis to restrict the state space (or `None` for full)
- `_solve(z, vsl, thsl)` — the per-batch solve; `self._w_solve(...)` gives you the
  divergence-guarded weighted iteration and `self._nres(...)` normalized residuals

The classes are verified per-record equivalent to the estimation paper's solver, so results slot
directly into its protocol.
