# Trusted meters with `fdia_graph.trust`

Which meters to secure so that a stealthy attack stops being stealthy.

```python
import fdia_graph as fg
from fdia_graph.trust import TrustedMeters, TrustedMetersDQN

train = fg.load("ieee14", split="train")
test = fg.load("ieee14", split="test", order="time")

tm = TrustedMeters(k=20).fit(train)  # the greedy row-reduction selection on the Jacobian
tm.order, tm.cost  # the meters secured in order, the attack cost after each
rep = tm.score(test)  # residual detection per family, with and without the trust

rl = TrustedMetersDQN(k=20, episodes=200).fit(train)  # the same selection as an MDP with a DQN policy
```

![tm.fit: the WLS Jacobian at the benign mean, the attack subspace left open by the secured rows, the attack cost, then the greedy or the DQN selection; tm.score: secured meters read their benign value, the WLS residual test at the benign alarm level, detection per family before and after](../figures/diagrams/trust_flow.png)

## The idea

A stealthy attack `a = H c` moves the meters the way a real state change `c` would, so the residual
test does not see it. A meter the attacker cannot write pins its row: `a_S = 0`, so `H_S c = 0`
and the attack lives in the null space of the secured rows. Every secured meter shrinks that space;
once `H_S` has full column rank there is no stealthy attack left. The attack cost is how many
meters the cheapest attack in the open subspace must touch (the sparsest row of its reduced row
echelon basis, an upper bound on the exact NP-hard minimum). Both selectors read the same kernel,
`fdia_graph.formulas.trust`.

| class | picks the next meter by | needs |
|---|---|---|
| `TrustedMeters` | the greedy rule: on the cheapest open attack, the meter whose protection leaves the attacker the costliest cheapest attack | `[se]` |
| `TrustedMetersDQN` | a Q-network trained on the MDP (state the secured set, reward the cost rise), rolled out greedily; one rollout of `m` forward passes picks a set where the greedy rule re-evaluates every candidate at every step | `[se]`, `[torch]` |

`score(test)` needs a time-ordered timeline view: on every attacked frame the secured meters are
set back to their benign reading (the attacker cannot write them) and a WLS residual test at the
benign false-alarm level says whether the attack now shows. `detected_before` is the residual test
with every meter writable, `detected_after` with the trust in place.

## Results

`python docs/trust/run_trust.py` (set `FG_SYSTEM`) writes `results/trust_<system>.json`: the
selection, the attack-cost curve, and the detection table. Read the detection table first: the
attack cost is the sparsest row of an echelon basis of the open subspace, an upper bound on the
true sparsest attack, so it is not monotone in the secured set (on IEEE-14 the greedy cost
plateaus at 54 after two meters, the DQN's walks 3, 49, 46, 31, 43, 7 and back to 3, while the
DQN's detection of Aq, Al and Am rises from 0.02, 0.00, 0.00 to 0.88, 0.79, 0.79).

WLS residual detection rate per family at a 1% benign alarm level, before and after securing 20
meters (v0.8.0 timelines, test split):

| system | selector | Aq | At | Al | Am | Ad / As / Ar |
|---|---|---|---|---|---|---|
| ieee14 | greedy | 0.02 → 0.76 | 0.01 → 0.25 | 0.00 → 0.82 | 0.00 → 0.69 | 1.00 / 1.00 / 0.95 → 1.00 / 1.00 / 0.92 |
| ieee14 | DQN | 0.02 → 0.88 | 0.01 → 0.49 | 0.00 → 0.79 | 0.00 → 0.79 | 1.00 / 1.00 / 0.95 → 1.00 / 1.00 / 0.91 |
| ieee118 | greedy | 0.00 → 0.00 | 0.01 → 0.01 | 0.00 → 0.02 | 0.01 → 0.01 | 1.00 / 1.00 / 0.98 → 1.00 / 1.00 / 0.98 |
| ieee118 | DQN | 0.00 → 0.32 | 0.01 → 0.02 | 0.00 → 0.45 | 0.01 → 0.27 | 1.00 / 1.00 / 0.98 → 1.00 / 1.00 / 0.98 |

## What the secured meters do for estimation and localization

The residual test is one consumer of a trusted set. `tm.secured_copy(ds, out, name)` writes a copy
of the timeline in which the selected meters read their benign value on every frame (the attacker
locked out of them, the tamper masks and the stored temporal features following), so every
estimator and localizer of the other two guides can be scored with the set secured;
`python docs/trust/run_secured.py` (set `FG_SYSTEM`, `FG_K`) does that for the greedy and the DQN
selections of `run_trust.py` and writes `results/secured_<system>.json`.

One rule came out of it: a localization gate must never down-weight a secured meter.
`GatedPrior` scales every meter of a flagged bus by a thousandth, and on a secured copy that
included the secured meters, the only true readings inside the attacked region, so every gate was
worse than no gate. `GatedPrior(secured=tm.select())` keeps them at full weight, and that arm wins.

IEEE-14, 20 secured meters, test split, node-F1 with the detection rate in parentheses:

| localizer | copy | Aq | At | Al | Am | macro-F1 / benign FA |
|---|---|---|---|---|---|---|
| residual | plain | 0.01 (0.04) | 0.01 (0.04) | 0.00 (0.02) | 0.00 (0.03) | 0.343 / 0.0027 |
| residual | greedy | 0.41 (0.82) | 0.06 (0.35) | 0.52 (0.88) | 0.48 (0.75) | 0.488 / 0.0027 |
| residual | DQN | 0.42 (0.90) | 0.25 (0.56) | 0.66 (0.89) | 0.68 (0.86) | 0.561 / 0.0027 |
| 1D CNN + Jacobian | plain | 0.71 (1.00) | 0.71 (0.92) | 0.97 (1.00) | 0.98 (0.99) | 0.870 / 0.0151 |
| 1D CNN + Jacobian | greedy | 0.68 (1.00) | 0.70 (0.91) | 0.98 (1.00) | 0.97 (0.99) | 0.873 / 0.0145 |
| 1D CNN + Jacobian | DQN | 0.68 (1.00) | 0.70 (0.92) | 0.97 (1.00) | 0.97 (0.99) | 0.871 / 0.0185 |

The trusted set opens the residual localizer on the stealthy families, the DQN's set more than the
greedy's on every one of them, and leaves the learned localizer where it was, since it already finds
those families from their onset.

Angle mean absolute error in degrees, geometric mean over the eight record classes:

| estimator | plain | greedy | DQN |
|---|---|---|---|
| prior + Huber | 0.050 | 0.041 | 0.033 |
| prior + Huber + CNN gate | 0.060 | 0.044 | 0.058 |
| prior + Huber + CNN gate, secured meters exempt | | 0.025 | 0.020 |
| prior + Huber + residual gate, secured meters exempt | | 0.031 | 0.020 |
| prior + Huber + oracle gate | 0.050 | 0.041 | 0.043 |
| prior + Huber + oracle gate, secured meters exempt | | 0.025 | 0.021 |

The DQN's 20 meters take the proposed estimator from 0.050 to 0.033 degrees with no gate and to
0.020 with the exempting gate, the stealthy families losing 58 to 83 percent of their error (Aq 0.219
to 0.036, At 0.077 to 0.020, Al 0.057 to 0.023, Am 0.048 to 0.020) with the benign error unchanged;
the residual gate, which needs no learned model, matches the CNN gate there because the secured
meters made the residual see those families. On IEEE-118 the same 20 meters are 4 percent of the
meters instead of 24, and the gain is within a percent; the budget has to scale with the system.

Source: the multi-snapshot attack and the trusted-PMU defence of [WU26] (`docs/reference/REFERENCES.md`).
