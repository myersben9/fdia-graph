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
plateaus at 54 after two meters, the DQN's walks 34, 54, 46, 43, 46, 11, 3, 9, 8, 9, 22 and 33,
while the DQN's detection of Aq, Al and Am rises from 0.00, 0.00, 0.00 to 0.93, 0.93, 0.90).

WLS residual detection rate per family at a 1% benign alarm level, before and after securing 20
meters (v0.8.1 timelines, test split):

| system | selector | Aq | At | Al | Am | Ad / As / Ar |
|---|---|---|---|---|---|---|
| ieee14 | greedy | 0.00 → 0.96 | 0.00 → 0.22 | 0.00 → 1.00 | 0.00 → 0.91 | 1.00 / 1.00 / 0.95 → 1.00 / 1.00 / 0.93 |
| ieee14 | DQN | 0.00 → 0.93 | 0.00 → 0.48 | 0.00 → 0.93 | 0.00 → 0.90 | 1.00 / 1.00 / 0.95 → 1.00 / 1.00 / 0.94 |
| ieee118 | greedy | 0.00 → 0.00 | 0.01 → 0.01 | 0.01 → 0.01 | 0.00 → 0.01 | 1.00 / 1.00 / 0.99 → 1.00 / 1.00 / 0.99 |
| ieee118 | DQN | 0.00 → 0.32 | 0.01 → 0.04 | 0.01 → 0.51 | 0.00 → 0.56 | 1.00 / 1.00 / 0.99 → 1.00 / 1.00 / 0.99 |

## What the secured meters do for estimation and localization

The residual test is one consumer of a trusted set. `tm.secured_copy(ds, out, name)` writes a copy
of the timeline in which the selected meters read their benign value on every frame (the attacker
locked out of them, the tamper masks and the stored temporal features following), so every
estimator and localizer of the other two guides can be scored with the set secured;
`python docs/trust/run_secured.py` (set `FG_SYSTEM`, `FG_K`) does that for the greedy and the DQN
selections of `run_trust.py` and writes `results/secured_<system>.json`.

One rule came out of it: a localization gate must never down-weight a secured meter.
`GatedPrior` scales every meter of a flagged bus by a thousandth, and on a secured copy that
included the secured meters, the only true readings inside the attacked region, so the CNN and
residual gates were worse than no gate on both secured copies. `GatedPrior(secured=tm.select())` keeps them at full weight, and that arm wins.

IEEE-14, 20 secured meters, test split, node-F1 with the detection rate in parentheses:

| localizer | copy | Aq | At | Al | Am | macro-F1 / benign FA |
|---|---|---|---|---|---|---|
| residual | plain | 0.01 (0.04) | 0.01 (0.05) | 0.01 (0.04) | 0.01 (0.05) | 0.388 / 0.0041 |
| residual | greedy | 0.48 (0.96) | 0.13 (0.33) | 0.59 (1.00) | 0.47 (0.94) | 0.495 / 0.0041 |
| residual | DQN | 0.45 (0.99) | 0.24 (0.56) | 0.66 (0.99) | 0.73 (0.93) | 0.541 / 0.0041 |
| 1D CNN + Jacobian | plain | 0.86 (1.00) | 0.81 (0.88) | 1.00 (1.00) | 0.99 (0.99) | 0.886 / 0.0158 |
| 1D CNN + Jacobian | greedy | 0.79 (1.00) | 0.78 (0.85) | 1.00 (1.00) | 0.99 (0.99) | 0.859 / 0.0165 |
| 1D CNN + Jacobian | DQN | 0.77 (1.00) | 0.77 (0.86) | 1.00 (1.00) | 0.99 (0.99) | 0.880 / 0.0169 |

The trusted set opens the residual localizer on the stealthy families, the DQN's set more than the
greedy's on `At`, `Al` and `Am`, and does not help the learned localizer, which already finds those
families from their onset: its `Aq` node-F1 drops from 0.86 to 0.79 and 0.77 on the secured copies.

Angle mean absolute error in degrees, geometric mean over the eight record classes:

| estimator | plain | greedy | DQN |
|---|---|---|---|
| prior + Huber | 0.049 | 0.039 | 0.033 |
| prior + Huber + CNN gate | 0.044 | 0.040 | 0.036 |
| prior + Huber + CNN gate, secured meters exempt | | 0.024 | 0.022 |
| prior + Huber + residual gate, secured meters exempt | | 0.030 | 0.025 |
| prior + Huber + oracle gate | 0.041 | 0.037 | 0.036 |
| prior + Huber + oracle gate, secured meters exempt | | 0.025 | 0.023 |

The DQN's 20 meters take the proposed estimator from 0.049 to 0.033 degrees with no gate and to
0.022 with the exempting gate, the stealthy families losing 53 to 82 percent of their error (Aq 0.248
to 0.045, At 0.077 to 0.030, Al 0.047 to 0.022, Am 0.049 to 0.022) with the benign error unchanged;
the residual gate, which needs no learned model, comes close to the CNN gate there (0.025) because
the secured meters made the residual see those families.

IEEE-118, the same 20 meters (20 of its 720 metered channels, under 3 percent, where they were 20 of
IEEE-14's 82, 24 percent):

| localizer | copy | Aq | At | Al | Am | macro-F1 / benign FA |
|---|---|---|---|---|---|---|
| residual | plain | 0.01 (0.46) | 0.01 (0.48) | 0.01 (0.50) | 0.02 (0.50) | 0.198 / 0.0075 |
| residual | greedy | 0.01 (0.47) | 0.01 (0.48) | 0.04 (0.56) | 0.04 (0.57) | 0.202 / 0.0075 |
| residual | DQN | 0.07 (0.74) | 0.04 (0.55) | 0.23 (0.83) | 0.25 (0.91) | 0.218 / 0.0075 |
| 1D CNN + Jacobian | plain | 0.57 (1.00) | 0.20 (0.83) | 0.84 (1.00) | 0.80 (1.00) | 0.421 / 0.0107 |
| 1D CNN + Jacobian | greedy | 0.55 (1.00) | 0.20 (0.85) | 0.83 (1.00) | 0.80 (1.00) | 0.395 / 0.0111 |
| 1D CNN + Jacobian | DQN | 0.55 (1.00) | 0.20 (0.85) | 0.83 (1.00) | 0.78 (1.00) | 0.414 / 0.0103 |

| estimator | plain | greedy | DQN |
|---|---|---|---|
| prior + Huber | 0.0106 | 0.0106 | 0.0104 |
| prior + Huber + CNN gate | 0.0108 | 0.0108 | 0.0114 |
| prior + Huber + CNN gate, secured meters exempt | | 0.0107 | 0.0106 |
| prior + Huber + residual gate, secured meters exempt | | 0.0189 | 0.0191 |
| prior + Huber + oracle gate | 0.0103 | 0.0103 | 0.0103 |
| prior + Huber + oracle gate, secured meters exempt | | 0.0101 | 0.0097 |

The greedy set barely moves the residual test on IEEE-118 (Al from 0.50 to 0.56 detected, Am 0.50 to
0.57, macro-F1 0.198 to 0.202) while the DQN set opens it on the stealthy families (Aq from 0.46 to
0.74 detected, Al 0.50 to 0.83, Am 0.50 to 0.91), but the residual localizer still points poorly on
a grid this size, the learned localizer does not move (0.421 to 0.414), and the estimator gains 2
percent with no gate (0.0106 to 0.0104). Twenty meters were a quarter of IEEE-14's channels and are under 3
percent of IEEE-118's; the budget has to scale with the system for the estimation gain to follow.

Source: the multi-snapshot attack and the trusted-PMU defence of [WU26] (`docs/reference/REFERENCES.md`).
