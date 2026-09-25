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
selection, the attack-cost curve, and the detection table. Read the detection table first. The
attack cost is the sparsest row of an echelon basis of the open subspace, an upper bound on the
true sparsest attack, so it is not monotone in the secured set. On IEEE-14 the greedy cost
plateaus at 54 and 55 from the second meter to the twentieth. The DQN selection's cost after each
of its twenty meters is 3 for the first nine, then 45, 49, 11, 11, 11, 11, 4, 7, 7, 7 and 16, yet
its detection of Aq, Al and Am rises from 0.00 to 0.86, 0.64 and 0.72.

WLS residual detection rate per family at a 1% benign alarm level, before and after securing 20
meters (data release v0.8.3, test split). The residual test's estimator is calibrated from
measurements only (`calibrate="measured"`):

| system | selector | Aq | At | Al | Am | Ad / As / Ar |
|---|---|---|---|---|---|---|
| ieee14 | greedy | 0.00 → 0.76 | 0.00 → 0.00 | 0.00 → 0.72 | 0.00 → 0.55 | 1.00 / 1.00 / 0.81 → 1.00 / 1.00 / 0.75 |
| ieee14 | DQN | 0.00 → 0.86 | 0.00 → 0.23 | 0.00 → 0.64 | 0.00 → 0.72 | 1.00 / 1.00 / 0.81 → 1.00 / 1.00 / 0.77 |
| ieee118 | greedy | 0.01 → 0.01 | 0.01 → 0.01 | 0.00 → 0.00 | 0.01 → 0.01 | 1.00 / 1.00 / 0.98 → 1.00 / 1.00 / 0.98 |
| ieee118 | DQN | 0.01 → 0.09 | 0.01 → 0.01 | 0.00 → 0.09 | 0.01 → 0.14 | 1.00 / 1.00 / 0.98 → 1.00 / 1.00 / 0.98 |

## What the secured meters do for estimation and localization

The residual test is one consumer of a trusted set. `tm.secured_copy(ds, out, name)` writes a copy
of the timeline in which the selected meters read their benign value on every frame (the attacker
locked out of them, the tamper masks and the stored temporal features following), so every
estimator and localizer of the other two guides can be scored with the set secured;
`python docs/trust/run_secured.py` (set `FG_SYSTEM`, `FG_K`) does that for the greedy and the DQN
selections of `run_trust.py` and writes `results/secured_<system>.json`.

Rule: a localization gate never down-weights a secured meter.
`GatedPrior` scales every meter of a flagged bus by a thousandth. On a secured copy those meters
include the secured ones, the only true readings inside the attacked region, so a plain gate throws
away the evidence the trusted set provides. `GatedPrior(secured=tm.select())` keeps the secured
meters at full weight, and that arm wins on IEEE-14.

IEEE-14, 20 secured meters, test split, node-F1 with the detection rate in parentheses:

| localizer | copy | Aq | At | Al | Am | macro-F1 / benign FA |
|---|---|---|---|---|---|---|
| residual | plain | 0.00 (0.03) | 0.00 (0.03) | 0.00 (0.03) | 0.01 (0.04) | 0.376 / 0.0017 |
| residual | greedy | 0.57 (0.94) | 0.13 (0.31) | 0.68 (1.00) | 0.60 (0.93) | 0.555 / 0.0017 |
| residual | DQN | 0.49 (0.95) | 0.11 (0.46) | 0.41 (0.77) | 0.35 (0.88) | 0.497 / 0.0017 |
| 1D CNN + Jacobian | plain | 0.89 (0.88) | 0.60 (0.71) | 0.98 (0.99) | 0.81 (0.77) | 0.835 / 0.0167 |
| 1D CNN + Jacobian | greedy | 0.87 (0.96) | 0.67 (0.73) | 1.00 (1.00) | 0.98 (0.98) | 0.826 / 0.0139 |
| 1D CNN + Jacobian | DQN | 0.87 (0.97) | 0.67 (0.75) | 0.99 (1.00) | 0.96 (0.96) | 0.854 / 0.0155 |

The trusted set opens the residual localizer on the stealthy families, the greedy set more than the
DQN's on `Aq`, `Al` and `Am` (macro-F1 0.376 to 0.555 and 0.497). The learned localizer already
finds the one-frame `Aq` without it; the trusted set lifts the sustained families, `Am` node-F1 from
0.81 to 0.98 and 0.96 and `At` from 0.60 to 0.67.

Angle mean absolute error in degrees, geometric mean over the eight record classes:

| estimator | plain | greedy | DQN |
|---|---|---|---|
| prior + Huber | 0.050 | 0.040 | 0.031 |
| prior + Huber + CNN gate | 0.044 | 0.041 | 0.033 |
| prior + Huber + CNN gate, secured meters exempt | | 0.026 | 0.017 |
| prior + Huber + residual gate, secured meters exempt | | 0.030 | 0.023 |
| prior + Huber + oracle gate | 0.041 | 0.038 | 0.030 |
| prior + Huber + oracle gate, secured meters exempt | | 0.026 | 0.017 |

The DQN's 20 meters take the proposed estimator from 0.050 to 0.031 degrees with no gate and to
0.017 with the exempting CNN gate, which matches the oracle gate. The stealthy families lose 65 to
91 percent of their error (Aq 0.355 to 0.033, At 0.066 to 0.019, Al 0.048 to 0.017, Am 0.046 to
0.016) with the benign error unchanged (0.0129 to 0.0130). The residual gate, which needs no
learned model, reaches 0.023 there because the secured meters made the residual see those families.

IEEE-118, the same budget of 20 meters (20 of its 720 metered channels, under 3 percent, where they
were 20 of IEEE-14's 82, 24 percent):

| localizer | copy | Aq | At | Al | Am | macro-F1 / benign FA |
|---|---|---|---|---|---|---|
| residual | plain | 0.01 (0.47) | 0.01 (0.54) | 0.01 (0.50) | 0.01 (0.53) | 0.231 / 0.0072 |
| residual | greedy | 0.02 (0.50) | 0.01 (0.54) | 0.02 (0.52) | 0.02 (0.55) | 0.233 / 0.0072 |
| residual | DQN | 0.07 (0.58) | 0.02 (0.56) | 0.05 (0.61) | 0.07 (0.64) | 0.238 / 0.0072 |
| 1D CNN + Jacobian | plain | 0.64 (0.98) | 0.13 (0.83) | 0.80 (1.00) | 0.15 (0.69) | 0.360 / 0.0116 |
| 1D CNN + Jacobian | greedy | 0.66 (0.97) | 0.12 (0.82) | 0.82 (1.00) | 0.23 (0.66) | 0.355 / 0.0119 |
| 1D CNN + Jacobian | DQN | 0.57 (0.95) | 0.10 (0.78) | 0.79 (1.00) | 0.32 (0.81) | 0.370 / 0.0130 |

| estimator | plain | greedy | DQN |
|---|---|---|---|
| prior + Huber | 0.0102 | 0.0102 | 0.0101 |
| prior + Huber + CNN gate | 0.0103 | 0.0103 | 0.0103 |
| prior + Huber + CNN gate, secured meters exempt | | 0.0101 | 0.0101 |
| prior + Huber + residual gate, secured meters exempt | | 0.0156 | 0.0153 |
| prior + Huber + oracle gate | 0.0100 | 0.0100 | 0.0100 |
| prior + Huber + oracle gate, secured meters exempt | | 0.0095 | 0.0097 |

On IEEE-118 twenty meters barely move anything. The DQN set raises the residual localizer's
stealthy detection rates by 2 to 11 points and its macro-F1 from 0.231 to 0.238, the learned
localizer's `Am` node-F1 rises from 0.15 to 0.32 while its `Aq` falls from 0.64 to 0.57, and the
estimator gains 1 percent with no gate (0.0102 to 0.0101). Twenty meters were a quarter of
IEEE-14's channels and are under 3 percent of IEEE-118's; the budget has to scale with the system
for the gains to follow.

Source: the multi-snapshot attack and the trusted-PMU defence of [WU26] (`docs/reference/REFERENCES.md`).
