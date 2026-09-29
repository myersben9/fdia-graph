# Plan: [WU26]'s trusted-PMU defense as the paper defines it

**Status: accepted 2026-09-29, with these changes: k = 1.2 default (k = 1.1 only for Table II), 1-minute snapshots, the DQN state and hyperparameters as stated in Fig. 1 and Sec. V, parallel harness. PR A in progress.**
- E1 is now fixed by the paper: `Δx_t` is the state deviation, and trust is cumulative from its slot.
- E2, E3 and E13-E15 were added or revised on the faithful re-prototype (section 3).
- The other decisions are open (section 7).

[WU26] defends against its multi-snapshot attack by making a few PMUs trusted (encrypted, so they cannot be
tampered with), one PMU per attack snapshot. It chooses them with two solvers:
- Solution 1, a row-transformation heuristic;
- Solution 2, a DQN over an MDP whose reward is the rise in the attacker's eq. (12)/(28) cost.

Our `fdia_graph.trust` implements something related but different: it secures meters rather than
PMUs, on a single-snapshot linear attack cost, statically. This plan does three things:
- extracts everything the paper states about the defense (section 1);
- tabulates where our code differs (section 2);
- proposes how to follow the paper so that its tables and figures can be regenerated and compared
  (sections 3 to 6).

Anything the paper does not state is marked UNSTATED, and every choice we make in its place is
labelled ours.

Source: S. Wu et al., IEEE Trans. Smart Grid 17(1), Jan. 2026, pp. 651-665. Page numbers are the
journal's.

## 1. What the paper states

### 1.1 The trusted PMU and the secure set (Sec. IV-A to IV-C, pp. 655-656)

- **What trusted means.** A trusted PMU cannot be tampered with (p. 655, and Sec. V, p. 658: "all
  configured PMUs ... can be converted into trusted PMUs through dynamic encryption ... thereby
  preventing tampering").
- **The secure set.** The secure set `h^S` holds the trusted PMUs' data: "voltage, phase angle, and
  branch current measurements, which are processed through the branch measurement conversion method
  to estimate neighboring node voltages" (p. 655). `h^S'` holds the non-trusted PMUs and all SCADA
  (eqs. 26-27). In eq. (27), `h^S` is the PMU voltage rows `-[H_P3 0; 0 H_P4]`: angle and
  `V^-1 dV` of the secured PMU nodes. Adopted (E2): a trusted PMU pins those two rows, its own |V| and
  θ, and its branch currents stay untrusted; the p. 655 reading, in which the currents also pin the
  converted voltages of its neighbours, is run as a sensitivity.
- **What Δx_t is.** Eq. (26), p. 655, is the SE linearization
  `[ΔP; ΔQ; (Δδ_PMU)_S'; (V^-1 ΔV_PMU)_S'; (Δδ_PMU)_S; (V^-1 ΔV_PMU)_S] = -H [Δδ; ΔV/V]`. So
  `Δx_t = [Δδ_t; ΔV_t/V_t]`. Eqs. (17)-(18) (`V_i,t + ΔV^a_i,t = V^a_i,t`,
  `δ_i,t + Δδ^a_i,t = δ^a_i,t`) and the nomenclature (p. 651) make it the attack's state deviation
  `x^a_t - x_t` at snapshot t. It is not an increment between snapshots.
- **The attack with a defense in place,** eqs. (28)-(32):
  - (28) `min || sum_{t=κ}^{κ+T} h^S'_t Δx_t ||_0`: the l0 norm of the attack over the window, with
    the sum inside the norm, as the Algorithm 1 image on p. 658 also shows.
  - (29) `h^S_t Δx_t = 0`: the secure rows are untouched exactly at snapshot t, with no noise
    threshold.
  - (30) `h^S'_t = h_{t-1} - h^S_t` and (31) `h^S_t = h^S_{t-1} + Δh^S_t`: trust accumulates. At
    snapshot t, every PMU trusted at or before t is pinned, and a PMU trusted later is still
    tamperable at t.
  - (32) `Δh^S_t = -[∂z^S_t/∂δ 0; 0 ∂z^S_t/∂(ΔV/V)]`: the new trusted PMU's rows, one angle row and
    one magnitude row.
  - The other constraints are (13)-(25), unchanged.
- **What a trusted PMU pins.** The paper supports two readings:
  - Eq. (27) is `h^S = -[H_P3 0; 0 H_P4]`, the `(Δδ_PMU)_S` and `(V^-1 ΔV_PMU)_S` rows only.
  - p. 656 says, verbatim, "In the n-th time period, the rank of hS becomes 2n × 2."
  - p. 655 says, verbatim, the elements of `h^S` are "voltage, phase angle, and branch current
    measurements, which are processed through the branch measurement conversion method to estimate
    neighboring node voltages".
  - Eq. (27), eq. (32) and the rank statement read as two rows per trusted PMU: its own |V| and θ.
    The p. 655 sentence reads as its currents also pinning the converted voltages of its neighbours.
    See E2.
- **Defense objective,** eq. (33): `max_S ( min ||Σ h^S'_t Δx_t||_0 - min ||Σ (z^a_t - h_A(x^a_t))||_0 )`.
  That is the rise in the attacker's minimum l0 cost, counted in measurements (rows of `h^S'`), that
  the trusted schedule buys against the undefended eq. (12) cost. It is the reward (p. 657) and the
  unit of "attack cost increasing" in Table II and Fig. 12.
- **Security level.** Configuration continues "until `h^S` is of full rank" (p. 656). In the
  experiments the budget is the PMU count: 4 on IEEE-14, 11 on IEEE-118.
- **Cost unit.** Eq. (33) is an l0 count over measurements, so the % is over channels. The results
  report extra devices alongside it ("extra-tampered SCADA", Table II; the histogram in Fig. 12).
  Table II's % cannot be devices. With his own counts:
  - Scenario 1: 7 devices plus 3 extra is +42.9%, where he reports 25.6% and 23.9%.
  - Scenario 2, Solution 1: 9 plus 5 is +55.6%, where he reports 35.2%.
  - Scenario 2, Solution 2: 9 plus 3 is +33.3%, where he reports 27.0%.

  This is consistent with channels, but his channel counts are not given, so it cannot be checked
  exactly.
- **The goal over the window,** eqs. (24)-(25):
  - (24) `sqrt((P_l,t+ΔP)^2 + (Q_l,t+ΔQ)^2) = S_l,t-1 + ΔS_l,t` defines the reported flow snapshot
    by snapshot.
  - (25) `S_l,κ + Σ_{t=κ+1}^{κ+T} ΔS_l,t ≥ S^max_l` requires only the accumulated flow at the
    window's end to reach the rating.
  - No per-snapshot path is required. Our linear ramp to the rating (D9) is ours.
  - On IEEE-14 all four PMUs are trusted by snapshot 8 of 20, so the overload must be reached with
    all four pinned.
- **Noise exclusion.** Sub-noise tampering is excluded from the l0 count (p. 659), as in our D8/D11.

### 1.2 Solution 1: row transformation (Sec. IV-D1, pp. 656-657)

Per dynamic attack interval:
1. Compute the Jacobian `h` and transform it to a full-row-rank `h_T`.
2. Reduce `h_T` to RREF by row swap, scaling and row addition.
3. Find the RREF row with the fewest nonzeros. Move its nonzeros to the end by column exchanges,
   keeping a tracking matrix of the exchanges.
4. Repeat 2 and 3 until that row and its nonzero count stop changing.
5. Multiply the sparsest row by the inverse of the tracking matrix. The result is the approximate
   sparsest attack.
6. Secure the PMUs of that attack's measurements (add them to `h^S`), and repeat on `h^S'` "until the
   desired security level".

The text uses the transpose of `h` [25] (p. 655). "Mathematical optimization" and "determinant
transformation" are named in the results (pp. 660, 663), but no formula is given for them: UNSTATED.
Step 5 secures "the PMUs corresponding to these measurements" (p. 657), so every PMU on the sparsest attack's support; with one configuration per slot in the case studies, which of several goes first is UNSTATED.

### 1.3 Solution 2: the MDP and the DQN (Sec. IV-D2, pp. 657-658; Sec. V, p. 658; Appendix, p. 663)

| item | stated value | where |
|---|---|---|
| MDP | tuple (O, U, P, R, γ) | p. 657 |
| state O_t | STATED in Fig. 1: the measurements (P_i,t, Q_i,t, P_ij,t, Q_ij,t, V_i,t, θ_i,t), the load rate of the target line S_ij, and the PMU/trusted-PMU configuration on the graph ("different states of the measurement matrix, varied h^S and h^S'"); only its tensor layout is not given | p. 657, Fig. 1 |
| action U_t | "selects one PMU per step for the trusted configuration" | p. 657 |
| transition | P(O_t+1 \| O_t, U_t), no model given; the loop "iterates over time snapshots" | p. 657 |
| reward R | "the increase in attack cost before and after adding trusted PMUs", i.e. eq. (33) per step | p. 657 |
| episode | Algorithm 1: for each iteration, initialize the state and generate attack vectors, then run steps t = 1..K. Break when defended cost < undefended cost (line 8, verbatim; see E6), otherwise store the reward | p. 658 |
| policy | u_t = argmax_u Q_θ(o_t, u) (34), (38); ε-greedy exploration | pp. 657-658 |
| loss | MSE to target (35); target y = r + γ max Q_θ'(o', u') (36); gradient step (37) | p. 658 |
| replay | experience replay, buffer D; Algorithm 1 line 15 "Select the smallest buffer sample from replay buffer D" (read as minibatch sampling) | p. 658 |
| target net | updated "every 20 iterations" (Alg. 1 lines 17-18: "if t mod d") | p. 658 |
| framework | TensorFlow 2.5, Adam | p. 658 |
| γ | 0.9 (sensitivity range 0.85-0.98 reasoned in the Appendix) | pp. 658, 663 |
| learning rate η | 0.005, chosen from {0.0005, 0.001, 0.005, 0.01} by Fig. 13 | pp. 658, 663 |
| buffer \|D\| | 2,500 | p. 658 |
| minibatch \|B\| | 25 | p. 658 |
| exploration | ε = e^(-0.002·ep) | p. 658 |
| episodes | about 250 (Fig. 13's x axis) | p. 663 |
| network architecture | UNSTATED: Fig. 1 draws a fully connected net but gives no sizes or activations | Fig. 1 |
| offline/online | trained offline on "dynamic grid scenarios", then deployed to new scenarios (Fig. 2) | p. 658 |
| "100 tests" | 14-bus: the time distributions (Figs. 5, 8). 118-bus: "each training session via Solution 2 is not identical", so the 100 tests are 100 training sessions (Figs. 11-12) | pp. 659, 662 |

### 1.4 The case studies and every number (Sec. V, pp. 658-663)

**Setup.**
- SE runs every 5 min, and attack snapshots are 30 s apart (p. 658).
- IEEE-14:
  - "10 minutes of power flow", i.e. 20 attack intervals (Fig. 4, p. 659);
  - PMUs at 1, 4, 6, 13;
  - loads scaled by U[0.9, 1] with per-load N(0, 0.03);
  - noise 0.03 pu SCADA and 0.01 pu PMU.
- IEEE-118:
  - "a 10-minute interval";
  - Fig. 10 numbers 10 configurations and Fig. 11 shows 10 snapshots, although 30 s snapshots over
    10 min would give 20 (UNSTATED which);
  - 11 PMUs: 76, 78, 80, 83, 89, 92, 94, 100, 105, 106, 110 (Fig. 9, Fig. 11 axis);
  - one trusted PMU per snapshot (p. 662).
- IEEE-14 trusted schedule: "one trusted PMU configuration operation at snapshots 2, 4, 6, and 8"
  (p. 659).

**Results.**

| item | value | where |
|---|---|---|
| attack, S1 lines 3-4 & 6-11 | SCADA 3,4,5,6,11 + PMU 4,6 (7 devices), max 0.22 pu | p. 659, Fig. 4 |
| attack, S2 lines 1-2 & 4-5 | SCADA 1,2,3,4,5,9 + PMU 1,4,6 (9 devices), max 0.17 pu | p. 659, Fig. 4 |
| attack time 14-bus | S1 3.8 s mean (1.1-5.3); S2 4.6 s (1.2-7.5); 100 tests per snapshot | p. 659, Fig. 5 |
| trusted sequence 14-bus (both solutions) | S1: 1→4→6→13; S2: 4→6→1→13 | p. 660, Fig. 6 |
| attack after defense | per-device boxplots over the 20 intervals; PMU channels still tampered before each is trusted | Fig. 7 |
| Table II extra SCADA | S1: Sol 1 {1,9,13}, Sol 2 {1,9,13}; S2: Sol 1 {7,11,12,13,14}, Sol 2 {7,13,14} | p. 660 |
| Table II cost increase | S1: Sol 1 25.6%, Sol 2 23.9%; S2: Sol 1 35.2%, Sol 2 27.0% | p. 660 |
| defense time 14-bus | S1: Sol 1 7.5 s (4.1-9.9), Sol 2 5.8 s (2.5-10.1); S2: Sol 1 7.3 s (2.4-12.7), Sol 2 6.3 s (2.3-10.5) | p. 660, Fig. 8 |
| Table III (detection method: detection rate / second column / extra time) | residual WLS, no PMU support: 37.4 / 22.1 / none; matrix-based: 88.6 / 7.1 / 7.4 s; DRL-based: 94.2 / 4.6 / 6.1 s. Both columns are headed "Detection rate (%)"; the second is read as the false-positive rate. The detector that uses the trusted PMUs is UNSTATED | p. 661 |
| Table IV attack solve time 118 | single snapshot 1.1 s (0.8-2.2); MS 6.2 s (4.7-8.0) | p. 661 |
| Table V attack effect 118 | single: SCADA 75,83,84,85,89,92,99,100,104,107,110 and PMU 83,89,92,94,100,105,110; DR 92.3%, FPR 11.9%. MS: SCADA 84,85,89,94,99,100,101,103,105 and PMU 83,89,100,105,106,110; DR 21.6%, FPR 37.6% | p. 661 |
| Fig. 10 sequence 118 (Sol 1) | read from the figure: 100 → 83 → 80 → 94 → 89 → 92 → 105 → 110 → 106 → 78. The labels at 92/94 (steps 4 and 6) are hard to separate; 76 is never chosen | p. 661 |
| Fig. 11 selection probability 118 (Sol 2, 100 tests) | snapshot 1: diffuse (100 ≈ 0.2, 83 ≈ 0.15). Snapshots 2-3: 80, 83, 100 at 0.25-0.28. Snapshots 4-6: 89, 92, 94 ≈ 0.22. Snapshots 7-8: 105, 110 ≈ 0.30. Snapshots 9-10: 76, 78 ≈ 0.30 and 106 ≈ 0.27 | p. 661 |
| Fig. 12 118 (Sol 2, 100 tests) | cost increase mean 16.4%, most 10-20%. Extra devices histogram, read from the figure: 0: ~3, 1: ~9, 2: ~21, 3: ~38, 4: ~24, 5: ~7; median 3 ("3-5 additional") | p. 662 |
| defense time 118 | Sol 1 16.5 s (11.3-22.4); Sol 2 8.7 s (6.9-13.6); Sol 2 47.3% faster | p. 662 |
| Table VI mean time | matrix: 1354-bus 43.9 s, 118-bus 16.5 s, 14-bus 7.4 s; DRL: 11.4, 8.7, 6.1 s | p. 662 |
| Table VII | qualitative comparison | p. 663 |
| Fig. 13 | reward vs episode (0-250) for 4 learning rates; 0.005 converges to about 4.7 | p. 663 |

## 2. Our implementation against the paper

State of `origin/main` at df53f90:
- `trust/base.py`: `TrustSelector`, `TrustedMeters`;
- `trust/dqn.py`: `TrustedMetersDQN`;
- `formulas/trust.py`: `attack_cost`, `greedy_trusted_meters`;
- `trust/secured.py`.

| aspect | [WU26] | ours | gap |
|---|---|---|---|
| what is trusted | a PMU's own \|V\| and θ rows (eqs. 27, 32; its branch currents stay in `h^S'`), the neighbour reading of p. 655 as a sensitivity (E2) | any single metered channel (a row of H, SCADA included) | wrong unit |
| candidates | the PMU buses (4 on 14, 11 on 118) | every metered channel (m ≈ 80-500) | wrong action space |
| attack cost | eq. (28): AC multi-snapshot l0 with the overload goal (24)-(25) and (13)-(23) | sparsest RREF row of the linear single-snapshot attack subspace {H c : H_S c = 0} at the benign mean, with no goal | different problem |
| time | secure set grows one PMU per snapshot inside the attack window (eqs. 30-31) | one static k-set for the whole dataset | no schedule |
| reward | rise in the eq. (33) cost per added PMU | rise in the linear cost, plus a bonus of m when the subspace closes | different cost |
| state | Fig. 1: P_i, Q_i, P_ij, Q_ij, V_i, θ_i, target-line load rate S_ij, trusted configuration on the graph | binary secured-channel vector | missing features |
| network | UNSTATED | MLP m→128→128→m, ReLU, masked Q | ours is an assumption either way |
| hyperparameters | γ 0.9, η 0.005, \|D\| 2500, \|B\| 25, target every 20 iterations, ε = e^(-0.002 ep), MSE, ~250 episodes | γ 0.95, lr 1e-3, buffer 5000, batch 64, target every 10 episodes, ε linear to 0.05 over 60%, smooth-L1, 200 episodes | all differ |
| training data | offline over dynamic grid scenarios, deployed to new ones; 100 training sessions on 118 | one Jacobian at the benign mean of one dataset | no scenario variation |
| Solution 1 | RREF of h_T, column exchanges with a tracking matrix iterated to a fixed point, secure the attack's PMUs, per interval | RREF of the attack subspace, secure the channel on the sparsest row's support that maximizes the next cost (greedy lookahead), static | related, not the same |
| evaluation | cost increase %, extra devices, sequences, times, detection/false-positive rates | residual detection before and after securing | different metrics |
| Jacobian | hybrid linear model with branch-current conversion (eqs. 1-3, 26) | WLS H in [V, P, Q, θ, Pf, Qf] | eq. (3) exists as `pmu_pseudo` since D10 |

The existing classes answer a different, useful question: which meters to secure against any
single-snapshot stealthy attack. The plan keeps them (E11) and adds the paper's defense beside them.

## 3. The attacker's cost with trusted PMUs: the minimizer as the reward

The reward needs the eq. (28) minimum under a trusted schedule. The fewest-tamper search already
solves eq. (12) (`engine/attacks/minimize.py: min_tamper`). What it lacks is constraint (29).

**How (29) enters our false-state model.** A false state moves only the voltages of its support S,
and `Δx_t = x^a_t - x_t`. Under E2's own-bus reading, pinning PMU b's |V| and θ at snapshot t means
b is held at its true voltage at t. So at each snapshot the free set is S minus the buses of the
PMUs trusted at or before t (eqs. 30-31). Under the neighbour reading, b's neighbours are held too.

**Measured with a prototype** of this constraint, then reproduced by PR A's implementation
(`tests/test_trusted_pmus.py` pins the counts).
- **Setup:** IEEE-14 on Wu's metering; 20 snapshots (the paper's 10 min at 30 s, here 20 pool
  frames); Wu's schedule 1→4→6→13 (Scenario 1) and 4→6→1→13 (Scenario 2) at snapshots 2, 4, 6, 8;
  ratings k × peak flow; the D16 bounds.
- **Exhaustive search:** budget 4096 exceeds IEEE-14's candidate count (1174 for Scenario 1, 454 for
  Scenario 2), so every search is exhaustive.
- **Not proven:** some candidate solves failed to converge ("unsolved"), so no result is proven
  minimal. An "infeasible" below means no candidate converged, not a proof.

| scenario, k | undefended: devices / channels | Wu's schedule, own bus: devices / channels | increase: devices / channels | Table II (Sol 1 / Sol 2) |
|---|---|---|---|---|
| S1 (3-4, 6-11), 1.1 | 5 / 8 | 6 / 11 | +20.0% / +37.5% | 25.6% / 23.9% |
| S2 (1-2, 4-5), 1.1 | 7 / 23 | 9 / 28 | +28.6% / +21.7% | 35.2% / 27.0% |
| S1, 1.2 | 8 / 26 | none converged (1174/1174 unsolved) | | |
| S2, 1.2 | 10 / 40 | none converged (454/454 unsolved) | | |
| S1, 1.2, no load cap | 8 / 23 | 9 / 37 | +12.5% / +60.9% | |
| S2, 1.2, no load cap | 9 / 32 | none converged (454/454 unsolved) | | |

- **Range:** at k = 1.1 both scenarios survive the defense, and the increases are in Table II's
  range.
- **Scenario 1 devices (k = 1.1):**
  - undefended: SCADA 2, 3, 4, 5 and PMU 4;
  - defended: SCADA 1, 2, 4, 5 and PMU 1, 4;
  - extra: SCADA 1 and PMU 1.

  The paper's extra devices are SCADA 1, 9, 13.
- **Scenario 2 devices (k = 1.1):**
  - undefended: SCADA 1-5 and PMU 1, 4;
  - defended: adds SCADA 6 and PMU 6.

  The paper's extra devices are SCADA 7, 11, 12, 13, 14 (Solution 1) or 7, 13, 14 (Solution 2).
- **Largest change:** 0.25 pu (Scenario 1) and 0.35 pu (Scenario 2) with the defense, against
  0.19 pu and 0.24 pu undefended.
- **Why k = 1.2 fails (the last snapshot solved alone with all four PMUs pinned on the largest possible support):**
  - The failure is at the window's end, where (25) itself binds, so it is not our ramp.
  - Scenario 1 converges once the load cap (ours, D16 [YUA11]) is removed.
  - Scenario 2 needs the generator and voltage limits (21)-(23), which are Wu's, removed as well.
  - So at k = 1.2 the defense does stop the attack under Wu's own operating limits. The ratings, and
    so k, are ours, since the paper states none.
- **The neighbour reading of E2** makes both scenarios infeasible at k = 1.1 (all 1174 and 444 of
  454 unsolved).
- **Order does not matter under a held support:** Wu's order and the swapped order give identical
  results. The defended optimal supports (2 3 5 11 and 2 5 6 11) avoid every PMU bus for the whole
  window, so the schedule acts only through its final set. Wu's attack chooses `Δx_t` per snapshot
  (the sum is inside (28)), and his Fig. 4 narrative has the tampered set shrink over the window.
  Our search holds one support for the window, which is ours (E13).
- **PMU 4 still counts as tampered** in the Scenario 1 defense under the own-bus reading. Its
  |V| and θ are pinned, but its branch-current channels at bus 4's end move when buses 3 and 5 move.
  Under eq. (27), those currents are not in `h^S`.

**Per-call time:** 16-35 s per 20-snapshot IEEE-14 search, exhaustive at budget 4096. IEEE-118 took
42-43 s at budget 4096 for a 10-snapshot window in the first prototype,
and about 3.1 s at budget 256 (#164's table).

**Caching.** The cost depends only on the window (its states and goal) and the trusted schedule, so
the key is (window id, tuple of (snapshot, PMU)).
- A DQN episode revisits prefixes constantly.
- IEEE-14's whole space is 4 PMUs over 4 slots: 64 schedule prefixes, so 64 solves per window.
- On 118, 11 PMUs over 10 snapshots is too many to enumerate, but a training run visits at most
  episodes × steps = 2,500 transitions.

The cache lives in the environment, not the minimizer.

## 4. What changes

1. **`min_tamper` takes a trusted schedule.** This is a new validated model, `TrustSchedule`: the
   PMU buses trusted per snapshot slot.
   - At snapshot t the free set is S minus the buses (E2) of the PMUs trusted at or before t, and
     those buses are held at their true voltage (`Δx_t = 0` there; eqs. 29-31).
   - E13 decides whether the support stays held for the window or is chosen per snapshot.
   - `MinimizerResult` already reports devices and channels, the latter the l0 of eq. (28) (E3).
2. **`trust.WuDefenseEnv`, the MDP.**
   - State: the paper's three feature groups, plus the secured pattern (E7).
   - Action: an untrusted PMU.
   - Reward: `cost(schedule + u_t) - cost(schedule)`, from the cached minimizer (eq. 33 per step).
   - The steps are the configuration slots (E5).
3. **`trust.TrustedPMUs`, Solution 1** as Sec. IV-D1 states it. It runs RREF on `h_T`, then
   column exchanges with a tracking matrix iterated to a fixed point. It recovers the sparsest
   attack, secures its PMUs, and repeats per slot. Where the paper is UNSTATED (which of several PMUs
   goes first when a slot takes one, and how `h` is made full-row-rank), the choice is ours and labelled.
4. **`trust.TrustedPMUsDQN`, Solution 2** as Algorithm 1 states it: MSE loss, the Sec. V
   hyperparameters, ε = e^(-0.002 ep), replay 2,500, batch 25, target every 20 iterations, about 250
   episodes, trained over scenario windows (E8). The network architecture is ours (E9).
5. **Scoring as the paper does:**
   - cost increase % (devices and channels);
   - extra-tampered devices;
   - the sequence;
   - solve time;
   - detection rate and false-positive rate. The residual test with the trusted PMUs reading truth
     is what `TrustSelector.score` does today (E10).

## 5. The reproduction harness (docs/wu26/, generated, CSV sidecars)

**Scenarios:**
- IEEE-14 S1 (3-4, 6-11) and S2 (1-2, 4-5), PMUs 1, 4, 6, 13, 20 snapshots, trust slots 2, 4, 6, 8;
- IEEE-118 (84-85, 99-100) on `WU26_ATTACK_AREA[118]`, its 11 PMUs, 10 snapshots, one slot each (E5).

Ratings are k × each target line's peak true flow (the paper states none; ours). Default k = 1.2 for
everything; k = 1.1 is run only for the Table II comparison on the two IEEE-14 scenarios (E14).
Attack snapshots are 1 minute, interpolated from the 5-minute pool (E4).

| paper item | regenerated as | figure form (Ben's rules: no legend, no in-plot text, caption carries keys) |
|---|---|---|
| Fig. 4 / Fig. 7 | per-device attack magnitude over the window, before and after the schedule | device × snapshot heatmap per scenario, small multiples before/after |
| Fig. 5, Fig. 8, Table VI | attack and defense solve times, 100 windows | dumbbell of min-mean-max per method and system |
| Fig. 6, Fig. 10 | the trusted sequence | numbered overlay on the system diagram (the paper's own form) |
| Table II | extra-tampered SCADA, cost increase % (devices and channels), Sol 1 vs Sol 2 | table |
| Table III, Table V | detection rate and false-positive rate of the residual WLS test: undefended, Sol 1, Sol 2; single snapshot (legacy recipe) vs MS | table |
| Table IV | single vs MS attack solve time on 118 | table |
| Fig. 11 | selection probability, PMU × snapshot over the DQN tests | heatmap |
| Fig. 12 | cost increase % per test and the extra-devices histogram | ranked dot strip with the 10-20% band shaded, plus the histogram |
| Fig. 13 | reward vs episode for η ∈ {0.0005, 0.001, 0.005, 0.01} | small multiples, one per η |

Every table also gets a "paper" column with the section 1 value, and each figure's caption states the
paper's value.

## 6. Sequence

1. **PR A:** `TrustSchedule` in `min_tamper` (E1-E3, E13), with tests:
   - at k = 1.1, the IEEE-14 scenarios stay feasible under 1→4→6→13 and 4→6→1→13 at slots 2, 4, 6, 8;
   - a trusted PMU's |V| and θ are exactly true from its slot on;
   - the prototype's device and channel counts are reproduced.
2. **PR B:** `WuDefenseEnv` with the cached cost oracle and the state features (E5, E7). Tests check
   reward equals the cost difference and cache hits are identical.
3. **PR C:** `TrustedPMUs` (Solution 1). Test: IEEE-14 sequence reported against the paper's.
4. **PR D:** `TrustedPMUsDQN` (Solution 2, Algorithm 1), with tests: seeded determinism, and a tiny
   case where the optimum is known.
5. **PR E:** the harness and figures in docs/wu26/, run at both rating margins.
6. **PR F:** docs.
   - The guides say which trust classes answer which question.
   - Update the existing `TrustedMeters*` docs so they no longer claim to be [WU26]'s defense.
   - They are its single-snapshot linear analogue.

## 7. Decisions

- **E1. Meaning of (29) over the window.** This is defined by the paper, so it is not a choice.
  - `Δx_t = x^a_t - x_t` (eq. 26 with eqs. 17-18).
  - Trust accumulates (eqs. 30-31): at snapshot t every PMU trusted at or before t is pinned, and a
    later one is still tamperable.
  - Implement exactly that. (An earlier draft recommended a "staged increment" reading. It is wrong
    and withdrawn.)
- **E2. What a trusted PMU pins.**
  - Own |V| and θ only: eq. (27) `h^S = -[H_P3 0; 0 H_P4]`, eq. (32) (one angle row, one magnitude
    row) and p. 656 "the rank of hS becomes 2n × 2".
  - Also its neighbours, through the currents: p. 655, trusted data "including voltage, phase angle,
    and branch current measurements, which are processed through the branch measurement conversion
    method to estimate neighboring node voltages".
  - Measured on IEEE-14 at k = 1.1: the own-bus reading leaves Wu's scenarios feasible, at +20% and
    +29% devices (+38% and +22% channels). The neighbour reading makes both infeasible, against
    Table II.
  - Recommend: own bus (eqs. 27, 32 and the rank statement), with the neighbour reading reported as
    a sensitivity.
- **E3. Cost unit.** Eq. (33) defines the defense effect as the rise in the l0 of eq. (28), which is
  over measurements (rows of `h^S'`).
  - Report the % in channels, and the extra devices separately as Table II and Fig. 12 do.
  - His own device counts give +42.9%, +55.6% and +33.3%, not his 25.6%, 35.2% and 27.0%. So the %
    is not over devices, which is consistent with channels.
- **E4. Snapshot cadence. DECIDED (Ben):** 1-minute attack snapshots interpolated from the 5-minute
  pool. Wu's SE/SCADA interval is also 5 minutes, but his attack snapshots are 30 s (p. 658); ours are
  twice as long, so a 20-snapshot IEEE-14 window spans 20 minutes against his 10. Stated in the harness.
- **E5. Slots on IEEE-118.** Fig. 10 and Fig. 11 show 10 configurations; 10 min at 30 s is 20
  snapshots.
  - Recommend: 10 snapshots with one slot each, matching the figures. At our 1-minute snapshots (E4)
    that is 10 minutes, which also matches the paper's "10-minute interval" for IEEE-118.
- **E6. Algorithm 1 line 8** breaks the episode when the defended cost is below the undefended cost,
  verbatim.
  - Read literally, that can only happen through solver non-optimality.
  - Recommend: implement it as written (end the episode, no reward) and log how often it fires.
- **E7. State. Wu's, STATED in Fig. 1 (p. 657):** the measurements P_i,t, Q_i,t, P_ij,t, Q_ij,t,
  V_i,t, θ_i,t, the load rate of the target lines S_ij, and the PMU/trusted configuration on the graph.
  Implement exactly those; only the flattening into the network's input is ours (per bus and branch
  in a fixed order, the trusted mask as a 0/1 per PMU).
- **E8. The 100 tests on 118** read as 100 training sessions. At about 2,500 cost calls × 3.1 s per
  training run, that is about 2 h each, so about 200 h in total.
  - Option 1: 100 sessions on 100 load windows at budget 256.
  - Option 2: 10 sessions × 10 test windows, about 20 h.
  - Option 3: one policy trained on many windows and 100 test windows, about 5 h.
  - Recommend Option 2, stated as a departure.
- **E9. Network architecture** is UNSTATED (Fig. 1 draws a fully connected net, no sizes). The
  hyperparameters are STATED and used as given: TensorFlow v2.5, Adam, γ 0.9, η 0.005, |D| 2,500,
  |B| 25, target update every 20, ε = e^(-0.002·ep) (p. 658).
  - Recommend, ours: an MLP state → 128 → 128 → |PMUs| with ReLU, masked Q, in torch (the paper
    used TensorFlow; the framework does not matter).
- **E10. Table III's "detection with PMU support"** is UNSTATED.
  - Recommend: the residual WLS test with the trusted PMUs reading truth, as `TrustSelector.score`
    does. Label it ours.
- **E11. Keep the existing classes?**
  - Recommend: keep `TrustedMeters` and `TrustedMetersDQN` as the single-snapshot linear analogue,
    renamed in the docs so nothing claims they are [WU26]'s defense.
  - Add `TrustedPMUs` and `TrustedPMUsDQN` as the paper's, mirroring the legacy-recipe decision.
- **E12. IEEE-1354 (Table VI).** It is not on our ladder.
  - Recommend: omit it, and say so in the harness.
- **E13. One support for the window, or one per snapshot.**
  - Eq. (28) sums `Δx_t` per snapshot inside one l0, and Fig. 4's narrative has the attacker narrow
    the tampered set over the window. The paper does not require one support for the window.
  - Our search holds one support (ours, from the attack plan). Under it the trust order does not
    matter: Wu's and the swapped schedule gave identical results, because the defended support
    avoids every PMU bus.
  - Recommend: allow the support to change at the trusted slots, while still counting the window's
    union of devices as eq. (28) does. Measure whether order then matters, as Table II and Fig. 6
    imply.
  - The cost is a search over a support per slot segment (5 segments on IEEE-14), which is more
    expensive. Prototype before committing.
- **E14. Rating margin for the defense harness.**
  - At k = 1.2 the defended attack does not converge on either scenario under Wu's limits (21)-(23),
    and it needs our load cap off even for Scenario 1.
  - At k = 1.1 both survive with increases in Table II's range.
  - DECIDED (Ben): k = 1.2 is the default for everything; k = 1.1 is run only for the Table II
    comparison on the two IEEE-14 scenarios. The k = 1.2 infeasibility is reported as a finding.
- **E15. Our linear ramp (D9)** is not in (24)-(25), which only require the rating at the window's
  end.
  - The k = 1.2 failures sit at the end snapshot, so dropping the ramp would not change them.
  - Keep D9 for generation, and note it in the harness.

## 8. Compute estimate

The harness runs as parallel processes (the searches are single-threaded since #164), about 26 of the
machine's 52 cores, with the cost cached per (window, trusted schedule). At search budget 256, per core:
- **IEEE-14:**
  - Solution 1: about 10 cost calls per window, under 1 min.
  - DQN: 64 cached schedules per window. 100 windows is 2-3 h in total, or about 17 h at budget
    4096.
- **IEEE-118:**
  - Solution 1: about 65 calls per window, about 3.5 min; 100 windows about 6 h.
  - DQN: about 20 h under E8 option 2, or about 5 h under option 3.
- **k = 1.1** only adds the two IEEE-14 Table II runs (minutes).
- **Total**: about 25-30 core-hours, so roughly 1-2 hours of wall time on 26 processes, with CSVs
  written as it goes so figures can be restyled without rerunning.
