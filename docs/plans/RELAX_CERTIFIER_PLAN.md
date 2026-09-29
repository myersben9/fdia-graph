# Plan: certifying the fewest-tamper attack by convex relaxation

**Status: shipped as an optional analysis tool; a valid but loose lower bound.** [WU26] eq. (12)
asks for the attack that tampers the fewest devices. The fewest-tamper search
(`engine/attacks/minimize.py`) returns a feasible exact-AC attack, so its device count is an upper
bound. It proves that bound optimal only when it exhausts its candidates or meets its forced-device
bound, which with new generation's defaults it seldom does for `Am` (`proven`, 0 to 8% of the
searches in the generation guide's runs). This plan adds a separate certifier: a convex relaxation of the same problem whose
optimum is a lower bound. When the two bounds meet, the search's attack is certified globally
optimal over the attacker's area; otherwise the certificate reports the gap. The certifier is an
analysis tool. Generation never calls it.

## 1. The problem the certifier bounds

The search's problem, for a window of T snapshots with true states x_t, over the attacker's area A
(`local_region(seeds, hops)`, the slack never in it):

- **Unknowns:** the false complex voltage V_i,t of every bus i in A. Every bus outside A keeps its
  true voltage (the search's supports lie in A and hold every other voltage true).
- **Count, eq. (12):** the devices d (one SCADA terminal and one PMU per bus, D1) with a channel m
  whose attack value a_m,t = h_m(V_t) - h_m(V_t^true) exceeds its noise sigma_m,t at some snapshot
  (D8 for `Am`, D7 for `At`). The channels are the metered P and Q injections, from-end flows,
  |V|, angles at PMU buses (D10) and the PMU branch-current channels.
- **Limits (21):** v_lo,i <= |V_i,t| <= v_hi,i, both widened to the true value and by 1e-3 pu as
  in `within_limits`. The relaxation's voltage box and every big-M constant come from these limits,
  so `certify` refuses knobs without finite voltage limits (`NoOperatingLimits`, checked by the
  input model `models.inputs.CertifiableLimits`): the search's voltages are then unbounded, and no
  box the relaxation invented would bound the same problem.
- **Zero injection:** a zero-injection bus keeps its true (zero) injection.
- **Goal, `Am` (24)-(25):** the noiseless apparent from-end flow |S_l,t| of each goal line equals
  its scheduled target at every snapshot (two lines by default, D17).
- **Injection bounds, `Am` (D14, D16):** every generator whose output the attack changes, in the
  support or on its edge, stays inside its limits (22)-(23), and every load bus it changes shows an
  active change of at most `load_cap` times its true load. **Goal, `At`:** each targeted bus's active injection
  moves by the designed load change, its reactive injection not at all, and every metered linear
  channel moves by at most its rated accuracy between snapshots (the stealth bound, D7).

## 2. The relaxation

Per snapshot, with W = V V^H (Jabr's variables):

- **Variables:** W_ii for i in A, a complex W_ij for every connected pair of A-buses, and the
  complex V_i for i in A. A bus outside A has its true V. A W_ij between an A-bus and a fixed bus is
  V_i conj(V_j^true), linear in V_i.
- **Linear physics:** injections S_i = sum_j conj(Y_ij) W_ij, from-end flows
  S_l = sum_n conj(Yf_ln) W_f(l),n and currents I = Yf V, Yt V are linear in (W, V). They are exact
  whenever W is rank one.
- **Cones (the relaxation):** |W_ij|^2 <= W_ii W_jj (rotated second-order cone) and
  |V_i|^2 <= W_ii. Dropping rank one is the only step that enlarges the feasible set.
- **Binaries and big-M:** one binary b_d per device, shared by the snapshots. Each linear channel
  gets -sigma - M b_d <= a <= sigma + M b_d. |V| gets (v - sigma)^2 - M b_d <= W_ii <=
  (v + sigma)^2 + M b_d. An angle gets |Im(V e^{-j theta})| <= tan(sigma) Re(V e^{-j theta}) + M b_d,
  the wedge the angle may not leave (exact for a voltage, since sigma is below 90 degrees).
- **Roundoff slack:** the search classifies channels in float32 (a flow as the difference of its
  false and true readings, each rounded first), so each sigma above is widened by
  2 eps32 (|true reading| + sigma) + 1e-6 (`formulas.relax.roundoff_slack`). At's step bound compares
  attack values of any size (near 100 on a late ramp frame), so each of its endpoints takes the slack
  of the most |a| can be at that snapshot, |A| r over the box, in place of sigma
  (`formulas.relax.step_roundoff_slack`); the frame before the window is data both share and adds
  none. Without these a channel the search counts within noise or within the step bound could lie
  beyond a fixed 1e-6 margin in exact arithmetic, and the relaxation would cut the search's attack.
  Every comparison with a float32-rounded search quantity carries such a margin: the count's linear
  channels (flows with their true reading, injections and currents with sigma), the |V| band, the
  angle wedge, the PMU caps (own |V| and angle, and branch currents) and At's step bound. The
  generator limits, the load cap, the voltage limits and the goal compare float64 quantities the
  search computes from the same stored states, with margins of 1e-3 (MW, MVAr, MVA, pu) against
  solves converged to 1e-9 pu.
- **Rigorous M:** the voltage limits bound W_ii, the cones bound |W_ij| by
  sqrt(W_ii,max W_jj,max) and |V_i| by v_max. M for a channel is sum_k |A_mk| r_k, with r_k the
  largest distance of variable k from its true value inside that box, so no feasible point is cut.
- **Flow goal:** |S_l,t| <= target is a cone. |S_l,t| >= target is not convex; it is replaced by K
  sectors with binaries z_k (sum z_k = 1): u_k . S >= target cos(pi / K) when z_k = 1. Every point
  on the circle lies in some sector, so the disjunction contains the true constraint.
- **Tying W to V where the meters pin V:** on a rank-one W the residual
  X_ij = W_ij - V_i conj(V0_j) - V0_i conj(V_j) + V0_i conj(V0_j) equals dV_i conj(dV_j), with
  dV = V - V^true. A variable r_i >= |dV_i| is capped while a device is untampered: by the bus's
  own PMU (|V| within sigma and angle within sigma put V in a small disc around its true value), and
  by a PMU at a neighbour that reads the current on the branch between them (the current is linear
  in both end voltages, so |dV_i| <= (|dI| + |y_a| rho_j) / |y_b|). Then |X_ij| <= R_j r_i,
  |X_ij| <= rho_i r_j + (R_i - rho_i) R_j b_d for each cap, X_ii <= rho_i^2 while untampered, and
  |X_ij|^2 <= X_ii X_jj (X = dV dV^H is positive semidefinite). R_i is the most |dV_i| can be in
  the voltage box. Without these cuts the paper scenarios' bound was 4 and 6 devices; with them 8 and
  7.
- **The search's support rule:** one binary y_i per area bus whose injection the search holds (every
  bus but the goal's free injections), shared by the snapshots: out of the support its voltage stays
  true (|dV_i| <= R_i y_i), in it its injection stays true (|dS_i| <= M_i (1 - y_i)). This makes the
  bound one on the search's own problem, the problem the generator solves.
- **Injection bounds (`Am`):** the D14 and D16 bounds as linear constraints on the injection
  changes, over the area and its edge (every bus whose injection a support in the area can change):
  a generator's implied output P_gen - dP, Q_gen - dQ inside its limits, widened to its true output,
  and a load bus's |dP| within `load_cap` times its true load. A bus the attack does not touch meets
  them at its true value, so they hold whatever the support.
- **Objective:** minimize sum_d b_d, with sum_d b_d >= 1 (the search never returns an empty attack).
- **Snapshots:** by default the relaxation keeps one snapshot, the one where the goal moves furthest
  from the true state. Keeping all sixty of a window made SCIP stop at its time limit with a weaker
  bound than the single snapshot, and on the `At` episodes eleven snapshots gave the same bound as
  one.

### 2.1 Cut families (`engine/attacks/relax_cuts.py`)

With cuts on, the relaxation admits at most one device fewer than the search's attack (the cutoff).
Infeasible, at bound tightening or at the end, it proves the search's count optimal. Feasible, its
optimum is the bound.

- **bounds:** optimization-based bound tightening of how far each area bus's voltage can move,
  |V_i - V_i^true| <= rho_i: the move is bounded in 8 directions over the mixed-integer relaxation
  at the cutoff, each solve stopped at 5 seconds (`CertifyOptions.tighten_limit`) and read by SCIP's dual bound (valid at any stop),
  so rho_i <= max direction bound / cos(pi / 8). The continuous relaxation tightened nothing, since
  relaxed binaries let every channel move. rho gives the angle bound asin(rho_i / |V_i^true|), used
  as wedges on V_i and on each W_ij, and a smaller big-M reach.
- **qc:** the QC relaxation. Per pair of area buses, the product of the magnitudes and the cosine
  and sine of the angle-difference move inside their convex envelopes, W_ij their McCormick
  products, and in rectangular form Re W_ij = e_i e_j + f_i f_j, Im W_ij = f_i e_j - e_i f_j with
  each product in its McCormick envelope over the box rho gives.
- **cycle:** one angle per area bus. Each pair's angle-difference move is the difference of its
  ends' moves, so the moves sum to zero around every cycle, and V_i is its bus's magnitude times the
  envelopes of its angle.

Each family holds at every AC point inside the bounds. A test per family tightens the bounds at the
search's own count, puts the search's attack and its polar quantities into the relaxation, and
checks that no constraint is violated.

## 3. Why the optimum is a lower bound

Every attack the search can return is a point of the relaxation. Its voltages give a rank-one W
that satisfies the linear physics and the cones with equality, and the residual cuts with
X = dV dV^H. Its devices give binaries that satisfy every big-M link, and its support gives the
support binaries. Its goal and limits are among the relaxation's constraints. The relaxation also
drops constraints that only shrink the search's set: the angle and |V| steps of the `At` stealth
bound, the generator limits of `At`'s support (its solve holds every non-target injection, so
they cannot bind), and the connectivity of the support. A test builds the relaxation on a window, puts the
search's attack in it and checks that no constraint is violated. Dropping a constraint only enlarges the set. The
relaxation minimizes the same count over a superset, so its optimum is at most the search's.
The bound covers the attacker's area, the region every search candidate lies in, not the whole
grid.

## 4. Solver and interface

cvxpy with SCIP (open source, mixed-integer second-order cone), in the optional extra
`[certify]`. `fdia_graph.engine.attacks.certify.certify(g, states, goal, k, prev)` runs the search,
builds the relaxation over the same area, and returns a `Certificate`: the upper and lower bound,
whether they meet, the solve time, SCIP's status, and two tightness measures of the relaxed point.
`cone_gap` is the largest relative slack of its cones, and `mismatch` is the largest injection
mismatch (MW) between its W and the exact injections of its voltages, zero only at an AC state. The
cones can be tight pair by pair while the mesh's cycles are not, so `mismatch` is the one that says
whether the relaxation found a real attack. When SCIP stops at its time limit, the lower bound is
its dual bound rounded with the margin below, still valid. The tests check that the channel maps
reproduce the search's attack vector and that the search's own attack satisfies every constraint of
the relaxation.

### 4.1 The tolerance guard

SCIP decides feasibility and integrality within a tolerance (numerics/feastol, 1e-6 by default), and
on these problems some answers sit inside it: two runs of the same cuts on one `At` episode have
disagreed, and a cut relaxation has returned a bound below the cone relaxation's, which cuts cannot
do. A certificate is therefore claimed only clear of the tolerances, and `Certificate.verdict` says
which case holds: "certified", "gap", or "uncertain" with the reason in `Certificate.reason`.

- **Rounding:** a bound b proves ceil(b - `bound_margin`) devices (0.01 by default), never a plain
  rounding, so a relaxed optimum just above an integer by the tolerances is not rounded past it.
- **Infeasibility:** an "infeasible at one device fewer" answer, from the final solve or from bound
  tightening, is accepted only when a re-solve with numerics/feastol loosened to `robust_feastol`
  (1e-4, 100 times SCIP's default; SCIP judges integrality by the same tolerance) is still
  infeasible. A problem that stays infeasible with every constraint relaxed by more than the
  solver's own tolerance was not made infeasible by rounding. Tightening the tolerance would test
  the opposite direction: it makes infeasibility easier to reach. When the loosened problem is
  feasible, its dual bound is the claim and the verdict is "uncertain". After bound tightening the
  re-solve runs on the untightened relaxation with the same cuts: the tightened bounds were found at
  the default tolerance, so confirming on them would check the tolerance with its own output.
- **Certifying optima:** an optimum whose bound reaches the search's count (a certificate without
  infeasibility, as the cone relaxation alone can give) is re-solved the same way and stands only
  when the loosened bound still reaches the count; otherwise the loosened bound is the claim and the
  verdict is "uncertain". Every would-be certificate thus survives the loosened tolerance or is not
  claimed.
- **Consistency between levels:** `certify` solves the cone relaxation alone (`Certificate.cone_lower`)
  and, with cut families, the cut relaxation at one device fewer. Cuts only shrink the relaxation, so
  a cut bound below the cone bound is a numerical contradiction: the smaller bound is kept and the
  verdict is "uncertain".
- **No floor from the search:** the search's forced-device bound (`_Window.lower_bound`) counts the
  goal's own changes crossing sigma in float64, while the search counts after float32 rounding, so
  it could exceed the true minimum; the certifier does not use it.

Both knobs are fields of `CertifyOptions` (`models.config`), checked on construction.

## 5. IEEE-14 runs

A valid but loose lower bound: it never exceeds the search's count, and on these episodes no
configuration certifies any episode once the tolerance guard is applied. IEEE-14, hybrid meters, new generation's defaults (pool ratings
at 1.25 times the peak flow, `load_cap` 0.5, two-line `Am`, budget 256), one kept snapshot per
window, each family on top of the previous one, every verdict through the tolerance guard (section
4.1, `bound_margin` 0.01, `robust_feastol` 1e-4). The paper scenarios run on [WU26]'s metering
(PMUs at buses 1, 4, 6 and 13, lines 3-4 and 6-11, then 1-2 and 4-5) at ratings 1.2 times the
window's peak flow, as `tests/test_wu_scenarios.py` sets them up. A gap is the search's count minus
the bound; the last row is `certify`'s default.

| family | paper | `Am` | `At` | `Am` gaps | median mismatch, MW (`Am`) | seconds |
|---|---|---|---|---|---|---|
| second-order cone | 0 of 2 certified (gaps 7, 6) | 0 of 10 certified | 0 certified, 1 uncertain | 3 to 11, median 4 | 24 | 5 to 35 solve |
| + bounds | 0 of 2 (7, 6) | 0 of 10 | 0 certified, 1 uncertain | 3 to 11, median 4 | 18 | 400 to 637 tightening, 7 to 52 solve |
| + qc | 0 of 2 (7, 6) | 0 of 10 | 0 certified, 1 uncertain | 3 to 11, median 4 | 20 | 9 to 68 solve |
| + cycle | 0 of 2 (7, 6) | 0 of 10 | 0 certified, 1 uncertain | 3 to 11, median 4 | 18 | 9 to 57 solve |

The search takes 6 to 29 seconds per `Am` window and under 1 per `At` window. No cut family moves an
`Am` bound: bound tightening at one device fewer than the search leaves the median voltage move at
0.7 to 2.1 pu, and only 1 to 7 of the area buses get an angle bound, so the QC and cycle envelopes
seldom bind. The D16 bounds bring the relaxed `Am` points much closer to an AC state than before
them (median mismatch 18 to 24 MW, against about 320 MW on the earlier recipe), but not close enough
to certify. Every `Am` and paper verdict is a plain gap: no claim there sits at the tolerances.

The guard changes the `At` column, and nowhere else:

- `At` episode 3 (loads at 5, 10, 12, 13 and 14): the cone relaxation's optimum reaches the
  search's 9 at SCIP's default tolerance, which without the guard certified it, but the re-solve
  with numerics/feastol at 1e-4 bounds 6 (5 in an earlier run of the same code path: the loosened
  problem is solved at a loose tolerance, and its bound moves by a device between runs), so it is
  "uncertain". Every cut level finds the relaxation infeasible at one device fewer, and the re-solve
  on the untightened relaxation is feasible too. No configuration certifies the episode.
- `At` episode 1 (loads at 4, 5, 11, 13 and 14): the cone relaxation bounds 9 devices, the bounds
  and qc levels 10 and the cycle level 9 (10 before At's step slack scaled with the attack values),
  gaps of 2 and 3. Before the roundoff slack (section 2) the cone relaxation bounded 10
  and the bounds and qc levels 9, which the guard reported as a contradiction: the fixed 1e-6 margin
  had cut part of the cone relaxation's set, so its 10 was too high. The slack removes the
  contradiction.
- On the earlier recipe, two runs of the same cuts on `At` episode 0 disagreed (a relaxed attack
  with 8 devices at 0.3 MW mismatch in one, infeasible in the other), a different draw from this
  set's episode 0, whose verdicts here are plain gaps.

## 6. What remains for `Am`

The relaxed optimum stays tens of MW from an AC state because the angles are mostly free. A certificate
for `Am` needs angle bounds, which need either a tighter cutoff model (the bound tightening sees the
same loose relaxation it is meant to tighten) or bounds from outside it: the SDP relaxation on the
area's cliques (a conic solver such as MOSEK or SCS, not SCIP), or a spatial branch on the bus
angles so each branch carries narrow envelopes. The bound stays over the attacker's area.
