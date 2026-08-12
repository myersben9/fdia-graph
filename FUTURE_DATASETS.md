# Future dataset / generation ideas

A running list of dataset variants and generation features to build with the
fdia-graph SDK. Each entry has the idea, why it matters, and rough notes on how
to implement it with the current `generate()` / `_core.FdiaGenerator`. Nothing
here is built yet; this is the backlog.

---

## 1. Bigger datasets (more attack and benign samples)

**Idea.** Scale up the per-family sample count and the benign majority well
beyond the current ~3k/family. Boyaci uses ~5.7k/family, PING uses ~10k/grid.

**Why.** More data almost certainly lifts localization on the hard stealthy
families (the models are not yet data-saturated, especially on 118/300). It also
tightens the BDD and residual statistics and makes held-out evaluation cleaner.

**How.** `generate(..., per_family=10000, n_benign=60000)`. The operating-point
pool is the bottleneck (it caps the number of distinct base states); to scale
past the pool size, either enlarge the pool (more init timesteps) or allow
multiple attacks per operating point with different attacked-bus sets. Watch the
IEEE-300 infeasibility rate (some attacks diverge; retry-to-target handles it but
costs wall-clock). Run a data-scaling study (F1 vs. #samples) to find the knee.

## 2. Attack-intensity scenarios

**Idea.** A dataset (or a per-record tag) that sweeps attack magnitude in tiers,
e.g. mild / moderate / strong, like Falas's 5 / 10 / 20 / 30 % ladder.

**Why.** Lets us plot performance vs. attack strength (detection and localization
degradation curves), and separates "small stealthy nudge" from "large but still
plausible" attacks. It is also the natural axis for a robustness claim.

**How.** Parameterize the magnitude knobs per tier: `attack_intensity` (the ±
load-shift bound for Ao/LRA), the ramp `rate`, and the As/Ad corruption scale.
Emit an `intensity_tier` field per record so a loader can filter or stratify.
Keep the plausibility cap active so "strong" still stays BDD-stealthy where it
should.

## 3. Zero-shot generalization dataset

**Idea.** A split/dataset designed for zero-shot evaluation: train on some
attack families or some systems, test on entirely unseen ones.

**Why.** The strongest generalization claim. Two flavors: (a) unseen-attack
(train without a family, test on it — we already have the As/Ar held-out flag,
extend to hold out a stealthy family), and (b) unseen-system (train on 14+118,
test on 300, or train on one topology and transfer). PING's whole thesis is
cross-system generalization; this dataset would let us measure it directly.

**How.** For unseen-attack, extend the `heldout` protocol to any family subset.
For unseen-system, ship the three systems with a documented transfer split and a
loader flag `zero_shot=("train_systems", "test_systems")`. Requires the model to
handle variable N/E (the ARMA + attention stack already does, per-graph).

## 4. Stealth-only dataset

**Idea.** A dataset containing only the BDD-evading families (benign, Ao, ramp,
LRA) — no detectable contrast set.

**Why.** This is the pure "ML-only" regime: every attack passes BDD, so a
classical detector is useless and only a learned model can localize. It is the
cleanest setting to state the ML-only thesis and to benchmark localizers without
the easy Ad/As/Ar inflating the aggregate score.

**How.** `generate(..., families=["Ao", "ramp", "LRA"])` (benign always
included). Trivial with the current knobs; mainly a packaging/release decision.

## 5. BDD-detectable-only dataset

**Idea.** The complement: only the classically-detectable families (benign, Ad,
As, Ar).

**Why.** A control / ablation dataset. Shows what a model learns when the signal
is a local measurement inconsistency (which BDD already catches), and lets us
quantify how much of a model's score comes from the easy families. Pairs with #4
for a clean stealthy-vs-detectable comparison.

**How.** `generate(..., families=["Ad", "As", "Ar"])`.

## 6. Mimicry attack (Am) — persistent, load-pattern-mimicking, operationally deadly  [NEXT VERSION]

**Idea.** The strongest stealthy attack: instead of a bounded nudge on the current
load, feed the operator a *different but individually-plausible load trajectory* on
the target buses — a load pattern the bus genuinely had at another time — blended in
smoothly and **held over a window** (persistent, not single-shot). Re-solve the power
flow so the whole state moves. The attacked bus is doing something believable on its
own, but jointly out-of-sync with the rest of the grid.

**Why it matters (the whole thesis, sharpened).** Reviewer-proof "ML-only dangerous":
- Evades **BDD** (spatially consistent — we re-solve) AND a **temporal-anomaly
  threshold** (smooth, in-range, no single-scan jump). Prototype on IEEE-118 at
  accuracy-class noise: Am flagged 8.3% by BDD and 8.3% by a 5%-FA temporal detector,
  i.e. at/below the benign floor on both. See `_am_prototype.py`.
- The only signal left is the **joint spatiotemporal correlation** (real regional loads
  ramp together; the mimic breaks that) — invisible to BDD and per-bus checks, catchable
  ONLY by a spatiotemporal GNN. That is the purest ML-only claim.
- Being **persistent**, it actually reaches the EMS applications and causes a wrong
  operator decision (masked overload) — unlike a single-shot spike, which the operator
  ignores. This is what makes it "deadly," measured by the operator-impact metric
  (line-overload masking, Yuan et al. 2011).

**Design knob (the research axis).** How far the joint deviation can be pushed while each
bus stays individually plausible: too consistent → undetectable by anything (useless);
too inconsistent → obvious. Target: individually plausible, jointly off just enough to be
localizable only by a learned model, and goal-directed (hide a real overload).

**How.** Reuse `profiles.generate_states` as the attacker (draw a plausible false
trajectory for the target buses from another time window), blend from the true previous
scan over a smooth onset, hold over K scans, re-solve with pinned dispatch + AGC. Add as
a NEW family (keep single-shot Aq/Al as the weaker contrast rung — the damage ladder
crude→single-shot→persistent is the argument). Three-detector eval: BDD, temporal
threshold, and the learned localizer.

**Why deferred to next version:** v0.4.0 already has a difficulty gradient (crude
Ad/As/Ar → stealthy single-shot Aq → harder At/Al) sufficient to carry the operator-impact
story. Am is a substantial new family + evaluation worth its own release, not a rush-in.

## 7. Detectability-vs-magnitude sweep for the load-moving stealthy attacks  [CORE DONE — see below]

**Status (2026-07-15):** the core sweep is RUN and in the report. `examples/_aq_sensitivity.py` sweeps Aq's
load-scale magnitude (+0.5% .. +50%) on 14/118/300, re-solving (pinned dispatch + AGC) and adding 1.7% meter noise,
and measures detectability as the ROC-AUC of separating attacked vs noise-only buses on the injection deviation, plus
the SNR = attack shift / noise sigma. Result (`results/aq_sensitivity.json`, report slide "How Stealthy Is Stealthy?"):
detectability collapses to chance (AUC 0.45-0.51) below ~1-2% load; SNR crosses 1 at a **topology-independent ~1.7%
load move** (the meter class); the AUC a detector can reach clears 0.75 at ~4% (IEEE-118, redundancy helps) vs ~8%
(14/300). That is the width of the ML-only-dangerous window. STILL TODO: overlay the three real detectors (BDD chi2,
swing catch %, trained localizer swF1/DR@FA) on the same magnitude axis, and run As (meter-scaling) alongside for the
BDD-detectable contrast. Original plan retained below.



**Idea.** Take our stealthy load-moving attack (Aq — scale the attacked buses' loads and
re-solve) and SWEEP the scale magnitude from a tiny nudge to a large shift, e.g.
`attack_intensity` giving multipliers ~1.02, 1.05, 1.10, 1.15, 1.25, 1.5, 2.0x (deliberately
going PAST the current plausibility cap to map the full curve). At each magnitude, measure how
well each detector catches it: (a) classical BDD (chi-square detect %), (b) the swing /
rate-of-change detector, and (c) the learned ML localizer/detector (swF1, DR at fixed FA). Also
log the residual footprint (median |ΔP| at attacked buses) and the operator impact per tier.

**Why.** This answers "how stealthy is stealthy" quantitatively — the point Ben wants: at what
load-move magnitude do our stealthy attacks stop being invisible and start getting caught? It
gives a detection-rate-vs-attack-strength CURVE with the ~5% false-alarm reference line, and
locates the CROSSOVER magnitude where each detector's catch rate lifts off the floor. That draws
the exact regime map: below the crossover only ML works (the ML-only-dangerous window), above it
even BDD suffices — directly sharpening the thesis and pinning down where the plausibility cap
should sit. Run As (meter-scaling) alongside for contrast: find the scale factor at which meter
scaling becomes BDD-detectable, versus the (larger) load-move magnitude at which the re-solved
Aq does.

**How.** Reuse the existing eval tooling per magnitude: generate Aq (and As) at each intensity
tier (holding attacked-bus count / timesteps fixed), then run `_bdd_release.py` (chi2 detect %),
`_roc_detector.py` (swing catch %), `_feature_sep.py` (per-feature AUC), and a trained localizer
(swF1 / DR@FA). Plot each metric vs magnitude on one figure (one line per detector) + the median
|ΔP| footprint, with the benign 5%-FA line. Emit an `intensity_tier` field (see #2) so the sweep
can share one shard. Expected shape: BDD/swing flat at ~5% until the footprint clears the noise
floor, then rising; ML detectable earlier (smaller magnitude) than BDD. Pairs with #2 (that
builds the tiered dataset; this is the analysis to run on it).

## 8. Availability-mask sensitivity analysis for state estimation  [EXPERIMENT]

**Idea.** Sweep the measurement AVAILABILITY, especially angle/PMU coverage, and measure the state-estimation
error (above all the angle MAE) as a function of it. Vary the per-quantity availability fraction (angle/PMU
coverage from ~10% to 100%, and separately the SCADA V/P/Q redundancy), and for each setting retrain the SE
estimator and record V and theta MAE per system. Anchor the curve at the two ends we already have, the
realistic-sparse baseline (~64-69% angle coverage) and the full-availability ceiling (every channel metered).

**Why.** Our SE results show the angle MAE piling up at a floor (~0.65 deg on IEEE-300) that no estimator beats
(WLS, plain ARMA, ARMA+attn, and PG-DGAT all land near it), and roughly a third of buses carry no angle meter at
all. The full-availability run in this session tests the extreme and points straight here. A SWEEP maps the whole
observability curve, so it says how much PMU coverage buys how much angle accuracy, where the knee sits, and
whether IEEE-300's penalty is an availability limit or a graph-size limit. That is the result an operator can act
on ("to hit X deg you need Y% PMU coverage"), and it separates the data limit from the model limit cleanly. It
belongs in the SE/PINN paper next to the "does physics help" question.

**How.** Reuse `examples/_se_arma_attn.py`, which already has a `FULL_AVAIL` mode. Generalize it to an
`AVAIL_FRAC` knob that keeps the angle (and optionally the |V|) mask at a target fraction, dropping the rest, with
the same dataset-matched noise on the kept meters. Sweep frac in {0.1,0.3,0.5,0.7,0.9,1.0} x systems x a few
seeds, plot theta-MAE vs coverage (one line per system) against the meter-noise floor, and save a CSV sidecar.
Each point is a small model and trains in minutes. A worthwhile second axis is the PMU PLACEMENT strategy at a
fixed budget (random vs degree-weighted vs observability-greedy), since where the PMUs go can matter as much as
how many. Pairs with #2 (intensity tiers) as another controlled-axis study.

## 9. Temporal load-redistribution attack (Atr) — zero-sum ramp, locally confined  [PROTOTYPED]

**Idea.** A new stealthy family that is At and Al at once: a slow temporal ramp whose per-scan
perturbation is a ZERO-SUM redistribution across the attacked buses (some ramp up while others ramp
down, summing to zero every scan) rather than a one-direction load scale. Think "load redistribution
(Al), but ramped over time."

**Why.** The current At scales the attacked loads in one direction, so the net load changes and the
re-solve pushes the imbalance onto the slack, perturbing the whole grid. Making the ramp zero-sum
means (a) total demand is conserved at every scan, so it is stealthier to a load/energy-balance
monitor, and (b) the slack absorbs nothing, so the state change stays LOCAL to the attacked region.
It combines At's temporal stealth (per-scan step within noise) with Al's spatial stealth
(conservation + locality), which should make it the hardest frontier attack in the set — exactly the
regime the SE/localization papers argue is the open problem.

**Prototype result** (`scratchpad/at_redistribute_proto.py`, real IEEE-118 pool): a zero-sum ramped
delta `s(t)*d0` over 5 buses re-solves with net load change ~2e-16 MW (machine zero) and a state
change 22x larger on the attacked buses than on the rest. Divergent up/down ramps are clearly
visible and self-contained (rise to peak, hold, fall back to baseline).

**How.** Add family code 7 (`Atr`) alongside the existing ramp. In `generation.py`, in the ramp
branch, replace the single-direction `Lp[atk] *= (1 +/- r_k)` with a fixed zero-sum base pattern d0
over the attacked buses (random signed shares, mean-subtracted so sum=0, peak-scaled into the
plausibility band) times the ramp envelope s(t), then `solve(Lp + s(t)*d0, Lq, Xt, Lp_true=Lp)`.
solve() already spreads only the NET load change to generators, so a zero-sum delta leaves generation
and slack untouched — no new physics needed. Keep it a distinct family (do not replace At), give it
its own family id, alias, and label so existing At results are unaffected. Regenerate all three
systems and cut a new release; verify per-bus magnitudes sit in the band and BDD-stealth holds, and
add the locality ratio to the release manifest. Pairs with #6 (mimicry) as the two "hard frontier"
additions.

---

## Cross-cutting notes

- Every variant should keep the seeded, reproducible pipeline (seed 123) and ship
  a manifest + BDD-verification numbers so the stealth split is documented per
  release.
- Consider a single `scenarios/` release that bundles #2–#5 as named sub-datasets
  loadable by name (`load("stealth_only")`, `load("intensity_strong")`, …) via the
  local-registry mechanism already in the SDK.
- Attacked-bus-set metadata (which buses, set size) should be exposed per record
  so analyses can condition on attack footprint (see the per-family set sizes in
  the report's attack-construction pages).
