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
