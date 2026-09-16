# Plan: readable functions, flat loops, formulas with a reference

Goal (Ben, 2026-09-16): keep every behaviour exactly as it is, but make the code readable by
someone who has not written it. Three rules, applied everywhere:

1. **One responsibility per function, measured, not counted in lines.** A line count says
   nothing about why a function is hard to read, so the limits are on the things that do:

   | measure | limit | why it matters |
   |---|---|---|
   | cyclomatic complexity (radon `cc`) | at most 10 per function | the number of independent paths a reader has to hold; radon grade A or B |
   | nesting depth, loops and branches together | at most 3 | every level is a condition the reader carries down the page |
   | nested functions that read variables of the enclosing function | none | a closure with hidden inputs cannot be understood from its signature or tested alone |
   | parameters | at most 7 on private functions; a public entry point with more takes a settings object | more than that and the call site is unreadable |
   | positional indexing of record tuples (`r[10]`) | none; named fields | the index says nothing, and it already caused one label bug |
   | one copy of every formula and policy | enforced in review | two copies means two places to read and to fix |

   Decision (Ben, 2026-09-16): no line limit. A function that scores inside these limits is
   long enough; one that does not is split along the comment headings it already has.
2. **Loops a reader can follow.** Nesting is sometimes the honest shape of an algorithm, so
   the rule is not "no nested loops". The rule is that a reader can say in one sentence what
   one iteration of each loop does, and the nesting measure above (depth 3, loops and branches
   together) is the budget. Section 3.7 lists the tools for getting there, extraction being
   only one of them, and maps every nested loop in the package today to the tool that fits.
3. **Formulas as named functions with a source.** Any operation that appears in a paper or a
   textbook gets its own function, even if it is called once, with the equation in the docstring
   and a reference key. The program's flow then reads as prose that calls named mathematics.

This document is the plan only. Nothing here has been done. It complements
`RESTRUCTURE_PLAN.md`, which already catalogues the formula extraction for `se/`,
`localization/` and `se/jacobian.py` (the `formulas/` kernel); this plan covers the generator,
the writers, the loader, and the function-shape rules that apply to every module.

## 0. The promise to people who depend on the package

Other research groups install `fdia-graph` from PyPI and build on it. A readability pass that
changes what they get is a failure, however clean the result. So, for the whole series:

- **No public name changes, no signature changes, no behaviour changes.** `fg.load`,
  `fg.load_stream`, `fg.get`, `fg.generate`, `fg.generate_stream`, `FAMILIES`,
  `STEALTHY_FAMILIES`, `FdiaGraph`, every class in `fdia_graph.se` and
  `fdia_graph.localization`, their constructor arguments and return shapes stay exactly as they
  are. The formula functions are additions.
- **Byte-identical outputs.** A shard or stream generated with the same arguments and seed is
  the same file before and after (section 5 is how that is proven). Estimates and scores from
  the estimator classes match the cached results to the last digit.
- **No shard or stream format change.** Files written by the current release load unchanged
  afterwards, and files written afterwards load in the current release.
- **Private names get a grace period.** Every private method or function that moves keeps its
  old name as a one-line alias that emits a `DeprecationWarning` naming the replacement. The
  aliases are removed one minor version later, listed in the changelog both times. Our own
  code that reaches into private names (the temporal harness in `scratchpad/temporal_se`, the
  deck scripts under `CPPSreport`) is updated in the same PR that adds the alias, so nothing
  of ours runs on the deprecated path. A grep on 2026-09-16 shows Ruslan's `FDIA_localization`
  imports nothing from the package; the private names in use are only ours (`_truth_of`,
  `_h`, `_w_solve`, `_z_of`, `_solve_plain`, `_om`, `_Ai`, `_pinv`, `_sw`).
- **Versioning.** The series ships as minor releases, one per completed step group, each with
  a changelog entry that says "no user-visible change" where that is true and lists the new
  formula functions where it is not. Nothing lands on `main` outside a reviewed PR.
- **The formula functions are public but provisional.** Their docs say so until a 1.0: the
  names may still be adjusted once, with the same alias-and-warn rule.

## 1. Where the code stands (measured 2026-09-16)

radon and a small AST script measured every function in `src/fdia_graph` against the limits in
rule 1 (225 functions and methods; average complexity 3.56, grade A, so the package is mostly
fine and the work is concentrated).

| measure | limit | functions outside it | worst |
|---|---|---:|---|
| cyclomatic complexity | 10 | 13 | `FdiaGraph.__init__` 32, `generate` 28, `generate_stream` 27, `FdiaGraph.to_numpy` 25 |
| nesting depth (loops and branches) | 3 | 10 | `line_outage_candidates` 6; `ensure_local`, `generate`, `corrupt` 5 |
| closures reading outer variables | 0 | 17 | `generate_stream._store` reads 11, `generate.make` 8, `generate_stream._attack_frame` 7 |
| parameters on private functions | 7 | 3 | `generate._fin` 10, `corrupt` 8, `generate_states` 8 (the public `generate` has 18, `generate_stream` 11) |
| positional record indexing | 0 | 5 functions, 33 sites | `generate_stream` 12, `_write` 7 |
| duplicated formulas and policies | 0 | 5 (found by reading, not by the script; no two six-statement blocks are textually identical) | attack frame, ramp profile, replay lag, swing scale, target draw, each in both generators |

The functions that fail several measures at once, and what is inside them:

| function | complexity | nesting | inside |
|---|---:|---:|---|
| `generation.generate` | 28 | 5 | swing-scale prefix sums, two closures (`_fin`, `make`) that read eleven outer variables between them, three draw loops (benign, single-shot with retry, ramp sequences with an inner step loop), the magnitude sidecar, registration |
| `streams.generate_stream` | 27 | 4 | the same physics as `generate.make` written a second time (`_attack_frame`), three closures reading twenty outer variables between them, the timeline walk with a benign-gap loop, a ramp-episode loop and a single-shot-episode loop, the result dict written twice (return and savez) |
| `engine.core.FdiaGenerator.__init__` | 15 | 3 | outage handling, base power flow, load and attackable tables, the meter plan, the edge index, the per-unit branch physics (with the series-admittance formula inline), admittance matrices, PTDF, generator dispatch, per-meter bias draw |
| `generation._write` | 16 | 3 | stacking the record tuples by position, file attributes, the graph group, the data group, the clean group, the scalar datasets |
| `dataset.FdiaGraph.__init__` | 32 | 4 | header, static graph, per-unit physics, feature flags, slack-bus rule, the record selection mask (where most of the 32 paths live), optional preload |
| `dataset.FdiaGraph.to_numpy` | 25 | 4 | field selection, the clean-layer resolution per timestep, the units conversion, all interleaved |

Two things the survey does not count but that hurt readability more than length:

- **Closures with hidden inputs.** `make` and `_fin` in `generate`, and `_attack_frame`,
  `_store`, `_emit_benign` in `generate_stream`, read `g`, `X`, `C`, `SCALE`, `rng`,
  `mag_log`, `replay_tau`, `attack_intensity`, `K`, `prev_nx` from the enclosing scope. A reader
  cannot tell from the signature what one record depends on.
- **The same physics written twice.** The per-family attack frame (re-solve for Aq/At, LRA
  re-solve for Al, corrupt-in-place for Ad/As/Ar, replay-lag policy, mask re-assertion) exists in
  `generate.make` and in `generate_stream._attack_frame`; the swing scale exists as an inline
  prefix-sum loop in `generate` and as `_swing_scale` in `streams`; the ramp profile (rise, hold,
  return) is written in both with slightly different variable names; target drawing is written in
  both. Two copies means two places to read and two places for a fix.

## 2. Formulas found in the generator and loader

These are the operations in the surveyed code that appear in papers or books and should become
named functions (the SE and localization formulas are in `RESTRUCTURE_PLAN.md` section 3). The
reference keys are for `docs/reference/REFERENCES.md`, proposed there.

| today (inline) | becomes | equation | source |
|---|---|---|---|
| `generate`: prefix-sum windowed std of scan-to-scan change; `streams._swing_scale` | `formulas.temporal.recent_change_scale(X, window)` | std over the last W scans of the per-bus one-step change in P and Q, floor 1e-3 | our federated localization paper (the swing feature) |
| `generate._fin`, `generate_stream._store` | `formulas.temporal.temporal_delta(x_t, x_prev, mask)` and `formulas.temporal.swing_zscore(delta, scale)` | delta = P_t − P_{t−1}; swing = delta / scale | same |
| ramp deviation in `generate` (rise, hold, return) and in `generate_stream` | `formulas.attacks.ramp_profile(i, rise, hold, rate_up, rate_down)` | dev(i) = rate_up·i for i < rise; peak for the hold; max(0, peak − rate_down·(i − rise − hold)) after | our dataset paper (the slow ramp At) |
| replay-lag choice (fixed tau, random lag ≥ 20, oldest) in both | `formulas.attacks.replay_frame(buffer, tau, rng)` | policy, not an equation, but it is the definition of Ar | our dataset paper |
| `FdiaGenerator.__init__`: `1/(r + jx)` with the zero-impedance guard | `formulas.network.series_admittance(r, x)` | y_s = g_s + j b_s = 1/(r + jx) | MATPOWER manual, branch model |
| `FdiaGenerator.__init__`: bias 0.968·SD and jitter 0.25·SD | `formulas.noise.bias_jitter_split(sd, jitter_frac)` | bias² + jitter² = SD² | Asprou et al. 2014 (accuracy class), our split |
| `FdiaGenerator.__init__`: meter plan draw | `engine.meters.meter_plan(rng, C, E, vbus_frac, pmu_frac, flow_frac, injection_buses)` | sampling, kept as orchestration, but named | |
| `FdiaGenerator.centrality_probs` | `formulas.network.centrality_targeting(degree, closeness, betweenness, strength)` | exponential tilt of a fused centrality | Doostinia et al. 2025 |
| `generation._chrono_split` | stays, documented as the split rule | | |
| `dataset.FdiaGraph.__init__`: slack from bus type, else the pinned angle | `formulas.network.reference_bus(bus_type, clean_angles)` | the one bus with std of clean angle < 1e-6 | pandapower convention |
| `dataset._admittances` | already listed in `RESTRUCTURE_PLAN.md` (`network.branch_admittances`) | | MATPOWER makeYbus |
| `engine.measurement.emit_from_state`, `se.base._h_t`, `clean_flows_from_states` | already listed (`network.ac_measurement`), one function, three callers | S = V ∘ conj(Y V) | Abur and Exposito, ch. 2 |

## 3. The split, function by function

Names are proposals. Every new function is module-level and private unless it is a formula,
takes its inputs as arguments, and returns what it produces. No closure captures.

### 3.1 `generation.generate` (291 lines) becomes about 40

```
generate(...)
    g, rng, X, scale, targets = _setup(system, seed, outage, redundancy, states, ...)
    recs, yield_ = [], {}
    recs += _draw_benign(g, X, scale, rng, n_benign)
    for fam in single_shot_families:
        got, yield_[fam] = _draw_single_shot(g, X, scale, rng, fam, per_family, targets, knobs)
        recs += got
    if ramp requested:
        got, yield_[5] = _draw_ramps(g, X, scale, rng, per_family, ramp_len, ramp_rate, targets, knobs)
        recs += got
    out = _write(g, recs, out, split, seed, yield_, pool=X)
    _write_magnitude_sidecar(out, mag_log)
    register_local(name, out, meta=_meta(...))
```

with, in a new module `engine/records.py` shared by shards and streams:

| function | replaces | one pass does |
|---|---|---|
| `attack_frame(g, X, t, fid, targets, mult, knobs, rng) -> Frame or None` | `generate.make` and `generate_stream._attack_frame` | one attacked scan for one family; returns observed, benign, mask, label, designed magnitudes |
| `_resolve_frame(...)`, `_lra_frame(...)`, `_corrupt_frame(...)`, `_benign_frame(...)` | the four branches of `make` | one family each, 15 to 25 lines |
| `record_features(nx, nm, X, t, scale) -> (delta, swing)` | `_fin` and `_store`'s feature lines | calls the two temporal formulas |
| `draw_targets(rng, attackable, fid, cent_p, intensity)` | the two inline target draws | the bus set and multipliers for one attempt |
| `replay_frame(buffer, tau, rng)` | the three-way if in both | the Ar/As source frame |

`_draw_single_shot` keeps the retry-until-quota `while` and calls `attack_frame` once per
attempt; `_draw_ramps` keeps the sequence `while` and calls `_ramp_sequence(...)` for one
sequence, which is the only place the step `for` lives. Loop nesting inside any one function
becomes 1.

The `Frame` return is a small named tuple (`nx, nm, ex, em, y, stealthy, mags`) so the writer
stops indexing tuples by position (`r[10]`, `r[-1]`).

### 3.2 `streams.generate_stream` (262 lines) becomes about 60

```
generate_stream(...)
    g, rng, X, scale = _setup(...)                     # shared with generate
    buf = _StreamBuffers(T, C, E)                      # the nine arrays, allocated once
    t = 0
    while t < T:
        if _want_benign_gap(buf.y, t, attacked_frac):
            t = _benign_gap(g, X, scale, buf, t, rng)  # one gap
        elif _use_ramp(...):
            t = _ramp_episode(g, X, scale, buf, t, rng, ramp_len, ramp_rate)   # one episode
        else:
            t = _single_shot_episode(g, X, scale, buf, t, rng, fid, knobs)     # one episode
    return buf.finish(g, X, out)                        # clean layer, static graph, masks, savez
```

`_ramp_episode` and `_single_shot_episode` call `engine.records.attack_frame`, the same function
the shard generator calls, so the stream's physics is the shard's by construction rather than by
a copied block. `_StreamBuffers.store(t, frame)` replaces `_store` and owns `prev_nx`. The result
dict and the `savez` call are built from one mapping instead of two lists.

### 3.3 `FdiaGenerator.__init__` (178 lines) becomes about 30

| function | lines today | does |
|---|---:|---|
| `_open_case(system, outage)` | 35 | load the pandapower case, apply and record the contingency, refuse islanding |
| `_load_tables(base)` | 15 | `load_bus`, `attackable_pos`, `zero_inj`, `load_genP` |
| `_meter_plan(rng, ...)` | 10 | `M`, `flow_meter` |
| `_edge_index(base)` | 8 | `ei`, `E`, `nl`, the deprecated `x_react` |
| `_branch_physics(ppc)` | 25 | the `edge_*` and `bus_shunt_*` arrays, calling `formulas.network.series_admittance` |
| `_admittances(ppc)` | 8 | `_Ybus`, `_Yf`, `_Yt`, `_lut`, `_fb`, PTDF slices |
| `_meter_bias(rng, sd_bias)` | 8 | the six bias vectors, calling `formulas.noise.bias_jitter_split` |

`__init__` then reads as the list above. The long comment blocks that explain *why* (BR_G column
23, float64 for Ybus reconstruction, the ppc lookup footgun) move into the docstrings of the
functions they belong to, unchanged in wording.

### 3.4 `generation._write` (149 lines) becomes about 20

`_stack_records(recs) -> dict of arrays`, `_shard_attrs(g, seed, yield_)`, `_write_graph(f, g)`,
`_write_data(f, arrays)`, `_write_clean(f, g, pool, tstep)`. The chunk-and-gzip pattern that is
repeated seven times becomes `_chunked(f, name, data)`.

### 3.5 `FdiaGraph.__init__` (130 lines) becomes about 25

`_read_header(f)`, `_read_static_graph(f)`, `_feature_flags(f)`, `formulas.network.reference_bus`,
`_select_records(fam, gap, sp, split, families, include_gaps, heldout) -> idx`, `_preload(path, idx)`.
The family-name-or-id translation, which also exists in `generate` (`FAM_ID`) and `streams`, becomes
one `family_ids(names_or_ids)` in `dataset.py` used by all three.

### 3.6 The rest of the list over 60 lines

| function | lines | plan |
|---|---:|---|
| `generation.make` (module level, the public one-call generator) | 92 | its 13 `if`s are argument validation; split into `_validate_args` and the call |
| `torch_data.torch_windows`, `pyg_stream` | 71, 63 | one function per output flavour, shared `_split_indices` |
| `engine.core.line_outage_candidates` | 63 | `_rank_lines(...)` and `_screen_islanding(...)` |
| `localization.base.score` | 63 | the metric formulas are already scheduled for `formulas.metrics` |
| `dataset.__getitem__` | 62 | `_record_arrays(i)`, `_to_torch(rec)`, `_to_pyg(rec)`, units conversion as `_to_pu(rec)` |
| `engine.attacks.corrupt` | 52 | one function per family code; the plausibility band as `formulas.attacks.plausibility_band` (already listed) |

### 3.7 Loops: when nesting stays, and the tools other than extraction

What makes a nested loop hard to read is rarely the second `for`. It is one of these: a
condition inside the body that does not change between iterations, so the reader re-evaluates
it every time; an if/else ladder inside the body that pushes the real work four levels deep;
index arithmetic that hides what is being paired with what; or a body long enough that the
loop variables have scrolled off the screen. Each has a cheaper fix than a new function. Try
them in this order, and stop at the first that makes the loop readable.

| tool | when | what it does to the depth |
|---|---|---|
| **Vectorise** | the inner loop is arithmetic over an array (a sum, a difference, a scatter into rows) | the inner loop disappears into a numpy call, which is also the formula's natural form |
| **Hoist the invariant** | a branch inside the body tests something fixed for the whole loop (an attack kind, a mode flag) | decide once outside; each branch owns a short loop of its own, or a dispatch table maps the value to a function |
| **Guard clauses** | the body is an if/else ladder deciding whether to keep, skip or reject the item | `continue` or `return` on each rejection, in order, so the accepted path runs at depth 1 |
| **Generator** | the inner loop produces items and the outer loop consumes them (files inside monthly zips, steps of a ramp) | the producer becomes a function that yields; the consumer loops once, flat |
| **zip, enumerate, itertools.product** | indices exist only to pair elements or to walk a grid | names the pairing; no index arithmetic to decode |
| **Merge `with` stacks** | resource nesting (session, response, file, progress bar) inflates the depth without any logic | one `with a, b, c:` or a small function for the streaming step |
| **Extract the body** | one iteration has a name of its own (one attempt, one ramp sequence, one episode, one screened line) | the loop stays, its body becomes a call that a reader can look up |
| **Accept it** | the nesting is the algorithm as the literature writes it (epochs over minibatches, chunks over rows), the body is a few lines, and the loop variables are the natural indices | leave it; one comment stating the invariant is enough |

Nested loops in the package today, with the tool that fits each:

| site | shape today | depth | tool | after |
|---|---|---:|---|---|
| `generate`: swing-scale prefix sums | `for t` with an `if n >= 3` around three lines of arithmetic | 2 | vectorise: window sums by slicing the cumulative arrays, one masked expression | `formulas.temporal.recent_change_scale`, no loop |
| `generate`: ramp sequences | `while` sequences quota, `for i` steps, `if` shape, `if r is None: break` | 5 | generator `_ramp_steps(t0, rise, hold, ...)` yields (t, multiplier); extract `_ramp_sequence` for one sequence | `while` calls `_ramp_sequence`, depth 2 |
| `generate`: single-shot draws | `while` quota, `if fam == 1` target rule, `if r is not None` | 3 | hoist the target rule into `draw_targets(fid, ...)`; guard clause on the rejected attempt | depth 2 |
| `generate_stream`: timeline walk | `while t`, `if gap / elif ramp / else episode`, each with its own `for` and `if res is None` | 4 | the walk is a state machine: `_benign_gap`, `_ramp_episode`, `_single_shot_episode` each advance `t` and return it | `while` with three calls, depth 2 |
| `corrupt` | `for b in atk` with `if kind == "Ad" / "As" / "Ar"` inside, then `for e in inc` with the same switch | 3 | hoist the invariant: `kind` never changes inside the loop; one function per family (`_corrupt_bias`, `_corrupt_scaling`, `_corrupt_replay`), and the bias and scaling bodies vectorise over `atk` and `inc` | dispatch once, depth 1 |
| `line_outage_candidates` | `for pos` with `if / try / else / if / elif / else / try` deciding the rejection reason | 6 | guard clauses in `_screen_line(net, idx, lut0) -> reason or None`: each test returns its reason as soon as it fails | `for` with one call and one `if why`, depth 2 |
| `_solve_states_chunk` | `for sf` timesteps, `try`, then `for b, ps, qs` subtracting shunt injections | 3 | vectorise the shunt subtraction (`np.subtract.at` on the bus rows) into `_remove_shunt_injections(z, ...)`, which is also a definition worth naming (SE excludes the shunt) | depth 2 |
| `_fetch_nyiso` | `for month`, `with zip`, `for file in zip` | 3 | generator `_daily_load_frames(start, end)` yields one frame per file; the caller becomes `pd.concat(list(...))` | depth 2 inside the generator, 0 in the caller |
| `_fit_stats` | `for epoch`, `for batch` | 2 | accept: this is the training loop as every paper writes it, eight lines, natural indices | unchanged |
| `ensure_local` | `with session`, `with response`, `with file`, `with bar`, `for chunk` | 5 | merge the `with` stack, or extract `_stream_to_file(response, path, total)` | depth 2 |
| `se.methods` Huber and gated passes | `for chunk`, `for pass` with an early `break` on settled weights | 4 | accept the two loops (chunks over passes is the algorithm); the settled test becomes `_weights_settled(a, prev, tol)` so the `if` reads as a sentence | depth 3 |

The two cases marked "accept" are the answer to the worry that a nesting ban would force
artificial splits: they stay as they are, and the measure allows them.

## 4. Rules for the comments

Ben's comment style stays: comments say why and what, where they help, not line by line. Two
mechanical changes make them easier to find:

- A comment block that explains a decision (the float64 Ybus rule, the BR_G column, the
  attackable-bus filter, the replay-lag policy) becomes the docstring of the function that owns
  the decision, in the same words.
- A comment that names an equation ("SCALE[t,b,:] = std over [t−W, t) of ...") becomes the
  docstring of the formula function, with the reference key, so the equation is stated once.

## 5. Verification, so the function does not change

Every step must reproduce the current outputs bit for bit. The safety net is built first, as
its own PR with no code changes (step 1 in section 6), and it has four parts:

- **Shard hash test.** `tests/test_sdk.py` already builds a small shard with a fixed seed. The
  new test hashes every dataset and attribute in that file and asserts the recorded hashes. It
  must cover every family and the ramp path; if the test shard's quota leaves a family out, the
  test generates a second tiny shard that includes it.
- **Stream hash test.** The same on a 300-frame IEEE-14 stream with every family enabled (300
  rather than 500: six episodes already cover every family, and the suite stays under a minute).
- **RNG-order check.** Byte-identical shards depend on every `rng` draw happening in the same
  order as today. The extraction keeps each draw inside the function that replaces its block,
  and the strict-mode shard comparison is the proof: the tiny shard takes 15 seconds to build,
  so a reordering shows up on the first local run of a PR, before it is pushed. (An earlier
  draft of this plan described a separate ten-record old-versus-new test; the frozen comparison
  is the same check with the old path frozen once, so no second harness is kept.) The shared
  attack module carries the invariant in its docstring: the power-flow re-solve and the
  measurement emission first, then the benign emission for streams; for the corrupt-in-place
  families the emission, then the replay-lag draw, then the corruption draws. Changing that
  order changes every released file.
- **Cross-platform tolerance.** CI runs on Linux and the references are written on Windows.
  Integer arrays are compared by kind and value (numpy's default integer is int32 on Windows and
  int64 on Linux), floating arrays to 1e-7 relative, and scores to 1e-4 relative, because the
  iterative estimators amplify last-bit power-flow differences and the tiny shard averages few
  records per family (the first CI run put the Huber Ad angle error 1.5e-5 relative from the
  frozen value). Bitwise equality is the local strict mode's job.
- **Estimator and localizer results.** `docs/se/results/*.json` and
  `docs/localization/results/*` are the reference numbers; the SE caches in
  `docs/se/results/cache` make the 14 and 118 checks a matter of minutes.

pyright and the existing tests run in CI already. The survey script becomes
`tools/readability.py` and runs in CI as a report on the whole package and as a gate on the
functions a PR touched (section 6, definition of done).

A short **exceptions list** lives at the top of `tools/readability.py`: a function may be
recorded there with a one-line reason when honest complexity is better kept together than
scattered, the shard reader that branches on schema version being the expected case. An
exception is a reviewed decision, not a way past the gate.

## 6. Sequencing

0. Agree the names in sections 2 and 3 and the reference keys (a review of this document).
1. **Safety net, no code changes.** The four tests of section 5, `tools/readability.py` with
   the CI report, and the docstring and reference template: one worked formula in
   `docs/reference/FORMULAS.md` and `REFERENCES.md` (the swing z-score, which has a source, a
   shape and a one-line equation), so every later formula is written the same way.
2. `engine/records.py` with `attack_frame` and its four per-family functions, proven equal to
   `generate.make` by the ten-record test. Nothing else changes.
3. `generate` rewritten on top of it; shard hash green. Then `generate_stream`; stream hash
   green. This step removes the largest duplication.
4. **One AC measurement function.** `engine.measurement.emit_from_state`,
   `clean_flows_from_states` and the docstring recipe in `dataset._admittances` collapse onto one
   numpy `formulas.network.ac_measurement`; the estimator's torch `_h_t` becomes a thin torch
   wrapper of the same expression, kept only because the Jacobian is taken by automatic
   differentiation. This removes the largest copy in the package without touching numerics; the
   analytic Jacobian that would remove torch entirely stays its own later PR, as decided.
5. `FdiaGenerator.__init__`, `_write`, `FdiaGraph.__init__`: pure splits, hash tests green.
6. The formula functions of section 2, each with a toy test, wired into the callers.
7. The rest of section 3.6 and the loop sites of 3.7, one function per PR, each with the
   relevant reference numbers.
8. `FORMULAS.md` and `REFERENCES.md` completed; the changelog for the series; the deprecation
   aliases scheduled for removal one minor version later.

**Definition of done for every PR in the series:** the functions the PR touched pass all six
measures or are on the exceptions list with a reason; the four safety-net tests are green; any
new formula has its row in `FORMULAS.md` and its key in `REFERENCES.md`; any moved private name
has its alias and warning, and our own callers are updated in the same PR; Copilot review and
CI green; the changelog entry is written. One PR per numbered step, smaller where a step
splits naturally.

**Milestones and effort.** Steps 1 to 3 are the first milestone: the safety net, the shared
attack module, and the two generators on top of it. Estimate three to four days including the
hash-test debugging, which is where surprises live. Re-estimate the rest after that milestone
lands; a first guess is step 4 one day, step 5 one day, step 6 two days, step 7 two days,
step 8 half a day, so about ten working days in all rather than the eight first written here.

## 6a. Status (updated as steps land)

| step | what | pull request | state |
|---|---|---|---|
| 1 | safety net, readability measures, formula catalogue | #64 | merged 2026-09-16 |
| 2, 3 | `engine/records.py`, both generators split, named `Record` | #65 | merged 2026-09-16 |
| 4 | one AC measurement function (`formulas.network`) | #66 | merged 2026-09-16 |
| 5 | generator constructor, loader constructor, `to_numpy` | #67 | merged 2026-09-16 |
| 6, 7 | `formulas.temporal`, `formulas.attacks`, the rest of the backlog | #68 | open |
| 8 | catalogue and changelog complete, aliases scheduled | with 6, 7 | |

Readability report: 26 functions outside a limit at the start of the series, 0 after steps 6
and 7. Every step passed the strict frozen tests (bit-identical seeded shard and stream, identical
estimator and localizer scores) before it was pushed.

## 7. Decisions for Ben

Decided 2026-09-16:

1. **Measures, not a line limit.** The six limits in rule 1. `radon cc` and the survey script
   (`scratchpad` for now; it moves to `tools/readability.py` in step 0 and runs in CI as a report,
   not a gate, until the backlog is cleared).
2. **`engine/records.py`**, a module of functions that take the generator `g`, holds the shared
   per-frame attack code. The generator class does not grow.
3. **The `Frame` named tuple** replaces positional record tuples, including in `_write`.
4. **The three temporal formulas start `formulas/temporal.py` now**, as the first residents of
   the kernel in `RESTRUCTURE_PLAN.md`.
5. **Reference keys** in the `[AE04, eq. 2.9]` style, one numbered `docs/reference/REFERENCES.md`.
6. **Loops are handled by the tool list in section 3.7, not by a ban on nesting.** Nesting that is
   the algorithm's own shape stays; the depth measure is the budget, and extraction is the last
   tool tried, not the first.
7. **The safety net is built before any code moves** (section 5, step 1), and every PR meets
   the definition of done in section 6.
8. **One AC measurement function now, analytic Jacobian later** (step 4): the numpy formula is
   the single source and the estimator's torch version becomes a wrapper of it.
9. **The compatibility promise in section 0 holds for the whole series**: public API, outputs
   and file formats unchanged, private names deprecated with a warning for one minor version.

Carried over from `RESTRUCTURE_PLAN.md`, also decided: the package is named `formulas`; the
autograd Jacobian stays for now and the analytic Jacobian is its own later PR; formula functions
are public; granularity is one function per equation with a short composing function per
algorithm.
