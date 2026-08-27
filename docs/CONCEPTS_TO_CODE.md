# Concepts to code — from the paper's equations to the functions

This maps the ideas and formulas in the FDIA papers to where they live in the code, so you can read an
equation and jump straight to the function that implements it. Pair it with `DATA_DICTIONARY.md` (what the
arrays mean).

Paths below are inside `fdia_graph_sdk/src/fdia_graph/` unless noted.

---

## 1. How the SDK is organized (the map)

The package looks big, but it is six files with clear jobs. Read them in this order:

| file            | job — "if you want to change X, look here" |
|-----------------|--------------------------------------------|
| `registry.py`   | **which dataset version** to use and where it lives (release tags, aliases, cache). Start here to understand `fg.load(..., release=...)`. |
| `download.py`   | fetch + cache a shard from a GitHub release (the bytes behind `fg.load`). |
| `_core.py`      | **the physics engine.** One class wraps a pandapower grid: builds Ybus, solves AC power flow, *emits* meter readings, *corrupts* them per attack family, builds redistribution attacks. Everything physical happens here. This is the dense file — see the section map below. |
| `generation.py` | **assembles the classification shard.** Loops over operating points, calls `_core` to build one labeled record per attack family, writes the HDF5. The "recipe." |
| `streams.py`    | **assembles a continuous timeline** (for LSTM/TGN): attacks as timed episodes, the three measurement layers, windowing. |
| `dataset.py`    | **the loader.** Turns a shard into PyTorch tensors / PyG graphs, applies units + splits, computes the per-record dict. What `fg.load()` returns. |
| `profiles.py`   | turn a real load time series (NYISO/CAISO) into the operating points the grid is driven from. |

Inside `_core.py`, the methods you will actually care about:

| method            | what it is |
|-------------------|------------|
| `emit_from_state` | given a true state, produce **noiseless-then-noisy meter readings** (node_x, edge_x + masks). The measurement function `h(x)`. |
| `emit` / `solve`  | re-solve AC power flow for a (possibly attacked) load, generation pinned to true dispatch. |
| `corrupt`         | apply the **meter-tamper** families Ad/As/Ar in place, keeping the change inside the plausibility band. |
| `lra_delta` / `_lra_for_line` | build the **load-redistribution** attack Al (load-conserving, per-bus bounded, aimed at a target line). |

---

## 2. State estimation and bad-data detection

**Weighted least squares (the operator's estimator).** The paper's
`x̂ = argmin_x (z − h(x))ᵀ W (z − h(x))`:
the measurement function `h(x)` is `_core.emit_from_state` (voltages, injections, branch flows from a
state). A full WLS *solver* built on top of it lives in the research code
(`scratchpad/full_dataset_se.py`, function `solve_batch`) — the SDK ships the measurements, you bring the
estimator.

**Bad-data detection (chi-square).** The paper's normalized residual `r_i = (z_i − h_i(x̂))/σ_i` and
statistic `J = Σ r_i²`: the per-measurement noise `σ_i` is set in `_core.py` (`self.SD`, the accuracy-class
model); the residual/threshold logic is applied in the validation scripts, not the loader. A record's
pass/fail is stored as the `stealthy` flag.

**Measurement noise (accuracy-class model).** `_core.py` `self.SD = {pf, qf, v, pi, qi, va}`. Each reading =
true value + a constant **per-meter bias** (drawn once) + a **per-scan jitter**. `|V|` and angle use the
class-0.2 instrument-transformer figures; `P/Q` use a larger ~1.7% power-measurement std.

---

## 3. The attack families (paper Table "Attack Families")

| family | paper symbol | construction | code |
|--------|--------------|--------------|------|
| Aq | `A_o` | scale load `L_a ← α·L_a`, α∈[1.05,1.20] at 1–6 buses, **re-solve** AC | `generation.py make()` (family 1) + `_core.solve` |
| At | `A_t` | slow ramp `(1 ± r_k·i)·L_a`, r_k≈0.002/scan, ~5 buses, ≤60 scans | `generation.py` ramp loop (family 5) |
| Al | `A_l` | load-conserving redistribution, `Σd=0`, `|d_i|≤0.20·L_i` | `_core.lra_delta` (family 6) |
| Ad | `A_d` | tamper reading `z_a ← z_a(1±u)`, u∈[0.02,0.20] | `_core.corrupt` (Ad) |
| As | `A_s` | scale reading `z_a ← β·z_a`, β∈[1.02,1.20] | `_core.corrupt` (As) |
| Ar | `A_r` | replay an earlier reading `z_a ← z_a(t−k)`, k≥20 | `_core.corrupt` (Ar) |

**Stealth condition.** The state-level families (Aq/At/Al) change the *load* and re-solve power flow, so the
readings stay a consistent AC state — they satisfy the seminal `a = Hc` invisibility condition *by
construction* (they emit `h(x_attacked)`), which is why the residual test cannot see them. The meter-level
families (Ad/As/Ar) tamper readings directly, which breaks consistency and is detectable.

**Plausibility band.** Every realized per-bus change is kept between a `2%` meter-noise floor
(`generation.NOISE_FLOOR`) and a `20%` cap (`attack_intensity`), so an attack is neither lost in noise nor
physically absurd.

---

## 4. The temporal feature (the fed paper's contribution)

**Scan-to-scan delta and swing.** `temporal_delta` = this scan minus the previous scan on `[P,Q]`.
`swing` = that delta divided by the bus's own recent volatility (a z-score over a trailing window). Built in
`generation.py` `_fin()` (per record) and `streams.py` `_store()` (per frame). The idea: any above-noise
attack forces a meter off its recent trajectory, so it shows up as a large `swing` regardless of where the
bus sits in the grid — localization becomes a per-bus temporal question, no graph needed. The slow ramp
`At` is the exception (it stays inside the natural swing), which is why it is the open case.

**Nodal power residual (KCL).** A per-bus consistency score = net injection minus the incident *metered*
branch flows. Reference implementation: `examples/_train_baselines.py` `kcl_residual` (and the fed harness
`handmade/cache_builder_sdk.py`). It is a *partial* balance where metering is sparse — see the docstring.

---

## 5. Metrics (paper "Benchmark Results")

- **Localization macro-F1** — per-bus F1 averaged over the attackable buses. This is the headline number.
- **DR / FA** — grid-level detection rate and false-alarm rate. **Always report FA with DR**: flagging every
  sample gives DR = 1 at FA = 1, which is useless. See the runnable examples in the top-level `README.md`.
- **Strict localization accuracy** — fraction of records whose predicted attacked set matches truth exactly.

Worked, runnable examples that compute these on the real data are in the top-level `README.md`
("Three runnable baselines"), including the 14-dim per-bus feature vector the papers use.

---

## 6. Where to start reading, by goal

- *"I just want tensors to train on"* → top-level `README.md` quickstart, then `DATA_DICTIONARY.md`.
- *"What does this column mean?"* → `DATA_DICTIONARY.md`.
- *"How is an attack built?"* → `generation.py make()` → `_core.corrupt` / `_core.lra_delta`.
- *"How are the measurements produced from a state?"* → `_core.emit_from_state`.
- *"How do I reproduce the paper's localizer?"* → `README.md` "Three runnable baselines".
