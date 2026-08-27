# Concepts to code

Paper ideas → the function that implements them. Pair with `DATA_DICTIONARY.md`.
Paths are under `src/fdia_graph/`.

## Modules

| file | job |
|------|-----|
| `registry.py` | dataset versions, aliases, cache |
| `download.py` | fetch + cache a shard |
| `_core.py` | physics engine: Ybus, AC solve, emit meters, corrupt, redistribution |
| `generation.py` | assemble the classification shard (the recipe) |
| `streams.py` | assemble a continuous timeline |
| `dataset.py` | loader → tensors / PyG (what `fg.load` returns) |
| `profiles.py` | real load series → operating points |

`_core.py` methods: `emit_from_state` = measurement function `h(x)`; `solve`/`emit` = re-solve AC;
`corrupt` = meter-tamper families; `lra_delta` = redistribution attack.

## State estimation

- WLS `x̂ = argmin (z−h(x))ᵀW(z−h(x))`: `h(x)` = `_core.emit_from_state`; a full solver is in
  `scratchpad/full_dataset_se.py` (`solve_batch`). The SDK ships measurements, you bring the estimator.
- Bad-data `r_i=(z_i−h_i)/σ_i`, `J=Σr_i²`: `σ_i` from `_core.SD`; pass/fail stored as the `stealthy` flag.
- Noise: `_core.SD` (accuracy-class), each reading = true + per-meter bias + per-scan jitter.

## Attack families

| family | paper | build | code |
|--------|-------|-------|------|
| Aq | `A_o` | scale load, re-solve | `generation.make` (1) + `_core.solve` |
| At | `A_t` | slow ramp, re-solve | `generation` ramp loop (5) |
| Al | `A_l` | load-conserving redistribution | `_core.lra_delta` (6) |
| Ad | `A_d` | `z ← z(1±u)` | `_core.corrupt` |
| As | `A_s` | `z ← βz` | `_core.corrupt` |
| Ar | `A_r` | replay `z(t−k)` | `_core.corrupt` |

Aq/At/Al re-solve power flow so readings stay a consistent AC state (invisible to the residual test by
construction). Ad/As/Ar tamper readings directly (detectable). All changes kept in a 2–20% plausibility
band (`generation.NOISE_FLOOR`, `attack_intensity`).

## Temporal feature

`temporal_delta` = scan-to-scan `[ΔP,ΔQ]`; `swing` = that as a z-score of recent volatility. Built in
`generation._fin` and `streams._store`. Any above-noise attack spikes the swing, so localization is per-bus
and needs no graph; the slow ramp `At` stays inside the swing, so it is the open case.

## Metrics

Localization macro-F1 (per-bus F1 over attackable buses); DR/FA (always report FA with DR); strict
localization accuracy. Runnable examples that compute these are in the top-level `README.md`.
