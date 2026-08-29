# Concepts to code

Paper ideas → the function that implements them. Pair with `DATA_DICTIONARY.md`.
Paths are under `src/fdia_graph/`.

## Modules

The paper's math lives in `src/fdia_graph/engine/`; at the top level, `generation.py`/`profiles.py`
drive the engine and the rest load and serve data.

| SDK file | job |
|------|-----|
| `registry.py` | dataset versions, aliases, cache |
| `download.py` | fetch + cache a shard |
| `generation.py` | assemble the classification shard (the recipe) |
| `streams.py` | assemble a continuous timeline |
| `dataset.py` | loader → tensors / PyG (what `fg.load` returns) |
| `profiles.py` | real load series → operating points |

| engine/ file | formula it implements |
|------|-----|
| `core.py` | `FdiaGenerator`: grid + noise setup (`__init__`), attack targeting; composes the three mixins |
| `measurement.py` | `emit_from_state` (the measurement function `h(x)`), `emit`, `state_from_net` |
| `physics.py` | `solve` / `resolve_states` — AC re-solve under new loads |
| `attacks.py` | `corrupt` (Ad/As/Ar) and `lra_delta` (Al redistribution) |

## State estimation

- WLS `x̂ = argmin (z−h(x))ᵀW(z−h(x))`: `h(x)` = `engine/measurement.emit_from_state`; full solvers ship
  in `fdia_graph.se` (`WLS` and the robust/prior classes) — see `../guides/state_estimation.md`.
- Bad-data `r_i=(z_i−h_i)/σ_i`, `J=Σr_i²`: `σ_i` from the engine FdiaGenerator.SD; pass/fail stored as the `stealthy` flag.
- Noise: engine FdiaGenerator.SD (accuracy-class), each reading = true + per-meter bias + per-scan jitter.

## Attack families

| family | paper | build | code |
|--------|-------|-------|------|
| Aq | `A_o` | scale load, re-solve | `generation.make` (1) + `engine/physics.solve` |
| At | `A_t` | slow ramp, re-solve | `generation` ramp loop (5) |
| Al | `A_l` | load-conserving redistribution | `engine/attacks.lra_delta` (6) |
| Ad | `A_d` | `z ← z(1±u)` | `engine/attacks.corrupt` |
| As | `A_s` | `z ← βz` | `engine/attacks.corrupt` |
| Ar | `A_r` | replay `z(t−k)` | `engine/attacks.corrupt` |

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
