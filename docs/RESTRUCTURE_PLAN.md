# Plan: separate the mathematics from the plumbing

Goal (Ben, 2026-09-11): every formula or algorithm that comes from a source gets its own named
function that represents that operation, so a reader can go from an equation in a paper to one
function in the code and back. The classes keep only orchestration: pulling data, chunking, caching,
the fit/estimate/score API.

Nothing in this document has been done. It is the plan to review.

## 1. What the package looks like today

| module | lines | what is in it |
|---|---:|---|
| `dataset.py` | 767 | HDF5 loading, splits, PyG packaging, units, **plus** the Ybus/Yf/Yt construction (`_admittances`) |
| `engine/measurement.py` | 93 | accuracy-class noise, **AC measurement function** (`emit_from_state`, `clean_flows_from_states`) |
| `engine/physics.py` | 91 | power-flow solve and re-solve through pandapower |
| `engine/attacks.py` | 139 | the six attack constructions and the plausibility band |
| `engine/core.py` | 371 | the generator class, line-outage candidates, centrality targeting |
| `generation.py` / `streams.py` / `profiles.py` | 581 / 397 / 315 | shard and stream writers, **plus** swing and temporal-delta features, the AR(1) fluctuation model |
| `se/base.py` | 380 | network build, **AC measurement function again** (`_h_t`), Jacobian, meter calibration, WLS step, batched linear algebra, normalized residuals, scoring |
| `se/methods.py` | 270 | Huber, largest-normalized-residual removal, subspace prior, Jacobian weighting, gated prior |
| `se/jacobian.py` | 158 | projection split, leverage, SVD, weak directions, per-bus aggregation |
| `localization/*` | 558 | threshold calibration, F1/DR/FR, KCL residual, the 14-dim vector, CNN/MLP |

Two symptoms of the mixing: the AC measurement function exists three times (engine, SE, and the
docstring recipe in `dataset._admittances`), and the SE paper's equations are spread across private
methods (`_solve_plain`, `_nres`, `fit`) whose names say nothing about the equation.

## 2. Target layout

A kernel of pure functions, numpy in and numpy out, no file I/O, no dataset objects, no torch:

```
fdia_graph/
  formulas/                 <- the mathematics, one function per operation
    __init__.py
    network.py              admittances, AC measurement function, Jacobian, KCL residual
    estimation.py           WLS step, normal equations, residual covariance, normalized residual,
                            Huber weights, largest-normalized-residual test, subspace basis and solve
    projection.py           P_H, explained/unexplained split, leverage, SVD and weak directions,
                            meter-to-bus aggregation           (the Jacobian-informed digest)
    temporal.py             temporal delta, swing z-score, W-frame displacement, diurnal reference,
                            CUSUM, onset innovation, separate-bias update, recoverability ratio
    attacks.py              the six attack vectors and the plausibility band, as functions of
                            (state, targets, knobs) -> attacked measurements
    noise.py                accuracy-class meter error (bias + jitter)
    linalg.py               guarded inverse, batched Cholesky inverse, condition estimate
    metrics.py              angle/voltage MAE, geometric mean, per-bus F1, macro F1, DR, FR,
                            window AUC with the chance baseline
  se/, localization/, engine/, streams.py ...   <- unchanged public API, now thin: they call formulas.*
docs/reference/FORMULAS.md  <- the catalogue: function, equation, source, where it is used
docs/reference/REFERENCES.md <- numbered sources cited by the docstrings
```

Conventions for every function in `formulas/`:

- Name is the operation, not the caller: `ac_measurement(x, ...)`, `wls_step(...)`, `huber_weights(...)`.
- Docstring carries the equation in plain text, the source with its equation number, and the shapes.
- Pure: same inputs, same outputs, nothing read or written. Chunking, caching and device handling
  stay in the classes.
- One test per function against a hand-checkable toy (the 3-meter example, a 2-bus line) or
  against the pandapower engine where that is the ground truth.
- No torch inside the kernel. The one place torch does mathematics today is the Jacobian by
  automatic differentiation; the analytic AC Jacobian (Abur and Exposito, chapter 2) is itself a
  formula from a source and can replace it. That is optional (see open decisions).

## 3. Catalogue: what moves where

Estimation (the SE paper):

| today | becomes | source |
|---|---|---|
| `SEBase._h_t`, `MeasurementMixin.emit_from_state`, `clean_flows_from_states` | `network.ac_measurement(V, theta, Ybus, Yf, from_bus)` (one function) | Abur and Exposito 2004, ch. 2 |
| `FdiaGraph._admittances` | `network.branch_admittances(r, x, b, g, tap, shift, status)` -> Ybus, Yf, Yt | MATPOWER makeYbus |
| `SEBase.fit` (jacrev) | `network.ac_jacobian(...)` | Abur and Exposito 2004, ch. 2 |
| `SEBase._solve_plain` | `estimation.wls_step(z, h0, H, W)` and `estimation.chord_newton(...)` | Schweppe 1970 |
| `SEBase.fit` (sigma from benign residual RMS) | `noise.meter_sigma_from_residuals(...)` | Asprou et al. 2014 |
| `SEBase.fit` (Omega, critical) | `estimation.residual_covariance_diag(H, W)`, `estimation.is_critical(...)` | Abur and Exposito, ch. 5 |
| `SEBase._nres` | `estimation.normalized_residual(r, omega)` | same |
| `AdaptiveWeighting._solve` | `estimation.huber_weights(r_n, c)` + `estimation.irls(...)` | Huber 1964; Mili et al. 1996 |
| `ResidualRemoval._solve` | `estimation.largest_normalized_residual(...)`, `estimation.observable(H, w)` | Handschin et al. 1975 |
| `SubspacePrior._fit_states/_basis/_solve` | `estimation.whitened_svd_basis(X, rank_frac)`, `estimation.reduced_wls_step(...)` | our estimation paper |
| `SEBase._normal_matrices/_inv/_inv_batch/_cond` | `linalg.*` | numerics, cite as implementation |
| `SEBase.score` | `metrics.angle_mae`, `metrics.voltage_mae`, `metrics.geo_mean` | |

Jacobian-informed features (the digest):

| today | becomes |
|---|---|
| `JacobianFeatures.fit` (pinv, whitening, leverage, SVD) | `projection.weighted_pseudoinverse`, `projection.hat_matrix`, `projection.leverage`, `projection.weak_directions` |
| `JacobianFeatures.delta_z/transform` | `projection.explained_unexplained(dz, H, W)`, `projection.weak_energy(dx, V_weak)`, `projection.meters_to_buses(...)` |
| `JacobianWeighting.weights` | `estimation.huber_weights` applied to the unexplained part |
| `GatedPrior.gated_weights`, `bus_incidence` | `network.bus_incidence(...)`, `estimation.gate_weights(W, flags, incidence, factor)` |

Localization (the fed and localization papers):

| today | becomes | source |
|---|---|---|
| `streams._swing_scale`, temporal delta in `generation.py` | `temporal.temporal_delta`, `temporal.swing_zscore` | fed paper |
| `learned.kcl_residual` | `network.kcl_residual` | physics |
| `learned.full14` | `localization/features.py` (assembly only, calls the formulas) | |
| `LocalizerBase.fit` (FA quantile) | `metrics.threshold_at_fa(scores, fa)` | Neyman-Pearson calibration |
| `base._perbus_f1/_micro_f1`, DR/FR | `metrics.*` | |
| `ResidualLocalizer` | uses `estimation.normalized_residual` | BDD |

Generation (dataset papers):

| today | becomes | source |
|---|---|---|
| `AttackMixin.corrupt` (bias, scaling, replay, stealthy re-solve, ramp) | `attacks.bias`, `attacks.scaling`, `attacks.replay`, `attacks.stealthy_resolve_targets`, `attacks.ramp_profile` | Liu et al. 2009; our dataset paper |
| `AttackMixin.lra_delta` | `attacks.load_redistribution` | Yuan et al. 2011 |
| the plausibility floor and cap | `attacks.plausibility_band(...)` | Yuan; Ashok; Kallitsis |
| `MeasurementMixin._n` | `noise.accuracy_class_error(...)` | Asprou et al. 2014 |
| `profiles._ar1_scale`, `generate_states` | `temporal.ar1_fluctuation(...)` (the model) + the solver loop stays in profiles | our pool model |
| `FdiaGenerator.centrality_probs` | `network.centrality_targeting(...)` | |

This week's scratchpad mathematics, which should graduate into the same kernel rather than stay in
`scratchpad/temporal_se/`:

| scratchpad | becomes | source |
|---|---|---|
| onset innovation and held bias (`online.py`) | `temporal.onset_innovation`, `temporal.separate_bias_update` | Friedland 1969; forecasting-aided SE |
| change-point state machine | stays orchestration, in a future `se/temporal.py` class | |
| WLS-residual stealthy switch | `estimation.normalized_residual` + `metrics.threshold_at_fa` | |
| W-frame displacement, diurnal reference, CUSUM (`ramp_gate.py`, `diurnal.py`) | `temporal.displacement`, `temporal.diurnal_reference`, `temporal.cusum` | Page 1954 |
| injection-to-state map (`ceil8.py`) | `network.injection_to_state(J, pv, slack, ...)` | power-flow Jacobian, PV reduction |
| window AUC with chance | `metrics.window_auc(...)` | |
| recoverability ratio | `temporal.recoverability_ratio(...)` | our result |

## 4. Sequencing, so nothing breaks

0. **Catalogue first, no code.** Write `docs/reference/FORMULAS.md` from the tables above and fill
   the source column properly (equation numbers). This is the review artifact; half the value is the
   table itself.
1. **Kernel in parallel.** Add `formulas/` with the functions and their tests, without touching a
   single caller. Each test also asserts equality with the existing private path on the tiny test
   shard, so the kernel is proven against the current numbers before anything switches.
2. **Switch callers one module at a time**, SE first (it has cached results for 14, 118 and 300 to
   reproduce bit for bit), then localization, then engine and streams. Old private names stay as
   one-line aliases for a release so nothing of Ruslan's breaks. One minor version per module.
3. **Graduate the temporal work** into `formulas/temporal.py` and public classes in `se/temporal.py`
   and `localization/temporal.py`, with the docs pages that go with them.
4. **Docs.** Every method table in the READMEs links its rows to the formula functions; the SE
   paper's equation list maps one-to-one onto function names.

Estimated effort: step 0 a day; step 1 three to four days including tests; step 2 a day per module;
step 3 two days; step 4 a day. Results must reproduce at every step from the existing caches.

## 5. Decisions for Ben

1. **Package name:** `formulas` (proposed), or `math`, `equations`, `theory`.
2. **Numpy-only kernel:** replace the autograd Jacobian with the analytic one from the textbook
   (removes torch from state estimation entirely; more work, cleaner story) or keep autograd.
3. **Public or private:** are the formula functions documented API (users can call
   `fg.formulas.huber_weights`) or internal? Proposed: public, since the point is traceability.
4. **Granularity:** one function per equation (`wls_step`, `residual_covariance_diag`) or per
   algorithm (`wls`, `bad_data_detection`). Proposed: per equation, with the algorithm as a short
   function that composes them.
5. **Citations:** a numbered `REFERENCES.md` that docstrings cite by key (`[AE04, eq. 2.9]`), so a
   reader can go paper -> key -> function -> code.
