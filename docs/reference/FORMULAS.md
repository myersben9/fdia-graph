# Formulas: from the equation in the source to the function in the code

Every operation in the package that appears in a paper or a textbook gets one named function,
even when it is called once, so a reader can go from an equation to the code and back. This
catalogue is the index: one row per formula, the function that implements it, the equation in
plain text, the source with a key from [`REFERENCES.md`](REFERENCES.md), and where it is used.

The plan behind it is [`../READABILITY_PLAN.md`](../plans/READABILITY_PLAN.md) (rule 3) and
[`../RESTRUCTURE_PLAN.md`](../plans/RESTRUCTURE_PLAN.md) (the `formulas/` kernel). Rows are added as
the functions land; a row whose function column says *today:* names where the operation lives
until its function exists.

## How a formula function is written

```python
def swing_zscore(delta: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """One-step injection change as a z-score of the bus's typical recent change [FED26, eq. 3].

        swing[b, c] = delta[b, c] / scale[b, c]

    delta : [N, 2]  current minus previous scan, active and reactive injection (MW, MVAr)
    scale : [N, 2]  std of the one-step change over the last SWING_W scans, floored at 1e-3
    returns [N, 2], dimensionless; large on a single-scan spike, near 1 on a slow ramp
    """
    return delta / scale
```

The rules: the name is the operation, not the caller; the first line of the docstring names the
quantity and cites the key with its equation number; the equation follows in plain text; every
argument has its shape and unit; the function is pure (numpy in, numpy out, no I/O, no dataset
objects); one test checks it on a case small enough to verify by hand.

## Catalogue

| formula | function | equation | source | used by |
|---|---|---|---|---|
| swing z-score and temporal delta | `formulas.temporal.temporal_delta`, `formulas.temporal.swing_zscore` (shards, via `generation._record_features`); the stream computes the same two lines against the previous emitted frame in `streams._StreamBuffers.store` | delta_t = z_t − z_{t−1} at injection-metered buses; swing = delta_t / scale_t | [FED26] | the swing and temporal_delta features, `SwingThreshold`, `DeltaThreshold` |
| recent-change scale | `formulas.temporal.recent_change_scale` (shards and streams, through `generation._swing_scale`) | scale_t = std over [t − W, t) of the one-step change, floor 1e-3 | [FED26] | the swing feature |
| replay policy | `engine.records.replay_frame` | fixed lag tau, else a random lag of at least 20 scans, else the oldest scan | [DAT26] | the Ar and As families |
| bus voltage phasors | `formulas.network.complex_voltages` | V = \|V\| e^{jθ} | [AE04, eq. 2.1] | every AC evaluation |
| bus injections | `formulas.network.bus_injections` (numpy); `se.base.SEBase._h_t` is its torch twin, kept for callers that differentiate through it and pinned by `tests/test_formulas.py` | S = V ∘ conj(Y V) | [AE04, eq. 2.6] | the estimator's h(x) |
| AC measurement function | `formulas.network.ac_measurement`, called by `SEBase._h` | h = [\|V\|, −Re S, −Im S, θ, Re S_f, Im S_f] | [AE04, eqs. 2.6, 2.8] | every estimator, `JacobianFeatures.delta_z` |
| AC measurement Jacobian | `formulas.network.ac_jacobian`, called by `SEBase._jacobian` (the chord Jacobian in `fit`), pinned to the autograd Jacobian by `tests/test_formulas.py` | ∂S/∂θ = j diag(V) conj(diag(I) − Y diag(V)); ∂S/∂\|V\| = diag(V) conj(Y diag(V/\|V\|)) + conj(diag(I)) diag(V/\|V\|); the same for S_f with C_f and Y_f | [AE04, ch. 2], [MP19, dSbus_dV, dSbr_dV] | every estimator |
| branch flows | `formulas.network.branch_flows`; called by `engine.measurement.emit_from_state`, `clean_flows_from_states` and `dataset._clean_flows_full` | S_f = V_from ∘ conj(Y_f V) | [AE04, eq. 2.8] | every emitted flow meter, the clean flow layers, the estimator's h(x) |
| branch admittances | `formulas.network.branch_admittances`; called by `dataset._admittances` (`ybus`, `yf`, `yt`) | y_tt = y_s + (g + jb)/2, y_ff = y_tt / (t conj t), y_ft = −y_s / conj t, y_tf = −y_s / t; shunts on the diagonal | [MP19, makeYbus] | the loader's admittance matrices |
| series admittance | `formulas.network.series_admittance`, called by `FdiaGenerator._branch_physics` and `branch_admittances` | y_s = 1 / (r + jx), zero when the branch has no impedance | [MP19, branch model] | the static edge physics |
| accuracy-class error split | `formulas.noise.bias_jitter_split`, called by `FdiaGenerator.__init__` | bias² + jitter² = SD², jitter = 0.25 SD | [ASP14], our split | every emitted measurement |
| ramp profile | `formulas.attacks.ramp_profile` (shards and streams) | dev(i) = rate_up·i for i < rise; peak on the hold; max(0, peak − rate_down·(i − rise − hold)) after | [DAT26] | the At family |
| Am schedule | `timeline._AmShape.under_floor`, the ramp of one Am episode | rate = am_rate · floor / max_b \|δ_b\| / \|L_b\|, rise = ⌈1 / rate⌉, the ramp profile above clipped at 1, so no bus's load moves by more than am_rate of the noise floor in one frame | [WU26], our closed form | the Am family |
| Am tamper set | `engine.records._beyond_noise`, called by `_am_frame` | a metered channel is written when its noiseless change from the true state exceeds am_sigma accuracy-class stds (relative for P, Q; absolute for \|V\|, θ); the rest read the un-attacked twin | [WU26] (the l0 objective with the noise floor as its threshold) | the Am family |
| open attack subspace | `formulas.trust.attack_subspace`, `sparse_basis` | {H c : H_S c = 0} = H · null(H_S), its basis in reduced row echelon form | [WU26], [AE04, ch. 5] | the trusted-meter selectors |
| attack cost | `formulas.trust.attack_cost` | the fewest meters the cheapest open attack touches: the sparsest row of the echelon basis (an upper bound on the sparsest vector of the subspace, NP-hard) | [WU26] | `TrustedMeters`, the DQN reward |
| greedy trusted meters | `formulas.trust.greedy_trusted_meters`, called by `trust.TrustedMeters` | secure, on the cheapest open attack, the meter whose protection leaves the costliest cheapest attack; ties to the meter the most basis attacks pass through | [WU26], the row-reduction selection | `TrustedMeters` |
| WLS step | `formulas.estimation.wls_step` (shared weights) and `wls_step_batched` (per-record weights), called by `SEBase._solve_plain` and `_w_solve` | Δc = (BᵀWB)⁻¹BᵀW (z − h(x)) | [SCH70, part II] | every estimator |
| gain matrix | `formulas.estimation.normal_matrix`; per record `formulas.linalg.batched_normal_matrices` | G = HᵀWH | [SCH70, part II] | every estimator, the observability guard |
| weighted objective | `formulas.estimation.weighted_objective`, the divergence guard of `SEBase._w_solve` | J = Σ w_i (z_i − h_i(x))² | [SCH70] | the robust arms |
| residual covariance | `formulas.estimation.residual_covariance_diag`, `critical_measurements`, `floored_covariance`, called by `SEBase.fit` | Ω = R − H G⁻¹ Hᵀ; critical when Ω_ii < 1e-6 R_ii | [HAN75] | normalized residuals, `ResidualRemoval` |
| normalized residual | `formulas.estimation.normalized_residual`, called by `SEBase._nres` | r_N,i = \|z_i − h_i(x̂)\| / √Ω_ii | [HAN75] | Huber, residual removal, `ResidualLocalizer` |
| Huber weight | `formulas.estimation.huber_weights`, called by `AdaptiveWeighting`, `SubspacePrior._huber_passes`, `JacobianWeighting` | a_i = min(1, c / \|r_N,i\|) | [HUB64] | `AdaptiveWeighting`, `SubspacePrior`, `JacobianWeighting` |
| operating-point prior | `formulas.estimation.whitened_svd_basis`, called by `SubspacePrior._fit_states` | whiten X per coordinate, SVD, keep the leading rank_frac directions, un-whiten, QR | [EST26] | `SubspacePrior`, `GatedPrior` |
| localization gate | `formulas.estimation.gate_weights`, called by `GatedPrior.gated_weights` | w_i ← factor · w_i for every meter incident to a flagged bus | [EST26] | `GatedPrior` |
| guarded inverse, condition number | `formulas.linalg.guarded_inverse`, `condition_number` (Cholesky, power and inverse iteration) | inverse when PD and cond < 1/(100 eps), else pseudo-inverse; cond = λ_max / λ_min | numerical guards, no source | every estimator, the observability guard |
| weighted pseudo-inverse | `formulas.projection.weighted_pseudoinverse` | H_W⁺ = G⁻¹HᵀW | [SCH70, part II] | Jacobian features |
| explained / unexplained split | `formulas.projection.explained_unexplained`, called by `JacobianFeatures.transform` | dx̂ = H_W⁺ Δz, r∥ = H dx̂, r⊥ = Δz − r∥ | [JAC26] | Jacobian features, `JacobianWeighting` |
| leverage | `formulas.projection.leverage` | diag of P = Hw G⁻¹ Hwᵀ, Hw = W^{1/2} H | [HAN75] | the leverage-weighted change feature |
| weak directions | `formulas.projection.weak_directions`, `direction_coefficients`, `weak_move` | SVD of Hw; α = Uᵀ W^{1/2} Δz; V_weak V_weakᵀ dx̂ | [JAC26] | the α_weak and weak-move features |
| meter-to-bus aggregation | `formulas.projection.bus_incidence`, `meters_to_buses` | a bus's own V, P, Q, θ channels plus the flows of its incident branches; sum for energies, max for changes | [JAC26] | the per-bus features, `GatedPrior`, `ResidualLocalizer` |
