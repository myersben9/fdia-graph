# Formulas: from the equation in the source to the function in the code

Every operation in the package that appears in a paper or a textbook gets one named function,
even when it is called once, so a reader can go from an equation to the code and back. This
catalogue is the index: one row per formula, the function that implements it, the equation in
plain text, the source with a key from [`REFERENCES.md`](REFERENCES.md), and where it is used.

The rules come from [`READABILITY_PLAN.md`](../plans/READABILITY_PLAN.md) (rule 3) and
[`RESTRUCTURE_PLAN.md`](../plans/RESTRUCTURE_PLAN.md) (the `formulas/` kernel). A new formula
function gets its row here in the same change.

## How a formula function is written

```python
def swing_zscore(nx: np.ndarray, prev: np.ndarray, scale_t: np.ndarray, metered: np.ndarray) -> np.ndarray:
    """The one-step injection change as a z-score of the bus's typical recent change [FED26]:
    spikes (Aq, Al) read large, the slow ramp and benign scans stay near 1.

        swing[b] = ([P_inj, Q_inj](t) - [P_inj, Q_inj](t-1)) / scale_t[b]

    nx, prev : [N, 4] current and previous scan; scale_t : [N, 2] from recent_change_scale
    metered  : [N] bool, buses with an injection meter
    returns  : [N, 2] float32, dimensionless
    """
    sw = np.zeros((nx.shape[0], 2), np.float32)
    sw[metered, 0] = (nx[metered, 1] - prev[metered, 1]) / scale_t[metered, 0]
    sw[metered, 1] = (nx[metered, 2] - prev[metered, 2]) / scale_t[metered, 1]
    return sw
```

The rules: the name is the operation, not the caller; the first line of the docstring names the
quantity and cites the key with its equation number; the equation follows in plain text; every
argument has its shape and unit; the function is pure (numpy in, numpy out, no I/O, no dataset
objects); one test checks it on a case small enough to verify by hand.

## Catalogue

| formula | function | equation | source | used by |
|---|---|---|---|---|
| regional subspace prior | `formulas.federated.block_diagonal_basis`, through `federated.RegionalPrior` | V[cols_k, block k] = V_k from each client's own `whitened_svd_basis`, zero elsewhere; orthonormal for disjoint column sets | [EST26], [FED26] | the federated fit of the estimator's prior |
| federated average | `formulas.federated.fedavg` | θ = Σ_k (n_k / n) θ_k, in float64, cast back; one client returns its own tensor | [MCM17] | the federated localizers |
| the buses of each client | `formulas.federated.interior_boundary`, `cut_edge_count`, `attackable_affinity`, `hop_distance`, `halo_nodes`, through `federated.spectral_partition` and `compute_nodes` | interior: every neighbour in the same client; cut edges: ends in different clients; W = A ⊙ √(m mᵀ), m = heavy at attackable buses; halo: other clients' buses within h hops, own buses first | [VLX07], [FED26] | the federated partition |
| pooled channel moments | `formulas.federated.channel_moments`, `pool_moments`, through `localization.learned.standardization` | f_a = n_a / n, f_b = n_b / n, mean = mean_a + (mean_b − mean_a) f_b, var = f_a var_a + f_b var_b + (mean_b − mean_a)² f_a f_b; one part returns unchanged | [CGL79] | the learned localizers' standardization, pooled across federated clients |
| per-bus rates and average precision | `formulas.metrics.perbus_rates`, `average_precision`, through `LocalizerBase.score_perbus` and `LearnedLocalizer.score_grid` | F1 = 2TP / (2TP + FP + FN), DR = TP / (TP + FN), FR = FP / (FP + TN); AP = Σ (R_n − R_{n−1}) P_n over distinct thresholds, highest first | [KEC25], [DG06] | the federated paper's node-wise table and grid detection |
| per-bus confusion counts and the papers' threshold | `formulas.metrics.perbus_counts`, `perbus_f1_from_counts`, `tau_from_counts`, through `LearnedLocalizer.tune_threshold` | F1 = 2TP / (2TP + FP + FN); tau = argmax over the grid of the mean per-bus F1 on the active buses | [KEC25], [FED26] | the learned localizers' validation threshold, summable across clients |
| swing z-score and temporal delta | `formulas.temporal.temporal_delta`, `formulas.temporal.swing_zscore`, through `timeline.write_temporal_layers` (each observed frame against the previous emitted frame, after the walk) | delta_t = z_t − z_{t−1} at injection-metered buses; swing = delta_t / scale_t | [FED26] | the swing and temporal_delta features, `SwingThreshold`, `DeltaThreshold` |
| recent-change scale | `formulas.temporal.recent_change_scale` (over the observed frames, through `timeline.write_temporal_layers`) | scale_t = std over [t − W, t) of the one-step change, plus 1e-3 | [FED26] | the swing feature |
| replay policy | `engine.records.replay_frame` | fixed lag tau, else a random lag of at least 20 scans, else the oldest scan | [DAT26] | the Ar family (the draw is made for every in-place family to keep the random stream fixed; only Ar uses it) |
| bus voltage phasors | `formulas.network.complex_voltages` | V = \|V\| e^{jθ} | [AE04, eq. 2.1] | every AC evaluation |
| bus injections | `formulas.network.bus_injections` (numpy); `se.base.SEBase._h_t` is its torch twin, kept for callers that differentiate through it and pinned by `tests/test_formulas.py` | S = V ∘ conj(Y V) | [AE04, eq. 2.6] | the estimator's h(x) |
| AC measurement function | `formulas.network.ac_measurement`, called by `SEBase._h` | h = [\|V\|, −Re S, −Im S, θ, Re S_f, Im S_f] | [AE04, eqs. 2.6, 2.8] | every estimator, `JacobianFeatures.delta_z` |
| AC measurement Jacobian | `formulas.network.ac_jacobian`, called by `SEBase._jacobian` (the chord Jacobian in `fit`), pinned to the autograd Jacobian by `tests/test_formulas.py` | ∂S/∂θ = j diag(V) conj(diag(I) − Y diag(V)); ∂S/∂\|V\| = diag(V) conj(Y diag(V/\|V\|)) + conj(diag(I)) diag(V/\|V\|); the same for S_f with C_f and Y_f | [AE04, ch. 2], [MP19, dSbus_dV, dSbr_dV] | every estimator |
| branch flows | `formulas.network.branch_flows`; called by `engine.measurement.emit_from_state`, `clean_flows_from_states` and `dataset._clean_flows_full` | S_f = V_from ∘ conj(Y_f V) | [AE04, eq. 2.8] | every emitted flow meter, the clean flow layers, the estimator's h(x) |
| branch admittances | `formulas.network.branch_admittances`; called by `dataset._admittances` (`ybus`, `yf`, `yt`) | y_tt = y_s + (g + jb)/2, y_ff = y_tt / (t conj t), y_ft = −y_s / conj t, y_tf = −y_s / t; shunts on the diagonal | [MP19, makeYbus] | the loader's admittance matrices |
| series admittance | `formulas.network.series_admittance`, called by `FdiaGenerator._branch_physics` and `branch_admittances` | y_s = 1 / (r + jx), zero when the branch has no impedance | [MP19, branch model] | the static edge physics |
| accuracy-class error split | `formulas.noise.bias_jitter_split`, called by `FdiaGenerator.__init__` | bias² + jitter² = SD², jitter = 0.25 SD | [ASP14], the jitter fraction set in this package | every emitted measurement |
| ramp profile | `formulas.attacks.ramp_profile`, through `timeline._ramp_dev` and `timeline._AmShape` | dev(i) = rate_up·i for i < rise; peak on the hold; max(0, peak − rate_down·(i − rise − hold)) after | [DAT26] | the At and Am families |
| attacker's subnetwork | `formulas.network.subnetwork`, `FdiaGenerator.local_region` | the buses within `hops` branches of the seeds (never the slack, grown over zero-injection boundary buses) and their boundary | [WU26] | every stealthy family |
| local false state | `formulas.network.local_ac_solve`, `FdiaGenerator.solve_local` | S_i(V) = V_i conj(Σ_j Y_ij V_j) = S_target_i on the interior, every other V held true; Newton on [θ_I, \|V\|_I], each step halved until the mismatch drops (the full step first, so a plain-Newton iteration is unchanged) | [WU26], [AE04, ch. 2] | Aq, At, Al, Am |
| Am schedule | `timeline._AmShape.under_floor`, the ramp of one Am episode | rate = am_rate · floor / max_b \|δ_b\| / \|L_b\|, rise = ⌈1 / rate⌉, the ramp profile above clipped at 1, so no bus's load moves by more than am_rate of the noise floor in one frame | [WU26], closed form derived here | the Am family |
| a scan's load at a generator bus | `formulas.attacks.bus_load`, through `engine.physics.PhysicsMixin.true_load` and `scan_generation` | P_load = P_inj + s P_gen,base = s P_load,base with s = P_inj / (P_load,base − P_gen,base), the pools scaling a bus's load and generation by one factor | [DAT26] | Aq, At, Al, Am, contingency re-solves |
| a bus's scan load over its load elements | `formulas.attacks.element_loads`, through `true_load` | P_e = P_bus(e) · P0_e / Σ P0 over the bus's elements, the pools scaling every load at a bus by one factor | [DAT26] | Aq, At, Al, Am, contingency re-solves |
| operating limits of a false state | `formulas.attacks.operating_limits`, `generator_output`, `within_limits`, checked by `_stealthy_frame` | per bus V_i^min, V_i^max and the generator limits from the case; every generator's implied output P_gen − (ΔP_inj − ΔP_load), Q_gen − ΔQ_inj within its limits, the true output recovered from the pool's common load and generation scale, ΔP_load the load change the attacker pretends; a true value already outside a limit is its own bound (the false state may not make it worse); a false state outside is rejected and its step halved | [WU26, eqs. 21-23] | Aq, At, Al, Am |
| attack vector | `engine.records._attack_vector`, called by `_stealthy_frame` | a = h(x_false) - h(x_true), the noiseless reading of the local false state minus that of the true state, added to the true scan so every meter keeps its own noise draw | [WU26] | Aq, At, Al, Am |
| open attack subspace | `formulas.trust.attack_subspace`, `sparse_basis` | {H c : H_S c = 0} = H · null(H_S), its basis in reduced row echelon form | [WU26], [AE04, ch. 5] | the trusted-meter selectors |
| attack cost | `formulas.trust.attack_cost` | the fewest meters the cheapest open attack touches: the sparsest row of the echelon basis (an upper bound on the sparsest vector of the subspace, NP-hard) | [WU26] | `TrustedMeters`, the DQN reward |
| greedy trusted meters | `formulas.trust.greedy_trusted_meters`, called by `trust.TrustedMeters` | secure, on the cheapest open attack, the meter whose protection leaves the costliest cheapest attack; ties to the meter the most basis attacks pass through | [WU26], the row-reduction selection | `TrustedMeters` |
| tamper set of a stealthy frame | `engine.records._changed_meters`, called by `_stealthy_frame` | the metered channels the attack vector moves: the interior's and boundary's injections and voltages, the flows on branches touching the interior | [WU26] | Aq, At, Al, Am |
| WLS step | `formulas.estimation.wls_step` (shared weights) and `wls_step_batched` (per-record weights), called by `SEBase._solve_plain` and `_w_solve` | Δc = (BᵀWB)⁻¹BᵀW (z − h(x)) | [SCH70, part II] | every estimator |
| gain matrix | `formulas.estimation.normal_matrix`; per record `formulas.linalg.batched_normal_matrices` | G = HᵀWH | [SCH70, part II] | every estimator, the observability guard |
| weighted objective | `formulas.estimation.weighted_objective`, the divergence guard of `SEBase._w_solve` | J = Σ w_i (z_i − h_i(x))² | [SCH70] | the robust arms |
| residual covariance | `formulas.estimation.residual_covariance_diag`, `critical_measurements`, `floored_covariance`, called by `SEBase.fit` | Ω = R − H G⁻¹ Hᵀ; critical when Ω_ii < 1e-6 R_ii | [HAN75] | normalized residuals, `ResidualRemoval` |
| accuracy-class sigma | `formulas.estimation.accuracy_class_sigma`, called by `SEBase._class_sigma` (`fit(calibrate="measured")`) | σ_i = c_i \|z̄_i\| + floor (relative), σ_i = c_i (absolute) | [ASP14] | the measurement-only calibration of the detector paths |
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
