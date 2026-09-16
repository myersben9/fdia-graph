# Formulas: from the equation in the source to the function in the code

Every operation in the package that appears in a paper or a textbook gets one named function,
even when it is called once, so a reader can go from an equation to the code and back. This
catalogue is the index: one row per formula, the function that implements it, the equation in
plain text, the source with a key from [`REFERENCES.md`](REFERENCES.md), and where it is used.

The plan behind it is [`../READABILITY_PLAN.md`](../READABILITY_PLAN.md) (rule 3) and
[`../RESTRUCTURE_PLAN.md`](../RESTRUCTURE_PLAN.md) (the `formulas/` kernel). Rows are added as
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
| swing z-score and temporal delta | `generation._record_features` (shards) and `streams._StreamBuffers.store` (streams, against the previous emitted frame); planned `formulas.temporal.swing_zscore`, `temporal_delta` | delta_t = z_t − z_{t−1} at injection-metered buses; swing = delta_t / scale_t | [FED26] | the swing and temporal_delta features, `SwingThreshold`, `DeltaThreshold` |
| recent-change scale | `generation._swing_scale` (shared by shards and streams); planned `formulas.temporal.recent_change_scale` | scale_t = std over [t − W, t) of the one-step change, floor 1e-3 | [FED26] | the swing feature |
| replay policy | `engine.records.replay_frame` | fixed lag tau, else a random lag of at least 20 scans, else the oldest scan | [DAT26] | the Ar and As families |
| bus voltage phasors | `formulas.network.complex_voltages` | V = \|V\| e^{jθ} | [AE04, eq. 2.1] | every AC evaluation |
| bus injections | `formulas.network.bus_injections` (numpy); `se.base.SEBase._h_t` is its torch twin for the autograd Jacobian, pinned by `tests/test_formulas.py` | S = V ∘ conj(Y V) | [AE04, eq. 2.6] | the estimator's h(x) |
| branch flows | `formulas.network.branch_flows`; called by `engine.measurement.emit_from_state`, `clean_flows_from_states` and `dataset._clean_flows_full` | S_f = V_from ∘ conj(Y_f V) | [AE04, eq. 2.8] | every emitted flow meter, the clean flow layers, the estimator's h(x) |
| branch admittances | `formulas.network.branch_admittances`; called by `dataset._admittances` (`ybus`, `yf`, `yt`) | y_tt = y_s + (g + jb)/2, y_ff = y_tt / (t conj t), y_ft = −y_s / conj t, y_tf = −y_s / t; shunts on the diagonal | [MP19, makeYbus] | the loader's admittance matrices |
| series admittance | `formulas.network.series_admittance`, called by `FdiaGenerator._branch_physics` and `branch_admittances` | y_s = 1 / (r + jx), zero when the branch has no impedance | [MP19, branch model] | the static edge physics |
| accuracy-class error split | `formulas.noise.bias_jitter_split`, called by `FdiaGenerator.__init__` | bias² + jitter² = SD², jitter = 0.25 SD | [ASP14], our split | every emitted measurement |
| ramp profile | `generation._ramp_profile` (shared by shards and streams); planned `formulas.attacks.ramp_profile` | dev(i) = rate_up·i for i < rise; peak on the hold; max(0, peak − rate_down·(i − rise − hold)) after | [DAT26] | the At family |
| WLS step | *today:* `se.base.SEBase._solve_plain`; planned `formulas.estimation.wls_step` | Δx = (HᵀWH)⁻¹HᵀW (z − h(x)) | [SCH70, part II] | every estimator |
| normalized residual | *today:* `se.base.SEBase._nres`; planned `formulas.estimation.normalized_residual` | r_N,i = (z_i − h_i(x̂)) / √Ω_ii, Ω = R − H G⁻¹ Hᵀ | [HAN75] | Huber, residual removal, `ResidualLocalizer` |
| Huber weight | *today:* inline in `se.methods`; planned `formulas.estimation.huber_weights` | a_i = min(1, c / |r_N,i|) | [HUB64] | `AdaptiveWeighting`, `SubspacePrior`, `JacobianWeighting` |
| explained / unexplained split | *today:* `se.jacobian.JacobianFeatures.transform`; planned `formulas.projection.explained_unexplained` | r∥ = H G⁻¹HᵀW Δz, r⊥ = Δz − r∥ | [JAC26] | Jacobian features, `JacobianWeighting` |
