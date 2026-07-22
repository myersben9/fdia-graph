#!/usr/bin/env python
"""Regenerate the ml-only FDIA dataset NOISE-FREE (exact h(x) / h(Xsolved) signal, zero meter noise) so the
accuracy-class noise sweep (_se_pinn_ac.py) can add controlled heteroscedastic noise at load time WITHOUT
re-solving power flows per noise level.

Why regenerate instead of re-noising the published shard: for the stealthy families (Aq/At/Al) the attack
re-solves a fake load and the state shift PROPAGATES to non-attacked metered buses (measured: Aq theta dev
0.22 deg at non-attacked metered buses vs 0.089 deg benign noise). Reconstructing those buses from the clean
pool would erase real stealth. Only a full re-solve reproduces the exact attacked measurement h(Xsolved).

Fidelity trick: the generator's noise primitive FdiaGenerator._n(s) draws exactly one standard normal then
scales by s. We rebind it to draw-one-standard-normal-then-return-0, so it CONSUMES the identical RNG budget
(keeping every attack target / multiplier / corrupt draw bit-identical to a published-shard regen) while
emitting a physically exact, noise-free measurement. Per-meter systematic biases are drawn in __init__ (RNG
advances) and then zeroed, matching the pure zero-mean accuracy-class model (no calibration bias).

Output: release_v0.4.1/ml_only_ieee{C}_clean.h5, same schema as the published shard.
Env: CASE (14). Seed 123. Knobs match _regen_release.py (per_family=6000, n_benign=36000, all defaults)."""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import fdia_graph as fg
import fdia_graph._core as core

C = int(os.environ.get("CASE", "14"))
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_v0.4.1")
POOL = os.path.join(REL, f"pool_ieee{C}.npz")
OUT = os.path.join(REL, f"ml_only_ieee{C}_clean.h5")

# ---- patch the generator to emit NOISE-FREE while preserving the RNG draw budget ----
# _n originally: return self.rng.normal(0, s)  (one standard-normal draw, scaled by s).
# Rebind: draw one standard normal (advances RNG identically) then return exactly 0.
core.FdiaGenerator._n = lambda self, s: float(self.rng.standard_normal()) * 0.0
_Orig_init = core.FdiaGenerator.__init__
def _clean_init(self, *a, **k):
    _Orig_init(self, *a, **k)                       # draws per-meter biases (RNG advances as published)
    for nm in ("bias_pi", "bias_qi", "bias_v", "bias_va", "bias_pf", "bias_qf"):
        getattr(self, nm)[:] = 0.0                  # zero the systematic bias -> pure signal
core.FdiaGenerator.__init__ = _clean_init

if __name__ == "__main__":
    states = np.load(POOL)["X"]
    print(f"[ieee{C}] pool {states.shape}; generating NOISE-FREE dataset (per_family=6000, n_benign=36000) ...", flush=True)
    t0 = time.time()
    if os.path.exists(OUT):
        os.remove(OUT)
    fg.generate(C, name=f"ieee{C}_clean", states=states, per_family=6000, n_benign=36000, out=OUT)
    print(f"[ieee{C}] DONE in {time.time()-t0:.0f}s -> {OUT}", flush=True)
