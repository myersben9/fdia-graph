#!/usr/bin/env python
"""Per-family measurement residuals on the NEW release shards, for the report's Figure 3.

residual = |measured injection - TRUE injection at the same timestep|. We report it two ways:
  dP/dQ      : RMS over ALL metered buses (whole-network aggregate deviation)
  dPatk      : RMS over the ATTACKED buses only (y==1) — the deviation where the attack actually acts
The true injection at a record's timestep comes from that system's operating-state pool (pool_ieee{C}.npz).
Writes results/ml_only_residuals_v2.npz. Env: SHARD_DIR (default release_v0.4.0), PER_FAM.
"""
import os, numpy as np, h5py
HERE = os.path.dirname(os.path.abspath(__file__))
SHARD_DIR = os.environ.get("SHARD_DIR", os.path.join(HERE, "release_v0.4.0"))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
PER_FAM = int(os.environ.get("PER_FAM", "1500"))
FAM = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
rng = np.random.default_rng(0)
out = {}

for C in (14, 118, 300):
    h5 = os.path.join(SHARD_DIR, f"ml_only_ieee{C}.h5"); pool = os.path.join(SHARD_DIR, f"pool_ieee{C}.npz")
    if not (os.path.exists(h5) and os.path.exists(pool)):
        print(f"IEEE-{C}: missing shard or pool, skipping", flush=True); continue
    X = np.load(pool)["X"]                                  # [T,N,4] = [Pinj,Qinj,|V|,theta] true states
    with h5py.File(h5, "r") as f:
        d = f["data"]; nx = d["node_x"][:]; nm = d["node_m"][:]; fam = d["family"][:]; ts = d["timestep"][:]; y = d["y"][:]
    idx = []
    for k in FAM:
        ii = np.where(fam == k)[0]
        if len(ii): idx.extend(rng.choice(ii, min(PER_FAM, len(ii)), replace=False).tolist())
    idx = np.array(sorted(idx))
    dP = np.full(len(idx), np.nan); dQ = np.full(len(idx), np.nan); dPatk = np.full(len(idx), np.nan)
    for j, i in enumerate(idx):
        t = int(ts[i])
        if t >= len(X): continue
        Xt = X[t]
        mP = nm[i, :, 1] > 0; mQ = nm[i, :, 2] > 0             # metered P / Q buses
        if mP.any(): dP[j] = np.sqrt(np.mean((nx[i, mP, 1] - Xt[mP, 0]) ** 2))
        if mQ.any(): dQ[j] = np.sqrt(np.mean((nx[i, mQ, 2] - Xt[mQ, 1]) ** 2))
        atk = (y[i] > 0) & mP                                  # attacked AND metered buses
        if atk.any(): dPatk[j] = np.sqrt(np.mean((nx[i, atk, 1] - Xt[atk, 0]) ** 2))
    fsel = fam[idx]
    for k, name in FAM.items():
        m = (fsel == k)
        if m.any():
            out[f"ieee{C}_{name}_dP"] = dP[m][np.isfinite(dP[m])]
            out[f"ieee{C}_{name}_dQ"] = dQ[m][np.isfinite(dQ[m])]
            out[f"ieee{C}_{name}_dPatk"] = dPatk[m][np.isfinite(dPatk[m])]
    print(f"IEEE-{C}: benign dP {np.nanmedian(dP[fsel==0]):.2f} | Aq dP {np.nanmedian(dP[fsel==1]):.2f} "
          f"(at attacked buses {np.nanmedian(dPatk[fsel==1]):.2f}) MW", flush=True)

np.savez_compressed(os.path.join(RES, "ml_only_residuals_v2.npz"), **out)
print(f"[done] results/ml_only_residuals_v2.npz  ({len(out)} arrays)", flush=True)
