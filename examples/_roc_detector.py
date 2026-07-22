#!/usr/bin/env python
"""Temporal rate-of-change detector on the v0.4.0 data, for the report page.

Detector: flag a record if ANY metered bus's injection jump this scan exceeds its TYPICAL recent-window jump
by more than a threshold (a per-bus rate-of-change z-score, max over buses). The window W is tuned to catch
the most attacks at a fixed 5% benign false-alarm rate. This is the SAME statistic fed to the model as the
'swing' feature. Writes results/roc_detector.json (per-system per-family catch + the window-tuning sweep).

Env: SEED (default 123; multi-seed error-bar runs point at ml_only_ieee{C}_s{SEED}.h5 / pool_ieee{C}_s{SEED}.npz
and write roc_detector_s{SEED}.json so per-seed results don't clobber each other or the seed-123 canonical file).
"""
import os, json, numpy as np, h5py
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SH = os.environ.get("FDIA_LOCAL_SHARDS", os.path.join(HERE, "release_v0.4.1"))
FAM = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
W_TUNED = 60
SEED = int(os.environ.get("SEED", "123"))
SEED_SUF = "" if SEED == 123 else f"_s{SEED}"
OUT_NAME = "roc_detector.json" if SEED == 123 else f"roc_detector_s{SEED}.json"
out = {"window_tuned": W_TUNED, "benign_fa_target": 5.0, "systems": {}}

for C in (14, 118, 300):
    h5 = os.path.join(SH, f"ml_only_ieee{C}{SEED_SUF}.h5"); pool = os.path.join(SH, f"pool_ieee{C}{SEED_SUF}.npz")
    if not (os.path.exists(h5) and os.path.exists(pool)):
        print(f"ieee{C}: missing shard/pool, skip", flush=True); continue
    X = np.load(pool)["X"]
    with h5py.File(h5, "r") as f:
        d = f["data"]; nx = d["node_x"][:]; nm = d["node_m"][:]; fam = d["family"][:]; ts = d["timestep"][:]
    rng = np.random.default_rng(0)
    idx = []
    for k in FAM:
        ii = np.where(fam == k)[0]; idx.extend(rng.choice(ii, min(800, len(ii)), replace=False).tolist())
    idx = np.array(idx); mP = nm[:, :, 1] > 0

    def stat(W):
        o = np.zeros(len(idx))
        for j, i in enumerate(idx):
            t = int(ts[i]); w0 = max(0, t - W)
            if t - w0 < 4 or t < 1: continue
            # Normalize each bus's jump by max(its natural scan-to-scan variation, its meter-noise level). The noise
            # floor is essential on large grids: the numerator compares the NOISY measured value to the noiseless true
            # previous state, so it carries ~1.7% relative meter noise. Without a proportional floor, a big flat-load
            # bus (hundreds of MW on IEEE-300) divides ~3 MW of noise by ~0 variation -> a spurious z-score in the
            # thousands that dominates the max and kills the detector. Flooring at the accuracy-class sigma fixes it.
            var = np.abs(np.diff(X[w0:t, :, :2], axis=0)).std(0)
            noise_floor = 0.017 * np.abs(X[t - 1, :, :2]) + 0.05          # P/Q accuracy-class sigma (~1.7%) + small abs MW floor
            scale = np.maximum(var, noise_floor)
            m = mP[i]
            if m.any(): o[j] = (np.abs(nx[i, m, 1] - X[t - 1, m, 0]) / scale[m, 0]).max()
        return o

    fsel = fam[idx]
    # tuning sweep (all-attack catch vs window)
    sweep = {}
    for W in (5, 10, 15, 20, 30, 45, 60, 90):
        s = stat(W); thr = np.percentile(s[fsel == 0], 95)
        sweep[W] = round(100 * float(np.mean(s[fsel > 0] > thr)), 1)
    # per-family catch at the tuned window
    s = stat(W_TUNED); thr = float(np.percentile(s[fsel == 0], 95))
    per_family = {FAM[k]: round(100 * float(np.mean(s[fsel == k] > thr)), 1) for k in FAM}
    out["systems"][f"ieee{C}"] = {"threshold": round(thr, 1), "per_family": per_family, "sweep": sweep,
                                  "all_attack_catch": round(100 * float(np.mean(s[fsel > 0] > thr)), 1)}
    print(f"ieee{C}: benign {per_family['benign']}%  Aq {per_family['Aq']}%  At {per_family['At']}%  Al {per_family['Al']}%  (all atk {out['systems'][f'ieee{C}']['all_attack_catch']}%)", flush=True)

json.dump(out, open(os.path.join(RES, OUT_NAME), "w"), indent=2)
print(f"[done] results/{OUT_NAME}", flush=True)
