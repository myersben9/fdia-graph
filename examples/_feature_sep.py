#!/usr/bin/env python
"""Measure WHY the engineered features work: per-feature separability of attacked vs benign buses, per family.

For each system we pull the test split and, over the attacked records of each family (EXCLUDING At — the temporal
family Ben wants left out so the picture is clean), score how well each per-bus feature distinguishes the buses that
are actually attacked (y==1) from the untouched buses (y==0) in the same records. Separability = ROC-AUC (0.5 = no
signal, 1.0 = perfect). This says, concretely, which feature carries the localization signal for which attack —
e.g. the windowed swing and temporal-delta light up on the abrupt families while the raw injection barely moves on
the stealthy ones. Writes results/feature_sep.json (the report reads it; no plotting here).
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, h5py
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
# Read the LOCAL v0.4.x release shards directly: they carry the swing feature, whereas fg.load still resolves to the
# published v0.3.0 cache (no swing) until we cut the v0.4.1 release. Same source the report's load_sys prefers.
SH = os.environ.get("FDIA_LOCAL_SHARDS", os.path.join(HERE, "release_v0.4.0"))
FAM = {1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 6: "Al"}     # non-At attack families (At=5 excluded on purpose)
out = {"note": "per-bus separability (ROC-AUC) of attacked vs untouched buses within each family's attacked records; At excluded",
       "systems": {}}

for C in (14, 118, 300):
    h5 = os.path.join(SH, f"ml_only_ieee{C}.h5")
    if not os.path.exists(h5):
        print(f"ieee{C}: no local shard, skip", flush=True); continue
    with h5py.File(h5, "r") as f:
        d = f["data"]
        nx = d["node_x"][:]; nm = d["node_m"][:].astype(bool); y = d["y"][:].astype(bool); fam = d["family"][:]
        sw = d["swing"][:] if "swing" in d else None
        td = d["temporal_delta"][:] if "temporal_delta" in d else None

    # per-bus candidate features (magnitude over P/Q channels where the feature is 2-wide)
    feats = {}
    feats["|P_inj| reading"] = np.abs(nx[:, :, 1])                       # raw metered injection magnitude
    if td is not None: feats["temporal delta"] = np.linalg.norm(td, axis=2)   # |current - previous scan|
    if sw is not None: feats["swing (windowed)"] = np.linalg.norm(sw, axis=2)  # recent-window rate-of-change z-score
    # metered-injection mask so we only score buses the model can actually see this feature on
    seen = nm[:, :, 1]

    sysres = {}
    for code, name in FAM.items():
        recs = np.where(fam == code)[0]
        if len(recs) == 0: continue
        yy = y[recs]; mm = seen[recs]                                   # [r, N]
        per_feat = {}
        for fname, F in feats.items():
            v = F[recs]
            lab = yy[mm]; val = v[mm]                                   # pool over metered buses of these records
            # need both classes present to score
            if lab.sum() < 20 or (~lab).sum() < 20:
                per_feat[fname] = None; continue
            try:
                per_feat[fname] = round(float(roc_auc_score(lab, val)), 3)
            except Exception:
                per_feat[fname] = None
        sysres[name] = {"n_records": int(len(recs)), "auc": per_feat}
    out["systems"][f"ieee{C}"] = sysres
    msg = "  ".join(f"{fm}:" + "/".join(f"{v}" for v in sysres[fm]["auc"].values()) for fm in sysres)
    print(f"ieee{C}  (feat order {list(feats)}):  {msg}", flush=True)

json.dump(out, open(os.path.join(RES, "feature_sep.json"), "w"), indent=2)
print("[done] results/feature_sep.json", flush=True)
