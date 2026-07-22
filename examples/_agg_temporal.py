#!/usr/bin/env python
"""Aggregate the non-graph temporal baseline campaign (results/temporal/<sys>_<model>_s<seed>.json) into one
mean+/-std-over-seeds table per (model, system), placed next to the existing corrected ladder for the paper.

Writes results/temporal_baselines.json: for each model in {gru, tcn, xgb} and system in {14,118,300},
n seeds, mean/std swf1, per-seed swf1, mean per-family swf1 (to confirm At stays the frozen frontier), plus
mean node_f1 / det_f1. Also prints a comparison against the frozen ladder numbers."""
import os, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
TDIR = os.path.join(RES, "temporal")
FAM_ORDER = ["Aq", "Ad", "As", "Ar", "At", "Al"]
# Frozen reference ladder (results/corrected_ladder_full.json, seeds 123-125, features held fixed).
LADDER = {"mlp_full14": {"14": 0.872, "118": 0.768, "300": 0.809},
          "gcn_full14": {"14": 0.731, "118": 0.590, "300": 0.577},
          "arma_attn":  {"14": 0.904, "118": 0.811, "300": 0.846}}


def main():
    files = sorted(glob.glob(os.path.join(TDIR, "*_s*.json")))
    recs = []
    for fp in files:
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        recs.append(r)
    # group by (model, system)
    out = {}
    for model in ("gru", "tcn", "xgb"):
        out[model] = {}
        for C in ("14", "118", "300"):
            sub = [r for r in recs if r.get("model") == model and r.get("system") == f"ieee{C}"]
            if not sub:
                continue
            sw = [r["swf1"] for r in sub]
            seeds = [r["seed"] for r in sub]
            fam = {f: float(np.mean([r["per_family_swf1"].get(f, np.nan) for r in sub])) for f in FAM_ORDER}
            out[model][C] = {
                "n": len(sub), "seeds": seeds,
                "swf1_mean": round(float(np.mean(sw)), 4), "swf1_std": round(float(np.std(sw)), 4),
                "swf1_per_seed": {s: round(v, 4) for s, v in zip(seeds, sw)},
                "node_f1_mean": round(float(np.mean([r["node_f1"] for r in sub])), 4),
                "det_f1_mean": round(float(np.mean([r["det_f1"] for r in sub])), 4),
                "per_family_swf1_mean": {k: round(v, 4) for k, v in fam.items()}}
    json.dump({"note": "Non-graph temporal baselines, seeds 123-125, swf1 = per-sample macro node-F1 on "
                        "attacked records, val-tuned threshold. gru/tcn consume a raw per-bus temporal window; "
                        "xgb consumes the 14-dim engineered feature vector.",
               "windowing": "gru/tcn: length-W per-bus window of raw [V,P,Q,theta]; history = benign pool "
                            "frames masked to current availability, last frame = the record's own scan. "
                            "xgb: meas4+mask4+kcl2+temporal_delta2+swing2 (== ARMA/MLP-full14 vector).",
               "models": out, "reference_ladder": LADDER},
              open(os.path.join(RES, "temporal_baselines.json"), "w"), indent=2)

    # ---- console table ----
    print(f"\n{'model':10s} {'sys':4s}  {'swf1 mean+/-std':16s}  At     {'per-family (Aq/Ad/As/Ar/At/Al)':30s}")
    print("-" * 92)
    for model in ("xgb", "tcn", "gru"):
        for C in ("14", "118", "300"):
            d = out.get(model, {}).get(C)
            if not d:
                continue
            fam = d["per_family_swf1_mean"]
            fstr = "/".join(f"{fam[f]:.2f}" for f in FAM_ORDER)
            print(f"{model:10s} {C:4s}  {d['swf1_mean']:.3f} +/- {d['swf1_std']:.3f} (n={d['n']})  "
                  f"{fam['At']:.3f}  {fstr}")
    print("\nReference ladder (frozen, seeds 123-125):")
    for k, v in LADDER.items():
        print(f"  {k:12s} 14={v['14']:.3f}  118={v['118']:.3f}  300={v['300']:.3f}")
    print("\nwrote results/temporal_baselines.json")


if __name__ == "__main__":
    main()
