#!/usr/bin/env python
"""Merge the per-system E7 Aq-detector-overlay results (results/_ao{C}.json, each written by a single-system
run of _ao_detector_overlay.py) into a combined results/ao_detector_overlay.json, and regenerate the figure
(results/fig_ao_detector_overlay.png|pdf) + CSV sidecar with BOTH systems. No recomputation, no fabricated
numbers: it only reshapes and plots the per-system results already computed on the pipeline."""
import os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SYSTEMS = [(14, "IEEE-14"), (118, "IEEE-118")]

parts = {}
for C, _ in SYSTEMS:
    p = os.path.join(RES, f"_ao{C}.json")
    if os.path.exists(p):
        parts[f"ieee{C}"] = json.load(open(p))
    else:
        print(f"[warn] missing {p} -- skipping ieee{C}")
if not parts:
    raise SystemExit("no per-system results found (results/_ao{14,118}.json)")

any_r = next(iter(parts.values()))
combined = {"attack": any_r["attack"], "detectors": any_r["detectors"], "fa_target_pct": any_r["fa_target_pct"],
            "k_buses": any_r["k_buses"], "note": any_r["note"], "systems": parts,
            "cross50": {k: v["cross50"] for k, v in parts.items()}}
json.dump(combined, open(os.path.join(RES, "ao_detector_overlay.json"), "w"), indent=2)
print("wrote results/ao_detector_overlay.json with systems:", list(parts.keys()))
for k, v in parts.items():
    print(f"  {k}: cross50 BDD {v['cross50']['bdd_chi2']}  swing {v['cross50']['swing']}  loc {v['cross50']['loc']}")

# ---- figure: one panel per system, three detector curves each ----
INK = "#222222"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "axes.linewidth": 0.8})
series = [("bdd_chi2_dr", "BDD chi-square", "#b2182b", "o-"),
         ("swing_dr", "swing detector (5% FA)", "#2166ac", "s-"),
         ("loc_dr5fa", "ARMA localizer DR@5%FA", "#1a9850", "^-")]
order = [k for k in ("ieee14", "ieee118") if k in parts]
fig, axes = plt.subplots(1, len(order), figsize=(4.7 * len(order), 3.4), sharey=True, squeeze=False)
for ax, k in zip(axes[0], order):
    curve = parts[k]["curve"]; xs = [c["scale_pct"] for c in curve]
    for key, lab, col, sty in series:
        ax.plot(xs, [c[key] for c in curve], sty, color=col, lw=1.8, ms=4.5, label=lab)
    ax.axhline(50, ls="--", lw=0.8, color=INK, alpha=0.45)
    ax.set_xscale("log"); ax.set_xlabel("Aq magnitude (load over-scaling at attacked buses, %)")
    ax.set_ylim(-3, 103); ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    ax.set_title(dict(SYSTEMS)[int(k[4:])], fontsize=9.2)
axes[0][0].set_ylabel("detection rate (%)")
axes[0][0].legend(fontsize=8.0, frameon=False, loc="center left")
fig.suptitle("Only measurement-aware detectors lift off; BDD never fires on stealthy Aq", fontsize=9.0, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_ao_detector_overlay.png"), dpi=175, bbox_inches="tight")
fig.savefig(os.path.join(RES, "fig_ao_detector_overlay.pdf"), bbox_inches="tight")
with open(os.path.join(RES, "sidecars", "ao_detector_overlay.csv"), "w") as f:
    f.write("system,scale_pct,factor,bdd_chi2_dr,swing_dr,loc_dr5fa,n\n")
    for k in order:
        for c in parts[k]["curve"]:
            f.write(f"{k},{c['scale_pct']},{c['factor']},{c['bdd_chi2_dr']},{c['swing_dr']},{c['loc_dr5fa']},{c['n']}\n")
print("wrote fig_ao_detector_overlay.(png|pdf) + sidecars/ao_detector_overlay.csv")
