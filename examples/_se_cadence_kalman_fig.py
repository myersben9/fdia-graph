#!/usr/bin/env python
"""Task 1 -- regenerate the seconds-cadence resolution figure so the CLASSICAL KALMAN baseline is
EXPLICIT (reviewer: "Fig 4's temporal win is measured against static WLS, the Kalman filter is
absent, exaggerating the benefit"). Rebuilt entirely from saved data, no experiment re-run.

The point the old temporal-win heatmap hid: the recursive estimator that earns the 42-77% win over
static WLS IS a plain classical Kalman filter, not our learned method. So this figure now shows both
named baselines as their own series and states the win as the CLASSICAL KALMAN's win over static WLS.

Panel (a): benign angle MAE vs measurement cadence, static WLS (dashed, cadence-blind) against the
           classical Kalman (solid), all three systems, with the Kalman->WLS gap shaded. The learned
           recursive benign point on IEEE-14 is overlaid to show it sits right on the Kalman -- the
           learned model adds no benign benefit over the Kalman, its value is attack robustness (Fig 5).
Panel (b): the classical-Kalman temporal-win heatmap (percent reduction vs static WLS), 3 systems x
           4 cadences, three-seed means -- same numbers the prose cites, now correctly attributed.

Sources: results/se_resolution_sensitivity.json (WLS vs Kalman, 3 seeds, 14/118/300),
         results/se_learned_recursive.json (learned benign point on IEEE-14).
Output: papers/pinn_se/figures/fig_resolution_sensitivity.pdf + .csv sidecar."""
import os, sys, csv, json
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/bm539044/desktop/fedpig")
from handmade.figures.paper_style import use_paper_style, save_figure_data, COL_WIDTH
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
OUT = r"C:/Users/bm539044/desktop/fedpig/papers/pinn_se/figures"
os.makedirs(OUT, exist_ok=True)
GREY = "#666666"
COLORS = {14: "#2166ac", 118: "#1a9850", 300: "#b2182b"}   # per-system, matches the rest of the paper
GREEN = "#8073ac"                                           # learned-method accent (distinct from systems)
use_paper_style(8.0)

rj = json.load(open(os.path.join(RES, "se_resolution_sensitivity.json")))["per_system"]
sysr = [14, 118, 300]; cads = [300, 60, 15, 2]
seeds = json.load(open(os.path.join(RES, "se_resolution_sensitivity.json"))).get("seeds", [123, 124, 125])

# learned recursive benign point on IEEE-14 (aggregate mean/std over seeds), at the 2 s stream
lr = json.load(open(os.path.join(RES, "se_learned_recursive.json")))
lr_benign = lr["aggregate"]["benign.th_learned"]["mean"]
lr_benign_std = lr["aggregate"]["benign.th_learned"]["std"]
lr_kalman_b = lr["aggregate"]["benign.th_kalman"]["mean"]

win = np.array([[rj[str(s)]["per_cadence"][str(c)]["gain_pct"] for c in cads] for s in sysr])
wls = np.array([[rj[str(s)]["per_cadence"][str(c)]["th_wls_deg"] for c in cads] for s in sysr])
rec = np.array([[rj[str(s)]["per_cadence"][str(c)]["th_recursive_deg"] for c in cads] for s in sysr])

fig, (axA, axB) = plt.subplots(2, 1, figsize=(COL_WIDTH, 4.35),
                               gridspec_kw=dict(height_ratios=[1.25, 1.0], hspace=0.62))

# ---- (a) absolute benign angle MAE: static WLS (dashed) vs CLASSICAL KALMAN (solid) ----
for i, s in enumerate(sysr):
    c = COLORS[s]
    axA.plot(cads, wls[i], "--", color=c, lw=0.9, marker="o", ms=3.0, mfc="white", mec=c, zorder=3)
    axA.plot(cads, rec[i], "-", color=c, lw=1.4, marker="o", ms=3.2, zorder=4)
    axA.fill_between(cads, rec[i], wls[i], color=c, alpha=0.08, zorder=1)
# learned recursive benign point on IEEE-14 at 2 s -- sits on the Kalman, no benign gain over it
axA.errorbar([2], [lr_benign], yerr=[lr_benign_std], fmt="D", ms=4.2, color=GREEN,
             mec="white", mew=0.5, ecolor=GREEN, elinewidth=0.8, capsize=2.0, zorder=6)
axA.annotate("Learned\n(benign)", xy=(2, lr_benign), xytext=(22, 0.0245),
             fontsize=6.2, color=GREEN, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.6))
axA.set_xscale("log"); axA.set_xticks(cads); axA.set_xticklabels([f"{c}s" for c in cads])
axA.invert_xaxis()   # coarse (300s) left -> fast (2s) right, matching panel (b)
axA.set_xlabel("Measurement Cadence (Coarse Scan to Fast Stream, Log Scale)")
axA.set_ylabel("Benign Angle MAE (deg)")
axA.grid(True, which="both", ls=":", lw=0.4, alpha=0.55)
axA.axvspan(2, 4, color="#2166ac", alpha=0.06, zorder=0)
axA.set_title("(a) Static WLS vs Classical Kalman", loc="left", color=GREY, pad=3)
sys_handles = [Line2D([], [], color=COLORS[s], lw=1.4, label=f"IEEE {s}-bus") for s in sysr]
style_handles = [Line2D([], [], color=GREY, lw=0.9, ls="--", marker="o", ms=3.0, mfc="white", label="Static WLS"),
                 Line2D([], [], color=GREY, lw=1.4, marker="o", ms=3.2, label="Classical Kalman")]
leg1 = axA.legend(handles=sys_handles, frameon=True, facecolor="white", framealpha=0.92,
                  edgecolor="none", fontsize=6.3, loc="upper right",
                  handlelength=1.2, labelspacing=0.2, borderpad=0.25, ncol=1)
axA.add_artist(leg1)
axA.legend(handles=style_handles, frameon=False, fontsize=6.3, loc="lower left",
           handlelength=1.6, labelspacing=0.2, borderpad=0.15)

# ---- (b) classical-Kalman temporal-win heatmap (percent reduction vs static WLS) ----
cmg = plt.get_cmap("YlGn")
im = axB.imshow(win, cmap=cmg, aspect="auto", vmin=0, vmax=win.max() * 1.02)
axB.set_xticks(range(len(cads))); axB.set_xticklabels([f"{c}s" for c in cads])
axB.set_yticks(range(len(sysr))); axB.set_yticklabels([f"{s}-bus" for s in sysr])  # heatmap row ticks (tight)
axB.set_xlabel("Measurement Cadence (Coarse Scan to Fast Stream)")
for r in range(win.shape[0]):
    for cc in range(win.shape[1]):
        v = win[r, cc]
        axB.text(cc, r, f"{v:.0f}%", ha="center", va="center",
                 color="white" if v > win.max() * 0.6 else "black", fontsize=7.2)
for sp in axB.spines.values():
    sp.set_visible(False)
axB.tick_params(length=0)
axB.set_title("(b) Classical Kalman Temporal Win over Static WLS", loc="left", color=GREY, pad=3)
cbar = fig.colorbar(im, ax=axB, fraction=0.045, pad=0.03, aspect=14)
cbar.set_label("Win (%)", fontsize=7.0)
cbar.ax.tick_params(length=0, labelsize=6.5)

fig.savefig(os.path.join(OUT, "fig_resolution_sensitivity.pdf"), bbox_inches="tight")
plt.close(fig)

# CSV sidecar: per system/cadence WLS, Kalman, win; plus the IEEE-14 learned benign point
with open(os.path.join(OUT, "fig_resolution_sensitivity.csv"), "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["system", "cadence_s", "wls_deg", "kalman_deg", "kalman_win_pct_vs_wls"])
    for i, s in enumerate(sysr):
        for j, c in enumerate(cads):
            wr.writerow([s, c, wls[i, j], rec[i, j], win[i, j]])
    wr.writerow([])
    wr.writerow(["# ieee14 benign at 2s: learned th MAE (deg), learned std, kalman th MAE (deg)"])
    wr.writerow(["ieee14_learned_benign", 2, lr_benign, lr_benign_std, lr_kalman_b])

print("wrote fig_resolution_sensitivity (pdf+csv) with explicit WLS + Classical Kalman", flush=True)
print(f"IEEE-14 benign @2s: WLS {wls[0,3]:.4f}  Kalman {rec[0,3]:.4f}  Learned {lr_benign:.4f}+-{lr_benign_std:.4f} deg", flush=True)
