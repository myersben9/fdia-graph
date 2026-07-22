#!/usr/bin/env python
"""Consolidate the two wide seconds-cadence figures (fig_seconds_sandbox + fig_learned_recursive)
into ONE compact single-column 2x2 small-multiple, Boyaci-style. Rebuilt ENTIRELY from saved
sidecars, no experiment re-run, no fabricated data.

  (a) two-second angle trajectory with the real five-minute anchors
  (b) static WLS vs classical Kalman angle error, benign-then-ramp timeline (shaded regimes)
  (c) learned recursive vs the two classical filters, benign-then-ramp (shaded + win-gap fill)
  (d) per-method mean angle MAE, benign vs attacked, as an annotated dumbbell (not a bar chart)

Data sources (all pre-computed):
  results/sidecars/seconds_sandbox.csv   + results/seconds_assets/seconds_ieee14.npz + pool anchors
  results/sidecars/learned_recursive.csv + results/se_learned_recursive.json
"""
import os, sys, csv, json
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/bm539044/desktop/fedpig")
from handmade.figures.paper_style import use_paper_style, save_figure_data, COL_WIDTH
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = r"C:/Users/bm539044/desktop/fedpig/papers/pinn_se/figures"
os.makedirs(OUT, exist_ok=True)
RED, BLUE, PURPLE, GREEN, GREY = "#b2182b", "#2166ac", "#8073ac", "#1a9850", "#666666"
use_paper_style(8.0)
# Pin the output to EXACTLY the single-column figsize (disable tight bbox) so the 8pt
# in-figure text renders at true 8pt when \includegraphics places it at \columnwidth.
matplotlib.rcParams["savefig.bbox"] = None


def read_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr = rows[0]
    data = np.array(rows[1:], dtype=float)
    return {h: data[:, i] for i, h in enumerate(hdr)}


# ---- load the two seconds-cadence sandboxes -----------------------------------------
sb = read_csv(os.path.join(RES, "sidecars", "seconds_sandbox.csv"))
tmin, ew, ek = sb["t_min"], sb["err_wls_deg"], sb["err_kalman_deg"]
RAMP0 = 0.55 * tmin[-1]                              # A0 = int(Tt*0.55) in the generator

lr = read_csv(os.path.join(RES, "sidecars", "learned_recursive.csv"))
t = lr["t_min"]; ewl, ekf, eln = lr["err_wls_deg"], lr["err_kalman_deg"], lr["err_learned_deg"]
R0 = 4.367                                           # deterministic rng(77): a0=131 at 2 s

lj = json.load(open(os.path.join(RES, "se_learned_recursive.json")))
ben, atk = lj["benign"], lj["attacked"]

az = np.load(os.path.join(RES, "seconds_assets", "seconds_ieee14.npz"))
keep = az["keep_bus_ppc"]; th_true = az["theta_true_deg"]; tt_s = az["t_s"]
POOL = np.load(os.path.join(HERE, "release_v0.4.1", "pool_ieee14.npz"))["X"].astype(float)
NA, T0, ANCHOR_DT = 25, 1000, 300.0
j = 3
th_anchor = POOL[T0:T0 + NA, keep[j], 3]; ta_min = np.arange(NA) * ANCHOR_DT / 60.0

# =====================================================================================
fig, axes = plt.subplots(2, 2, figsize=(COL_WIDTH, 3.5))
(aA, aB), (aC, aD) = axes
fig.subplots_adjust(left=0.145, right=0.965, top=0.925, bottom=0.11, wspace=0.46, hspace=0.66)


def panel_tag(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", color=GREY, pad=3)


# ---- (a) real-anchored trajectory ---------------------------------------------------
aA.plot(tt_s / 60.0, th_true[:, j], "-", color=BLUE, lw=0.7, zorder=2)
aA.plot(ta_min, th_anchor, "o", color=RED, ms=2.6, zorder=3)
aA.set_xlabel("Time (min)"); aA.set_ylabel("Bus Angle (deg)")
aA.set_xticks([0, 60, 120])
aA.legend(handles=[Line2D([], [], color=BLUE, lw=0.9, label="2-s Stream"),
                   Line2D([], [], color=RED, marker="o", ls="none", ms=2.6, label="5-min Anchors")],
          frameon=False, loc="lower center", handlelength=1.1, handletextpad=0.4,
          borderpad=0.1, labelspacing=0.2)
panel_tag(aA, "a", "Real-Anchored Trajectory")

# ---- (b) WLS vs Kalman over the benign-then-ramp timeline ---------------------------
aB.axvspan(0, RAMP0, color=BLUE, alpha=0.05, zorder=0)
aB.axvspan(RAMP0, tmin[-1], color=RED, alpha=0.06, zorder=0)
aB.plot(tmin, ew, color=RED, lw=0.6, zorder=2)
aB.plot(tmin, ek, color=BLUE, lw=0.6, zorder=3)
aB.axvline(RAMP0, color=GREY, ls=":", lw=0.7, zorder=1)
ymaxB = max(ew.max(), ek.max())
aB.set_ylim(0, ymaxB * 1.12)
aB.set_xticks([0, 60, 120])
# (ramp-onset and "Kalman follows" callouts removed; series identified by legend, message in caption)
aB.legend(handles=[Line2D([], [], color=RED, lw=0.9, label="WLS"),
                   Line2D([], [], color=BLUE, lw=0.9, label="Kalman")],
          frameon=False, loc="upper left", handlelength=1.0, handletextpad=0.4,
          borderpad=0.1, labelspacing=0.2)
aB.set_xlabel("Time (min)"); aB.set_ylabel("Angle Error (deg)")
panel_tag(aB, "b", "WLS vs Kalman")

# ---- (c) learned recursive vs the two classical filters -----------------------------
aC.axvspan(0, R0, color=BLUE, alpha=0.05, zorder=0)
aC.axvspan(R0, t[-1], color=RED, alpha=0.06, zorder=0)
matk = t >= R0
aC.fill_between(t[matk], eln[matk], ekf[matk], where=ekf[matk] > eln[matk],
                color=GREEN, alpha=0.16, zorder=1)
aC.plot(t, ewl, color=RED, lw=0.55, zorder=2)
aC.plot(t, ekf, color=PURPLE, lw=0.55, zorder=3)
aC.plot(t, eln, color=GREEN, lw=1.1, zorder=4)
aC.axvline(R0, color=GREY, ls=":", lw=0.7, zorder=1)
ymaxC = max(ewl.max(), ekf.max(), eln.max())
aC.set_ylim(0, ymaxC * 1.16)
aC.set_xlabel("Time (min)"); aC.set_ylabel("Angle Error (deg)")
aC.legend(handles=[Line2D([], [], color=RED, lw=0.9, label="WLS"),
                   Line2D([], [], color=PURPLE, lw=0.9, label="Kalman"),
                   Line2D([], [], color=GREEN, lw=1.3, label="Learned")],
          frameon=False, loc="upper left", handlelength=1.0, handletextpad=0.4,
          borderpad=0.1, labelspacing=0.2, ncol=1)
# ("learned margin" callout removed; the shaded gap carries it, explained in the caption)
panel_tag(aC, "c", "Learned Recursive")

# ---- (d) per-method mean MAE, benign vs attacked, as a dumbbell ---------------------
methods = [("WLS", RED, ben["th_wls"], atk["atk_wls"]),
           ("Kalman", PURPLE, ben["th_kalman"], atk["atk_kalman"]),
           ("Learned", GREEN, ben["th_learned"], atk["atk_learned"])]
ypos = [2, 1, 0]
for y, (name, col, b, a) in zip(ypos, methods):
    aD.plot([b, a], [y, y], color=col, lw=1.3, zorder=2, solid_capstyle="round")
    aD.plot(b, y, "o", mfc="white", mec=col, mew=1.0, ms=5.0, zorder=3)
    aD.plot(a, y, "o", color=col, ms=5.0, zorder=3)
    aD.annotate(f"+{a - b:.2f}", xy=((b + a) / 2, y + 0.20), ha="center", va="bottom",
                color=GREY)
aD.set_yticks(ypos); aD.set_yticklabels([m[0] for m in methods])
aD.set_ylim(-0.55, 2.62)
aD.set_xlim(0, max(atk["atk_wls"], atk["atk_kalman"]) * 1.12)
aD.set_xlabel("Mean Angle MAE (deg)")
# open marker = benign, filled = under ramp (stated in the caption); the "+delta" over each
# dumbbell is the ramp damage, exactly the benign-to-attacked jump also listed in Table IV.
panel_tag(aD, "d", "Mean MAE per Method")

fig.savefig(os.path.join(OUT, "fig_seconds_panels.pdf"), bbox_inches=None)
plt.close(fig)

save_figure_data(os.path.join(OUT, "fig_seconds_panels.csv"), {
    "panel": (["a"] * len(tt_s) + ["b"] * len(tmin) + ["c"] * len(t) + ["d"] * 3),
    "x": (list(tt_s / 60.0) + list(tmin) + list(t) + [0, 1, 2]),
    "series1": (list(th_true[:, j]) + list(ew) + list(ewl) + [ben["th_wls"], ben["th_kalman"], ben["th_learned"]]),
    "series2": (["" ] * len(tt_s) + list(ek) + list(ekf) + [atk["atk_wls"], atk["atk_kalman"], atk["atk_learned"]]),
    "series3": (["" ] * (len(tt_s) + len(tmin)) + list(eln) + ["", "", ""]),
})
print("wrote fig_seconds_panels (pdf+csv) to papers/pinn_se/figures", flush=True)
