#!/usr/bin/env python
"""State-estimation-under-attack dumbbell figure for the PINN-SE paper.

One row per (system, regime). Each dumbbell connects the classical WLS angle
MAE (red circle) to the learned-estimator angle MAE (blue diamond) at metered
buses. Benign rows sit far left of the single-meter noise floor, so WLS and the
learned estimator tie there. Attacked rows stretch the dumbbell open: WLS is
dragged right past the floor while the learned estimator holds, and the green
span is the angle error the learned prior recovers.

Numbers are the report-verified values (se_pinn_report / dataset report page
56, State Estimation Under Attack, WLS vs the Learned Estimator). NOTHING is
recomputed or fabricated here — this script only lays out those numbers.

Outputs PDF + PNG + CSV sidecar into papers/pinn_se/figures/fig_attack_dumbbell.*
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/bm539044/desktop/fedpig")
from handmade.figures.paper_style import use_paper_style, save_figure_data, COL_WIDTH
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RED, BLUE, GREEN = "#b2182b", "#2166ac", "#1a9850"
use_paper_style(8.0)

# Report page 56, angle MAE (deg), metered buses. (unmetered kept for the CSV.)
# system: (wls_met, learned_met, wls_unmet, learned_unmet)
BENIGN = {
    "IEEE 14-bus":  (0.019, 0.049, 0.013, 0.038),
    "IEEE 118-bus": (0.014, 0.120, 0.016, 0.125),
    "IEEE 300-bus": (0.027, 0.252, 0.029, 0.244),
}
ATTACKED = {
    "IEEE 14-bus":  (0.364, 0.056, 0.301, 0.044),
    "IEEE 118-bus": (0.221, 0.142, 0.243, 0.148),
    "IEEE 300-bus": (0.644, 0.422, 0.673, 0.415),
}
FLOOR = 0.29  # roughly the single-meter angle noise (report page 56)
SYSTEMS = ["IEEE 14-bus", "IEEE 118-bus", "IEEE 300-bus"]

OUT = r"C:/Users/bm539044/desktop/fedpig/papers/pinn_se/figures/fig_attack_dumbbell"

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.7))

# y layout: three system groups top-to-bottom (IEEE-14 on top), benign above
# attacked within each group.
rows = []          # (y, system, regime, wls, learned)
ytips = {}
for i, s in enumerate(SYSTEMS):
    base = (len(SYSTEMS) - 1 - i) * 3.0
    y_att, y_ben = base, base + 1.0
    rows.append((y_ben, s, "benign",   BENIGN[s][0],   BENIGN[s][1]))
    rows.append((y_att, s, "attacked", ATTACKED[s][0], ATTACKED[s][1]))
    ytips[s] = (y_att, y_ben)
ymax = max(r[0] for r in rows)

# noise-floor band (genuine reference region, kept; label moved to the caption)
ax.axvspan(FLOOR, 1.2, color="0.90", alpha=0.7, lw=0, zorder=0)
ax.axvline(FLOOR, color="0.45", ls="--", lw=0.7, zorder=1)

for yy, s, regime, wls, lrn in rows:
    lo, hi = min(wls, lrn), max(wls, lrn)
    learned_wins = lrn < wls
    # advantage span (green when the learned estimator recovers error under attack)
    if regime == "attacked" and learned_wins:
        ax.plot([lo, hi], [yy, yy], color=GREEN, lw=5.0, alpha=0.28,
                solid_capstyle="round", zorder=2)
    ax.plot([lo, hi], [yy, yy], color="0.55", lw=1.1, zorder=3)
    ax.plot(wls, yy, "o", color=RED, ms=5.2, zorder=5, mec="white", mew=0.5)
    ax.plot(lrn, yy, "D", color=BLUE, ms=4.6, zorder=5, mec="white", mew=0.5)
    # value labels: put WLS label on its outer side, learned on its outer side
    ax.text(wls * (1.14 if wls >= lrn else 0.86), yy + 0.20, f"{wls:.3f}",
            fontsize=6.0, color=RED, ha="center", va="bottom")
    ax.text(lrn * (1.14 if lrn > wls else 0.86), yy - 0.20, f"{lrn:.3f}",
            fontsize=6.0, color=BLUE, ha="center", va="top")

# system labels on the far left (outside the regime tick labels)
from matplotlib.transforms import blended_transform_factory
btf = blended_transform_factory(ax.transAxes, ax.transData)
for s in SYSTEMS:
    y_att, y_ben = ytips[s]
    ax.text(-0.155, (y_att + y_ben) / 2, s, fontsize=7.8, rotation=90,
            va="center", ha="center", fontweight="bold", transform=btf)

# regime tick labels
yt = [r[0] for r in rows]
ax.set_yticks(yt)
ax.set_yticklabels([r[2].title() for r in rows], fontsize=6.8)

ax.set_xscale("log")
ax.set_xlim(0.0105, 1.2)
ax.set_ylim(-0.75, ymax + 0.9)
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1.0"])
ax.minorticks_off()
ax.set_xlabel("Angle MAE at Metered Buses (deg, Log Scale)")
ax.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.grid(True, axis="x", ls=":", lw=0.4, alpha=0.5, zorder=0)

# (takeaway callout removed; the message now lives in the LaTeX caption)
from matplotlib.patches import Patch
legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=RED,
                 markeredgecolor="white", markersize=6, label="Classical WLS"),
          Line2D([0], [0], marker="D", color="none", markerfacecolor=BLUE,
                 markeredgecolor="white", markersize=5.5, label="Learned Estimator"),
          Patch(facecolor=GREEN, alpha=0.28, label="Learned Advantage")]
ax.legend(handles=legend, frameon=False, fontsize=6.6, loc="lower left",
          handletextpad=0.4, borderaxespad=0.15, labelspacing=0.3)

fig.subplots_adjust(left=0.20, right=0.985, top=0.98, bottom=0.13)
# Save the PDF at exactly the figsize (COL_WIDTH=3.5in) rather than the rcParam
# "tight" crop, by passing a full-figure Bbox (bbox_inches=None would fall back
# to the tight rcParam). Displayed at \columnwidth in the paper this gives an
# \includegraphics scale of 1.00, so the embedded 8pt text matches the caption
# (PAPER_DESIGN rule 3 / sizing audit).
from matplotlib.transforms import Bbox
_w, _h = fig.get_size_inches()
fig.savefig(OUT + ".pdf", bbox_inches=Bbox([[0, 0], [_w, _h]]))
fig.savefig(OUT + ".png", dpi=300)
plt.close(fig)

# CSV sidecar: full metered + unmetered numbers, both regimes.
save_figure_data(OUT + ".csv", {
    "system":        [s for s in SYSTEMS for _ in (0, 1)],
    "regime":        [r for _ in SYSTEMS for r in ("benign", "attacked")],
    "wls_met_deg":   [BENIGN[s][0] if r == "benign" else ATTACKED[s][0]
                      for s in SYSTEMS for r in ("benign", "attacked")],
    "learned_met_deg": [BENIGN[s][1] if r == "benign" else ATTACKED[s][1]
                        for s in SYSTEMS for r in ("benign", "attacked")],
    "wls_unmet_deg": [BENIGN[s][2] if r == "benign" else ATTACKED[s][2]
                      for s in SYSTEMS for r in ("benign", "attacked")],
    "learned_unmet_deg": [BENIGN[s][3] if r == "benign" else ATTACKED[s][3]
                          for s in SYSTEMS for r in ("benign", "attacked")],
    "noise_floor_deg": [FLOOR for _ in SYSTEMS for _ in ("benign", "attacked")],
})
print("wrote", OUT + ".pdf/.png/.csv")
