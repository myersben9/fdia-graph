#!/usr/bin/env python
"""Task 4 — RESTYLE the four seconds-cadence SE figures to gsp paper spec, from SAVED DATA (no re-running of
any experiment). Paper-ready copies go to results/paper/ (originals in results/ are kept untouched).

  fig_recursive_poc      <- results/se_recursive_poc.json                 (cadence curve, IEEE-14 DC PoC)
  fig_seconds_sandbox    <- results/sidecars/seconds_sandbox.csv          (error timeseries)
                          + results/seconds_assets/seconds_ieee14.npz     (deterministic display trajectory)
                          + pool anchors                                   (real 5-min points, no RNG)
  fig_learned_recursive  <- results/sidecars/learned_recursive.csv + se_learned_recursive.json
  fig_graph_recursive    <- results/se_graph_recursive.json               (grouped bars, zero-shot transfer)

Paper spec: handmade.figures.paper_style (Times 8 pt, embedded Type-42 fonts), single-column COL_WIDTH unless
the figure is inherently wide (2-panel -> DBL_WIDTH), NO in-figure title (the caption carries it). Each figure
re-saves PDF + PNG + a CSV sidecar next to it."""
import os, sys, csv, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/bm539044/desktop/fedpig")
from handmade.figures.paper_style import use_paper_style, save_figure_data, COL_WIDTH, DBL_WIDTH
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "paper"); os.makedirs(OUT, exist_ok=True)
RED, BLUE, PURPLE, GREEN = "#b2182b", "#2166ac", "#8073ac", "#1a9850"
use_paper_style(8.0)


def read_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr = rows[0]; data = np.array(rows[1:], dtype=float)
    return {h: data[:, i] for i, h in enumerate(hdr)}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png")); fig.savefig(os.path.join(OUT, name + ".pdf")); plt.close(fig)


# ---------- 1. recursive PoC: benign angle MAE vs cadence (IEEE-14 DC) ----------
poc = json.load(open(os.path.join(RES, "se_recursive_poc.json")))
cads = sorted((int(k) for k in poc["per_cadence"]))
yw = [poc["per_cadence"][str(c)]["th_wls_deg"] for c in cads]
yr = [poc["per_cadence"][str(c)]["th_recursive_deg"] for c in cads]
fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.3))
ax.plot(cads, yw, "o-", color=RED, ms=3.5, label="static WLS")
ax.plot(cads, yr, "s-", color=BLUE, ms=3.5, label="recursive estimator")
ax.set_xscale("log"); ax.set_xlabel("measurement cadence (s, log scale)"); ax.set_ylabel("benign angle MAE (deg)")
ax.axvspan(2, 4, color=BLUE, alpha=0.08); ax.text(2.5, min(yr) * 1.02, "real EMS\ncadence", va="bottom", fontsize=6.5, color=BLUE)
ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6); ax.legend(frameon=False, loc="center right")
fig.tight_layout(); save(fig, "fig_recursive_poc")
save_figure_data(os.path.join(OUT, "fig_recursive_poc.csv"),
                 {"cadence_s": cads, "th_wls_deg": yw, "th_recursive_deg": yr,
                  "gain_pct": [poc["per_cadence"][str(c)]["gain_pct"] for c in cads]})

# ---------- 2. seconds sandbox: real-anchored trajectory + benign-then-fooled error ----------
sb = read_csv(os.path.join(RES, "sidecars", "seconds_sandbox.csv"))
az = np.load(os.path.join(RES, "seconds_assets", "seconds_ieee14.npz"))
keep = az["keep_bus_ppc"]; th_true = az["theta_true_deg"]; tt_s = az["t_s"]
POOL = np.load(os.path.join(HERE, "release_v0.4.1", "pool_ieee14.npz"))["X"].astype(float)
NA, T0, ANCHOR_DT = 25, 1000, 300.0
j = 3                                                                         # one representative free-angle bus
th_anchor = POOL[T0:T0 + NA, keep[j], 3]; ta_min = np.arange(NA) * ANCHOR_DT / 60.0
tmin = sb["t_min"]; ew, ek = sb["err_wls_deg"], sb["err_kalman_deg"]
a0 = int(len(tmin) * 0.55); a1 = a0 + int(120 / 2.0)                          # sandbox ramp window (steps)
fig, ax = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.4))
ax[0].plot(ta_min, th_anchor, "o", color=RED, ms=3.5, label="real 5-min anchors (NYISO)")
ax[0].plot(tt_s / 60.0, th_true[:, j], "-", color=BLUE, lw=0.9, label="generated 2-s trajectory")
ax[0].set_xlabel("time (min)"); ax[0].set_ylabel("bus angle (deg)"); ax[0].legend(frameon=False, fontsize=6.6)
ax[1].axvspan(tmin[a0], tmin[a1], color=RED, alpha=0.08)
ax[1].plot(tmin, ew, color=RED, lw=1.0, label="static WLS")
ax[1].plot(tmin, ek, color=BLUE, lw=1.0, label="classical Kalman")
ax[1].annotate("stealthy ramp", xy=(tmin[a1], 0.32), xytext=(96, 0.12),
               ha="center", va="center", fontsize=6.6, color=RED,
               arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
ax[1].set_xlabel("time (min)"); ax[1].set_ylabel("angle error vs true state (deg)"); ax[1].legend(frameon=False, loc="upper left")
fig.tight_layout(); save(fig, "fig_seconds_sandbox")
save_figure_data(os.path.join(OUT, "fig_seconds_sandbox.csv"),
                 {"t_min": tmin, "err_wls_deg": ew, "err_kalman_deg": ek})

# ---------- 3. learned recursive: one attacked sequence, all three estimators ----------
lr = read_csv(os.path.join(RES, "sidecars", "learned_recursive.csv"))
lj = json.load(open(os.path.join(RES, "se_learned_recursive.json")))["attacked"]
t = lr["t_min"]
fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.4))
ax.plot(t, lr["err_wls_deg"], color=RED, lw=1.0, label=f"static WLS ({lj['atk_wls']:.3f} deg)")
ax.plot(t, lr["err_kalman_deg"], color=PURPLE, lw=1.0, label=f"classical Kalman ({lj['atk_kalman']:.3f} deg)")
ax.plot(t, lr["err_learned_deg"], color=GREEN, lw=1.3, label=f"learned recursive ({lj['atk_learned']:.3f} deg)")
ax.set_xlabel("time (min)"); ax.set_ylabel("angle error vs true benign state (deg)")
ax.legend(frameon=False, fontsize=6.6, loc="upper left"); ax.grid(True, ls=":", lw=0.4, alpha=0.6)
fig.tight_layout(); save(fig, "fig_learned_recursive")
save_figure_data(os.path.join(OUT, "fig_learned_recursive.csv"),
                 {"t_min": t, "err_wls_deg": lr["err_wls_deg"], "err_kalman_deg": lr["err_kalman_deg"],
                  "err_learned_deg": lr["err_learned_deg"]})

# ---------- 4. graph recursive: zero-shot transfer, grouped bars per system ----------
gr = json.load(open(os.path.join(RES, "se_graph_recursive.json")))["per_system"]
syst = [14, 118, 300]; x = np.arange(len(syst)); w = 0.26
fig, ax = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.4))
for a, reg, ttl in zip(ax, ("benign", "attack"), ("Benign (clean)", "Under stealthy ramp")):
    for k, (name, col, lab) in enumerate([("wls", RED, "static WLS"), ("kalman", PURPLE, "classical Kalman"),
                                          ("learned", GREEN, "learned (train@14)")]):
        a.bar(x + (k - 1) * w, [gr[str(c)][reg][name] for c in syst], w, color=col, label=lab)
    a.set_xticks(x); a.set_xticklabels([f"IEEE-{c}" + ("" if c == 14 else "\n(zero-shot)") for c in syst])
    a.set_ylabel("angle MAE (deg)"); a.grid(True, axis="y", ls=":", lw=0.4, alpha=0.6)
    vmax = max(gr[str(c)][reg][name] for c in syst for name in ("wls", "kalman", "learned"))
    a.set_ylim(0, vmax * 1.25)
    a.text(0.5, 0.97, ttl, transform=a.transAxes, ha="center", va="top", fontsize=7.5)
ax[0].legend(frameon=False, fontsize=6.4, loc="upper left")
fig.tight_layout(); save(fig, "fig_graph_recursive")
with open(os.path.join(OUT, "fig_graph_recursive.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["system", "regime", "wls", "kalman", "learned"])
    for c in syst:
        for reg in ("benign", "attack"):
            r = gr[str(c)][reg]; wr.writerow([c, reg, r["wls"], r["kalman"], r["learned"]])

print("wrote results/paper/fig_{recursive_poc,seconds_sandbox,learned_recursive,graph_recursive}.(png|pdf|csv)", flush=True)
