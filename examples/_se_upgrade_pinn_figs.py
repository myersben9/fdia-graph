#!/usr/bin/env python
"""Upgrade the four pinn_se paper figures away from basic line/bar plots, rebuilt ENTIRELY from
saved sidecars (no experiment re-run, no fabricated data). Outputs go straight to
papers/pinn_se/figures as PDF + a CSV sidecar next to each.

  fig_sweep                 -> stacked benign/attacked HEATMAPs of %-change vs w=0 (diverging)
  fig_resolution_sensitivity-> temporal-win HEATMAP + enriched cadence curves (win-gap shaded)
  fig_seconds_sandbox       -> enriched 2-panel: real-anchored trajectory + WLS/Kalman error with
                               shaded benign/ramp regimes and event annotations
  fig_learned_recursive     -> enriched 3-estimator error with shaded regimes and win-gap fill

Data sources (all pre-computed):
  results/sidecars/pinn_sweep.csv, results/pinn_sweep_agg.json
  results/se_resolution_sensitivity.json
  results/sidecars/seconds_sandbox.csv + results/seconds_assets/seconds_ieee14.npz + pool anchors
  results/sidecars/learned_recursive.csv + results/se_learned_recursive.json
"""
import os, sys, csv, json
import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:/Users/bm539044/desktop/fedpig")
from handmade.figures.paper_style import use_paper_style, save_figure_data, COL_WIDTH, DBL_WIDTH
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = r"C:/Users/bm539044/desktop/fedpig/papers/pinn_se/figures"
os.makedirs(OUT, exist_ok=True)
RED, BLUE, PURPLE, GREEN, GREY = "#b2182b", "#2166ac", "#8073ac", "#1a9850", "#666666"
use_paper_style(8.0)


def read_csv(path):
    with open(path) as f:
        rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
    hdr = rows[0]
    # numeric columns only; keep string first column separately if present
    try:
        data = np.array(rows[1:], dtype=float)
        return {h: data[:, i] for i, h in enumerate(hdr)}
    except ValueError:
        cols = list(zip(*rows[1:]))
        out = {}
        for i, h in enumerate(hdr):
            try:
                out[h] = np.array(cols[i], dtype=float)
            except ValueError:
                out[h] = np.array(cols[i])
        return out


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)


# =====================================================================================
# 1. fig_sweep -- stacked benign/attacked heatmaps of percent change vs w_phys = 0
# =====================================================================================
sw = read_csv(os.path.join(RES, "sidecars", "pinn_sweep.csv"))
sys_keys = ["ieee14", "ieee118", "ieee300"]
sys_lab = {"ieee14": "14-bus", "ieee118": "118-bus", "ieee300": "300-bus"}  # heatmap row ticks (tight)
weights = sorted(set(sw["w_phys"]))
wtxt = [("0" if w == 0 else f"{w:g}") for w in weights]


def pct_matrix(col):
    M = np.zeros((len(sys_keys), len(weights)))
    for r, sk in enumerate(sys_keys):
        m = sw["system"] == sk
        w = sw["w_phys"][m]; v = sw[col][m]
        order = np.argsort(w); w, v = w[order], v[order]
        base = v[np.argmin(np.abs(w - 0.0))]
        M[r] = (v - base) / base * 100.0
    return M


Mb = pct_matrix("benign_th_deg")
Ma = pct_matrix("attacked_th_deg")
vmax = max(np.abs(Mb).max(), np.abs(Ma).max())
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
cmap = plt.get_cmap("RdBu_r")

fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH, 2.95), sharex=True,
                         gridspec_kw=dict(hspace=0.16, height_ratios=[1, 1]))
for ax, M, tag in zip(axes, (Mb, Ma), ("Benign", "Attacked")):
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(weights))); ax.set_xticklabels(wtxt)
    ax.set_yticks(range(len(sys_keys))); ax.set_yticklabels([sys_lab[s] for s in sys_keys])
    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            val = M[r, c]
            txt = "0" if abs(val) < 0.05 else f"{val:+.0f}"
            shade = cmap(norm(val))
            lum = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2]
            ax.text(c, r, txt, ha="center", va="center",
                    color="white" if lum < 0.5 else "black")
    ax.text(-0.02, 1.04, tag, transform=ax.transAxes, ha="left", va="bottom",
            color=GREY, fontweight="bold")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
axes[1].set_xlabel(r"Physics Weight $w_p$   (Percent Change in Angle MAE vs $w_p{=}0$)")
# (takeaway banner removed; the message now lives in the LaTeX caption)
cbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.02, aspect=28)
cbar.set_label("Angle-MAE Change (%)")
cbar.ax.tick_params(length=0)
fig.savefig(os.path.join(OUT, "fig_sweep.pdf"), bbox_inches="tight")
plt.close(fig)
rows = []
for r, sk in enumerate(sys_keys):
    for c, w in enumerate(weights):
        rows.append([sk, w, round(Mb[r, c], 2), round(Ma[r, c], 2)])
with open(os.path.join(OUT, "fig_sweep.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["system", "w_phys", "benign_pct_vs_w0", "attacked_pct_vs_w0"])
    wr.writerows(rows)

# =====================================================================================
# 2. fig_resolution_sensitivity -- temporal-win heatmap + enriched cadence curves
# =====================================================================================
rj = json.load(open(os.path.join(RES, "se_resolution_sensitivity.json")))["per_system"]
sysr = ["14", "118", "300"]
cads = [300, 60, 15, 2]
win = np.array([[rj[s]["per_cadence"][str(c)]["gain_pct"] for c in cads] for s in sysr])
wls = np.array([[rj[s]["per_cadence"][str(c)]["th_wls_deg"] for c in cads] for s in sysr])
rec = np.array([[rj[s]["per_cadence"][str(c)]["th_recursive_deg"] for c in cads] for s in sysr])

fig, axh = plt.subplots(figsize=(COL_WIDTH, 2.35))
cmg = plt.get_cmap("YlGn")
im = axh.imshow(win, cmap=cmg, aspect="auto", vmin=0, vmax=win.max() * 1.02)
axh.set_xticks(range(len(cads))); axh.set_xticklabels([f"{c}s" for c in cads])
axh.set_yticks(range(len(sysr))); axh.set_yticklabels([f"{s}-bus" for s in sysr])
axh.set_xlabel("Measurement Cadence   (Coarse Scan to Fast Stream)")
for r in range(win.shape[0]):
    for c in range(win.shape[1]):
        v = win[r, c]
        axh.text(c, r, f"{v:.0f}%", ha="center", va="center",
                 color="white" if v > win.max() * 0.6 else "black")
for s in axh.spines.values():
    s.set_visible(False)
axh.tick_params(length=0)
# (takeaway banner removed; the message now lives in the LaTeX caption)
cbar = fig.colorbar(im, ax=axh, fraction=0.045, pad=0.03, aspect=18)
cbar.set_label("Temporal Win (%)")
cbar.ax.tick_params(length=0)
fig.savefig(os.path.join(OUT, "fig_resolution_sensitivity.pdf"), bbox_inches="tight")
plt.close(fig)
with open(os.path.join(OUT, "fig_resolution_sensitivity.csv"), "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["system", "cadence_s", "wls_deg", "recursive_deg", "win_pct"])
    for i, s in enumerate(sysr):
        for j, c in enumerate(cads):
            wr.writerow([s, c, wls[i, j], rec[i, j], win[i, j]])

# =====================================================================================
# 3. fig_seconds_sandbox -- enriched 2-panel: trajectory + WLS/Kalman error, shaded regimes
# =====================================================================================
sb = read_csv(os.path.join(RES, "sidecars", "seconds_sandbox.csv"))
az = np.load(os.path.join(RES, "seconds_assets", "seconds_ieee14.npz"))
keep = az["keep_bus_ppc"]; th_true = az["theta_true_deg"]; tt_s = az["t_s"]
POOL = np.load(os.path.join(HERE, "release_v0.4.1", "pool_ieee14.npz"))["X"].astype(float)
NA, T0, ANCHOR_DT, CAD = 25, 1000, 300.0, 2.0
j = 3
th_anchor = POOL[T0:T0 + NA, keep[j], 3]; ta_min = np.arange(NA) * ANCHOR_DT / 60.0
tmin = sb["t_min"]; ew, ek = sb["err_wls_deg"], sb["err_kalman_deg"]
RAMP0_MIN = 0.55 * tmin[-1]                     # A0 = int(Tt*0.55) in the generator
RAMP1_MIN = RAMP0_MIN + 120.0 / 60.0           # 120 s ramp

fig, ax = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.45), gridspec_kw=dict(wspace=0.22))
# left: real anchors + generated 2 s trajectory
ax[0].plot(tt_s / 60.0, th_true[:, j], "-", color=BLUE, lw=0.8, label="generated 2-s trajectory", zorder=2)
ax[0].plot(ta_min, th_anchor, "o", color=RED, ms=4.0, label="real 5-min anchors (NYISO)", zorder=3)
ax[0].set_xlabel("time (min)"); ax[0].set_ylabel("bus angle (deg)")
ax[0].legend(frameon=False, loc="lower left")
ax[0].annotate("seconds-scale motion the fast\nsampling has to resolve",
               xy=(0.5, 0.97), xycoords="axes fraction", ha="center", va="top", color=GREY)
# right: error with shaded benign / attacked regimes and event annotations
ax[1].axvspan(0, RAMP0_MIN, color=BLUE, alpha=0.05, zorder=0)
ax[1].axvspan(RAMP0_MIN, tmin[-1], color=RED, alpha=0.06, zorder=0)
mben = tmin <= RAMP0_MIN
ax[1].fill_between(tmin[mben], ek[mben], ew[mben], where=ew[mben] > ek[mben],
                   color=BLUE, alpha=0.16, zorder=1)
ax[1].plot(tmin, ew, color=RED, lw=0.8, label="static WLS", zorder=2)
ax[1].plot(tmin, ek, color=BLUE, lw=0.8, label="classical Kalman", zorder=3)
ax[1].axvline(RAMP0_MIN, color=GREY, ls=":", lw=0.8, zorder=1)
ymax = max(ew.max(), ek.max())
ax[1].set_ylim(0, ymax * 1.10)
ax[1].annotate("ramp onset", xy=(RAMP0_MIN, ymax * 0.58), xytext=(RAMP0_MIN - 24, ymax * 0.72),
               ha="center", va="center", color=GREY,
               arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))
ax[1].annotate("Kalman follows\nthe attacker", xy=(RAMP1_MIN + 3, 0.30),
               xytext=(RAMP0_MIN + 32, ymax * 0.44), ha="center", va="center", color=BLUE,
               arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.6))
ax[1].text((0 + RAMP0_MIN) / 2, ymax * 0.98, "Kalman holds\nthe noise floor (benign)",
           ha="center", va="top", color=BLUE)
ax[1].set_xlabel("time (min)"); ax[1].set_ylabel("angle error vs true state (deg)")
ax[1].legend(frameon=False, loc="lower right")
fig.savefig(os.path.join(OUT, "fig_seconds_sandbox.pdf"), bbox_inches="tight")
plt.close(fig)
save_figure_data(os.path.join(OUT, "fig_seconds_sandbox.csv"),
                 {"t_min": tmin, "err_wls_deg": ew, "err_kalman_deg": ek})

# =====================================================================================
# 4. fig_learned_recursive -- enriched 3-estimator error, shaded regimes, win-gap fill
# =====================================================================================
lr = read_csv(os.path.join(RES, "sidecars", "learned_recursive.csv"))
lj = json.load(open(os.path.join(RES, "se_learned_recursive.json")))["attacked"]
t = lr["t_min"]; ewl, ekf, eln = lr["err_wls_deg"], lr["err_kalman_deg"], lr["err_learned_deg"]
R0, R1 = 4.367, 5.90                             # deterministic rng(77): a0=131, dur=46 steps at 2 s

fig, ax = plt.subplots(figsize=(COL_WIDTH, 2.42))
ax.axvspan(0, R0, color=BLUE, alpha=0.05, zorder=0)
ax.axvspan(R0, t[-1], color=RED, alpha=0.06, zorder=0)
matk = t >= R0
ax.fill_between(t[matk], eln[matk], ekf[matk], where=ekf[matk] > eln[matk],
                color=GREEN, alpha=0.16, zorder=1)
ax.plot(t, ewl, color=RED, lw=0.8, label="static WLS", zorder=2)
ax.plot(t, ekf, color=PURPLE, lw=0.8, label="classical Kalman", zorder=3)
ax.plot(t, eln, color=GREEN, lw=1.3, label="learned recursive", zorder=4)
ax.axvline(R0, color=GREY, ls=":", lw=0.8, zorder=1)
ymax = max(ewl.max(), ekf.max(), eln.max())
ax.set_ylim(0, ymax * 1.14)
ax.annotate("ramp onset", xy=(R0, 0.03), xytext=(R0 - 1.7, ymax * 0.40), ha="center", va="center",
            color=GREY, arrowprops=dict(arrowstyle="->", color=GREY, lw=0.6))
ax.annotate("learned holds while\nclassical filters climb", xy=(7.0, eln[t >= 6.9][0]),
            xytext=(5.3, ymax * 1.06), ha="center", va="top", color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=0.6))
# final mean-MAE readout in the clear attacked-bottom band
ax.text(t[-1], 0.005,
        f"mean MAE over run (deg)\nWLS {lj['atk_wls']:.3f}   Kalman {lj['atk_kalman']:.3f}   learned {lj['atk_learned']:.3f}",
        ha="right", va="bottom", color=GREY)
ax.set_xlabel("time (min)"); ax.set_ylabel("angle error vs true benign state (deg)")
ax.legend(frameon=False, loc="upper left")
ax.grid(True, ls=":", lw=0.4, alpha=0.4)
fig.savefig(os.path.join(OUT, "fig_learned_recursive.pdf"), bbox_inches="tight")
plt.close(fig)
save_figure_data(os.path.join(OUT, "fig_learned_recursive.csv"),
                 {"t_min": t, "err_wls_deg": ewl, "err_kalman_deg": ekf, "err_learned_deg": eln})

print("wrote fig_sweep, fig_resolution_sensitivity, fig_seconds_sandbox, fig_learned_recursive "
      "(pdf+csv) to papers/pinn_se/figures", flush=True)
