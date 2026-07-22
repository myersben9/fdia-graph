#!/usr/bin/env python
"""Aggregate the flat-noise w_phys sweep (results/sweep_pinn/se_{c}_wp{wp}_s{seed}.json) into the
pinn_se paper's Table (flat rows) and fig:sweep. Groups the 3 seeds per (system, w_phys), reports
mean +/- std of the SE state error, benign and attacked, and picks the best interior w_phys.

State error we report: the swing/angle channel th_mae_se_metered (deg) is the channel that actually
moves; the voltage channel V_mae_se_metered (pu) is reported alongside. sp-MAE (pu) is the combined
per-unit state error, angle converted to radians and RMS-combined with |V|, matching the paper's
"state MAE (sp-MAE, p.u.)". No fabricated numbers: only the seeds present on disk are aggregated,
and the count is reported so partial runs are visible. CPU only, no GPU, no network."""
import os, json, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SWEEP = os.path.join(RES, "sweep_pinn")
SYS = [(14, "IEEE-14"), (118, "IEEE-118"), (300, "IEEE-300")]
WPS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
DEG2RAD = math.pi / 180.0


def load(c, wp, seed):
    # python writes wp0.0 for 0, wp0.1 etc; match the filename the sweep produced
    for tag in (f"{wp:.1f}", f"{wp:g}"):
        p = os.path.join(SWEEP, f"se_{c}_wp{tag}_s{seed}.json")
        if os.path.exists(p):
            return json.load(open(p))
    return None


def sp_mae(v_pu, th_deg):                        # combined per-unit state error (angle -> rad)
    return math.sqrt(0.5 * (v_pu ** 2 + (th_deg * DEG2RAD) ** 2))


def cell(c, wp):
    """mean/std over available seeds for (system, w_phys). Returns dict or None if no seeds."""
    rows = [load(c, wp, s) for s in (123, 124, 125)]
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    def agg(fn):
        vals = [fn(r) for r in rows]
        return float(np.mean(vals)), float(np.std(vals)), len(vals)
    bV = agg(lambda r: r["per_family"]["benign"]["V_mae_se_metered"])
    bT = agg(lambda r: r["per_family"]["benign"]["th_mae_se_metered"])
    aV = agg(lambda r: r["overall"]["V_mae_se_metered_attacked"])
    aT = agg(lambda r: r["overall"]["th_mae_se_metered_attacked"])
    bSP = agg(lambda r: sp_mae(r["per_family"]["benign"]["V_mae_se_metered"], r["per_family"]["benign"]["th_mae_se_metered"]))
    aSP = agg(lambda r: sp_mae(r["overall"]["V_mae_se_metered_attacked"], r["overall"]["th_mae_se_metered_attacked"]))
    return dict(n_seeds=bT[2], benign_V=bV, benign_th=bT, benign_sp=bSP,
                atk_V=aV, atk_th=aT, atk_sp=aSP)


out = {"noise": "flat ~1% (shard-baked)", "per_system": {}}
for c, name in SYS:
    cells = {wp: cell(c, wp) for wp in WPS}
    have = {wp: v for wp, v in cells.items() if v is not None}
    if not have:
        print(f"{name}: no results yet"); continue
    base = have.get(0.0)
    # best interior w_phys by attacked angle MAE (the regime physics is meant to help); exclude wp=0
    interior = {wp: v for wp, v in have.items() if wp > 0}
    best_wp = min(interior, key=lambda wp: interior[wp]["atk_th"][0]) if interior else 0.0
    best = have[best_wp]
    def delta(metric):  # percent improvement of best vs wp=0, positive = physics better
        if base is None: return None
        b0 = base[metric][0]; bb = best[metric][0]
        return round(100.0 * (b0 - bb) / b0, 1) if b0 else None
    out["per_system"][f"ieee{c}"] = dict(
        name=name, seeds=base["n_seeds"] if base else 0, best_wp=best_wp,
        wp0=base, best=best,
        d_benign_sp=delta("benign_sp"), d_atk_sp=delta("atk_sp"),
        d_benign_th=delta("benign_th"), d_atk_th=delta("atk_th"),
        curve={f"{wp:g}": dict(benign_th=have[wp]["benign_th"][0], atk_th=have[wp]["atk_th"][0],
                               benign_sp=have[wp]["benign_sp"][0], atk_sp=have[wp]["atk_sp"][0],
                               n=have[wp]["n_seeds"]) for wp in sorted(have)})
    print(f"{name}: seeds/cell={base['n_seeds'] if base else 0}, best wp={best_wp}, "
          f"benign th {base['benign_th'][0]:.3f}->{best['benign_th'][0]:.3f} deg (d={delta('benign_th')}%), "
          f"attacked th {base['atk_th'][0]:.3f}->{best['atk_th'][0]:.3f} deg (d={delta('atk_th')}%)")

json.dump(out, open(os.path.join(RES, "pinn_sweep_agg.json"), "w"), indent=2)

# ---- fig:sweep — attacked + benign angle MAE vs w_phys, one color per system ----
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.edgecolor": "#222", "axes.linewidth": 0.8})
cols = {"ieee14": "#1a9850", "ieee118": "#2166ac", "ieee300": "#b2182b"}
fig, ax = plt.subplots(figsize=(4.8, 3.3))
for c, name in SYS:
    k = f"ieee{c}"
    if k not in out["per_system"]: continue
    cv = out["per_system"][k]["curve"]; xs = [float(w) for w in cv]
    ax.plot(xs, [cv[w]["atk_th"] for w in cv], "o-", color=cols[k], lw=1.7, ms=4, label=f"{name} (attacked)")
    ax.plot(xs, [cv[w]["benign_th"] for w in cv], "o--", color=cols[k], lw=1.2, ms=3, alpha=0.55, label=f"{name} (benign)")
ax.set_xlabel(r"physics weight $w_{\mathrm{phys}}$"); ax.set_ylabel("SE angle MAE (deg)")
ax.set_title("State error vs physics weight (flat ~1% noise)", fontsize=8.8)
ax.legend(fontsize=7, frameon=False, ncol=1, loc="best"); ax.grid(True, ls=":", lw=0.4, alpha=0.6)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_sweep.png"), dpi=175); fig.savefig(os.path.join(RES, "fig_sweep.pdf"))
with open(os.path.join(RES, "sidecars", "pinn_sweep.csv"), "w") as f:
    f.write("system,w_phys,benign_th_deg,attacked_th_deg,benign_sp_pu,attacked_sp_pu,n_seeds\n")
    for c, name in SYS:
        k = f"ieee{c}"
        if k not in out["per_system"]: continue
        for w, v in out["per_system"][k]["curve"].items():
            f.write(f"{k},{w},{v['benign_th']},{v['atk_th']},{v['benign_sp']},{v['atk_sp']},{v['n']}\n")
print("wrote results/pinn_sweep_agg.json + fig_sweep.(png|pdf) + sidecars/pinn_sweep.csv")
