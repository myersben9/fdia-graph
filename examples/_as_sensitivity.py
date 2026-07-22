#!/usr/bin/env python
"""E8 -- As (meter-scaling attack) BDD sensitivity, the contrast to the Aq/Ao load-move sweep.

As multiplies the metered P/Q of the attacked buses (and their incident branch flows) by a gain factor s
near 1. Unlike Aq (which re-solves a fully power-flow-consistent counterfactual and stays on the physics
manifold), As tampers the measurements in place: the scaled injections/flows no longer agree with the
neighbours' un-scaled readings, so the WLS residual blows up. The story this figure tells: As becomes
BDD-detectable at a MUCH smaller magnitude than the load-move Aq needs, because As breaks physics
consistency while Aq does not.

For a grid of scale factors s (0.2% .. 50% over-reading), we apply As to k random active-load buses on real
benign operating points, emit the measurement graph with the identical accuracy-class meter model used
everywhere, run the pandapower WLS + chi-square bad-data test, and report the fraction of records flagged.
The s=1.0 row is the benign false-positive floor. Numbers are computed, never fabricated.

Env: CASE (14/118/300, default 118), N_SAMP (samples per magnitude), SHARD_DIR (default release_v0.4.1).
CPU-safe. Writes results/as_sensitivity.json + results/fig_as_sensitivity.(png|pdf) + a CSV sidecar.
"""
import os, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdia_graph._core import FdiaGenerator
import pandapower as pp, pandapower.networks as pn
from pandapower.create import create_measurement
from pandapower.estimation import chi2_analysis

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
C = int(os.environ.get("CASE", "118")); NS = int(os.environ.get("N_SAMP", "80")); K = 3
SH = os.environ.get("SHARD_DIR", os.path.join(HERE, "release_v0.4.1"))
NET = {14: pn.case14, 118: pn.case118, 300: pn.case300}[C]
# accuracy-class meter std-devs used to weight the WLS measurements (identical to _bdd_release.py / the generator)
SD = dict(pf=0.017, qf=0.017, v=0.0012, pi=0.017, qi=0.017, va=0.00168)
# scale-factor tiers: fine near 1 (to catch the detection knee) up to the shard As range (1.25-1.5)
TIERS = [1.0, 1.002, 1.005, 1.01, 1.02, 1.03, 1.05, 1.08, 1.12, 1.20, 1.35, 1.50]

base = NET(); pp.runpp(base); nl = len(base.line); ntr = len(base.trafo)


def build_meas(nx, nm, ex, em):
    """Assemble a pandapower state-estimation net from an emitted measurement graph (same construction the
    BDD release script uses): flow meters on lines/trafos, P/Q injection + |V| + angle meters on buses."""
    est = NET()
    for e in range(nl):
        if em[e, 0]: create_measurement(est, "p", "line", ex[e, 0], max(abs(ex[e, 0]) * SD["pf"], 1e-3), element=e, side="from")
        if em[e, 1]: create_measurement(est, "q", "line", ex[e, 1], max(abs(ex[e, 1]) * SD["qf"], 1e-3), element=e, side="from")
    for ti in range(ntr):
        e = nl + ti
        if em[e, 0]: create_measurement(est, "p", "trafo", ex[e, 0], max(abs(ex[e, 0]) * SD["pf"], 1e-3), element=ti, side="hv")
        if em[e, 1]: create_measurement(est, "q", "trafo", ex[e, 1], max(abs(ex[e, 1]) * SD["qf"], 1e-3), element=ti, side="hv")
    for b in range(len(nx)):
        if nm[b, 1]: create_measurement(est, "p", "bus", nx[b, 1], max(abs(nx[b, 1]) * SD["pi"], 1e-3), element=b)
        if nm[b, 2]: create_measurement(est, "q", "bus", nx[b, 2], max(abs(nx[b, 2]) * SD["qi"], 1e-3), element=b)
        if nm[b, 0]: create_measurement(est, "v", "bus", nx[b, 0], SD["v"], element=b)
        if nm[b, 3]: create_measurement(est, "va", "bus", nx[b, 3], np.degrees(SD["va"]), element=b)
    return est


def bdd_detect(nx, nm, ex, em):
    """Return (detected, converged). detected = chi-square bad-data test fires; a WLS non-convergence is
    treated as a detection (a gross inconsistency the operator would notice), matching _bdd_release.py."""
    est = build_meas(nx, nm, ex, em)
    try:
        return bool(chi2_analysis(est, init="flat")), True
    except Exception:
        return True, False


def run_system():
    g = FdiaGenerator(C)
    X = np.load(os.path.join(SH, f"pool_ieee{C}.npz"))["X"]
    rng = np.random.default_rng(0); ts = rng.choice(len(X), min(NS, len(X)), replace=False)
    apos = g.attackable_pos
    out = []
    for s in TIERS:
        det = 0; conv = 0; n = 0
        for t in ts:
            nx, nm, ex, em = g.emit_from_state(X[t])                  # benign emitted measurements (accuracy-class noise)
            if s != 1.0:
                a = rng.choice(apos, min(K, len(apos)), replace=False)
                abus = g.load_bus[a]                                  # array-index -> bus-index (attack indexes by bus)
                inc = [e for e in range(g.E) if g.ei[0, e] in abus or g.ei[1, e] in abus]
                for b in abus: nx[b, 1:3] *= s                       # scale injected P/Q at the attacked buses
                for e in inc: ex[e] *= s                             # scale the incident branch flows (As convention)
            d, ok = bdd_detect(nx, nm, ex, em)
            det += int(d); conv += int(ok); n += 1
        out.append({"scale_pct": round((s - 1) * 100, 1), "factor": s, "n": int(n),
                    "chi2_detect_pct": round(100 * det / n, 1), "conv_fail_pct": round(100 * (n - conv) / n, 1)})
        print(f"ieee{C}  x{s:.3f} (+{(s-1)*100:5.1f}%)  ->  chi2 detect {100*det/n:5.1f}%   (n={n})", flush=True)
    return out


res = {"attack": "As (meter-scaling)", "system": f"ieee{C}", "k_buses": K,
       "note": "BDD chi-square detection rate vs As scale factor. As tampers measurements in place so it breaks "
               "physics consistency and lifts off the noise floor at a far smaller magnitude than the on-manifold "
               "Aq load-move. s=1.0 row is the benign false-positive floor. All numbers computed from the pipeline.",
       "sigma_P_pct": SD["pi"] * 100, "tiers": run_system()}
json.dump(res, open(os.path.join(RES, "as_sensitivity.json"), "w"), indent=2)

# ---- crossing point (magnitude at which detection first reaches ~50%) ----
xs = [r["scale_pct"] for r in res["tiers"]]; ys = [r["chi2_detect_pct"] for r in res["tiers"]]
cross = next((r["scale_pct"] for r in res["tiers"] if r["chi2_detect_pct"] >= 50.0), None)
res["cross50_scale_pct"] = cross
json.dump(res, open(os.path.join(RES, "as_sensitivity.json"), "w"), indent=2)

# ---- figure ----
INK = "#222222"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(4.8, 3.3))
ax.plot(xs, ys, "o-", color="#b2182b", lw=1.8, ms=4.5, label="BDD chi-square (As)")
ax.axhline(50, ls="--", lw=0.8, color=INK, alpha=0.5)
if cross is not None:
    ax.axvline(cross, ls=":", lw=0.9, color="#b2182b", alpha=0.7)
    ax.text(cross * 1.05, 8, f"50% at +{cross:g}%", fontsize=7.5, color="#b2182b")
ax.set_xscale("log"); ax.set_xlabel("As magnitude (meter over-reading, %)")
ax.set_ylabel("BDD chi-square detection rate (%)"); ax.set_ylim(-3, 103)
ax.set_title("Meter-scaling (As) breaks physics and is BDD-detectable early", fontsize=8.8)
ax.legend(fontsize=8.5, frameon=False, loc="lower right"); ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_as_sensitivity.png"), dpi=175); fig.savefig(os.path.join(RES, "fig_as_sensitivity.pdf"))
with open(os.path.join(RES, "sidecars", "as_sensitivity.csv"), "w") as f:
    f.write("system,scale_pct,factor,chi2_detect_pct,conv_fail_pct,n\n")
    for r in res["tiers"]:
        f.write(f"ieee{C},{r['scale_pct']},{r['factor']},{r['chi2_detect_pct']},{r['conv_fail_pct']},{r['n']}\n")
print(f"[done] results/as_sensitivity.json + fig_as_sensitivity.(png|pdf); 50% crossing at +{cross}% ", flush=True)
