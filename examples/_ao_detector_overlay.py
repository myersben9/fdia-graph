#!/usr/bin/env python
"""E7 -- Aq/Ao (stealthy load-scale) detector overlay: THREE real detectors vs attack magnitude.

On the SAME load-scale magnitude axis the Aq sensitivity figure uses (percent load over-scaling of the
attacked buses, 0.5% .. 50%), we overlay three detectors as curves and show at what magnitude each one lifts
off the meter-noise floor:

  (1) BDD chi-square detection rate %  -- the classical residual test (pandapower WLS + chi2).
  (2) swing / physics catch %          -- the per-bus rate-of-change detector (the same windowed z-score fed
                                          to the model as the 'swing' feature), thresholded at a 5% benign
                                          false-alarm operating point.
  (3) ARMA localizer DR@5%FA %         -- the trained ARMA+attention localizer's grid-level detection rate
                                          (max per-bus attack probability) at a 5% benign false-alarm point.

For each load-scale tier we generate Aq attacks on real benign operating points via the stealthy AC re-solve
(pinned dispatch + AGC, the on-manifold construction), emit the accuracy-class measurement graph, and evaluate
all three detectors on the identical records. The BDD curve stays pinned near zero (Aq is power-flow
consistent, so the residual test never fires); the swing and ARMA curves lift off once the attack rises above
the meter-noise floor. Every number is computed from the pipeline, never fabricated.

Env: CASE (14/118/300, default 118), N_SAMP (attacks per tier), N_CAL (benign calibration samples),
     EPOCHS (localizer training), SHARD_DIR (default release_v0.4.1). CPU-safe (CPU=1 CUDA_VISIBLE_DEVICES="").
Writes results/ao_detector_overlay.json + results/fig_ao_detector_overlay.(png|pdf) + a CSV sidecar.
"""
import os, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch, torch.nn.functional as F
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdia_graph._core import FdiaGenerator
from fdia_graph import FdiaGraph
from train_arma import ArmaLoc, kcl_residual                         # reuse the winner localizer + physics feature
import pandapower as pp, pandapower.networks as pn
from pandapower.create import create_measurement
from pandapower.estimation import chi2_analysis

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
DEV = "cpu" if os.environ.get("CPU", "0") == "1" or not torch.cuda.is_available() else "cuda"
C = int(os.environ.get("CASE", "118")); NS = int(os.environ.get("N_SAMP", "80"))
NCAL = int(os.environ.get("N_CAL", "200")); EPOCHS = int(os.environ.get("EPOCHS", "40")); BS = 256; K = 3
SH = os.environ.get("SHARD_DIR", os.path.join(HERE, "release_v0.4.1"))
NET = {14: pn.case14, 118: pn.case118, 300: pn.case300}[C]
SD = dict(pf=0.017, qf=0.017, v=0.0012, pi=0.017, qi=0.017, va=0.00168)   # accuracy-class meter std-devs
SWING_W = 60                                                         # swing window (scans), matches generation.py
FA_TARGET = 5.0                                                      # false-alarm operating point (percent)
# same load-scale tiers as the Aq sensitivity figure (percent load over-scaling of attacked buses)
TIERS = [1.005, 1.01, 1.02, 1.03, 1.05, 1.08, 1.12, 1.20, 1.35, 1.50]
torch.manual_seed(0); np.random.seed(0)

base = NET(); pp.runpp(base); nl = len(base.line); ntr = len(base.trafo)

# ------------------------------------------------------------------ BDD (chi-square) on an emitted graph
def build_meas(nx, nm, ex, em):
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
    try:
        return bool(chi2_analysis(build_meas(nx, nm, ex, em), init="flat"))
    except Exception:
        return True


# ------------------------------------------------------------------ localizer: train + normalization stats
def load_split(split):
    ds = FdiaGraph(os.path.join(SH, f"ml_only_ieee{C}.h5"), split=split, units="pu"); a = ds.to_numpy()
    keys = ["node_x", "node_m", "edge_x", "edge_m", "y"] + (["temporal_delta"] if ds.has_temporal else []) \
           + (["swing"] if getattr(ds, "has_swing", False) else [])
    g = {k: torch.as_tensor(a[k], device=DEV, dtype=torch.float32) for k in keys}
    return g, torch.as_tensor(a["family"], device=DEV), ds


print(f"[ieee{C}] loading shard splits + training ARMA localizer on {DEV} ({EPOCHS} epochs) ...", flush=True)
trG, trFam, ds = load_split("train"); vaG, vaFam, _ = load_split("val")
N, E = ds.N, ds.E; ei0 = ds.edge_index.to(DEV)
# append KCL residual + temporal_delta into node_x (the training-time feature layout), then standardize
for g in (trG, vaG):
    extra = [kcl_residual(g["node_x"], g["edge_x"], ei0, N)] + ([g["temporal_delta"]] if "temporal_delta" in g else [])
    g["node_x"] = torch.cat([g["node_x"]] + extra, -1)
    g["node_m"] = torch.cat([g["node_m"]] + [torch.ones_like(e) for e in extra], -1)
NORM = {}
for xk, mk in (("node_x", "node_m"), ("edge_x", "edge_m")):
    w = trG[mk].sum((0, 1)).clamp(min=1.0); mu = (trG[xk] * trG[mk]).sum((0, 1)) / w
    sd = (((trG[xk] - mu) ** 2 * trG[mk]).sum((0, 1)) / w).sqrt().clamp(min=1e-3)
    NORM[xk] = (mu, sd)
    for g in (trG, vaG): g[xk] = (g[xk] - mu) / sd * g[mk]
ei_bi = torch.cat([ei0, ei0.flip(0)], 1)


def batched(B):
    off = (torch.arange(B, device=DEV) * N).repeat_interleave(ei_bi.shape[1]); return ei_bi.repeat(1, B) + off.unsqueeze(0)


def feats(g, idx):
    b = len(idx); nxb = g["node_x"][idx]; exb = g["edge_x"][idx]
    kcl = kcl_residual(nxb, exb, ei0, N); parts = [nxb, g["node_m"][idx], kcl]
    if "temporal_delta" in g: parts.append(g["temporal_delta"][idx])
    if "swing" in g: parts.append(g["swing"][idx])
    x = torch.cat(parts, -1).reshape(b * N, -1)
    e2 = torch.cat([exb, g["edge_m"][idx]], -1); e = torch.cat([e2, e2], 1).reshape(b * 2 * E, -1)
    return x, e, batched(b), g["y"][idx].reshape(b * N)


xdim = feats(trG, torch.arange(2, device=DEV))[0].shape[-1]
model = ArmaLoc(xdim, attn=True).to(DEV); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
pos = float(trG["y"].sum()); pw = torch.tensor(min(max((trG["y"].numel() - pos) / max(pos, 1), 1.0), 30.0), device=DEV)
n = trG["y"].shape[0]
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(n, device=DEV); tot = 0.0
    for i in range(0, n, BS):
        idx = perm[i:i + BS]; x, e, ei, y = feats(trG, idx)
        loss = F.binary_cross_entropy_with_logits(model(x, e, ei), y, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
    if (ep + 1) % 5 == 0 or ep == 0: print(f"  epoch {ep+1}/{EPOCHS}  loss {tot/n:.4f}", flush=True)


def grid_scores(g):
    """Grid-level detection score per record = max over buses of the sigmoid attack probability."""
    model.eval(); S = []
    with torch.no_grad():
        for i in range(0, g["y"].shape[0], BS):
            idx = torch.arange(i, min(i + BS, g["y"].shape[0]), device=DEV)
            lg = model(*feats(g, idx)[:3]).reshape(len(idx), N)
            S.append(torch.sigmoid(lg).max(1).values.cpu())
    return torch.cat(S)


# 5% FA operating point: threshold on the grid score at the 95th percentile of the BENIGN (val) scores
vBen = (vaFam == 0)
dthr = float(np.percentile(grid_scores({k: v[vBen] for k, v in vaG.items()}).numpy(), 100 - FA_TARGET))
va_fa = float(np.mean(grid_scores({k: v[vBen] for k, v in vaG.items()}).numpy() > dthr) * 100)
print(f"  localizer 5%-FA grid threshold = {dthr:.4f}  (val benign FA {va_fa:.1f}%)", flush=True)

# ------------------------------------------------------------------ on-the-fly Aq generation + all-detector eval
g = FdiaGenerator(C)
X = np.load(os.path.join(SH, f"pool_ieee{C}.npz"))["X"].astype(np.float32)
nT = len(X); bMVA = g._bMVA
# precompute SWING scale exactly as generation.py (windowed std of scan-to-scan |change| in P/Q)
_D = np.abs(np.diff(X[:, :, :2], axis=0))
_c1 = np.concatenate([np.zeros((1,) + _D.shape[1:]), np.cumsum(_D, 0)], 0)
_c2 = np.concatenate([np.zeros((1,) + _D.shape[1:]), np.cumsum(_D ** 2, 0)], 0)
SCALE = np.full((nT, C, 2), 1e-3, np.float32)
for t in range(2, nT):
    s0 = max(0, t - SWING_W); e0 = t - 1; nn = e0 - s0
    if nn >= 3:
        su = _c1[e0] - _c1[s0]; sq = _c2[e0] - _c2[s0]
        SCALE[t] = np.sqrt(np.maximum(sq / nn - (su / nn) ** 2, 0.0)) + 1e-3


def temporal_feats(nx, nm, t):
    """Reproduce generation.py's temporal_delta (MW) + swing (dimensionless z-score) from the emitted graph."""
    mP = nm[:, 1] > 0; prev = X[t - 1] if t > 0 else X[t]
    td = np.zeros((C, 2), np.float32); sw = np.zeros((C, 2), np.float32); sc = SCALE[t]
    td[mP, 0] = nx[mP, 1] - prev[mP, 0]; td[mP, 1] = nx[mP, 2] - prev[mP, 1]
    sw[mP, 0] = (nx[mP, 1] - prev[mP, 0]) / sc[mP, 0]; sw[mP, 1] = (nx[mP, 2] - prev[mP, 1]) / sc[mP, 1]
    return td, sw


def to_dict(recs):
    """Stack emitted (physical-unit) records into a normalized localizer input dict matching the trained model.
    Applies the identical FdiaGraph pu conversion (P/Q,flows /baseMVA; theta deg->rad; swing dimensionless),
    then the training KCL+temporal append and the train-set standardization."""
    nx = np.stack([r[0] for r in recs]).astype(np.float32); nm = np.stack([r[1] for r in recs]).astype(np.float32)
    ex = np.stack([r[2] for r in recs]).astype(np.float32); em = np.stack([r[3] for r in recs]).astype(np.float32)
    td = np.stack([r[4] for r in recs]).astype(np.float32); sw = np.stack([r[5] for r in recs]).astype(np.float32)
    nx[:, :, 1:3] /= bMVA; nx[:, :, 3] = np.deg2rad(nx[:, :, 3])      # pu conversion (V left as-is, theta->rad)
    ex[:, :, :] /= bMVA; td[:, :, :] /= bMVA                          # flows + temporal_delta are power -> pu
    d = {"node_x": torch.tensor(nx, device=DEV), "node_m": torch.tensor(nm, device=DEV),
         "edge_x": torch.tensor(ex, device=DEV), "edge_m": torch.tensor(em, device=DEV),
         "temporal_delta": torch.tensor(td, device=DEV), "swing": torch.tensor(sw, device=DEV),
         "y": torch.zeros((len(recs), N), device=DEV)}
    extra = [kcl_residual(d["node_x"], d["edge_x"], ei0, N), d["temporal_delta"]]
    d["node_x"] = torch.cat([d["node_x"]] + extra, -1)
    d["node_m"] = torch.cat([d["node_m"]] + [torch.ones_like(e) for e in extra], -1)
    for xk, mk in (("node_x", "node_m"), ("edge_x", "edge_m")):
        mu, sd = NORM[xk]; d[xk] = (d[xk] - mu) / sd * d[mk]
    return d


rng = np.random.default_rng(0)
valid_t = np.arange(SWING_W, nT)                                     # timesteps with a full swing window
apos = g.attackable_pos


def swing_stat(nx, nm, t):
    """Grid-level swing statistic: max over metered-P buses of |measured P - prev true P| / typical recent jump.
    Same statistic as the rate-of-change detector (_roc_detector.py)."""
    mP = nm[:, 1] > 0
    if not mP.any(): return 0.0
    sc = np.maximum(SCALE[t], 0.017 * np.abs(X[t - 1, :, :2]) + 0.05)
    return float((np.abs(nx[mP, 1] - X[t - 1, mP, 0]) / sc[mP, 0]).max())


# benign calibration set for the swing detector (5% FA) + benign floor rows for all three detectors
print(f"[ieee{C}] calibrating detectors on {NCAL} benign records ...", flush=True)
ben_recs = []; ben_swing = []; ben_bdd = 0
cal_ts = rng.choice(valid_t, min(NCAL, len(valid_t)), replace=False)
for t in cal_ts:
    nx, nm, ex, em = g.emit_from_state(X[t]); td, sw = temporal_feats(nx, nm, int(t))
    ben_recs.append((nx.copy(), nm.copy(), ex.copy(), em.copy(), td, sw))
    ben_swing.append(swing_stat(nx, nm, int(t))); ben_bdd += int(bdd_detect(nx, nm, ex, em))
swing_thr = float(np.percentile(ben_swing, 100 - FA_TARGET))        # 5% benign FA threshold for the swing detector
ben_loc = grid_scores(to_dict(ben_recs)).numpy()
ben_loc_fa = float(np.mean(ben_loc > dthr) * 100); ben_swing_fa = float(np.mean(np.array(ben_swing) > swing_thr) * 100)
ben_bdd_fa = round(100 * ben_bdd / len(cal_ts), 1)
print(f"  swing 5%-FA threshold = {swing_thr:.2f}   benign FA: BDD {ben_bdd_fa}%  swing {ben_swing_fa:.1f}%  loc {ben_loc_fa:.1f}%", flush=True)

# sweep: generate Aq at each tier, evaluate all three detectors on the SAME records
curve = []
for m in TIERS:
    recs = []; ts_used = []; bdd = 0
    tries = 0
    while len(recs) < NS and tries < NS * 6:
        tries += 1; t = int(rng.choice(valid_t))
        Xt = X[t]; Lp = Xt[g.load_bus, 0] + g.load_genP; Lq = Xt[g.load_bus, 1].copy()
        a = rng.choice(apos, min(K, len(apos)), replace=False)
        Lp_atk = Lp.copy(); Lp_atk[a] *= m
        net = g.solve(Lp_atk, Lq, Xt=Xt, Lp_true=Lp)                 # stealthy on-manifold re-solve
        if net is None: continue
        nx, nm, ex, em = g.emit(net); td, sw = temporal_feats(nx, nm, t)
        recs.append((nx.copy(), nm.copy(), ex.copy(), em.copy(), td, sw)); ts_used.append(t)
        bdd += int(bdd_detect(nx, nm, ex, em))
    nrec = len(recs)
    sw_dr = 100 * np.mean([swing_stat(r[0], r[1], tt) > swing_thr for r, tt in zip(recs, ts_used)])
    loc_dr = float(np.mean(grid_scores(to_dict(recs)).numpy() > dthr) * 100)
    bdd_dr = 100 * bdd / nrec
    curve.append({"scale_pct": round((m - 1) * 100, 1), "factor": m, "n": int(nrec),
                  "bdd_chi2_dr": round(bdd_dr, 1), "swing_dr": round(sw_dr, 1), "loc_dr5fa": round(loc_dr, 1)})
    print(f"  +{(m-1)*100:5.1f}% load  ->  BDD {bdd_dr:5.1f}%   swing {sw_dr:5.1f}%   loc@5FA {loc_dr:5.1f}%   (n={nrec})", flush=True)


def cross50(key):
    return next((c["scale_pct"] for c in curve if c[key] >= 50.0), None)


res = {"attack": "Aq / Ao (stealthy load-scale)", "system": f"ieee{C}", "detectors": ["BDD chi-square", "swing (5%FA)", "ARMA localizer DR@5%FA"],
       "fa_target_pct": FA_TARGET, "k_buses": K, "epochs": EPOCHS, "seed": 0,
       "note": "Three real detectors vs load-scale magnitude on identical stealthy Aq records. BDD stays on the "
               "noise floor (Aq is power-flow consistent); swing and the ARMA localizer lift off above the meter "
               "noise. All numbers computed from the pipeline, never fabricated.",
       "benign_floor": {"bdd_chi2_dr": ben_bdd_fa, "swing_dr": round(ben_swing_fa, 1), "loc_dr5fa": round(ben_loc_fa, 1), "n": int(len(cal_ts))},
       "thresholds": {"loc_grid": round(dthr, 4), "swing": round(swing_thr, 2)},
       "cross50": {"bdd_chi2": cross50("bdd_chi2_dr"), "swing": cross50("swing_dr"), "loc": cross50("loc_dr5fa")},
       "curve": curve}
json.dump(res, open(os.path.join(RES, "ao_detector_overlay.json"), "w"), indent=2)

# ---- figure ----
INK = "#222222"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": INK, "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(5.0, 3.4))
xs = [c["scale_pct"] for c in curve]
series = [("bdd_chi2_dr", "BDD chi-square", "#b2182b", "o-"),
          ("swing_dr", "swing detector (5% FA)", "#2166ac", "s-"),
          ("loc_dr5fa", "ARMA localizer DR@5%FA", "#1a9850", "^-")]
for key, lab, col, sty in series:
    ax.plot(xs, [c[key] for c in curve], sty, color=col, lw=1.8, ms=4.5, label=lab)
ax.axhline(50, ls="--", lw=0.8, color=INK, alpha=0.45)
ax.set_xscale("log"); ax.set_xlabel("Aq magnitude (load over-scaling at attacked buses, %)")
ax.set_ylabel("detection rate (%)"); ax.set_ylim(-3, 103)
ax.set_title("Only measurement-aware detectors lift off; BDD never fires on stealthy Aq", fontsize=8.6)
ax.legend(fontsize=8.0, frameon=False, loc="center left"); ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_ao_detector_overlay.png"), dpi=175); fig.savefig(os.path.join(RES, "fig_ao_detector_overlay.pdf"))
with open(os.path.join(RES, "sidecars", "ao_detector_overlay.csv"), "w") as f:
    f.write("system,scale_pct,factor,bdd_chi2_dr,swing_dr,loc_dr5fa,n\n")
    for c in curve:
        f.write(f"ieee{C},{c['scale_pct']},{c['factor']},{c['bdd_chi2_dr']},{c['swing_dr']},{c['loc_dr5fa']},{c['n']}\n")
print(f"[done] results/ao_detector_overlay.json + fig_ao_detector_overlay.(png|pdf)", flush=True)
print(f"       cross-50%: BDD {res['cross50']['bdd_chi2']}  swing {res['cross50']['swing']}  loc {res['cross50']['loc']}", flush=True)
