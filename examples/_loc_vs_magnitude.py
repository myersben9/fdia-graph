#!/usr/bin/env python
"""Localization score vs attack magnitude, one curve per system (IEEE 14/118/300).

The key figure for the magnitude-dependent story. We train the winning ARMA+attention localizer (reusing the
exact model and feature pipeline from train_arma.py) on each system, then on the ATTACKED test samples we bin
the per-sample localization F1 by the per-sample attack magnitude and plot the curve. Magnitude is the RMS
angle shift the attack introduces at the attacked buses, |theta_measured - theta_true| in degrees, computed
from the raw shard against the true pool state. This shows directly how localization degrades as attacks
shrink toward the meter-noise floor, and it is the same axis the intensity-tier datasets use.

Reuses train_arma.ArmaLoc / kcl_residual / f1 so the localizer matches the released benchmark. GPU. Seed 0."""
import os, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from train_arma import ArmaLoc, kcl_residual                      # reuse the winner model + physics feature
from fdia_graph import FdiaGraph

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
EPOCHS = int(os.environ.get("EPOCHS", "40")); BS = 256
BINS = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 1e9])  # percent injection deviation; last bin is the tail
CENTERS = np.array([0.35, 0.7, 1.4, 2.8, 5.6, 11.0, 22.0, 45.0])  # representative x per bin (percent)


def per_sample_f1(P, Y):                                          # [n,N] binary -> [n] F1 of predicted set vs truth
    tp = (P * Y).sum(1); fp = (P * (1 - Y)).sum(1); fn = ((1 - P) * Y).sum(1)
    return (2 * tp / (2 * tp + fp + fn + 1e-9))


def run_system(sysname, c):
    torch.manual_seed(0)
    shard = os.path.join(HERE, "release_v0.4.1", f"ml_only_ieee{c}.h5")
    def load(split):
        ds = FdiaGraph(shard, split=split, units="pu"); a = ds.to_numpy()
        keys = ["node_x", "node_m", "edge_x", "edge_m", "y"] + (["temporal_delta"] if ds.has_temporal else []) \
               + (["swing"] if getattr(ds, "has_swing", False) else [])
        g = {k: torch.as_tensor(a[k], device=DEV, dtype=torch.float32) for k in keys}
        return g, torch.as_tensor(a["family"], device=DEV), ds, a
    trG, trFam, ds, _ = load("train"); vaG, vaFam, _, _ = load("val"); teG, teFam, _, teA = load("test")
    N, E = ds.N, ds.E; ei0 = ds.edge_index.to(DEV)
    for g in (trG, vaG, teG):
        extra = [kcl_residual(g["node_x"], g["edge_x"], ei0, N)] + ([g["temporal_delta"]] if "temporal_delta" in g else [])
        g["node_x"] = torch.cat([g["node_x"]] + extra, -1)
        g["node_m"] = torch.cat([g["node_m"]] + [torch.ones_like(e) for e in extra], -1)
    for xk, mk in (("node_x", "node_m"), ("edge_x", "edge_m")):
        w = trG[mk].sum((0, 1)).clamp(min=1.0); mu = (trG[xk] * trG[mk]).sum((0, 1)) / w
        sd = (((trG[xk] - mu) ** 2 * trG[mk]).sum((0, 1)) / w).sqrt().clamp(min=1e-3)
        for g in (trG, vaG, teG): g[xk] = (g[xk] - mu) / sd * g[mk]
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
        model.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, BS):
            idx = perm[i:i + BS]; x, e, ei, y = feats(trG, idx)
            loss = F.binary_cross_entropy_with_logits(model(x, e, ei), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()

    def collect(g):
        model.eval(); LG = []
        with torch.no_grad():
            for i in range(0, g["y"].shape[0], BS):
                idx = torch.arange(i, min(i + BS, g["y"].shape[0]), device=DEV)
                LG.append(model(*feats(g, idx)[:3]).reshape(len(idx), N).float().cpu())
        return torch.cat(LG)
    vL = collect(vaG); vY = vaG["y"].cpu(); vatk = (vaFam.cpu() > 0)
    thr = float(max(torch.linspace(-2, 3, 26), key=lambda t: per_sample_f1((vL[vatk] > t).float(), vY[vatk]).mean()))
    L = collect(teG); Y = teG["y"].cpu(); Fm = teFam.cpu(); P = (L > thr).float()

    # per-sample attack magnitude: RMS over METERED attacked buses of the RELATIVE active-power injection deviation
    # in PERCENT, |P_meas - P_true| / |P_true|. Physical units (MW) so it is unit-consistent with the pool; metered
    # buses only, so the unmetered-channel zeros do not pollute it; relative so it is comparable across systems.
    POOLX = np.load(os.path.join(HERE, "release_v0.4.1", f"pool_ieee{c}.npz"))["X"].astype(np.float32)  # [T,N,4], P col 1 MW
    teP = FdiaGraph(shard, split="test", units="physical").to_numpy()   # same split/order, PHYSICAL units (MW)
    P_meas = teP["node_x"][:, :, 1]                                # [n,N] measured/attacked injection MW
    P_true = POOLX[teA["timestep"].astype(int)][:, :, 0]          # [n,N] true injection MW (POOLX order is [P,Q,V,theta])
    msk = teA["y"].astype(bool) & teP["node_m"][:, :, 1].astype(bool)   # attacked AND P-injection metered
    rel = np.abs(P_meas - P_true) / (np.abs(P_true) + 1.0)         # relative dev (1 MW floor guards near-zero injections)
    mag = np.array([100.0 * np.sqrt((rel[i][msk[i]] ** 2).mean()) if msk[i].any() else np.nan for i in range(len(rel))])  # percent

    atk = (Fm > 0).numpy()
    psf1 = per_sample_f1(P, Y).numpy()
    ma = mag[atk]; fa = psf1[atk]
    curve = []
    for lo, hi, ctr in zip(BINS[:-1], BINS[1:], CENTERS):
        sel = (ma >= lo) & (ma < hi)
        curve.append(dict(center_deg=float(ctr), lo=float(lo), hi=float(hi), n=int(sel.sum()),
                          swf1=round(float(fa[sel].mean()), 3) if sel.sum() >= 20 else None))
    overall = round(float(fa.mean()), 3)
    print(f"IEEE-{c}: overall attacked swF1 {overall:.3f} (thr {thr:.2f}); curve " +
          " ".join(f"{ci['center_deg']:.2f}deg:{ci['swf1']}({ci['n']})" for ci in curve), flush=True)
    return dict(system=f"ieee{c}", overall_swf1=overall, threshold=round(thr, 2), curve=curve)


res = {"model": "ARMA+attn localizer", "seed": 0, "magnitude": "RMS |theta_meas-theta_true| over attacked buses (deg)", "per_system": {}}
for c, s in [(14, "ieee14"), (118, "ieee118"), (300, "ieee300")]:
    res["per_system"][f"ieee{c}"] = run_system(s, c)
json.dump(res, open(os.path.join(RES, "loc_vs_magnitude.json"), "w"), indent=2)

# ---- figure ----
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.edgecolor": "#222", "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(4.8, 3.3))
cols = {"ieee14": "#1a9850", "ieee118": "#2166ac", "ieee300": "#b2182b"}
for s in ("ieee14", "ieee118", "ieee300"):
    cv = res["per_system"][s]["curve"]
    xs = [ci["center_deg"] for ci in cv if ci["swf1"] is not None]
    ys = [ci["swf1"] for ci in cv if ci["swf1"] is not None]
    ax.plot(xs, ys, "o-", color=cols[s], lw=1.7, ms=4.5, label=f"IEEE-{s[4:]}")
ax.set_xscale("log"); ax.set_xlabel("attack magnitude (RMS injection deviation at attacked buses, %)")
ax.set_ylabel("localization swF1 (attacked)"); ax.set_ylim(0, 1.02)
ax.axvspan(0.5, 1.7, color="#888", alpha=0.10); ax.text(0.53, 0.06, "meter\nnoise floor", fontsize=6.8, color="#555")
ax.set_title("Localization degrades as the attack shrinks toward the noise floor", fontsize=8.8)
ax.legend(fontsize=8.5, frameon=False, loc="lower right"); ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_loc_vs_magnitude.png"), dpi=175); fig.savefig(os.path.join(RES, "fig_loc_vs_magnitude.pdf"))
with open(os.path.join(RES, "sidecars", "loc_vs_magnitude.csv"), "w") as f:
    f.write("system,magnitude_deg,swf1,n\n")
    for s in ("ieee14", "ieee118", "ieee300"):
        for ci in res["per_system"][s]["curve"]:
            f.write(f"{s},{ci['center_deg']},{ci['swf1']},{ci['n']}\n")
print("wrote results/loc_vs_magnitude.json + fig_loc_vs_magnitude.(png|pdf)", flush=True)
