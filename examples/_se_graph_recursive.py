#!/usr/bin/env python
"""Phase 3a: a GRAPH-AWARE, SIZE-INVARIANT learned recursive estimator, trained on ONE grid and applied
ZERO-SHOT to others. Tests the cross-grid-transfer claim (train on IEEE-14, run on 118 and 300 unchanged).

Design. Same win-both recipe as Phase 2 (a learned residual on top of a per-node steady-state Kalman), but the
correction network is now per-NODE with SHARED weights plus a graph-aggregated neighbor term, so it is
independent of the number of buses and can transfer across topologies. Per node and per step it sees its own
Kalman estimate, its WLS measurement, its innovation, and the neighbor-averaged innovation (the joint context
that lets it flag a ramp that is locally plausible but jointly inconsistent). Trained ONLY on IEEE-14 benign +
ramp-attacked sequences with the true benign state as target, then evaluated on 14 (in-domain) and on 118 and
300 (zero-shot, weights frozen, each with its own Kalman baseline).

DC linear SE (unbiased BLUE). Output: results/se_graph_recursive.json + fig_graph_recursive.(png|pdf) + CSV.
Seed 123."""
import os, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from scipy.interpolate import CubicSpline
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
torch.manual_seed(123); np.random.seed(123)
CAD, ANCHOR_DT, NA, L = 2.0, 300.0, 6, 256
SIG_F, TAU = 0.004, 30.0; PHI = np.exp(-CAD / TAU); SF = SIG_F * np.sqrt(1 - PHI ** 2)
NETS = {14: pn.case14, 118: pn.case118, 300: pn.case300}


def build_case(c):
    net = NETS[c](); pp.rundcpp(net); ppc = net._ppc; br = ppc["branch"]; bus = ppc["bus"]
    NBp = bus.shape[0]; fb = br[:, 0].real.astype(int); tb = br[:, 1].real.astype(int); bl = 1.0 / br[:, 3].real
    B = np.zeros((NBp, NBp))
    for f, t, b in zip(fb, tb, bl):
        B[f, f] += b; B[t, t] += b; B[f, t] -= b; B[t, f] -= b
    slack = int(np.where(bus[:, 1] == 3)[0][0]); keep = [i for i in range(NBp) if i != slack]; NS = len(keep)
    pos = {b: i for i, b in enumerate(keep)}
    rows, kinds = [], []
    for i in range(NBp): rows.append(B[i, keep]); kinds.append("inj")
    for (f, t, b) in zip(fb, tb, bl):
        r = np.zeros(NS)
        if f in keep: r[pos[f]] += b
        if t in keep: r[pos[t]] -= b
        rows.append(r); kinds.append("flow")
    pmu = sorted(np.random.default_rng(7).choice(keep, int(round(0.65 * NS)), replace=False))
    for i in pmu:
        r = np.zeros(NS); r[pos[i]] = 1.0; rows.append(r); kinds.append("pmu")
    H = np.array(rows); sig = np.array([0.02 if k != "pmu" else 0.005 for k in kinds])
    Ginv = np.linalg.solve((H.T * (1 / sig ** 2)) @ H, H.T * (1 / sig ** 2))
    # row-normalized adjacency over kept nodes (for the neighbor term)
    A = np.zeros((NS, NS))
    for (f, t) in zip(fb, tb):
        if f in keep and t in keep: A[pos[f], pos[t]] = 1; A[pos[t], pos[f]] = 1
    A = A / np.maximum(A.sum(1, keepdims=True), 1)
    POOLX = np.load(os.path.join(HERE, "release_v0.4.1", f"pool_ieee{c}.npz"))["X"].astype(np.float32)
    baseMVA = float(ppc["baseMVA"]); load_ppc = net._pd2ppc_lookups["bus"][net.load.bus.values].astype(int)
    lp = np.array([pos.get(b, -1) for b in load_ppc])                     # kept-node index per load (or -1 if slack)
    base_load_pu = net.load.p_mw.values / baseMVA
    Bred = B[np.ix_(keep, keep)]
    d = dict(c=c, NS=NS, H=H, Ginv=Ginv, sig=sig, A=A, Bred=Bred, keep=keep, POOLX=POOLX,
             load_ppc=load_ppc, base_load_pu=base_load_pu)
    return d


def make_seq(cd, rs, attack):
    NS = cd["NS"]; keep = cd["keep"]; POOLX = cd["POOLX"]
    t0 = rs.integers(0, POOLX.shape[0] - NA - 1)
    tha = np.deg2rad(POOLX[t0:t0 + NA, keep, 3].astype(float)); ta = np.arange(NA) * ANCHOR_DT; tt = np.arange(L) * CAD
    slow = np.stack([CubicSpline(ta, tha[:, j])(tt) for j in range(NS)], 1)
    oul = np.zeros((L, len(cd["base_load_pu"])))
    for t in range(1, L): oul[t] = PHI * oul[t - 1] + rs.normal(0, SF, len(cd["base_load_pu"]))
    dP = np.zeros((L, POOLX.shape[1]))
    for li, b in enumerate(cd["load_ppc"]): dP[:, b] += -oul[:, li] * cd["base_load_pu"][li]
    th_true = slow + np.linalg.solve(cd["Bred"], dP[:, keep].T).T
    ms = th_true.copy()
    if attack:
        nb = rs.integers(1, 4); ab = rs.choice(NS, nb, replace=False)
        a0 = rs.integers(int(L * 0.3), int(L * 0.6)); dur = rs.integers(int(30 / CAD), int(120 / CAD))
        mag = rs.uniform(np.deg2rad(0.6), np.deg2rad(1.8)); prog = np.clip((np.arange(L) - a0) / dur, 0, 1)
        for b in ab: ms[:, b] += prog * mag
    z = ms @ cd["H"].T + rs.normal(0, cd["sig"], (L, cd["H"].shape[0]))
    return th_true.astype(np.float32), (z @ cd["Ginv"].T).astype(np.float32)


def kalman_gain(q, r):
    K = np.zeros_like(q)
    for i in range(len(q)):
        P = r[i]
        for _ in range(400): Pm = P + q[i]; K[i] = Pm / (Pm + r[i]); P = (1 - K[i]) * Pm
    return K


def calib_kalman(cd):
    rs = np.random.default_rng(555); T, W = [], []
    for _ in range(40):
        tr, wl = make_seq(cd, rs, False); T.append(tr); W.append(wl)
    q = np.mean([np.diff(t, 0).var(0) for t in T], 0)
    r = np.mean([np.maximum((np.diff(w, 0).var(0) - np.diff(t, 0).var(0)) / 2, 1e-12) for t, w in zip(T, W)], 0)
    return kalman_gain(q, r)


def kf_np(wl, K):
    out = wl.copy()
    for t in range(1, len(wl)): out[t] = out[t - 1] + K * (wl[t] - out[t - 1])
    return out


class GraphResidual(nn.Module):
    """Per-node shared GRU + neighbor-aggregated innovation. Size-invariant: same weights for any bus count."""
    def __init__(self, hid=96):
        super().__init__()
        self.gru = nn.GRUCell(4, hid)                                     # per node: [kf, wls, innov, nbr_innov]
        self.out = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, z, kf, A):                                          # z,kf [B,L,N]; A [N,N]
        Bn, Ln, N = z.shape; h = z.new_zeros(Bn * N, self.gru.hidden_size); outs = []
        for t in range(Ln):
            innov = z[:, t] - kf[:, t]                                    # [B,N]
            nbr = innov @ A.T                                             # neighbor-averaged innovation (joint context)
            feat = torch.stack([kf[:, t], z[:, t], innov, nbr], -1).reshape(Bn * N, 4)
            h = self.gru(feat, h)
            corr = self.out(h).reshape(Bn, N)
            outs.append(kf[:, t] + corr)
        return torch.stack(outs, 1)


# ---- build cases, calibrate Kalman per case ----
CASES = {c: build_case(c) for c in (14, 118, 300)}
KG = {c: calib_kalman(CASES[c]) for c in (14, 118, 300)}
print("built cases 14/118/300, calibrated per-case Kalman", flush=True)


def make_set(cd, K, n, seed):
    rs = np.random.default_rng(seed); T, W, F = [], [], []
    for i in range(n):
        tr, wl = make_seq(cd, rs, attack=(i % 2 == 0)); T.append(tr); W.append(wl); F.append(kf_np(wl, K))
    return (torch.tensor(np.array(T)), torch.tensor(np.array(W)), torch.tensor(np.array(F)))


# ---- train ONLY on IEEE-14 ----
cd14 = CASES[14]; K14 = KG[14]; A14 = torch.tensor(cd14["A"], dtype=torch.float32, device=DEV)
TRt, TRw, TRf = [x.to(DEV) for x in make_set(cd14, K14, 512, 123)]
model = GraphResidual().to(DEV); opt = torch.optim.Adam(model.parameters(), 2e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
print(f"device {DEV}; graph residual ({sum(p.numel() for p in model.parameters()):,} params); train on IEEE-14 only", flush=True)
for ep in range(600):
    perm = torch.randperm(TRt.shape[0], device=DEV); model.train(); tot = 0.0
    for i in range(0, len(perm), 64):
        idx = perm[i:i + 64]; out = model(TRw[idx], TRf[idx], A14); loss = ((out - TRt[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    sched.step()
    if (ep + 1) % 150 == 0: print(f"  epoch {ep+1}  train MSE {tot/(len(perm)//64+1):.2e}", flush=True)


# ---- evaluate on each grid (14 in-domain, 118/300 zero-shot) ----
def eval_case(c, nseq=80):
    cd = CASES[c]; K = KG[c]; A = torch.tensor(cd["A"], dtype=torch.float32, device=DEV); to_deg = 180 / np.pi
    def run(attack):
        rs = np.random.default_rng(4000 + c + int(attack)); e_w, e_k, e_l, p_w, p_k, p_l = [], [], [], [], [], []
        for _ in range(nseq):
            tr, wl = make_seq(cd, rs, attack); kf = kf_np(wl, K)
            with torch.no_grad():
                ln = model(torch.tensor(wl[None]).to(DEV), torch.tensor(kf[None]).to(DEV), A).cpu().numpy()[0]
            e_w.append(np.abs(wl - tr).mean() * to_deg); e_k.append(np.abs(kf - tr).mean() * to_deg); e_l.append(np.abs(ln - tr).mean() * to_deg)
            if attack:
                s = slice(int(L * 0.7), L)
                p_w.append(np.abs(wl[s] - tr[s]).mean() * to_deg); p_k.append(np.abs(kf[s] - tr[s]).mean() * to_deg); p_l.append(np.abs(ln[s] - tr[s]).mean() * to_deg)
        return e_w, e_k, e_l, p_w, p_k, p_l
    bw, bk, bl_, _, _, _ = run(False); aw, ak, al, pw, pk, pl = run(True)
    return dict(benign=dict(wls=round(np.mean(bw), 4), kalman=round(np.mean(bk), 4), learned=round(np.mean(bl_), 4)),
                attack=dict(wls=round(np.mean(pw), 4), kalman=round(np.mean(pk), 4), learned=round(np.mean(pl), 4)))


res = {"model": "graph residual (train on 14)", "seed": 123, "device": DEV, "per_system": {}}
for c in (14, 118, 300):
    res["per_system"][c] = eval_case(c); tag = "in-domain" if c == 14 else "ZERO-SHOT"
    r = res["per_system"][c]
    print(f"IEEE-{c} ({tag}): benign WLS {r['benign']['wls']:.3f} KF {r['benign']['kalman']:.3f} learned {r['benign']['learned']:.3f} | "
          f"attack WLS {r['attack']['wls']:.3f} KF {r['attack']['kalman']:.3f} learned {r['attack']['learned']:.3f}", flush=True)
json.dump(res, open(os.path.join(RES, "se_graph_recursive.json"), "w"), indent=2)

# ---- figure: grouped bars, attack-phase angle error, per system ----
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.edgecolor": "#222", "axes.linewidth": 0.8})
fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.1))
syst = [14, 118, 300]; x = np.arange(len(syst)); w = 0.26
for a, reg, ttl in zip(ax, ("benign", "attack"), ("Benign (clean)", "Under stealthy ramp")):
    for k, (name, col) in enumerate([("wls", "#b2182b"), ("kalman", "#8073ac"), ("learned", "#1a9850")]):
        vals = [res["per_system"][c][reg][name] for c in syst]
        a.bar(x + (k - 1) * w, vals, w, color=col, label={"wls": "static WLS", "kalman": "classical Kalman", "learned": "learned (train@14)"}[name])
    a.set_xticks(x); a.set_xticklabels([f"IEEE-{c}" + ("" if c == 14 else "\n(zero-shot)") for c in syst], fontsize=8)
    a.set_ylabel("angle MAE (deg)"); a.set_title(ttl, fontsize=9.5); a.grid(True, axis="y", ls=":", lw=0.4, alpha=0.6)
ax[0].legend(fontsize=7.4, frameon=False)
fig.suptitle("Graph-aware recursive SE, trained on IEEE-14, applied zero-shot to 118 and 300", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(RES, "fig_graph_recursive.png"), dpi=175); fig.savefig(os.path.join(RES, "fig_graph_recursive.pdf"))
with open(os.path.join(RES, "sidecars", "graph_recursive.csv"), "w") as f:
    f.write("system,regime,wls,kalman,learned\n")
    for c in syst:
        for reg in ("benign", "attack"):
            r = res["per_system"][c][reg]; f.write(f"{c},{reg},{r['wls']},{r['kalman']},{r['learned']}\n")
print("wrote results/se_graph_recursive.json + fig_graph_recursive.(png|pdf)", flush=True)
