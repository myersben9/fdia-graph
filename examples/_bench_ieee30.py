#!/usr/bin/env python
"""Head-to-head: run OUR ARMA graph localizer on the external IEEE-30 multi-scenario FDIA dataset
(IEEE DataPort, CC BY 4.0). Characterizes that benchmark with our approach and shows the intensity->difficulty
gradient (Scenario 1 -> 5), giving a real comparison for our dataset paper.

Their data per scenario: X_*_3d_scaled [N,16,41] = 16-step windows of 41 DC active-power BRANCH FLOWS
(StandardScaler'd); y_*_windowed [N,29] = per-bus attack labels for the 29 non-slack buses (multi-label).

Mapping (stated assumptions, since their column order is not documented): their P_Branch_1..41 map to
pandapower case30 lines 0..40 in order, and the 29 label columns map to buses 1..29 (bus 0 is the slack).
The intensity->difficulty gradient is robust to any residual mapping error; exact per-bus numbers assume the
mapping above. Our localizer here uses ONLY branch flows as edge features, because their dataset provides
nothing else (no |V|, no injections, no PMU angles) -- which is itself the measurement-model limitation our
own dataset removes.

Output: results/bench_ieee30.json + fig_bench_ieee30.(png|pdf) + CSV. GPU. Seed 123."""
import os, json, glob, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import ARMAConv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
DP = "C:/Users/bm539044/AppData/Local/Temp/claude/C--Users-bm539044-desktop-fedpig/330c32b6-e5dd-4ce0-9578-6af29753b676/scratchpad/dp30/full"
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
torch.manual_seed(123); np.random.seed(123)

# ---- IEEE-30 topology ----
net = pn.case30(); pp.rundcpp(net)
fb = net.line.from_bus.values.astype(int); tb = net.line.to_bus.values.astype(int); NB = len(net.bus); NBR = len(net.line)
ei = torch.tensor(np.concatenate([np.stack([fb, tb]), np.stack([tb, fb])], 1), dtype=torch.long, device=DEV)  # bidir
LABEL_BUSES = list(range(1, 30))                                           # 29 non-slack buses (slack = 0)
# incidence matrix bus x branch (+1 from, -1 to) for aggregating branch flows to node features
INC = np.zeros((NB, NBR), np.float32)
for e in range(NBR): INC[fb[e], e] += 1; INC[tb[e], e] -= 1
INC_t = torch.tensor(INC, device=DEV)


def scen_dir(s):
    d = glob.glob(os.path.join(DP, f"Scenario_{s}_*[!0-9]_rar", "Scenario_*"))
    return [x for x in d if os.path.isdir(x)][0]


def load_scen(s):
    d = scen_dir(s)
    Xtr = np.load(os.path.join(d, f"X_train_3d_scaled_S{s}.npy")).astype(np.float32)   # [Ntr,16,41]
    ytr = np.load(os.path.join(d, f"y_train_windowed_S{s}.npy")).astype(np.float32)    # [Ntr,29]
    Xte = np.load(os.path.join(d, f"X_test_3d_scaled_S{s}.npy")).astype(np.float32)
    yte = np.load(os.path.join(d, f"y_test_windowed_S{s}.npy")).astype(np.float32)
    return Xtr, ytr, Xte, yte


def feats(X):
    """window [B,16,41] -> node feats [B,NB,3] (incidence-agg last, mean, |last|) + edge feats [B,41,3]."""
    last = X[:, -1, :]; mean = X.mean(1); delta = last - mean                # [B,41] each
    ef = np.stack([last, mean, delta], -1)                                   # [B,41,3] edge feats
    nf = np.stack([last @ INC.T, mean @ INC.T, np.abs(last) @ np.abs(INC).T], -1)  # [B,NB,3] node feats
    return torch.tensor(nf), torch.tensor(ef)


class ArmaLoc(nn.Module):
    def __init__(self, nin=3, ein=3, hid=128):
        super().__init__()
        self.nenc = nn.Linear(nin, hid); self.eenc = nn.Linear(ein, hid)
        self.b1 = ARMAConv(hid, hid, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.05, act=F.relu)
        self.b2 = ARMAConv(hid, hid, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.05, act=F.relu)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, nf, ef):                                               # nf [B,NB,3], ef [B,41,3]
        B = nf.shape[0]; h = F.relu(self.nenc(nf))                           # [B,NB,hid]
        he = F.relu(self.eenc(ef)); he = torch.cat([he, he], 1)             # bidir edge msgs [B,82,hid]
        h = h + torch.zeros_like(h).index_add_(1, ei[1], he)                 # scatter edge msgs to dst nodes
        hh = h.reshape(B * NB, -1)
        off = (torch.arange(B, device=DEV) * NB).repeat_interleave(ei.shape[1]); eib = ei.repeat(1, B) + off
        hh = self.b1(hh, eib); hh = self.b2(hh, eib)
        return self.head(hh).reshape(B, NB)[:, LABEL_BUSES]                  # [B,29] logits


def _auc(pos, neg):                                                         # rank-based ROC-AUC, no sklearn dep
    if len(pos) == 0 or len(neg) == 0: return 0.5
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def metrics(logit, y):
    """Detection judged with FALSE-ALARM CONTROL (AUC + DR at 5% FA). Localization scored on attacked windows
    at a per-bus threshold set to 5% per-bus false alarm on benign windows. No threshold tuned on the positives."""
    prob = torch.sigmoid(logit); atkw = (y.sum(1) > 0); benw = ~atkw
    score = prob.max(1).values                                              # per-window detection score
    sa = score[atkw].cpu().numpy(); sb = score[benw].cpu().numpy()
    auc = _auc(sa, sb)
    thr = float(np.quantile(sb, 0.95)) if len(sb) else 0.5                   # threshold giving 5% window FA on benign
    DR5 = float((sa > thr).mean()) if len(sa) else 0.0
    # localization at a PER-BUS threshold giving 5% per-bus FA on benign windows
    pb = prob[benw]
    thr_bus = torch.quantile(pb, 0.95, dim=0) if benw.any() else torch.full((29,), 0.5, device=prob.device)
    p = (prob[atkw] > thr_bus).float(); ya = y[atkw]
    tp = (p * ya).sum(0); fp = (p * (1 - ya)).sum(0); fn = ((1 - p) * ya).sum(0)
    macro = (2 * tp / (2 * tp + fp + fn + 1e-9)).mean().item()
    tps = (p * ya).sum(1); pps = p.sum(1); ys = ya.sum(1)
    swf1 = (2 * tps / (pps + ys + 1e-9)).mean().item()
    exact = (p == ya).all(1).float().mean().item()
    return dict(det_auc=round(auc, 3), DR_at_FA5=round(DR5, 3), macro_f1=round(macro, 3), sample_f1=round(swf1, 3), exact_match=round(exact, 3))


res = {"benchmark": "IEEE DataPort IEEE-30 multi-scenario FDIA", "model": "our ARMA graph localizer",
       "device": DEV, "per_scenario": {}}
NAMES = {1: "High-Intensity", 2: "Medium", 3: "High-Stealth", 4: "Combined", 5: "Advanced"}
for s in range(1, 6):
    Xtr, ytr, Xte, yte = load_scen(s)
    ntr_f, etr_f = feats(Xtr); nte_f, ete_f = feats(Xte)
    ntr_f, etr_f, Ytr = ntr_f.to(DEV), etr_f.to(DEV), torch.tensor(ytr, device=DEV)
    nte_f, ete_f, Yte = nte_f.to(DEV), ete_f.to(DEV), torch.tensor(yte, device=DEV)
    pos = Ytr.sum() / (Ytr.numel() - Ytr.sum() + 1e-9)
    model = ArmaLoc().to(DEV); opt = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-5)
    pw = torch.full((29,), float((1 / pos).clamp(1, 5)), device=DEV)        # mild pos-weight; avoids predict-all collapse
    N = ntr_f.shape[0]
    for ep in range(60):
        model.train(); perm = torch.randperm(N, device=DEV)
        for i in range(0, N, 256):
            idx = perm[i:i + 256]
            out = model(ntr_f[idx], etr_f[idx]); loss = F.binary_cross_entropy_with_logits(out, Ytr[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        lo = torch.cat([model(nte_f[i:i + 512], ete_f[i:i + 512]) for i in range(0, nte_f.shape[0], 512)])
    m = metrics(lo, Yte)
    res["per_scenario"][f"S{s}"] = dict(name=NAMES[s], n_train=int(N), n_test=int(nte_f.shape[0]), **m)
    print(f"S{s} {NAMES[s]:14s} | det-AUC {m['det_auc']:.3f} DR@FA5% {m['DR_at_FA5']:.2f} | "
          f"loc macro-F1 {m['macro_f1']:.3f} sample-F1 {m['sample_f1']:.3f} exact {m['exact_match']:.3f}", flush=True)
json.dump(res, open(os.path.join(RES, "bench_ieee30.json"), "w"), indent=2)

# ---- figure: difficulty gradient across scenarios ----
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.edgecolor": "#222", "axes.linewidth": 0.8})
S = [f"S{s}" for s in range(1, 6)]; x = np.arange(5)
fig, ax = plt.subplots(figsize=(5.6, 3.3))
for key, col, lab in [("macro_f1", "#1a9850", "per-bus macro-F1"), ("sample_f1", "#2166ac", "sample-wise F1"), ("exact_match", "#b2182b", "strict exact-match")]:
    ax.plot(x, [res["per_scenario"][s][key] for s in S], "o-", color=col, lw=1.6, ms=5, label=lab)
ax.set_xticks(x); ax.set_xticklabels([f"S{s}\n{NAMES[s]}" for s in range(1, 6)], fontsize=7.5)
ax.set_ylabel("localization score"); ax.set_ylim(0, 1)
ax.set_title("Our ARMA localizer on the IEEE-30 benchmark: stealth->difficulty gradient", fontsize=9.3)
ax.legend(fontsize=8, frameon=False, loc="upper right"); ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.6)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_bench_ieee30.png"), dpi=175); fig.savefig(os.path.join(RES, "fig_bench_ieee30.pdf"))
with open(os.path.join(RES, "sidecars", "bench_ieee30.csv"), "w") as f:
    f.write("scenario,name,det_auc,DR_at_FA5,macro_f1,sample_f1,exact_match\n")
    for s in range(1, 6):
        r = res["per_scenario"][f"S{s}"]; f.write(f"S{s},{r['name']},{r['det_auc']},{r['DR_at_FA5']},{r['macro_f1']},{r['sample_f1']},{r['exact_match']}\n")
print("wrote results/bench_ieee30.json + fig_bench_ieee30.(png|pdf)", flush=True)
