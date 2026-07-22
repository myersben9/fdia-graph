#!/usr/bin/env python
"""Head-to-head v2 on the IEEE-30 DataPort FDIA dataset (CC BY 4.0), fixing the issues in v1:
  - TOPOLOGY-FREE temporal model (2-layer GRU over the 16-step window -> per-bus heads). Removes the branch/bus
    order-mapping assumption entirely, and matches how the dataset is meant to be used (LSTM/CNN).
  - Localization threshold TUNED on a validation split (not a naive 0.5 or 5%-FA), reported on test.
  - Adds micro-F1 (standard for multi-label) alongside macro-F1, sample-F1, exact-match.
  - Detection judged with false-alarm control (ROC-AUC + DR at 5% FA).
  - SEED-AVERAGED over 3 seeds, reported as mean +/- std.

Output: results/bench_ieee30b.json + fig_bench_ieee30b.(png|pdf) + CSV. GPU."""
import os, json, glob, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
DP = "C:/Users/bm539044/AppData/Local/Temp/claude/C--Users-bm539044-desktop-fedpig/330c32b6-e5dd-4ce0-9578-6af29753b676/scratchpad/dp30/full"
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
NAMES = {1: "High-Intensity", 2: "Medium", 3: "High-Stealth", 4: "Combined", 5: "Advanced"}
SEEDS = [0, 1, 2]


def scen_dir(s):
    d = glob.glob(os.path.join(DP, f"Scenario_{s}_*[!0-9]_rar", "Scenario_*"))
    return [x for x in d if os.path.isdir(x)][0]


def load_scen(s):
    d = scen_dir(s)
    return (np.load(os.path.join(d, f"X_train_3d_scaled_S{s}.npy")).astype(np.float32),
            np.load(os.path.join(d, f"y_train_windowed_S{s}.npy")).astype(np.float32),
            np.load(os.path.join(d, f"X_test_3d_scaled_S{s}.npy")).astype(np.float32),
            np.load(os.path.join(d, f"y_test_windowed_S{s}.npy")).astype(np.float32))


class GRULoc(nn.Module):
    def __init__(self, fin=41, hid=160, nb=29):
        super().__init__()
        self.gru = nn.GRU(fin, hid, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * hid, hid), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hid, nb))

    def forward(self, x):                              # x [B,16,41]
        o, _ = self.gru(x); return self.head(o[:, -1])  # last step -> [B,29] logits


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return 0.5
    a = np.concatenate([pos, neg]); r = np.empty(len(a)); r[a.argsort()] = np.arange(1, len(a) + 1)
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def loc_metrics(prob, y, thr):
    atk = y.sum(1) > 0; p = (prob[atk] > thr).astype(np.float32); ya = y[atk]
    tp = (p * ya).sum(0); fp = (p * (1 - ya)).sum(0); fn = ((1 - p) * ya).sum(0)
    micro = 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-9)
    macro = np.mean(2 * tp / (2 * tp + fp + fn + 1e-9))
    tps = (p * ya).sum(1); swf1 = np.mean(2 * tps / (p.sum(1) + ya.sum(1) + 1e-9))
    exact = np.mean((p == ya).all(1))
    return float(micro), float(macro), float(swf1), float(exact)


def run_seed(s, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, Xte, yte = load_scen(s)
    n = len(Xtr); idx = np.random.permutation(n); nv = int(0.15 * n)
    vi, ti = idx[:nv], idx[nv:]
    Xt = torch.tensor(Xtr[ti], device=DEV); Yt = torch.tensor(ytr[ti], device=DEV)
    Xv = torch.tensor(Xtr[vi], device=DEV); Yv = ytr[vi]
    Xe = torch.tensor(Xte, device=DEV)
    pos = Yt.sum() / (Yt.numel() - Yt.sum() + 1e-9); pw = torch.full((29,), float((1 / pos).clamp(1, 8)), device=DEV)
    model = GRULoc().to(DEV); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 100)
    for ep in range(100):
        model.train(); perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(perm), 256):
            j = perm[i:i + 256]; out = model(Xt[j]); loss = F.binary_cross_entropy_with_logits(out, Yt[j], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(torch.cat([model(Xv[i:i + 1024]) for i in range(0, len(Xv), 1024)])).cpu().numpy()
        pe = torch.sigmoid(torch.cat([model(Xe[i:i + 1024]) for i in range(0, len(Xe), 1024)])).cpu().numpy()
    # tune localization threshold on val to maximize micro-F1
    best_thr, best = 0.5, -1
    for thr in np.linspace(0.1, 0.9, 33):
        m = loc_metrics(pv, Yv, thr)[0]
        if m > best: best, best_thr = m, thr
    # detection with FA control on TEST
    sc = pe.max(1); atk = yte.sum(1) > 0
    A = auc(sc[atk], sc[~atk]); thrd = np.quantile(sc[~atk], 0.95) if (~atk).any() else 0.5
    DR5 = float((sc[atk] > thrd).mean()) if atk.any() else 0.0
    micro, macro, swf1, exact = loc_metrics(pe, yte, best_thr)
    # LESSON-comparable accuracy metrics (over ALL test windows, at the tuned threshold)
    p_all = (pe > best_thr).astype(np.float32)
    meter_acc = float((p_all == yte).mean())           # per-bus (per-meter) prediction accuracy, all windows
    row_acc = float((p_all == yte).all(1).mean())       # exact-match row accuracy, all windows
    return dict(det_auc=A, DR_at_FA5=DR5, meter_acc=meter_acc, row_acc=row_acc,
                micro_f1=micro, macro_f1=macro, sample_f1=swf1, exact_match=exact)


res = {"benchmark": "IEEE-30 DataPort FDIA (CC BY 4.0)", "model": "topology-free 2-layer BiGRU", "seeds": SEEDS, "per_scenario": {}}
print(f"device {DEV}; BiGRU localizer, {len(SEEDS)} seeds")
for s in range(1, 6):
    runs = [run_seed(s, sd) for sd in SEEDS]
    agg = {k: (round(float(np.mean([r[k] for r in runs])), 3), round(float(np.std([r[k] for r in runs])), 3)) for k in runs[0]}
    res["per_scenario"][f"S{s}"] = dict(name=NAMES[s], **{k: v[0] for k, v in agg.items()}, **{k + "_std": v[1] for k, v in agg.items()})
    r = res["per_scenario"][f"S{s}"]
    print(f"S{s} {NAMES[s]:13s} | AUC {r['det_auc']:.3f} | meter-acc {r['meter_acc']:.3f} row-acc {r['row_acc']:.3f} "
          f"(LESSON-style) | micro-F1 {r['micro_f1']:.3f} macro-F1 {r['macro_f1']:.3f} sample-F1 {r['sample_f1']:.3f} exact(atk) {r['exact_match']:.3f}", flush=True)
json.dump(res, open(os.path.join(RES, "bench_ieee30b.json"), "w"), indent=2)

# figure
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.edgecolor": "#222", "axes.linewidth": 0.8})
S = [f"S{i}" for i in range(1, 6)]; x = np.arange(5)
fig, ax = plt.subplots(figsize=(5.9, 3.3))
for key, col, mk, lab in [("det_auc", "#b2182b", "o", "detection AUC"), ("DR_at_FA5", "#8073ac", "s", "DR @ 5% FA"), ("micro_f1", "#1a9850", "^", "loc micro-F1")]:
    y = [res["per_scenario"][s][key] for s in S]; e = [res["per_scenario"][s][key + "_std"] for s in S]
    ax.errorbar(x, y, yerr=e, marker=mk, color=col, lw=1.6, ms=5, capsize=2.5, label=lab)
ax.axhline(0.5, color="#888", ls=":", lw=1.0); ax.text(3.3, 0.52, "chance", fontsize=7, color="#555")
ax.set_xticks(x); ax.set_xticklabels([f"S{i+1}\n{NAMES[i+1]}" for i in range(5)], fontsize=7.3)
ax.set_ylabel("score"); ax.set_ylim(0, 1.05)
ax.set_title("Our BiGRU localizer on IEEE-30 (3 seeds): detection to chance as attacks shrink", fontsize=8.8)
ax.legend(fontsize=7.8, frameon=False, loc="center left"); ax.grid(True, axis="y", ls=":", lw=0.4, alpha=0.6)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_bench_ieee30b.png"), dpi=175); fig.savefig(os.path.join(RES, "fig_bench_ieee30b.pdf"))
with open(os.path.join(RES, "sidecars", "bench_ieee30b.csv"), "w") as f:
    f.write("scenario,name,det_auc,det_auc_std,DR_at_FA5,micro_f1,micro_f1_std,macro_f1,sample_f1,exact_match\n")
    for s in range(1, 6):
        r = res["per_scenario"][f"S{s}"]; f.write(f"S{s},{r['name']},{r['det_auc']},{r['det_auc_std']},{r['DR_at_FA5']},{r['micro_f1']},{r['micro_f1_std']},{r['macro_f1']},{r['sample_f1']},{r['exact_match']}\n")
print("wrote results/bench_ieee30b.json + fig", flush=True)
