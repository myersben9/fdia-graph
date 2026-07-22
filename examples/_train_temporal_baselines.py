#!/usr/bin/env python
"""NON-GRAPH TEMPORAL localizer baselines on the v0.4.1 shards (reviewer-requested; PAPER 2 ml_centralized).

THESIS UNDER TEST: FDIA localization is fundamentally a TEMPORAL problem. Any attack that pushes a meter
beyond the noise floor makes a per-bus temporal SPIKE (a discontinuity vs the recent-normal trajectory) that
survives spatial/BDD stealth; the only doubly-stealthy family is the slow ramp At. If that is true, a
NON-GRAPH model given the temporal information should localize about as well as the graph (ARMA+attn), and it
should still fail on At. These baselines test exactly that with three model classes that are NOT graphs:

  lstm  per-bus LSTM over a RAW temporal window (learns temporal features itself; no KCL/graph, no
        hand-engineered temporal_delta/swing given)
  tcn   per-bus 1-D dilated causal TCN over the same RAW temporal window
  xgb   gradient-boosted trees (XGBoost) on the SAME 14-dim engineered per-bus feature vector the ARMA
        model consumes (meas4 + mask4 + KCL2 + temporal_delta2 + swing2) -- tests model-class invariance

WINDOWING SETUP (the part the reviewer wanted clarified) -- read carefully:
  The shards store only ONE family as contiguous multi-scan sequences: At (ramp), 100 sequences x 60 scans,
  each wholly inside one train/val/test split. Benign and every other attack family (Aq/Ad/As/Ar/Al) are
  stored as INDEPENDENT single scans (seq_id = -1); their temporal context is not stored as extra records.
  It lives instead in (a) the engineered features temporal_delta (current-minus-previous-scan injection) and
  swing (a z-score of the current jump vs the bus's own recent SWING_W=60-scan typical jump), and (b) the
  benign operating-point POOL X[t] (pool_ieee{C}.npz, [nT,N,4]) that every record's `timestep` indexes into
  and that the generator injected each attack onto.

  So for the SEQUENCE models (lstm/tcn) we reconstruct, per record at global timestep t, a length-W per-bus
  window of RAW measurements [V(pu), P(pu), Q(pu), theta(rad)]:
     frames t-W+1 .. t-1  = the benign POOL states X[.]      (recent-normal history)
     frame  t            = the record's OWN node_x           (the current, possibly-attacked scan)
  The history frames are masked with the CURRENT record's availability mask so unmeasured channels are zeroed
  identically to the current scan (no clean/masked mismatch). This is exactly what a streaming per-bus
  rate-of-change monitor sees: a recent-normal history then the live scan. It is FAITHFUL across families:
  an abrupt attack (Aq/Ad/As/Ar/Al) makes the current frame jump far off the history (per-bus P deviation
  ~1.3-4.8 p.u. vs a ~0.2 benign floor at 14-bus), while At's current-step deviation (~0.28) sits at the
  noise floor -- so the window keeps At hard, as the thesis predicts. The history is a benign rolling
  baseline (input context only); the label and the discriminative current frame come from the record itself.

  The `xgb` model does NOT use the window; it uses the 14-dim engineered feature vector directly, identical to
  train_arma.py.feats() / _train_baselines.py full14, so it is a features-held-fixed, graph-removed ablation.

Metrics, splits, thresholding are IDENTICAL to _train_baselines.py / train_arma.py: per-sample macro node-F1
(swf1) is the headline, threshold tuned on VAL attacked records and reported on TEST attacked records, plus
node_f1, per-family swf1, and grid-level detection F1. Trains on release_v0.4.1 shards, per-unit.

  python _train_temporal_baselines.py --system ieee14 --model lstm --seed 123 --out results/tmp.json
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from fdia_graph.dataset import FdiaGraph, FAMILIES

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SHARDS = os.path.join(HERE, "release_v0.4.1"); EPOCHS = int(os.environ.get("EPOCHS", "40"))
WIN = int(os.environ.get("WIN", "8"))                 # temporal window length W (scans) for lstm/tcn
dev = "cuda" if torch.cuda.is_available() else "cpu"


def f1(pred, tgt, sample=False):
    tp = (pred*tgt).sum(-1); fp = (pred*(1-tgt)).sum(-1); fn = ((1-pred)*tgt).sum(-1)
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); fv = 2*p*r/(p+r+1e-9)
    return fv.mean().item() if sample else (2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()+1e-9)).item()


def kcl_residual(node_x, edge_x, ei, N):
    """Per-bus P/Q power-balance residual from the measurements (inflow - injection); identical to
    train_arma.py / _train_baselines.py so xgb's full14 vector matches the graph model's bit-for-bit."""
    n = node_x.shape[0]; fr, to = ei[0], ei[1]
    inP = torch.zeros(n, N, device=node_x.device); inQ = torch.zeros(n, N, device=node_x.device)
    inP.index_add_(1, to, edge_x[:, :, 0]); inP.index_add_(1, fr, -edge_x[:, :, 0])
    inQ.index_add_(1, to, edge_x[:, :, 1]); inQ.index_add_(1, fr, -edge_x[:, :, 1])
    return torch.stack([inP - node_x[:, :, 1], inQ - node_x[:, :, 2]], -1)


# ------------------------------------------------------------------ sequence models (non-graph) --------
class GRULoc(nn.Module):
    """Per-bus GRU over the [W,4] raw measurement window -> per-bus attack logit. No graph, no engineered
    temporal features: the recurrence must learn the delta/swing signal itself. One GRU layer (faster and
    more stable than a deep LSTM at this window length, which under-trained/collapsed to all-positive)."""
    def __init__(self, cin=4, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.GRU(cin, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):                     # x [(B*N), W, 4]
        out, _ = self.rnn(x)                  # final hidden state summarizes the window
        return self.head(out[:, -1]).squeeze(-1)


class TCNLoc(nn.Module):
    """Per-bus 1-D dilated CAUSAL temporal conv net over the [W,4] window. Exponentially-growing dilation
    gives a wide receptive field over the window with few params; causal left-padding keeps it streaming."""
    def __init__(self, cin=4, hidden=64, levels=3, k=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(); c = cin
        for i in range(levels):
            d = 2 ** i
            self.blocks.append(nn.ModuleDict({
                "pad": nn.ConstantPad1d(((k - 1) * d, 0), 0.0),   # causal left-pad
                "conv": nn.Conv1d(c, hidden, k, dilation=d),
                "drop": nn.Dropout(dropout)}))
            c = hidden
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):                     # x [(B*N), W, 4]
        h = x.transpose(1, 2)                 # -> [(B*N), C, W]
        for b in self.blocks:
            h = b["drop"](F.relu(b["conv"](b["pad"](h))))
        return self.head(h[:, :, -1]).squeeze(-1)   # take the last (current) timestep's features


# ------------------------------------------------------------------ data loading ----------------------
_SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def fast_numpy(shard, split, fields):
    """Fast replacement for FdiaGraph.to_numpy(units='pu'): read each field FULLY (sequential gzip
    decompress, ~0.1s) then index the split rows in numpy. The SDK's per-row fancy h5py indexing
    re-decompresses gzip chunks and costs ~200s/split here; read-all is ~170x faster and bit-identical.
    Applies the same p.u. conversion as dataset._to_units (P,Q /baseMVA; theta deg->rad; edge/td /baseMVA)."""
    import h5py
    out = {}
    with h5py.File(shard, "r") as f:
        d = f["data"]; base = float(f.attrs.get("baseMVA", 100.0))
        idx = np.nonzero(d["split"][:] == _SPLIT_CODE[split])[0]
        for k in fields:
            a = d[k][:][idx]
            if k == "node_x":
                a = a.astype(np.float32); a[..., 1] /= base; a[..., 2] /= base; a[..., 3] = np.deg2rad(a[..., 3])
            elif k in ("edge_x", "temporal_delta"):
                a = a.astype(np.float32) / base
            out[k] = a
    return out


def load_split(shard, split, keys):
    a = fast_numpy(shard, split, keys + ["timestep", "family"])
    g = {k: torch.as_tensor(a[k], device=dev, dtype=torch.float32) for k in keys}
    g["timestep"] = torch.as_tensor(a["timestep"], device=dev, dtype=torch.long)
    return g, torch.as_tensor(a["family"].astype(np.int64), device=dev)


def load_pool(C, seed):
    """Benign operating-point pool [nT,N,4]. On disk column order is [P,Q,V,theta] in PHYSICAL units;
    return [V(pu),P(pu),Q(pu),theta(rad)] to match node_x (units='pu')."""
    p = os.path.join(SHARDS, f"pool_ieee{C}.npz" if seed == 123 else f"pool_ieee{C}_s{seed}.npz")
    z = np.load(p)["X"].astype(np.float32)                  # [nT,N,4] = [P_MW, Q_MVAr, V_pu, theta_deg]
    base = 100.0
    out = np.stack([z[:, :, 2], z[:, :, 0] / base, z[:, :, 1] / base, np.deg2rad(z[:, :, 3])], -1)
    return torch.as_tensor(out, device=dev, dtype=torch.float32)   # [nT,N,4] = [V,P,Q,theta] pu/rad


def build_windows(pool, g, N, W, mu, sd, chunk=4096):
    """Precompute the standardized raw temporal window for EVERY record in a split, ONCE.
    Returns Xflat [(n*N), W, 4] and yflat [(n*N)] on device. History = benign POOL frames (masked to the
    record's current availability), current frame = the record's own node_x. Building the window per batch on
    the fly is dominated by kernel-launch overhead on a shared GPU; materializing once (windows are small --
    even 300-bus is <2 GB) makes training a plain fast per-row sequence classification."""
    n = g["y"].shape[0]; nT = pool.shape[0]
    ks = torch.arange(W, device=dev) - (W - 1)             # offsets -(W-1)..0 (0 = current scan)
    outX = torch.empty(n * N, W, 4, device=dev, dtype=torch.float32)
    for i in range(0, n, chunk):                            # chunk over records to cap peak memory
        sl = slice(i, min(i + chunk, n)); b = sl.stop - sl.start
        t = g["timestep"][sl]                              # [b] global timestep
        times = (t[:, None] + ks[None, :]).clamp(0, nT - 1)  # [b,W]
        win = pool[times]                                  # [b,W,N,4] benign history
        m = g["node_m"][sl]                                # [b,N,4] current availability mask
        win = win * m[:, None, :, :]                       # mask history like the current scan
        win[:, -1] = g["node_x"][sl]                       # last frame = the real (possibly attacked) scan
        win = (win - mu) / sd * m[:, None, :, :]           # standardize; masked entries stay 0
        outX[sl.start * N: sl.stop * N] = win.permute(0, 2, 1, 3).reshape(b * N, W, 4)
    return outX, g["y"].reshape(n * N)


# ------------------------------------------------------------------ training paths --------------------
def run_seq(C, Model, tag, seed, shard, ei0, N):
    keys = ["node_x", "node_m", "y"]
    trG, trF = load_split(shard, "train", keys); vaG, vaF = load_split(shard, "val", keys)
    teG, teF = load_split(shard, "test", keys)
    pool = load_pool(C, seed)
    # per-channel standardization stats from TRAIN measured entries (measurements are physically the same
    # quantity as the pool history, so one set of stats standardizes the whole window consistently)
    w = trG["node_m"].sum((0, 1)).clamp(min=1.0)
    mu = (trG["node_x"] * trG["node_m"]).sum((0, 1)) / w
    sd = (((trG["node_x"] - mu) ** 2 * trG["node_m"]).sum((0, 1)) / w).sqrt().clamp(min=1e-3)
    Xtr, ytr = build_windows(pool, trG, N, WIN, mu, sd)
    Xva, _ = build_windows(pool, vaG, N, WIN, mu, sd)
    Xte, _ = build_windows(pool, teG, N, WIN, mu, sd)
    nva = vaG["y"].shape[0]; nte = teG["y"].shape[0]

    torch.manual_seed(seed)
    model = Model().to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    pos = float(trG["y"].sum()); pw = torch.tensor(min(max((trG["y"].numel()-pos)/max(pos, 1), 1.0), 30.0), device=dev)
    R = Xtr.shape[0]; B = 512 * N                           # 512 records' worth of bus-rows (kept ~constant per epoch across N)
    print(f"[ieee{C}/{tag}/s{seed}] train {trG['y'].shape[0]:,} records ({R:,} bus-rows)  N={N}  W={WIN}  "
          f"pos_weight={pw.item():.1f}  epochs={EPOCHS} [{dev}]", flush=True)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(R, device=dev); tot = torch.zeros((), device=dev)
        for i in range(0, R, B):
            idx = perm[i:i+B]
            loss = F.binary_cross_entropy_with_logits(model(Xtr[idx]), ytr[idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.detach() * idx.numel()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{EPOCHS}  loss {(tot/R).item():.4f}", flush=True)

    def collect(X, nrec):
        model.eval(); LG = []
        with torch.no_grad():
            for i in range(0, X.shape[0], B):
                LG.append(model(X[i:i+B]))
        return torch.cat(LG).reshape(nrec, N)
    vL = collect(Xva, nva); vY = vaG["y"]; vatk = vaF > 0
    thr = float(max(torch.linspace(-2, 3, 26), key=lambda th: f1((vL[vatk] > th).float(), vY[vatk])))
    L = collect(Xte, nte); Y = teG["y"]; P = (L > thr).float(); atk = teF > 0
    return finalize(C, tag, seed, thr, P, Y, L, vL, teF, vaF, extra={"window": WIN, "channels": "V,P,Q,theta(raw)"})


def run_xgb(C, seed, shard, ei0, N):
    import xgboost as xgb
    keys = ["node_x", "node_m", "edge_x", "y", "temporal_delta", "swing"]
    def full14(g):
        kcl = kcl_residual(g["node_x"], g["edge_x"], ei0, N)
        return torch.cat([g["node_x"], g["node_m"], kcl, g["temporal_delta"], g["swing"]], -1)  # [n,N,14]
    trG, trF = load_split(shard, "train", keys); vaG, vaF = load_split(shard, "val", keys)
    teG, teF = load_split(shard, "test", keys)
    Xtr = full14(trG).reshape(-1, 14).cpu().numpy(); ytr = trG["y"].reshape(-1).cpu().numpy()
    Xva = full14(vaG).reshape(-1, 14).cpu().numpy(); Xte = full14(teG).reshape(-1, 14).cpu().numpy()
    Nv = vaG["y"].shape[0]; Nt = teG["y"].shape[0]
    pos = float(ytr.sum()); spw = min(max((len(ytr) - pos) / max(pos, 1), 1.0), 30.0)   # same cap as the NN pos_weight
    clf = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
                            colsample_bytree=0.8, tree_method="hist", device=dev,
                            scale_pos_weight=spw, eval_metric="logloss", n_jobs=6,
                            random_state=seed)
    clf.fit(Xtr, ytr)
    pv = clf.predict_proba(Xva)[:, 1].reshape(Nv, N); pt = clf.predict_proba(Xte)[:, 1].reshape(Nt, N)
    vL = torch.as_tensor(pv); L = torch.as_tensor(pt)      # probabilities in [0,1]
    vY = vaG["y"].cpu(); Y = teG["y"].cpu(); vatk = (vaF > 0).cpu(); teF_c = teF.cpu(); vaF_c = vaF.cpu()
    thr = float(max(torch.linspace(0.05, 0.95, 19), key=lambda th: f1((vL[vatk] > th).float(), vY[vatk])))
    P = (L > thr).float(); atk = teF_c > 0
    return finalize(C, "xgb", seed, thr, P, Y, L, vL, teF_c, vaF_c, prob=True,
                    extra={"features": "full14(meas4+mask4+kcl2+temporal_delta2+swing2)",
                           "n_estimators": 400, "max_depth": 6})


def finalize(C, tag, seed, thr, P, Y, L, vL, teF, vaF, prob=False, extra=None):
    """Shared metric block: swf1 headline, node_f1, per-family swf1, grid-level detection F1 (val-tuned)."""
    P = P.cpu(); Y = Y.cpu(); L = L.cpu(); vL = vL.cpu(); teF = teF.cpu(); vaF = vaF.cpu()
    atk = teF > 0
    swf1 = f1(P[atk], Y[atk], sample=True); node_f1 = f1(P[atk], Y[atk], sample=False)
    perfam = {name: round(f1(P[teF == k], Y[teF == k], sample=True), 4) for k, name in FAMILIES.items()
              if k and (teF == k).any()}
    # grid-level detection F1 (max bus score), val-tuned. For NN paths L is a logit -> sigmoid; for xgb L is
    # already a probability, so score directly.
    sc = (lambda z: z.max(1).values) if prob else (lambda z: torch.sigmoid(z).max(1).values)
    def detf1(S, lab, t):
        pr = (S > t).float(); tp = (pr*lab).sum(); fp = (pr*(1-lab)).sum(); fn = ((1-pr)*lab).sum()
        p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); return (2*p*r/(p+r+1e-9)).item()
    vS = sc(vL); vlab = (vaF > 0).float()
    lo, hi = (0.05, 0.95) if prob else (0.05, 0.95)
    dthr = float(max(torch.linspace(lo, hi, 19), key=lambda t: detf1(vS, vlab, float(t))))
    det = detf1(sc(L), (teF > 0).float(), dthr)
    print(f"[ieee{C}/{tag}/s{seed}] swF1 {swf1:.3f}  node-F1 {node_f1:.3f}  det-F1 {det:.3f}  thr {thr:+.3f}  "
          f"perfam {perfam}", flush=True)
    out = {"system": f"ieee{C}", "model": tag, "seed": seed, "threshold": round(thr, 4),
           "swf1": round(swf1, 4), "node_f1": round(node_f1, 4), "det_f1": round(det, 4),
           "per_family_swf1": perfam}
    if extra: out.update(extra)
    return out


def main():
    ap = argparse.ArgumentParser(description="Non-graph temporal localizer baselines (lstm/tcn/xgb).")
    ap.add_argument("--system", required=True, help="ieee14|ieee118|ieee300")
    ap.add_argument("--model", required=True, choices=["gru", "tcn", "xgb"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(6)
    C = int(args.system.replace("ieee", ""))
    shard = os.path.join(SHARDS, f"ml_only_ieee{C}.h5" if args.seed == 123 else f"ml_only_ieee{C}_s{args.seed}.h5")
    if not os.path.exists(shard):
        print(f"shard missing: {shard}", flush=True); sys.exit(2)
    ds0 = FdiaGraph(shard); N = ds0.N; ei0 = ds0.edge_index.to(dev)
    if args.model == "xgb":
        res = run_xgb(C, args.seed, shard, ei0, N)
    else:
        Model = (lambda: GRULoc()) if args.model == "gru" else (lambda: TCNLoc())
        res = run_seq(C, Model, args.model, args.seed, shard, ei0, N)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2); print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
