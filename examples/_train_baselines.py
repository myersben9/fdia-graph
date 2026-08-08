#!/usr/bin/env python
"""Simple localizer baselines (per-bus MLP, plain 2-layer GCN) on the v0.4.1 shards — a complexity LADDER
next to the ARMA+attention hybrid, so the report shows whether the fancy model earns its complexity.

Feature sets (``--features``):
  raw8    node_x(4) + node_m(4)                                    -- the NAIVE per-bus input (paper's Fig-1
          baseline). NOT standardized, to reproduce the published baselines_{C}.json exactly.
  full14  meas(4) + mask(4) + KCL residual(2) + temporal_delta(2) + swing(2) = 14 dims, standardized.
          This is the SAME per-bus feature vector train_arma.py's feats() builds, so an MLP/GCN on full14
          is a features-held-fixed ablation that isolates the value of the graph model itself.
  swing   windowed swing only (2 dims), standardized -- the single engineered temporal feature whose
          per-bus ROC-AUC is ~0.93-1.00 (results/feature_sep.json).

Same sample-wise F1 metric, pos-weighted BCE and val-tuned threshold as train_arma.py.
No-argument invocation preserves the legacy behaviour: MLP+GCN on raw8 over IEEE-14/118/300, written to
results/baselines_{C}.json {mlp:{swf1,det_f1}, gcn:{...}} -- so existing results still reproduce.
Single-job invocation (``--system ieee300 --model mlp --features full14 --seed 123 --out path.json``) writes
one rich per-(system,model,features,seed) record for the corrected multi-seed ladder / feature ablation.
Env: EPOCHS (default 40). Trains on the LOCAL release_v0.4.1 shards, per-unit.
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fdia_graph.dataset import FdiaGraph, FAMILIES

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SHARDS = os.environ.get("FDIA_SHARDS", os.path.join(HERE, "release_v0.4.1")); EPOCHS = int(os.environ.get("EPOCHS", "40"))
dev = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_SETS = ("raw8", "full14", "swing")
# raw8 is left UN-standardized on purpose so the legacy baselines_{C}.json numbers (0.108/0.130/0.603)
# reproduce bit-for-bit; the engineered-feature sets are standardized to match train_arma.py's recipe.
STANDARDIZE = {"raw8": False, "full14": True, "swing": True}


def f1(pred, tgt, sample=False):
    tp = (pred*tgt).sum(-1); fp = (pred*(1-tgt)).sum(-1); fn = ((1-pred)*tgt).sum(-1)
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); fv = 2*p*r/(p+r+1e-9)
    return fv.mean().item() if sample else (2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()+1e-9)).item()


def kcl_residual(node_x, edge_x, ei, N):
    """Per-bus real/reactive power-balance residual from the MEASUREMENTS (inflow - injection); identical to
    train_arma.py.kcl_residual. node_x[...,1]/[...,2] are metered P/Q injection, edge_x[...,0]/[...,1] P/Q flow."""
    n = node_x.shape[0]; fr, to = ei[0], ei[1]
    inP = torch.zeros(n, N, device=node_x.device); inQ = torch.zeros(n, N, device=node_x.device)
    inP.index_add_(1, to, edge_x[:, :, 0]); inP.index_add_(1, fr, -edge_x[:, :, 0])
    inQ.index_add_(1, to, edge_x[:, :, 1]); inQ.index_add_(1, fr, -edge_x[:, :, 1])
    return torch.stack([inP - node_x[:, :, 1], inQ - node_x[:, :, 2]], -1)


class MLP(nn.Module):                                  # per-bus MLP, NO graph (pure lower bound)
    def __init__(self, fin, h=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(fin, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x, ei): return self.net(x).squeeze(-1)   # ei ignored


class GCN(nn.Module):                                  # plain 2-layer GCN (simplest graph model)
    def __init__(self, fin, h=128):
        super().__init__(); self.c1 = GCNConv(fin, h); self.c2 = GCNConv(h, h); self.head = nn.Linear(h, 1)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei)); x = F.relu(self.c2(x, ei)); return self.head(x).squeeze(-1)


def build_feats(g, ei0, N, features):
    """Return per-record per-bus feature tensor [R,N,F] for the chosen feature set (see module docstring)."""
    nx, nm = g["node_x"], g["node_m"]
    if features == "raw8":
        return torch.cat([nx, nm], -1)                                              # [R,N,8]
    if features == "full14":
        kcl = kcl_residual(nx, g["edge_x"], ei0, N)                                 # [R,N,2]
        return torch.cat([nx, nm, kcl, g["temporal_delta"], g["swing"]], -1)        # [R,N,14]
    if features == "swing":
        return g["swing"]                                                           # [R,N,2]
    raise ValueError(f"unknown feature set {features!r}")


def run(C, Model, tag, seed=123, features="raw8"):
    def gpu(split):
        # per-unit measurements (V p.u., P/Q p.u., theta rad) for consistent feature scales — matches train_arma.py
        a = FdiaGraph(os.path.join(SHARDS, f"ml_only_ieee{C}.h5"), split=split, units="pu").to_numpy()
        keys = ["node_x", "node_m", "y"] + (["edge_x", "temporal_delta", "swing"] if features == "full14"
                                            else ["swing"] if features == "swing" else [])
        g = {k: torch.as_tensor(a[k], device=dev, dtype=torch.float32) for k in keys}
        return g, torch.as_tensor(a["family"], device=dev)
    trG, trF = gpu("train"); vaG, vaF = gpu("val"); teG, teF = gpu("test")
    N = trG["y"].shape[1]; ds0 = FdiaGraph(os.path.join(SHARDS, f"ml_only_ieee{C}.h5")); ei0 = ds0.edge_index.to(dev)
    ei_bi = torch.cat([ei0, ei0.flip(0)], 1)
    # materialize the chosen feature tensor once per split, then (optionally) z-score with TRAIN stats
    Xtr = build_feats(trG, ei0, N, features); Xva = build_feats(vaG, ei0, N, features); Xte = build_feats(teG, ei0, N, features)
    if STANDARDIZE[features]:
        mu = Xtr.mean((0, 1)); sd = Xtr.std((0, 1)).clamp(min=1e-3)
        Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd; Xte = (Xte - mu) / sd
    Fdim = Xtr.shape[-1]

    def batched(B):
        off = (torch.arange(B, device=dev)*N).repeat_interleave(ei_bi.shape[1]); return ei_bi.repeat(1, B)+off.unsqueeze(0)
    def feats(X, Y, idx):
        b = len(idx); x = X[idx].reshape(b*N, -1); return x, batched(b), Y[idx].reshape(b*N)
    torch.manual_seed(seed)
    model = Model(Fdim).to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    pos = float(trG["y"].sum()); pw = torch.tensor(min(max((trG["y"].numel()-pos)/max(pos, 1), 1.0), 30.0), device=dev)
    n = trG["y"].shape[0]; B = 256
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, B):
            idx = perm[i:i+B]; x, ei, y = feats(Xtr, trG["y"], idx)
            loss = F.binary_cross_entropy_with_logits(model(x, ei), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    def collect(X, Y):
        model.eval(); LG = []
        with torch.no_grad():
            for i in range(0, Y.shape[0], B):
                idx = torch.arange(i, min(i+B, Y.shape[0]), device=dev); x, ei, _ = feats(X, Y, idx)
                LG.append(model(x, ei).reshape(len(idx), N))
        return torch.cat(LG)
    vL = collect(Xva, vaG["y"]); vY = vaG["y"]; vatk = vaF > 0
    thr = float(max(torch.linspace(-2, 3, 26), key=lambda t: f1((vL[vatk] > t).float(), vY[vatk])))
    L = collect(Xte, teG["y"]); Y = teG["y"]; P = (L > thr).float(); atk = teF > 0
    swf1 = f1(P[atk], Y[atk], sample=True); node_f1 = f1(P[atk], Y[atk], sample=False)
    perfam = {name: round(f1(P[teF == k], Y[teF == k], sample=True), 4) for k, name in FAMILIES.items()
              if k and (teF == k).any()}
    # detection F1 (grid-level max-prob), val-tuned
    def dscore(x): return torch.sigmoid(x).max(1).values
    def detf1(S, lab, t):
        pr = (S > t).float(); tp = (pr*lab).sum(); fp = (pr*(1-lab)).sum(); fn = ((1-pr)*lab).sum()
        p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); return (2*p*r/(p+r+1e-9)).item()
    vS = dscore(vL); vlab = (vaF > 0).float()
    dthr = float(max(torch.linspace(0.05, 0.95, 19), key=lambda t: detf1(vS, vlab, float(t))))
    det = detf1(dscore(L), (teF > 0).float(), dthr)
    print(f"[ieee{C}/{tag}/{features}] swF1 {swf1:.3f}  node-F1 {node_f1:.3f}  det-F1 {det:.3f}  thr {thr:+.2f}  Fdim {Fdim}", flush=True)
    return {"system": f"ieee{C}", "model": tag, "features": features, "seed": seed, "Fdim": int(Fdim),
            "threshold": round(thr, 3), "swf1": round(swf1, 4), "node_f1": round(node_f1, 4),
            "det_f1": round(det, 4), "per_family_swf1": perfam}


MODELS = {"mlp": MLP, "gcn": GCN}


def main():
    ap = argparse.ArgumentParser(description="Per-bus MLP / plain-GCN localizer baselines on v0.4.1 shards.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed (default 123, the canonical run).")
    ap.add_argument("--features", choices=FEATURE_SETS, default="raw8",
                    help="per-bus feature set (default raw8 = naive baseline; full14 = ARMA's feature vector).")
    ap.add_argument("--system", default=None, help="ieee14|ieee118|ieee300 for a SINGLE job; omit for the legacy loop.")
    ap.add_argument("--model", choices=["mlp", "gcn", "both"], default="both", help="which baseline(s) to train.")
    ap.add_argument("--out", default=None, help="write the single-job result dict to this JSON path.")
    args = ap.parse_args()

    if args.system is None:
        # ---- legacy path: MLP+GCN on raw8 over all three systems, canonical filenames (reproduces published JSONs) ----
        sfx = "" if (args.seed == 123 and args.features == "raw8") else f"_{args.features}_s{args.seed}"
        for C in (14, 118, 300):
            shard = os.path.join(SHARDS, f"ml_only_ieee{C}.h5")
            if not os.path.exists(shard): print(f"ieee{C} shard missing, skip"); continue
            print(f"[data] ieee{C} reading {shard}  (seed={args.seed}, features={args.features})", flush=True)
            out = {"mlp": {k: run(C, MLP, "mlp", args.seed, args.features)[k] for k in ("swf1", "det_f1")},
                   "gcn": {k: run(C, GCN, "gcn", args.seed, args.features)[k] for k in ("swf1", "det_f1")}}
            json.dump(out, open(os.path.join(RES, f"baselines_{C}{sfx}.json"), "w"), indent=2)
            print(f"[done] results/baselines_{C}{sfx}.json", flush=True)
        print("[all] simple baselines done", flush=True)
        return

    # ---- single-job path for the corrected ladder / feature ablation ----
    C = int(args.system.replace("ieee", ""))
    shard = os.path.join(SHARDS, f"ml_only_ieee{C}.h5")
    if not os.path.exists(shard): print(f"ieee{C} shard missing, abort"); sys.exit(2)
    which = ["mlp", "gcn"] if args.model == "both" else [args.model]
    res = {m: run(C, MODELS[m], m, args.seed, args.features) for m in which}
    payload = res[which[0]] if len(which) == 1 else res
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(payload, open(args.out, "w"), indent=2); print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
