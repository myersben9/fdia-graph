#!/usr/bin/env python
"""Simple localizer baselines (per-bus MLP, plain 2-layer GCN) on the v0.4.1 shards — a complexity LADDER
next to the ARMA+attention hybrid, so the report shows whether the fancy model earns its complexity.

Same node features (node_x + node_m, 8 dims/bus), same sample-wise F1 metric, same pos-weighted BCE and
val-tuned threshold as train_arma.py. Writes results/baselines_{C}.json {mlp:{swf1,det_f1}, gcn:{...}}.
Env: EPOCHS (default 40). Trains on the LOCAL release_v0.4.1 shards for all three systems, per-unit.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fdia_graph.dataset import FdiaGraph, FAMILIES

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SHARDS = os.path.join(HERE, "release_v0.4.1"); EPOCHS = int(os.environ.get("EPOCHS", "40"))
dev = "cuda" if torch.cuda.is_available() else "cpu"


def f1(pred, tgt, sample=False):
    tp = (pred*tgt).sum(-1); fp = (pred*(1-tgt)).sum(-1); fn = ((1-pred)*tgt).sum(-1)
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); fv = 2*p*r/(p+r+1e-9)
    return fv.mean().item() if sample else (2*tp.sum()/(2*tp.sum()+fp.sum()+fn.sum()+1e-9)).item()


class MLP(nn.Module):                                  # per-bus MLP, NO graph (pure lower bound)
    def __init__(self, fin, h=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(fin, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x, ei): return self.net(x).squeeze(-1)   # ei ignored


class GCN(nn.Module):                                  # plain 2-layer GCN (simplest graph model)
    def __init__(self, fin, h=128):
        super().__init__(); self.c1 = GCNConv(fin, h); self.c2 = GCNConv(h, h); self.head = nn.Linear(h, 1)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei)); x = F.relu(self.c2(x, ei)); return self.head(x).squeeze(-1)


def run(C, Model, tag):
    def gpu(split):
        # per-unit measurements (V p.u., P/Q p.u., theta rad) for consistent feature scales — matches train_arma.py
        a = FdiaGraph(os.path.join(SHARDS, f"ml_only_ieee{C}.h5"), split=split, units="pu").to_numpy()
        g = {k: torch.as_tensor(a[k], device=dev, dtype=torch.float32) for k in ("node_x", "node_m", "y")}
        return g, torch.as_tensor(a["family"], device=dev)
    trG, trF = gpu("train"); vaG, vaF = gpu("val"); teG, teF = gpu("test")
    N = trG["y"].shape[1]; ds0 = FdiaGraph(os.path.join(SHARDS, f"ml_only_ieee{C}.h5")); ei0 = ds0.edge_index.to(dev)
    ei_bi = torch.cat([ei0, ei0.flip(0)], 1)
    def batched(B):
        off = (torch.arange(B, device=dev)*N).repeat_interleave(ei_bi.shape[1]); return ei_bi.repeat(1, B)+off.unsqueeze(0)
    def feats(g, idx):
        b = len(idx); x = torch.cat([g["node_x"][idx], g["node_m"][idx]], -1).reshape(b*N, -1)
        return x, batched(b), g["y"][idx].reshape(b*N)
    torch.manual_seed(123)
    model = Model(8).to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    pos = float(trG["y"].sum()); pw = torch.tensor(min(max((trG["y"].numel()-pos)/max(pos, 1), 1.0), 30.0), device=dev)
    n = trG["y"].shape[0]; B = 256
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, B):
            idx = perm[i:i+B]; x, ei, y = feats(trG, idx)
            loss = F.binary_cross_entropy_with_logits(model(x, ei), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
    def collect(g):
        model.eval(); LG = []
        with torch.no_grad():
            for i in range(0, g["y"].shape[0], B):
                idx = torch.arange(i, min(i+B, g["y"].shape[0]), device=dev); x, ei, _ = feats(g, idx)
                LG.append(model(x, ei).reshape(len(idx), N))
        return torch.cat(LG)
    vL = collect(vaG); vY = vaG["y"]; vatk = vaF > 0
    thr = float(max(torch.linspace(-2, 3, 26), key=lambda t: f1((vL[vatk] > t).float(), vY[vatk])))
    L = collect(teG); Y = teG["y"]; P = (L > thr).float(); atk = teF > 0
    swf1 = f1(P[atk], Y[atk], sample=True)
    # detection F1 (grid-level max-prob), val-tuned
    def dscore(x): return torch.sigmoid(x).max(1).values
    def detf1(S, lab, t):
        pr = (S > t).float(); tp = (pr*lab).sum(); fp = (pr*(1-lab)).sum(); fn = ((1-pr)*lab).sum()
        p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); return (2*p*r/(p+r+1e-9)).item()
    vS = dscore(vL); vlab = (vaF > 0).float()
    dthr = float(max(torch.linspace(0.05, 0.95, 19), key=lambda t: detf1(vS, vlab, float(t))))
    det = detf1(dscore(L), (teF > 0).float(), dthr)
    print(f"[ieee{C}/{tag}] swF1 {swf1:.3f}  det-F1 {det:.3f}", flush=True)
    return {"swf1": round(swf1, 3), "det_f1": round(det, 3)}


for C in (14, 118, 300):
    shard = os.path.join(SHARDS, f"ml_only_ieee{C}.h5")
    if not os.path.exists(shard): print(f"ieee{C} shard missing, skip"); continue
    print(f"[data] ieee{C} reading {shard}", flush=True)
    out = {"mlp": run(C, MLP, "mlp"), "gcn": run(C, GCN, "gcn")}
    json.dump(out, open(os.path.join(RES, f"baselines_{C}.json"), "w"), indent=2)
    print(f"[done] results/baselines_{C}.json", flush=True)
print("[all] simple baselines done", flush=True)
