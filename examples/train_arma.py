#!/usr/bin/env python
"""Stronger localizer baseline: an edge-fused ARMA graph convolution net (Bianchi et al., IEEE TPAMI 2021).

Motivation: a 2-3 hop message-passing GNN has a narrow receptive field, so it struggles to localize the
sparse, spatially-distributed FDIA attacks on large grids (IEEE-118/300). ARMA convolution uses a RATIONAL
(auto-regressive moving-average) spectral filter with a much broader, adaptive receptive field. Pure
ARMAConv is node-spectral and would ignore the branch P/Q FLOW features that carry the attack signal, so we
first FUSE the edge features into the node representation (edge-encoding scattered to both endpoints), then
stack ARMA filters, then decode per-bus attack logits. Per-record (spatial) model; same optimized recipe as
the other examples (feature standardization, GPU-resident data, bf16, auto pos_weight, val-tuned threshold).

    python train_arma.py --system ieee118 --epochs 40
"""
import argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from contextlib import nullcontext
import fdia_graph as fg
from fdia_graph.dataset import FAMILIES
from torch_geometric.nn import ARMAConv


class ArmaLoc(nn.Module):
    def __init__(self, hidden=128, stacks=3, layers=2, dropout=0.1):
        super().__init__()
        self.nenc = nn.Linear(4 * 2, hidden)                     # node feat (4) concat mask (4)
        self.eenc = nn.Linear(2 * 2, hidden)                     # edge feat (2) concat mask (2)
        # two ARMA blocks: each is `stacks` parallel ARMA_1 filters, each `layers` recursive steps -> wide receptive field
        self.arma1 = ARMAConv(hidden, hidden, num_stacks=stacks, num_layers=layers, shared_weights=True, dropout=dropout, act=F.relu)
        self.arma2 = ARMAConv(hidden, hidden, num_stacks=stacks, num_layers=layers, shared_weights=True, dropout=dropout, act=F.relu)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, e, ei):
        # x [n,8]  e [m,4]  ei [2,m] (bidirectional, batched). Returns per-node logit [n].
        h = F.relu(self.nenc(x))
        he = F.relu(self.eenc(e))
        h = h + torch.zeros_like(h).index_add_(0, ei[1], he)     # fuse edge-flow signal into destination nodes
        h = self.arma1(h, ei); h = self.arma2(h, ei)
        return self.head(h).squeeze(-1)


def f1(pred, tgt, sample=False):
    tp = (pred * tgt).sum(-1); fp = (pred * (1 - tgt)).sum(-1); fn = ((1 - pred) * tgt).sum(-1)
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); fv = 2 * p * r / (p + r + 1e-9)
    return fv.mean().item() if sample else (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-9)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--release", default=None); ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev == "cuda" else nullcontext

    def gpu(split):
        ds = fg.load(args.system, split=split, release=args.release); a = ds.to_numpy()
        g = {k: torch.as_tensor(a[k], device=dev, dtype=torch.float32) for k in ("node_x", "node_m", "edge_x", "edge_m", "y")}
        return g, torch.as_tensor(a["family"], device=dev), ds
    trG, trFam, ds = gpu("train"); vaG, vaFam, _ = gpu("val"); teG, teFam, _ = gpu("test")
    N, E = ds.N, ds.E
    ei0 = ds.edge_index.to(dev)                                   # [2,E] per-graph
    # feature standardization (train stats, metered entries) — essential on large grids
    for xk, mk in (("node_x", "node_m"), ("edge_x", "edge_m")):
        w = trG[mk].sum((0, 1)).clamp(min=1.0); mu = (trG[xk] * trG[mk]).sum((0, 1)) / w
        sd = (((trG[xk] - mu) ** 2 * trG[mk]).sum((0, 1)) / w).sqrt().clamp(min=1e-3)
        for g in (trG, vaG, teG): g[xk] = (g[xk] - mu) / sd * g[mk]

    # precompute a BIDIRECTIONAL, batched edge_index for a given batch size (PyG-style: B disjoint grid copies)
    ei_bi = torch.cat([ei0, ei0.flip(0)], 1)                      # [2, 2E] undirected (spectral filters need symmetry)
    def batched(B):
        off = (torch.arange(B, device=dev) * N).repeat_interleave(ei_bi.shape[1])
        return ei_bi.repeat(1, B) + off.unsqueeze(0)             # [2, B*2E]
    def feats(g, idx):
        b = len(idx)
        x = torch.cat([g["node_x"][idx], g["node_m"][idx]], -1).reshape(b * N, -1)      # [b*N, 8]
        e2 = torch.cat([g["edge_x"][idx], g["edge_m"][idx]], -1)                          # [b,E,4]
        e = torch.cat([e2, e2], 1).reshape(b * 2 * E, -1)                                 # bidirectional [b*2E,4]
        return x, e, batched(b), g["y"][idx].reshape(b * N)

    model = ArmaLoc().to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    pos = float(trG["y"].sum()); pw = torch.tensor(min(max((trG["y"].numel() - pos) / max(pos, 1), 1.0), 30.0), device=dev)
    n = trG["y"].shape[0]
    print(f"{args.system}: N={N} E={E}  train {n:,}  ARMA(stacks=3,layers=2)  pos_weight={pw.item():.1f}  batch={args.batch} [{dev}]")
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]; x, e, ei, y = feats(trG, idx)
            with autocast():
                loss = F.binary_cross_entropy_with_logits(model(x, e, ei), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0 or ep == 0: print(f"epoch {ep+1}/{args.epochs}  loss {tot/n:.4f}")

    def collect(g, fam):
        model.eval(); LG = []
        with torch.no_grad(), autocast():
            for i in range(0, g["y"].shape[0], args.batch):
                idx = torch.arange(i, min(i + args.batch, g["y"].shape[0]), device=dev)
                x, e, ei, _ = feats(g, idx)
                LG.append(model(x, e, ei).reshape(len(idx), N).float().cpu())
        return torch.cat(LG), g["y"].cpu(), fam.cpu()
    vL, vY, vFm = collect(vaG, vaFam); vatk = vFm > 0
    thr = float(max(torch.linspace(-2, 3, 26), key=lambda t: f1((vL[vatk] > t).float(), vY[vatk])))
    L, Y, Fm = collect(teG, teFam); P = (L > thr).float(); atk = Fm > 0
    res = {"system": args.system, "N": int(N), "E": int(E), "model": "ARMA", "threshold": round(thr, 2),
           "overall": {"n": int(atk.sum()), "node_f1": f1(P[atk], Y[atk]), "swf1": f1(P[atk], Y[atk], sample=True)}, "per_family": {}}
    print(f"tuned threshold {thr:.2f}\noverall (attacked): node-F1 {res['overall']['node_f1']:.3f}  swF1 {res['overall']['swf1']:.3f}")
    for k, name in FAMILIES.items():
        m = Fm == k
        if m.any():
            r = {"n": int(m.sum()), "node_f1": f1(P[m], Y[m]), "swf1": f1(P[m], Y[m], sample=True)}
            res["per_family"][name] = r; print(f"  {name:6s} n={r['n']:5d}  node-F1 {r['node_f1']:.3f}  swF1 {r['swf1']:.3f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2); print("wrote", args.out)


if __name__ == "__main__":
    main()
