#!/usr/bin/env python
"""End-to-end example: train a GNN attack-localizer on an fdia-graph dataset — the whole point of the SDK.

A researcher runs THIS and nothing else: the SDK downloads/loads the data and yields ready batches; the
model below is a small edge-conditioned message-passing GNN that predicts, per bus, whether that bus is
attacked (multi-label localization). We train on the chronological train split and report per-attack-type
node-F1 on the test split — so the hard stealthy families (Ao/ramp/LRA) are visible, not hidden behind the
easy detectable ones. Runs on CPU for IEEE-14; pass a bigger system + GPU for 118/300.

    python train_gnn.py --system ieee14 --epochs 5
"""
from __future__ import annotations

import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from typing import Any, Dict, Tuple
import fdia_graph as fg
from fdia_graph.dataset import FAMILIES


class EdgeMPNN(nn.Module):
    """Edge-conditioned message-passing net: each layer messages per directed edge from
    (src, dst, edge feats) and mean-aggregates at the destination, so line P/Q flows inform the
    incident buses. Availability masks are concatenated so the model knows which meters are present."""

    def __init__(self, n_node_feat: int = 4, n_edge_feat: int = 2, hidden: int = 64, layers: int = 3) -> None:
        super().__init__()
        self.enc = nn.Linear(n_node_feat * 2, hidden)                  # *2: feature concatenated with its mask
        self.edge_enc = nn.Linear(n_edge_feat * 2, hidden)
        # one MLP per message-passing round: [h_src, h_dst, h_edge] -> hidden
        self.msg = nn.ModuleList([nn.Linear(hidden * 3, hidden) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 1)                               # per-node attack logit

    def forward(self, node_x: torch.Tensor, node_m: torch.Tensor, edge_x: torch.Tensor,
                edge_m: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        B, N, _ = node_x.shape; E = edge_index.shape[1]
        src, dst = edge_index[0], edge_index[1]
        h = F.relu(self.enc(torch.cat([node_x, node_m], -1)))          # [B,N,H]
        he = F.relu(self.edge_enc(torch.cat([edge_x, edge_m], -1)))    # [B,E,H]
        src2 = torch.cat([src, dst]); dst2 = torch.cat([dst, src]); he2 = torch.cat([he, he], 1)  # bidirectional
        for msg, upd in zip(self.msg, self.upd):
            m = F.relu(msg(torch.cat([h[:, src2], h[:, dst2], he2], -1)))   # message per edge
            agg = torch.zeros_like(h)
            agg.index_add_(1, dst2, m)
            deg = torch.zeros(N, device=h.device).index_add_(0, dst2, torch.ones(2 * E, device=h.device))
            agg = agg / deg.clamp(min=1).view(1, N, 1)                 # mean aggregation
            h = F.relu(upd(torch.cat([h, agg], -1)))
        return self.head(h).squeeze(-1)                               # [B,N] logits


def node_f1(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """Micro node-F1 pooled across the selected records (mask). Threshold logit>0."""
    pred = (logits[mask] > 0).float(); tgt = y[mask]
    tp = (pred * tgt).sum(); fp = (pred * (1 - tgt)).sum(); fn = ((1 - pred) * tgt).sum()
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return (2 * p * r / (p + r + 1e-9)).item()


def sample_f1(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """Sample-wise F1 (Boyaci et al., IEEE T-SG 2022): F1 over buses within each record, averaged across
    records. Stricter than pooled node-F1 as it rewards the per-sample bus SET. Over attacked records."""
    pred = (logits[mask] > 0).float(); tgt = y[mask]                 # [n_rec, N]
    tp = (pred * tgt).sum(1); fp = (pred * (1 - tgt)).sum(1); fn = ((1 - pred) * tgt).sum(1)
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return (2 * p * r / (p + r + 1e-9)).mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)         # large batch — target the GPU, not memory
    ap.add_argument("--release", default=None)               # pin a dataset version if desired
    ap.add_argument("--pos_weight", type=float, default=-1)   # -1 = auto (neg/pos; scales with grid size)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from contextlib import nullcontext
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":                                        # GPU throughput knobs
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev == "cuda" else nullcontext

    train = fg.load(args.system, split="train", release=args.release)
    test = fg.load(args.system, split="test", release=args.release)
    ei = train.edge_index.to(dev)
    print(f"train {len(train):,} / test {len(test):,} records on {args.system} (N={train.N}, E={train.E}) [{dev}]")

    # Pre-load whole splits onto the GPU once so batches are pure GPU gathers (train.loader() streams if VRAM is tight).
    def gpu(ds: Any) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        a = ds.to_numpy()
        return ({k: torch.as_tensor(a[k], device=dev, dtype=torch.float32) for k in ("node_x", "node_m", "edge_x", "edge_m", "y")},
                torch.as_tensor(a["family"], device=dev))
    trG, _ = gpu(train); teG, teFam = gpu(test); n = trG["y"].shape[0]
    # Standardize per channel using TRAIN stats (metered entries only); essential as raw MW is large on IEEE-300.
    for xk, mk in (("node_x", "node_m"), ("edge_x", "edge_m")):
        w = trG[mk].sum((0, 1)).clamp(min=1.0)
        mu = (trG[xk] * trG[mk]).sum((0, 1)) / w
        sd = (((trG[xk] - mu) ** 2 * trG[mk]).sum((0, 1)) / w).sqrt().clamp(min=1e-3)
        trG[xk] = (trG[xk] - mu) / sd * trG[mk]; teG[xk] = (teG[xk] - mu) / sd * teG[mk]

    model = EdgeMPNN().to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3)
    # auto BCE pos-class weight = neg/pos ratio (attacked-bus rate falls as the grid grows).
    pos = float(trG["y"].sum())
    pwv = args.pos_weight if args.pos_weight > 0 else float(min(max((trG["y"].numel() - pos) / max(pos, 1), 1.0), 100.0))
    pw = torch.tensor(pwv, device=dev); print(f"pos_weight = {pwv:.1f}  batch={args.batch}  bf16={dev=='cuda'}")
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            with autocast():
                logits = model(trG["node_x"][idx], trG["node_m"][idx], trG["edge_x"][idx], trG["edge_m"][idx], ei)
                loss = F.binary_cross_entropy_with_logits(logits, trG["y"][idx], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        print(f"epoch {ep+1}/{args.epochs}  train loss {tot/n:.4f}")

    # evaluate: overall + per-attack-type node-F1 + swF1 (GPU-resident, bf16)
    model.eval(); allL = []
    nt = teG["y"].shape[0]
    with torch.no_grad(), autocast():
        for i in range(0, nt, args.batch):
            sl = slice(i, i + args.batch)
            allL.append(model(teG["node_x"][sl], teG["node_m"][sl], teG["edge_x"][sl], teG["edge_m"][sl], ei).float().cpu())
    L = torch.cat(allL); Y = teG["y"].cpu(); Fam = teFam.cpu()
    attacked = Fam > 0
    print(f"\noverall (attacked records):  node-F1 {node_f1(L, Y, attacked):.3f}   swF1 {sample_f1(L, Y, attacked):.3f}")
    print(f"{'family':8s} {'n':>6s}   node-F1   swF1   (swF1 = Boyaci sample-wise localization F1)")
    for k, name in FAMILIES.items():
        if k == 0:
            continue
        m = Fam == k
        if m.any():
            print(f"  {name:6s} {int(m.sum()):6d}    {node_f1(L, Y, m):.3f}   {sample_f1(L, Y, m):.3f}")


if __name__ == "__main__":
    main()
