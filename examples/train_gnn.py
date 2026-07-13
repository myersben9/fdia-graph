#!/usr/bin/env python
"""End-to-end example: train a GNN attack-localizer on an fdia-graph dataset — the whole point of the SDK.

A researcher runs THIS and nothing else: the SDK downloads/loads the data and yields ready batches; the
model below is a small edge-conditioned message-passing GNN that predicts, per bus, whether that bus is
attacked (multi-label localization). We train on the chronological train split and report per-attack-type
node-F1 on the test split — so the hard stealthy families (Ao/ramp/LRA) are visible, not hidden behind the
easy detectable ones. Runs on CPU for IEEE-14; pass a bigger system + GPU for 118/300.

    python train_gnn.py --system ieee14 --epochs 5
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import fdia_graph as fg
from fdia_graph.dataset import FAMILIES


class EdgeMPNN(nn.Module):
    """Minimal edge-conditioned message-passing net. Each layer builds a message per directed edge from
    (source node, dest node, edge features) and mean-aggregates it at the destination — so line P/Q flows
    (edge features) directly inform the incident buses, which is exactly the physics a localizer needs.
    Availability masks are concatenated to the raw features so the model knows which meters are present."""

    def __init__(self, n_node_feat=4, n_edge_feat=2, hidden=64, layers=3):
        super().__init__()
        self.enc = nn.Linear(n_node_feat * 2, hidden)                  # *2: features are concatenated with their mask
        self.edge_enc = nn.Linear(n_edge_feat * 2, hidden)
        # one MLP per message-passing round: input = [h_src, h_dst, h_edge] -> hidden
        self.msg = nn.ModuleList([nn.Linear(hidden * 3, hidden) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(layers)])
        self.head = nn.Linear(hidden, 1)                               # per-node attack logit

    def forward(self, node_x, node_m, edge_x, edge_m, edge_index):
        B, N, _ = node_x.shape; E = edge_index.shape[1]
        src, dst = edge_index[0], edge_index[1]
        h = F.relu(self.enc(torch.cat([node_x, node_m], -1)))          # [B,N,H]
        he = F.relu(self.edge_enc(torch.cat([edge_x, edge_m], -1)))    # [B,E,H]
        # make edges bidirectional so information flows both ways along a line
        src2 = torch.cat([src, dst]); dst2 = torch.cat([dst, src]); he2 = torch.cat([he, he], 1)  # [B,2E,H]
        for msg, upd in zip(self.msg, self.upd):
            m = F.relu(msg(torch.cat([h[:, src2], h[:, dst2], he2], -1)))   # [B,2E,H] message per edge
            agg = torch.zeros_like(h)
            agg.index_add_(1, dst2, m)                                 # sum messages into destination nodes
            deg = torch.zeros(N, device=h.device).index_add_(0, dst2, torch.ones(2 * E, device=h.device))
            agg = agg / deg.clamp(min=1).view(1, N, 1)                 # mean aggregation
            h = F.relu(upd(torch.cat([h, agg], -1)))                   # node update with residual-style concat
        return self.head(h).squeeze(-1)                               # [B,N] logits


def node_f1(logits, y, mask):
    """Micro node-F1 over the buses of the selected records (mask), threshold at logit>0 (prob>0.5)."""
    pred = (logits[mask] > 0).float(); tgt = y[mask]
    tp = (pred * tgt).sum(); fp = (pred * (1 - tgt)).sum(); fn = ((1 - pred) * tgt).sum()
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    return (2 * p * r / (p + r + 1e-9)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--release", default=None)               # pin a dataset version if desired
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- the only data code a user writes: load train/test, get a ready DataLoader ---
    train = fg.load(args.system, split="train", release=args.release)
    test = fg.load(args.system, split="test", release=args.release)
    tl = train.loader(batch_size=args.batch, shuffle=True)
    ei = train.edge_index.to(dev)
    print(f"train {len(train):,} / test {len(test):,} records on {args.system} (N={train.N}, E={train.E}) [{dev}]")

    model = EdgeMPNN().to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for b in tl:
            logits = model(b["node_x"].to(dev), b["node_m"].to(dev), b["edge_x"].to(dev), b["edge_m"].to(dev), ei)
            loss = F.binary_cross_entropy_with_logits(logits, b["y"].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(b["y"])
        print(f"epoch {ep+1}/{args.epochs}  train loss {tot/len(train):.4f}")

    # --- evaluate: overall + PER-ATTACK-TYPE node-F1 (the metric that matters) ---
    model.eval(); allL, allY, allF = [], [], []
    with torch.no_grad():
        for b in test.loader(batch_size=args.batch, shuffle=False):
            allL.append(model(b["node_x"].to(dev), b["node_m"].to(dev), b["edge_x"].to(dev), b["edge_m"].to(dev), ei).cpu())
            allY.append(b["y"]); allF.append(b["family"])
    L = torch.cat(allL); Y = torch.cat(allY); Fam = torch.cat(allF)
    attacked = Fam > 0
    print(f"\nnode-F1 (attacked records): {node_f1(L, Y, attacked):.3f}")
    print("per-attack-type node-F1:")
    for k, name in FAMILIES.items():
        if k == 0:
            continue
        m = Fam == k
        if m.any():
            print(f"  {name:6s} (n={int(m.sum()):5d}): {node_f1(L, Y, m):.3f}")


if __name__ == "__main__":
    main()
