#!/usr/bin/env python
"""Temporal Graph NN localizer benchmark on an fdia-graph dataset.

Architecture: a spatial edge-conditioned message-passing GNN encodes each timestep's measurement graph into
per-bus embeddings, then a GRU runs over a sliding window of W consecutive timesteps (records are ordered by
`timestep`, so a window is a short slice of the operating timeline) and the final hidden state is decoded to
per-bus attack logits for the LAST timestep in the window. This gives the model temporal context — crucial
for the ramp family, whose per-snapshot perturbation is tiny but whose trend over time is the tell.

Reports pooled node-F1 and Boyaci sample-wise F1 (swF1) per attack family. Emits a JSON result the report
reads. Runs on GPU if available.

    python train_tgnn.py --system ieee118 --epochs 4 --window 8
"""
from __future__ import annotations

import argparse, json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict, Optional, Tuple
import fdia_graph as fg
from fdia_graph.dataset import FAMILIES


class SpatialEncoder(nn.Module):
    """Edge-conditioned message passing -> per-bus embedding (shared across timesteps)."""
    def __init__(self, hidden: int = 64, layers: int = 2, n_node: int = 4, n_edge: int = 2) -> None:
        super().__init__()
        self.enc = nn.Linear(n_node * 2, hidden)                     # *2: feature concatenated with its mask
        self.edge_enc = nn.Linear(n_edge * 2, hidden)
        self.msg = nn.ModuleList([nn.Linear(hidden * 3, hidden) for _ in range(layers)])
        self.upd = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(layers)])

    def forward(self, node_x: torch.Tensor, node_m: torch.Tensor, edge_x: torch.Tensor,
                edge_m: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # node_x [B,N,4] -> [B,N,H]; B is (batch * window) flattened.
        B, N, _ = node_x.shape; E = edge_index.shape[1]
        src, dst = edge_index[0], edge_index[1]
        h = F.relu(self.enc(torch.cat([node_x, node_m], -1)))
        he = F.relu(self.edge_enc(torch.cat([edge_x, edge_m], -1)))
        s2 = torch.cat([src, dst]); d2 = torch.cat([dst, src]); he2 = torch.cat([he, he], 1)  # bidirectional
        for msg, upd in zip(self.msg, self.upd):
            m = F.relu(msg(torch.cat([h[:, s2], h[:, d2], he2], -1)))
            agg = torch.zeros_like(h).index_add_(1, d2, m)
            deg = torch.zeros(N, device=h.device).index_add_(0, d2, torch.ones(2 * E, device=h.device))
            h = F.relu(upd(torch.cat([h, agg / deg.clamp(min=1).view(1, N, 1)], -1)))
        return h


class TemporalGNN(nn.Module):
    """Spatial GNN per timestep + GRU over the window -> per-bus logits at the last step."""
    def __init__(self, hidden: int = 128) -> None:              # wider+deeper: localizing ~4 of 300 buses is hard
        super().__init__()
        self.enc = SpatialEncoder(hidden, layers=3)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, nx: torch.Tensor, nm: torch.Tensor, ex: torch.Tensor,
                em: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        # nx [B,W,N,4] -> encode every (B*W) graph, GRU per node over W, decode last step
        B, W, N, _ = nx.shape
        flat = lambda t: t.reshape(B * W, *t.shape[2:])
        h = self.enc(flat(nx), flat(nm), flat(ex), flat(em), ei)     # [B*W, N, H]
        h = h.reshape(B, W, N, -1).permute(0, 2, 1, 3).reshape(B * N, W, -1)  # [B*N, W, H]
        out, _ = self.gru(h)                                         # temporal aggregation per bus
        return self.head(out[:, -1]).reshape(B, N)                   # logits for the window's last timestep


def windows(arr_idx: np.ndarray, W: int) -> np.ndarray:
    """Contiguous sliding windows (stride 1) of length W over timestep-ordered record indices."""
    return np.stack([arr_idx[i:i + W] for i in range(len(arr_idx) - W + 1)]) if len(arr_idx) >= W else np.empty((0, W), int)


def f1(pred: torch.Tensor, tgt: torch.Tensor, sample: bool = False) -> float:
    tp = (pred * tgt).sum(-1); fp = (pred * (1 - tgt)).sum(-1); fn = ((1 - pred) * tgt).sum(-1)
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); f = 2 * p * r / (p + r + 1e-9)
    return f.mean().item() if sample else (2 * (tp.sum() * 1.0) / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-9)).item()


def load_split(system: str, split: str, W: int, release: Optional[str]
               ) -> Tuple[Dict[str, np.ndarray], np.ndarray, int, int, np.ndarray]:
    """Return timestep-ordered windows (X tensors) + the LAST-step label/family for a split."""
    ds = fg.load(system, split=split, release=release)
    a = ds.to_numpy()
    order = np.argsort(a["timestep"])                                # chronological
    for k in ("node_x", "node_m", "edge_x", "edge_m", "y", "family"):
        a[k] = a[k][order]
    win = windows(np.arange(len(order)), W)
    return a, win, ds.N, ds.E, ds.edge_index_np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee14")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)           # large batch — the GPU is the target, not memory-bound
    ap.add_argument("--pos_weight", type=float, default=-1)      # -1 = auto (neg/pos from labels; scales with grid size)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--release", default=None)
    ap.add_argument("--out", default=None)                           # JSON results path (for the report)
    args = ap.parse_args()
    from contextlib import nullcontext
    torch.manual_seed(args.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":                                            # GPU throughput knobs
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    amp = dev == "cuda"                                          # bf16 autocast (Blackwell); bf16 needs no GradScaler
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else nullcontext
    W = args.window

    tr, trw, N, E, ei_np = load_split(args.system, "train", W, args.release)
    va, vaw, _, _, _ = load_split(args.system, "val", W, args.release)     # for threshold calibration
    te, tew, _, _, _ = load_split(args.system, "test", W, args.release)
    ei = torch.as_tensor(ei_np, dtype=torch.long, device=dev)
    print(f"{args.system}: N={N} E={E}  train windows {len(trw):,} / test {len(tew):,}  W={W} [{dev}]")

    # Pre-load each split ONCE onto the GPU so a batch is a pure GPU gather (no per-step CPU->GPU transfer).
    def to_gpu(a: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        d = {k: torch.as_tensor(a[k], device=dev, dtype=torch.float32) for k in ("node_x", "node_m", "edge_x", "edge_m")}
        d["y"] = torch.as_tensor(a["y"], device=dev, dtype=torch.float32); return d
    trG, vaG, teG = to_gpu(tr), to_gpu(va), to_gpu(te)
    trwG = torch.as_tensor(trw, device=dev); vawG = torch.as_tensor(vaw, device=dev); tewG = torch.as_tensor(tew, device=dev)

    # Standardize per channel using TRAIN stats (metered entries only); essential as raw MW is large on IEEE-300 (else bf16 NaN).
    def _stats(g: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        s: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for xk, mk in (("node_x", "node_m"), ("edge_x", "edge_m")):
            w = g[mk].sum((0, 1)).clamp(min=1.0)
            mu = (g[xk] * g[mk]).sum((0, 1)) / w
            sd = (((g[xk] - mu) ** 2 * g[mk]).sum((0, 1)) / w).sqrt().clamp(min=1e-3)
            s[xk] = (mu, sd)
        return s
    NST = _stats(trG)
    for g in (trG, vaG, teG):
        for xk, mk in (("node_x", "node_m"), ("edge_x", "edge_m")):
            mu, sd = NST[xk]; g[xk] = (g[xk] - mu) / sd * g[mk]      # keep masked (unmetered) entries at 0

    def batch_tensors(g: Dict[str, torch.Tensor], widx: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # widx [B,W] GPU long -> [B,W,N,·] GPU gathers
        return g["node_x"][widx], g["node_m"][widx], g["edge_x"][widx], g["edge_m"][widx], g["y"][widx[:, -1]]

    model = TemporalGNN().to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3)
    yt = tr["y"][trw[:, -1]]; pos = float(yt.sum()); pw_val = args.pos_weight if args.pos_weight > 0 else float(np.clip((yt.size - pos) / max(pos, 1), 1.0, 100.0))
    pw = torch.tensor(pw_val, device=dev)
    print(f"pos_weight = {pw_val:.1f}  (positive-label rate {pos/yt.size:.3%})  batch={args.batch}  bf16={amp}")
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(trwG), device=dev); tot = 0.0
        for i in range(0, len(perm), args.batch):
            wb = trwG[perm[i:i + args.batch]]
            nx, nm, ex, em, y = batch_tensors(trG, wb)
            with autocast():
                logit = model(nx, nm, ex, em, ei)
                loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(wb)
        print(f"epoch {ep+1}/{args.epochs}  loss {tot/len(trwG):.4f}")

    # collect raw per-bus logits for a set of windows (family = last record's family). GPU-resident + bf16.
    def collect(gG: Dict[str, torch.Tensor], gWins: torch.Tensor, npA: Dict[str, np.ndarray]
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model.eval(); LG, Y, Fm = [], [], []
        with torch.no_grad(), autocast():
            for i in range(0, len(gWins), args.batch):
                wb = gWins[i:i + args.batch]
                nx, nm, ex, em, y = batch_tensors(gG, wb)
                LG.append(model(nx, nm, ex, em, ei).float().cpu()); Y.append(y.cpu())
                Fm.append(torch.as_tensor(npA["family"][wb[:, -1].cpu().numpy()]))
        return torch.cat(LG), torch.cat(Y), torch.cat(Fm)

    # tune the decision threshold on VALIDATION (the high pos_weight shifts logits, so a fixed 0 is miscalibrated)
    vL, vY, vFm = collect(vaG, vawG, va); vatk = vFm > 0
    ths = torch.linspace(-2.0, 3.0, 26)
    thr = float(max(ths, key=lambda t: f1((vL[vatk] > t).float(), vY[vatk])))
    tL, Y, Fm = collect(teG, tewG, te); P = (tL > thr).float()
    atk = Fm > 0                                                  # overall is over ATTACKED records (benign has no positives)
    print(f"tuned threshold (val) = {thr:.2f}")
    res = {"system": args.system, "N": int(N), "E": int(E), "window": W, "threshold": round(thr, 2),
           "overall": {"n": int(atk.sum()), "node_f1": f1(P[atk], Y[atk]), "swf1": f1(P[atk], Y[atk], sample=True)},
           "per_family": {}}
    print(f"\noverall (attacked): node-F1 {res['overall']['node_f1']:.3f}  swF1 {res['overall']['swf1']:.3f}")
    for k, name in FAMILIES.items():
        m = Fm == k
        if m.any():
            r = {"n": int(m.sum()), "node_f1": f1(P[m], Y[m]), "swf1": f1(P[m], Y[m], sample=True)}
            res["per_family"][name] = r
            print(f"  {name:6s} n={r['n']:5d}  node-F1 {r['node_f1']:.3f}  swF1 {r['swf1']:.3f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
