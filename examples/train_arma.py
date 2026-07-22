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
from fdia_graph.dataset import FAMILIES, FdiaGraph
from torch_geometric.nn import ARMAConv, GATv2Conv


def kcl_residual(node_x, edge_x, ei, N):
    """Per-bus real/reactive power-balance residual from the MEASUREMENTS (a cheap BDD-like inconsistency
    signal, no WLS): inflow - injection. ~0 for a consistent state (Ao/ramp/LRA pass BDD), but spikes at
    buses whose incident meters were corrupted (Ad/As/Ar break KCL) -> the model gets the physics filter as
    a feature and can flag the crude attacks it was weak on, instead of competing with BDD."""
    n = node_x.shape[0]; fr, to = ei[0], ei[1]
    inP = torch.zeros(n, N, device=node_x.device); inQ = torch.zeros(n, N, device=node_x.device)
    inP.index_add_(1, to, edge_x[:, :, 0]); inP.index_add_(1, fr, -edge_x[:, :, 0])   # + into to-bus, - out of from-bus
    inQ.index_add_(1, to, edge_x[:, :, 1]); inQ.index_add_(1, fr, -edge_x[:, :, 1])
    return torch.stack([inP - node_x[:, :, 1], inQ - node_x[:, :, 2]], -1)             # [n,N,2] residual (P,Q)


class ArmaLoc(nn.Module):
    def __init__(self, n_in=8, hidden=128, stacks=3, layers=2, num_blocks=2, dropout=0.1, attn=True):
        super().__init__()
        self.attn = attn
        self.nenc = nn.Linear(n_in, hidden)                      # node feats: measurements(4)+mask(4)+KCL(2)+temporal(2)+swing(2)
        self.eenc = nn.Linear(2 * 2, hidden)                     # edge feat (2) concat mask (2)
        # `num_blocks` stacked ARMA blocks (graph depth); each is `stacks` parallel ARMA_1 filters, each `layers`
        # recursive steps -> wide receptive field. Default num_blocks=2 reproduces the original arma1->arma2 stack.
        self.blocks = nn.ModuleList([
            ARMAConv(hidden, hidden, num_stacks=stacks, num_layers=layers, shared_weights=True, dropout=dropout, act=F.relu)
            for _ in range(num_blocks)])
        if attn:
            # PHYSICS-BIASED ATTENTION: the ARMA trunk is spectral (isotropic per hop); a parallel GATv2 head lets
            # the model attend ANISOTROPICALLY along the branches that matter. Its attention logits are a learned
            # function of both endpoints' node features (which carry the KCL residual + temporal delta) and the
            # edge P/Q flow, so it focuses on physically-suspicious edges. Fused with a learnable gate initialised
            # at 0 (tanh) -> the net starts identical to plain ARMA and only opens the branch if it helps.
            self.gat = GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=2 * 2, dropout=dropout, add_self_loops=False)
            self.gate = nn.Parameter(torch.zeros(1))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, e, ei):
        # x [n,8]  e [m,4]  ei [2,m] (bidirectional, batched). Returns per-node logit [n].
        h = F.relu(self.nenc(x))
        he = F.relu(self.eenc(e))
        h = h + torch.zeros_like(h).index_add_(0, ei[1], he)     # fuse edge-flow signal into destination nodes
        for blk in self.blocks: h = blk(h, ei)                   # stacked ARMA graph-conv blocks
        if self.attn:
            h = h + torch.tanh(self.gate) * self.gat(h, ei, edge_attr=e)
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
    ap.add_argument("--shard", default=None, help="path to a local .h5 shard to train on (overrides fg.load/release)")
    ap.add_argument("--no_attn", action="store_true", help="disable the physics-biased attention branch (plain ARMA)")
    # Architecture-depth knobs (defaults reproduce the original ARMA+attn recipe exactly).
    ap.add_argument("--hidden", type=int, default=128, help="hidden width (default 128; keep divisible by 4 when attention is on)")
    ap.add_argument("--arma-stacks", type=int, default=3, dest="arma_stacks", help="parallel ARMA_1 filter stacks per block (default 3)")
    ap.add_argument("--arma-layers", type=int, default=2, dest="arma_layers", help="recursive ARMA steps per block (default 2)")
    ap.add_argument("--num-blocks", type=int, default=2, dest="num_blocks", help="number of stacked ARMA graph-conv blocks / graph depth (default 2)")
    # Feature-ablation toggles for the feats() channels (default = all on, i.e. current behavior).
    ap.add_argument("--no-kcl", action="store_true", dest="no_kcl", help="ablate the KCL power-balance residual feature")
    ap.add_argument("--no-temporal", action="store_true", dest="no_temporal", help="ablate the temporal-delta (change-since-last-scan) feature")
    ap.add_argument("--no-swing", action="store_true", dest="no_swing", help="ablate the windowed relative-swing feature")
    # Train in per-unit by default: node_x becomes [V p.u., P p.u., Q p.u., theta rad] so all channels are O(0.1-30)
    # instead of V~1 next to P/Q/KCL~1000. Without input normalization the physical MW channels otherwise dominate
    # the linear encoder. "physical" reproduces the older mixed-scale behaviour.
    ap.add_argument("--units", default="pu", choices=["pu", "physical"], help="unit system for measurements (default pu)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev == "cuda" else nullcontext

    def gpu(split):
        # local shard (v0.4.0 regen) takes precedence over the downloadable release when --shard is given
        ds = FdiaGraph(args.shard, split=split, units=args.units) if args.shard else fg.load(args.system, split=split, release=args.release, units=args.units)
        a = ds.to_numpy()
        keys = ["node_x", "node_m", "edge_x", "edge_m", "y"] + (["temporal_delta"] if ds.has_temporal else []) \
               + (["swing"] if getattr(ds, "has_swing", False) else [])
        g = {k: torch.as_tensor(a[k], device=dev, dtype=torch.float32) for k in keys}
        return g, torch.as_tensor(a["family"], device=dev), ds
    trG, trFam, ds = gpu("train"); vaG, vaFam, _ = gpu("val"); teG, teFam, _ = gpu("test")
    N, E = ds.N, ds.E
    ei0 = ds.edge_index.to(dev)                                   # [2,E] per-graph
    # PHYSICS-INFORMED FEATURES: append the per-bus KCL power-balance residual (flags crude measurement attacks)
    # and, when present (v0.3+), the stored temporal-delta (current vs previous scan — catches replay Ar & drift).
    # ablation flags (default all-on reproduces current behavior); gate BOTH the baked-in node_x augmentation here
    # and the fresh recompute in feats() so a feature is fully removed when toggled off.
    use_kcl = not args.no_kcl; use_temporal = not args.no_temporal; use_swing = not args.no_swing
    for g in (trG, vaG, teG):
        extra = ([kcl_residual(g["node_x"], g["edge_x"], ei0, N)] if use_kcl else []) \
                + ([g["temporal_delta"]] if (use_temporal and "temporal_delta" in g) else [])
        g["node_x"] = torch.cat([g["node_x"]] + extra, -1)
        g["node_m"] = torch.cat([g["node_m"]] + [torch.ones_like(e) for e in extra], -1)
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
        nxb = g["node_x"][idx]; exb = g["edge_x"][idx]                                    # [b,N,4], [b,E,2]
        parts = [nxb, g["node_m"][idx]]                                                   # measurements + mask
        if use_kcl: parts.append(kcl_residual(nxb, exb, ei0, N))                          # [b,N,2] measurement KCL residual
        if use_temporal and "temporal_delta" in g: parts.append(g["temporal_delta"][idx]) # [b,N,2] change-since-last-scan
        if use_swing and "swing" in g: parts.append(g["swing"][idx])                     # [b,N,2] windowed relative swing
        x = torch.cat(parts, -1).reshape(b * N, -1)                                       # [b*N, F]
        e2 = torch.cat([exb, g["edge_m"][idx]], -1)                                       # [b,E,4]
        e = torch.cat([e2, e2], 1).reshape(b * 2 * E, -1)                                 # bidirectional [b*2E,4]
        return x, e, batched(b), g["y"][idx].reshape(b * N)

    xdim = feats(trG, torch.arange(2, device=dev))[0].shape[-1]                           # actual node-feature width
    model = ArmaLoc(xdim, hidden=args.hidden, stacks=args.arma_stacks, layers=args.arma_layers,
                    num_blocks=args.num_blocks, attn=not args.no_attn).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    pos = float(trG["y"].sum()); pw = torch.tensor(min(max((trG["y"].numel() - pos) / max(pos, 1), 1.0), 30.0), device=dev)
    n = trG["y"].shape[0]; arch = "ARMA+attn" if not args.no_attn else "ARMA"
    abl = "".join(t for t, on in [("-kcl", not use_kcl), ("-temporal", not use_temporal), ("-swing", not use_swing)] if on)
    print(f"{args.system}: N={N} E={E}  train {n:,}  {arch}(hidden={args.hidden},blocks={args.num_blocks},stacks={args.arma_stacks},layers={args.arma_layers}{' ablate'+abl if abl else ''})  pos_weight={pw.item():.1f}  batch={args.batch} [{dev}]")
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
    res = {"system": args.system, "N": int(N), "E": int(E), "model": arch, "threshold": round(thr, 2),
           "config": {"seed": args.seed, "hidden": args.hidden, "arma_stacks": args.arma_stacks,
                      "arma_layers": args.arma_layers, "num_blocks": args.num_blocks, "attn": not args.no_attn,
                      "feats": {"kcl": use_kcl, "temporal": use_temporal, "swing": use_swing}},
           "overall": {"n": int(atk.sum()), "node_f1": f1(P[atk], Y[atk]), "swf1": f1(P[atk], Y[atk], sample=True)}, "per_family": {}}
    print(f"tuned threshold {thr:.2f}\noverall (attacked): node-F1 {res['overall']['node_f1']:.3f}  swF1 {res['overall']['swf1']:.3f}")
    for k, name in FAMILIES.items():
        m = Fm == k
        if m.any():
            r = {"n": int(m.sum()), "node_f1": f1(P[m], Y[m]), "swf1": f1(P[m], Y[m], sample=True)}
            res["per_family"][name] = r; print(f"  {name:6s} n={r['n']:5d}  node-F1 {r['node_f1']:.3f}  swF1 {r['swf1']:.3f}")
    # ---- DETECTION (grid-level, Boyaci-style S = max bus prob); this is what the "ML-only" claim rests on ----
    def dscore(logits): return torch.sigmoid(logits).max(1).values           # per-record grid attack score
    def detf1(S, lab, t):
        pr = (S > t).float(); tp = (pr * lab).sum(); fp = (pr * (1 - lab)).sum(); fn = ((1 - pr) * lab).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); return (2 * p * r / (p + r + 1e-9)).item()
    vS = dscore(vL); vlab = (vFm > 0).float()                                 # tune detection threshold on VAL
    dthr = float(max(torch.linspace(0.05, 0.95, 19), key=lambda t: detf1(vS, vlab, float(t))))
    tS = dscore(L); tlab = (Fm > 0).float(); pr = (tS > dthr).float()
    DR = (pr * tlab).sum().item() / tlab.sum().item()                         # detection rate  (recall on attacked)
    FA = (pr * (1 - tlab)).sum().item() / (1 - tlab).sum().item()            # false-alarm rate (on benign)
    res["detection"] = {"DR": round(DR, 4), "FA": round(FA, 4), "det_f1": round(detf1(tS, tlab, dthr), 4), "threshold": round(dthr, 3), "per_family_DR": {}}
    print(f"\nDETECTION (grid-level): DR {DR:.3f}  FA {FA:.3f}  det-F1 {res['detection']['det_f1']:.3f}  (threshold {dthr:.2f})")
    print("per-family detection rate (DR):")
    for k, name in FAMILIES.items():
        m = Fm == k
        if k and m.any():
            res["detection"]["per_family_DR"][name] = round(pr[m].mean().item(), 3)
            print(f"  {name:6s} DR {pr[m].mean().item():.3f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2); print("wrote", args.out)


if __name__ == "__main__":
    main()
