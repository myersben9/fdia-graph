#!/usr/bin/env python
"""Per-epoch train/val history logging + t-SNE embedding dumps for the localizer complexity ladder.
Writes results/history_{model}_{C}.json and results/embeds_{model}_{C}.npz.
"""
import os, sys, json, random, argparse
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from contextlib import nullcontext
from sklearn.metrics import average_precision_score
from torch_geometric.nn import GCNConv

from fdia_graph.dataset import FdiaGraph, FAMILIES
from train_arma import ArmaLoc, kcl_residual

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SHARDS = os.path.join(HERE, "release_v0.4.1")
EPOCHS_DEFAULT = int(os.environ.get("EPOCHS", "40"))
BATCH = 256
SEED = 123
THR_LOGIT = 0.0
FAM_KEEP_EMBED = [0, 1, 2, 3, 4, 6]
N_PER_FAMILY = 400


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def f1(pred, tgt, sample=False):
    tp = (pred * tgt).sum(-1); fp = (pred * (1 - tgt)).sum(-1); fn = ((1 - pred) * tgt).sum(-1)
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); fv = 2 * p * r / (p + r + 1e-9)
    return fv.mean().item() if sample else (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-9)).item()


class MLP(nn.Module):
    def __init__(self, fin, h=128):
        super().__init__(); self.net = nn.Sequential(nn.Linear(fin, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(self, x, ei): return self.net(x).squeeze(-1)


class GCN(nn.Module):
    def __init__(self, fin, h=128):
        super().__init__(); self.c1 = GCNConv(fin, h); self.c2 = GCNConv(h, h); self.head = nn.Linear(h, 1)
    def forward(self, x, ei):
        x = F.relu(self.c1(x, ei)); x = F.relu(self.c2(x, ei)); return self.head(x).squeeze(-1)


def mlp_embed(model, x):
    h = model.net[0](x); h = model.net[1](h); h = model.net[2](h); h = model.net[3](h)
    return h


def gcn_embed(model, x, ei):
    x = F.relu(model.c1(x, ei)); x = F.relu(model.c2(x, ei))
    return x


def arma_embed(model, x, e, ei):
    h = F.relu(model.nenc(x))
    he = F.relu(model.eenc(e))
    h = h + torch.zeros_like(h).index_add_(0, ei[1], he)
    h = model.arma1(h, ei); h = model.arma2(h, ei)
    if model.attn:
        h = h + torch.tanh(model.gate) * model.gat(h, ei, edge_attr=e)
    return h


def _detf1(gmax, lab, thr):
    pr = (gmax >= thr).float(); tp = (pr * lab).sum(); fp = (pr * (1 - lab)).sum(); fn = ((1 - pr) * lab).sum()
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); return float((2 * p * r / (p + r + 1e-9)).item())


def tune_thr(vaL, vaF):
    # Pick the grid-level probability threshold that MAXIMIZES val detection-F1 — the real operating point the
    # benchmark uses. A fixed 0.5 makes a weak model look like it flags everything (FA=1); the tuned threshold gives
    # the honest FA. (This is the fix for the misleading DR=FA=1 curves.)
    gmax = torch.sigmoid(vaL).max(1).values; lab = (vaF > 0).float()
    best_t, best_d = 0.5, -1.0
    for t in np.linspace(0.5, 0.99, 26):
        d = _detf1(gmax, lab, float(t))
        if d > best_d: best_t, best_d = float(t), d
    return best_t


def snapshot(logits, y, fam, thr=0.5, loss=0.0):
    prob = torch.sigmoid(logits)
    pred = (logits > THR_LOGIT).float()
    atk = fam > 0; ben = ~atk
    out = {}
    out["f1"] = f1(pred[atk], y[atk], sample=True) if atk.any() else 0.0
    out["auprc"] = float(average_precision_score(y.reshape(-1).cpu().numpy(), prob.reshape(-1).cpu().numpy()))
    gmax = prob.max(1).values
    # DR/FA at the TUNED operating threshold (not a fixed 0.5) so FA reflects the real detector, not a flag-all point.
    out["dr"] = float((gmax[atk] >= thr).float().mean().item()) if atk.any() else 0.0
    out["fa"] = float((gmax[ben] >= thr).float().mean().item()) if ben.any() else 0.0
    out["loss"] = float(loss)                                # BCE loss for the convergence curve
    return out


def append_hist(hist, split, m):
    for k in ("f1", "auprc", "dr", "fa", "loss"):
        hist[split][k].append(m[k])


def new_hist():
    return {"epochs": [], "train": {"f1": [], "auprc": [], "dr": [], "fa": [], "loss": []},
            "val": {"f1": [], "auprc": [], "dr": [], "fa": [], "loss": []}}


def balanced_embed_idx(fam_np, rng):
    idx = []
    for k in FAM_KEEP_EMBED:
        pool = np.nonzero(fam_np == k)[0]
        if len(pool) == 0:
            print(f"  WARNING: family {FAMILIES[k]} ({k}) has ZERO test records -- embed dump will be missing this family")
            continue
        take = min(N_PER_FAMILY, len(pool))
        idx.append(rng.choice(pool, size=take, replace=False))
    return np.sort(np.concatenate(idx)) if idx else np.array([], dtype=np.int64)


def run_arma(C, epochs, out_hist, out_embed):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev == "cuda" else nullcontext
    shard = os.path.join(SHARDS, f"ml_only_ieee{C}.h5")
    print(f"[data] arma/ieee{C} reading {shard}  units=pu", flush=True)

    def gpu(split):
        ds = FdiaGraph(shard, split=split, units="pu")
        a = ds.to_numpy()
        keys = ["node_x", "node_m", "edge_x", "edge_m", "y"] + (["temporal_delta"] if ds.has_temporal else []) \
               + (["swing"] if getattr(ds, "has_swing", False) else [])
        g = {k: torch.as_tensor(a[k], device=dev, dtype=torch.float32) for k in keys}
        return g, torch.as_tensor(a["family"], device=dev), ds

    trG, trFam, ds = gpu("train"); vaG, vaFam, _ = gpu("val"); teG, teFam, _ = gpu("test")
    N, E = ds.N, ds.E
    ei0 = ds.edge_index.to(dev)
    for g in (trG, vaG, teG):
        extra = [kcl_residual(g["node_x"], g["edge_x"], ei0, N)] + ([g["temporal_delta"]] if "temporal_delta" in g else [])
        g["node_x"] = torch.cat([g["node_x"]] + extra, -1)
        g["node_m"] = torch.cat([g["node_m"]] + [torch.ones_like(e) for e in extra], -1)
    for xk, mk in (("node_x", "node_m"), ("edge_x", "edge_m")):
        w = trG[mk].sum((0, 1)).clamp(min=1.0); mu = (trG[xk] * trG[mk]).sum((0, 1)) / w
        sd = (((trG[xk] - mu) ** 2 * trG[mk]).sum((0, 1)) / w).sqrt().clamp(min=1e-3)
        for g in (trG, vaG, teG): g[xk] = (g[xk] - mu) / sd * g[mk]

    ei_bi = torch.cat([ei0, ei0.flip(0)], 1)

    def batched(B):
        off = (torch.arange(B, device=dev) * N).repeat_interleave(ei_bi.shape[1])
        return ei_bi.repeat(1, B) + off.unsqueeze(0)

    def feats(g, idx):
        b = len(idx)
        nxb = g["node_x"][idx]; exb = g["edge_x"][idx]
        kcl = kcl_residual(nxb, exb, ei0, N)
        parts = [nxb, g["node_m"][idx], kcl]
        if "temporal_delta" in g: parts.append(g["temporal_delta"][idx])
        if "swing" in g: parts.append(g["swing"][idx])
        x = torch.cat(parts, -1).reshape(b * N, -1)
        e2 = torch.cat([exb, g["edge_m"][idx]], -1)
        e = torch.cat([e2, e2], 1).reshape(b * 2 * E, -1)
        return x, e, batched(b), g["y"][idx].reshape(b * N)

    xdim = feats(trG, torch.arange(2, device=dev))[0].shape[-1]
    model = ArmaLoc(xdim, attn=True).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    pos = float(trG["y"].sum()); pw = torch.tensor(min(max((trG["y"].numel() - pos) / max(pos, 1), 1.0), 30.0), device=dev)
    n = trG["y"].shape[0]
    print(f"[arma/ieee{C}] N={N} E={E} train={n} val={vaG['y'].shape[0]} test={teG['y'].shape[0]} xdim={xdim} pw={pw.item():.1f} dev={dev}", flush=True)

    rng = np.random.RandomState(SEED)
    sub_size = min(2000, n)
    sub_idx = torch.as_tensor(np.sort(rng.choice(n, size=sub_size, replace=False)), device=dev)
    val_idx = torch.arange(vaG["y"].shape[0], device=dev)

    def collect(g, idx):
        model.eval(); LG = []
        with torch.no_grad(), autocast():
            for i in range(0, len(idx), BATCH):
                bidx = idx[i:i + BATCH]; x, e, ei, _ = feats(g, bidx)
                LG.append(model(x, e, ei).reshape(len(bidx), N).float())
        return torch.cat(LG)

    hist = new_hist(); best_val_f1 = -1.0; best_state = None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev); ep_loss = 0.0; nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]; x, e, ei, y = feats(trG, idx)
            with autocast():
                loss = F.binary_cross_entropy_with_logits(model(x, e, ei), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); ep_loss += float(loss.item()); nb += 1

        trL = collect(trG, sub_idx); trY = trG["y"][sub_idx]; trF = trFam[sub_idx]
        vaL = collect(vaG, val_idx); vaY = vaG["y"][val_idx]; vaF = vaFam[val_idx]
        thr = tune_thr(vaL, vaF)                              # honest operating point (not fixed 0.5)
        valoss = float(F.binary_cross_entropy_with_logits(vaL, vaY, pos_weight=pw).item())
        m_tr = snapshot(trL, trY, trF, thr, ep_loss / max(nb, 1)); m_va = snapshot(vaL, vaY, vaF, thr, valoss)
        hist["epochs"].append(ep + 1); append_hist(hist, "train", m_tr); append_hist(hist, "val", m_va)
        if m_va["f1"] > best_val_f1:
            best_val_f1 = m_va["f1"]; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            tf1 = m_tr["f1"]; taup = m_tr["auprc"]; vf1 = m_va["f1"]; vaup = m_va["auprc"]; vdr = m_va["dr"]; vfa = m_va["fa"]
            print(f"  ep {ep+1}/{epochs}  train f1={tf1:.3f} auprc={taup:.3f}  val f1={vf1:.3f} auprc={vaup:.3f} dr={vdr:.3f} fa={vfa:.3f}", flush=True)

    hist_out = {"model": "arma", "system": f"ieee{C}", **hist}
    json.dump(hist_out, open(out_hist, "w"), indent=2)
    print(f"[arma/ieee{C}] wrote {out_hist}  best val f1={best_val_f1:.3f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    fam_np = teFam.cpu().numpy()
    eidx_np = balanced_embed_idx(fam_np, np.random.RandomState(SEED))
    eidx = torch.as_tensor(eidx_np, device=dev)
    Zs, Xs = [], []
    with torch.no_grad(), autocast():
        for i in range(0, len(eidx), BATCH):
            bidx = eidx[i:i + BATCH]; b = len(bidx)
            x, e, ei, _ = feats(teG, bidx)
            h = arma_embed(model, x, e, ei).float().reshape(b, N, -1)
            Zs.append(h.mean(1).cpu())
            Xs.append(x.float().reshape(b, N, -1).mean(1).cpu())
    Z = torch.cat(Zs).numpy().astype(np.float32) if Zs else np.zeros((0, 0), np.float32)
    Xin = torch.cat(Xs).numpy().astype(np.float32) if Xs else np.zeros((0, 0), np.float32)
    fam_sel = fam_np[eidx_np]
    np.savez(out_embed, Z=Z, Xin=Xin, family=fam_sel.astype(np.int64))
    fams_present = sorted(set(fam_sel.tolist()))
    print(f"[arma/ieee{C}] wrote {out_embed}  Z={Z.shape} Xin={Xin.shape} n={len(fam_sel)} families={fams_present}", flush=True)


def run_baseline(tag, C, epochs, out_hist, out_embed):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    shard = os.path.join(SHARDS, f"ml_only_ieee{C}.h5")
    print(f"[data] {tag}/ieee{C} reading {shard}  units=pu", flush=True)

    def gpu(split):
        a = FdiaGraph(shard, split=split, units="pu").to_numpy()
        g = {k: torch.as_tensor(a[k], device=dev, dtype=torch.float32) for k in ("node_x", "node_m", "y")}
        return g, torch.as_tensor(a["family"], device=dev)

    trG, trFam = gpu("train"); vaG, vaFam = gpu("val"); teG, teFam = gpu("test")
    N = trG["y"].shape[1]; ds0 = FdiaGraph(shard, units="pu"); ei0 = ds0.edge_index.to(dev)
    ei_bi = torch.cat([ei0, ei0.flip(0)], 1)

    def batched(B):
        off = (torch.arange(B, device=dev) * N).repeat_interleave(ei_bi.shape[1])
        return ei_bi.repeat(1, B) + off.unsqueeze(0)

    def feats(g, idx):
        b = len(idx); x = torch.cat([g["node_x"][idx], g["node_m"][idx]], -1).reshape(b * N, -1)
        return x, batched(b), g["y"][idx].reshape(b * N)

    torch.manual_seed(SEED)
    ModelCls = MLP if tag == "mlp" else GCN
    embed_fn = mlp_embed if tag == "mlp" else gcn_embed
    model = ModelCls(8).to(dev); opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    pos = float(trG["y"].sum()); pw = torch.tensor(min(max((trG["y"].numel() - pos) / max(pos, 1), 1.0), 30.0), device=dev)
    n = trG["y"].shape[0]
    ntr_val = vaG["y"].shape[0]; ntr_te = teG["y"].shape[0]
    print(f"[{tag}/ieee{C}] N={N} train={n} val={ntr_val} test={ntr_te} pw={pw.item():.1f} dev={dev}", flush=True)

    rng = np.random.RandomState(SEED)
    sub_size = min(2000, n)
    sub_idx = torch.as_tensor(np.sort(rng.choice(n, size=sub_size, replace=False)), device=dev)
    val_idx = torch.arange(vaG["y"].shape[0], device=dev)

    def collect(g, idx):
        model.eval(); LG = []
        with torch.no_grad():
            for i in range(0, len(idx), BATCH):
                bidx = idx[i:i + BATCH]; x, ei, _ = feats(g, bidx)
                LG.append(model(x, ei).reshape(len(bidx), N))
        return torch.cat(LG)

    hist = new_hist(); best_val_f1 = -1.0; best_state = None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev); ep_loss = 0.0; nb = 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]; x, ei, y = feats(trG, idx)
            loss = F.binary_cross_entropy_with_logits(model(x, ei), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); ep_loss += float(loss.item()); nb += 1

        trL = collect(trG, sub_idx); trY = trG["y"][sub_idx]; trF = trFam[sub_idx]
        vaL = collect(vaG, val_idx); vaY = vaG["y"][val_idx]; vaF = vaFam[val_idx]
        thr = tune_thr(vaL, vaF)                              # honest operating point (not fixed 0.5)
        valoss = float(F.binary_cross_entropy_with_logits(vaL, vaY, pos_weight=pw).item())
        m_tr = snapshot(trL, trY, trF, thr, ep_loss / max(nb, 1)); m_va = snapshot(vaL, vaY, vaF, thr, valoss)
        hist["epochs"].append(ep + 1); append_hist(hist, "train", m_tr); append_hist(hist, "val", m_va)
        if m_va["f1"] > best_val_f1:
            best_val_f1 = m_va["f1"]; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            tf1 = m_tr["f1"]; taup = m_tr["auprc"]; vf1 = m_va["f1"]; vaup = m_va["auprc"]; vdr = m_va["dr"]; vfa = m_va["fa"]
            print(f"  ep {ep+1}/{epochs}  train f1={tf1:.3f} auprc={taup:.3f}  val f1={vf1:.3f} auprc={vaup:.3f} dr={vdr:.3f} fa={vfa:.3f}", flush=True)

    hist_out = {"model": tag, "system": f"ieee{C}", **hist}
    json.dump(hist_out, open(out_hist, "w"), indent=2)
    print(f"[{tag}/ieee{C}] wrote {out_hist}  best val f1={best_val_f1:.3f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    fam_np = teFam.cpu().numpy()
    eidx_np = balanced_embed_idx(fam_np, np.random.RandomState(SEED))
    eidx = torch.as_tensor(eidx_np, device=dev)
    Zs, Xs = [], []
    with torch.no_grad():
        for i in range(0, len(eidx), BATCH):
            bidx = eidx[i:i + BATCH]; b = len(bidx)
            x, ei, _ = feats(teG, bidx)
            if tag == "gcn":
                h = embed_fn(model, x, ei).reshape(b, N, -1)
            else:
                h = embed_fn(model, x).reshape(b, N, -1)
            Zs.append(h.mean(1).cpu())
            Xs.append(x.reshape(b, N, -1).mean(1).cpu())
    Z = torch.cat(Zs).numpy().astype(np.float32) if Zs else np.zeros((0, 0), np.float32)
    Xin = torch.cat(Xs).numpy().astype(np.float32) if Xs else np.zeros((0, 0), np.float32)
    fam_sel = fam_np[eidx_np]
    np.savez(out_embed, Z=Z, Xin=Xin, family=fam_sel.astype(np.int64))
    fams_present = sorted(set(fam_sel.tolist()))
    print(f"[{tag}/ieee{C}] wrote {out_embed}  Z={Z.shape} Xin={Xin.shape} n={len(fam_sel)} families={fams_present}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["arma", "gcn", "mlp"])
    ap.add_argument("--system", required=True, type=int, choices=[14, 118, 300])
    ap.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    args = ap.parse_args()
    seed_everything(SEED)
    C = args.system
    out_hist = os.path.join(RES, f"history_{args.model}_{C}.json")
    out_embed = os.path.join(RES, f"embeds_{args.model}_{C}.npz")
    if args.model == "arma":
        run_arma(C, args.epochs, out_hist, out_embed)
    else:
        run_baseline(args.model, C, args.epochs, out_hist, out_embed)
    print("[done]", out_hist, out_embed, flush=True)


if __name__ == "__main__":
    main()
