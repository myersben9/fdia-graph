#!/usr/bin/env python
"""Hyperparameter optimization for the ARMA FDIA-localization model, in a Boyaci-style search space.

Optuna study (seed 123, matching the project convention). Each trial samples an ARMAConv configuration +
training hyperparameters, trains on the TRAIN split, and is scored by sample-wise F1 (swF1) on the VAL split
(threshold tuned on val) — so the test split is never touched during search. The GPU data is loaded and
standardized ONCE and reused across trials. Prints the best config + its val score.

    python hpo_arma.py --system ieee118 --trials 40 --epochs 50
"""
import argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from contextlib import nullcontext
import optuna, fdia_graph as fg
from fdia_graph.dataset import FAMILIES
from torch_geometric.nn import ARMAConv

DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
AC = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if DEV == "cuda" else nullcontext


class ArmaLoc(nn.Module):
    """Edge-fused ARMA localizer following Boyaci's stack (Joint, IEEE TSG 2022): input encoder -> L ARMA_K
    blocks (each K parallel stacks x T recursive layers, ReLU) -> dense -> per-bus sigmoid. Dims set by HPO."""
    def __init__(self, hidden, stacks, layers, blocks, dropout, shared_weights):
        super().__init__()
        self.nenc = nn.Linear(8, hidden); self.eenc = nn.Linear(4, hidden)
        self.blocks = nn.ModuleList([ARMAConv(hidden, hidden, num_stacks=stacks, num_layers=layers,
                                              shared_weights=shared_weights, dropout=dropout, act=F.relu) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, e, ei):
        h = F.relu(self.nenc(x)); he = F.relu(self.eenc(e))
        h = h + torch.zeros_like(h).index_add_(0, ei[1], he)
        for blk in self.blocks: h = blk(h, ei)
        return self.head(h).squeeze(-1)


def f1(pred, tgt, sample=False):
    tp = (pred * tgt).sum(-1); fp = (pred * (1 - tgt)).sum(-1); fn = ((1 - pred) * tgt).sum(-1)
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); fv = 2 * p * r / (p + r + 1e-9)
    return fv.mean().item() if sample else (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-9)).item()


def load(system, release):
    def g(split):
        ds = fg.load(system, split=split, release=release); a = ds.to_numpy()
        d = {k: torch.as_tensor(a[k], device=DEV, dtype=torch.float32) for k in ("node_x", "node_m", "edge_x", "edge_m", "y")}
        return d, torch.as_tensor(a["family"], device=DEV), ds
    trG, _, ds = g("train"); vaG, vaFam, _ = g("val"); teG, teFam, _ = g("test")
    for xk, mk in (("node_x", "node_m"), ("edge_x", "edge_m")):        # standardize on train stats
        w = trG[mk].sum((0, 1)).clamp(min=1.0); mu = (trG[xk] * trG[mk]).sum((0, 1)) / w
        sd = (((trG[xk] - mu) ** 2 * trG[mk]).sum((0, 1)) / w).sqrt().clamp(min=1e-3)
        for gg in (trG, vaG, teG): gg[xk] = (gg[xk] - mu) / sd * gg[mk]
    return trG, vaG, vaFam, teG, teFam, ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="ieee118"); ap.add_argument("--trials", type=int, default=35)
    ap.add_argument("--epochs", type=int, default=150); ap.add_argument("--release", default=None)  # max; early stopping caps it
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    trG, vaG, vaFam, teG, teFam, ds = load(args.system, args.release)
    N, E = ds.N, ds.E; ei0 = ds.edge_index.to(DEV); ei_bi = torch.cat([ei0, ei0.flip(0)], 1)
    pos = float(trG["y"].sum()); base_pw = min(max((trG["y"].numel() - pos) / max(pos, 1), 1.0), 40.0)
    n = trG["y"].shape[0]; vatk = (vaFam > 0)

    def batched(B):
        off = (torch.arange(B, device=DEV) * N).repeat_interleave(ei_bi.shape[1]); return ei_bi.repeat(1, B) + off.unsqueeze(0)
    def feats(g, idx):
        b = len(idx); x = torch.cat([g["node_x"][idx], g["node_m"][idx]], -1).reshape(b * N, -1)
        e2 = torch.cat([g["edge_x"][idx], g["edge_m"][idx]], -1); e = torch.cat([e2, e2], 1).reshape(b * 2 * E, -1)
        return x, e, batched(b), g["y"][idx].reshape(b * N)

    pw = torch.tensor(base_pw, device=DEV); B = 256; va = vatk.cpu()      # Boyaci fixes batch=256
    def val_eval(model):                                                  # returns (val BCE loss, val logits [nv,N])
        model.eval(); LG = []; vloss = 0.0; nv = vaG["y"].shape[0]
        with torch.no_grad(), AC():
            for i in range(0, nv, 512):
                idx = torch.arange(i, min(i + 512, nv), device=DEV); x, e, ei, y = feats(vaG, idx)
                lo = model(x, e, ei); vloss += F.binary_cross_entropy_with_logits(lo, y, pos_weight=pw).item() * len(idx)
                LG.append(lo.reshape(len(idx), N).float().cpu())
        return vloss / nv, torch.cat(LG)

    # ---- Boyaci's Table III search axes (Joint, IEEE TSG 2022); lr/wd/dropout/shared grounded in Bianchi et al. ----
    def objective(trial):
        hp = dict(blocks=trial.suggest_int("blocks", 1, 3),               # Boyaci "layers L" {1,2,3}
                  hidden=trial.suggest_categorical("hidden", [16, 32, 64, 128]),   # Boyaci "units"
                  stacks=trial.suggest_int("stacks", 2, 4),               # Boyaci "K" {2,3,4}
                  layers=trial.suggest_int("layers", 2, 5),               # Boyaci "iteration T" {2..5}
                  dropout=trial.suggest_float("dropout", 0.0, 0.5),       # Bianchi 0.25-0.75
                  lr=trial.suggest_float("lr", 1e-3, 1e-2, log=True),     # Bianchi 1e-2 (Adam)
                  wd=trial.suggest_float("wd", 1e-5, 1e-3, log=True),     # Bianchi L2 5e-4
                  shared=trial.suggest_categorical("shared", [False, True]))
        torch.manual_seed(123)
        model = ArmaLoc(hp["hidden"], hp["stacks"], hp["layers"], hp["blocks"], hp["dropout"], hp["shared"]).to(DEV)
        opt = torch.optim.Adam(model.parameters(), hp["lr"], weight_decay=hp["wd"])
        best = 1e9; patience = 0                                          # Boyaci early stopping: patience 16, min-delta 1e-4
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(n, device=DEV)
            for i in range(0, n, B):
                idx = perm[i:i + B]; x, e, ei, y = feats(trG, idx)
                with AC(): loss = F.binary_cross_entropy_with_logits(model(x, e, ei), y, pos_weight=pw)
                opt.zero_grad(); loss.backward(); opt.step()
            vl, _ = val_eval(model)
            if vl < best - 1e-4: best = vl; patience = 0
            else: patience += 1
            if patience >= 16: break
        _, L = val_eval(model); Y = vaG["y"].cpu()
        thr = float(max(torch.linspace(-2, 3, 26), key=lambda t: f1((L[va] > t).float(), Y[va])))
        return f1((L[va] > thr).float(), Y[va], sample=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=123))
    study.enqueue_trial({"blocks": 2, "hidden": 16, "stacks": 3, "layers": 5,      # Boyaci's IEEE-118 optimum (warm start)
                         "dropout": 0.1, "lr": 5e-3, "wd": 5e-4, "shared": True})
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    bp = study.best_params
    print(f"\n[{args.system}] best val swF1 = {study.best_value:.3f}\nbest params: {json.dumps(bp)}")

    # ---- retrain the best config and evaluate on TEST (threshold tuned on val) -> final benchmark entry ----
    torch.manual_seed(123)
    model = ArmaLoc(bp["hidden"], bp["stacks"], bp["layers"], bp["blocks"], bp["dropout"], bp["shared"]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), bp["lr"], weight_decay=bp["wd"]); best = 1e9; patience = 0
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, B):
            idx = perm[i:i + B]; x, e, ei, y = feats(trG, idx)
            with AC(): loss = F.binary_cross_entropy_with_logits(model(x, e, ei), y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        vl, _ = val_eval(model)
        if vl < best - 1e-4: best = vl; patience = 0
        else: patience += 1
        if patience >= 16: break
    _, vL = val_eval(model); vY = vaG["y"].cpu()
    thr = float(max(torch.linspace(-2, 3, 26), key=lambda t: f1((vL[va] > t).float(), vY[va])))
    model.eval(); LG = []; nt = teG["y"].shape[0]
    with torch.no_grad(), AC():
        for i in range(0, nt, 512):
            idx = torch.arange(i, min(i + 512, nt), device=DEV); x, e, ei, _ = feats(teG, idx)
            LG.append(model(x, e, ei).reshape(len(idx), N).float().cpu())
    L = torch.cat(LG); Y = teG["y"].cpu(); Fm = teFam.cpu(); P = (L > thr).float(); atk = Fm > 0
    res = {"system": args.system, "N": int(N), "E": int(E), "model": "ARMA-HPO", "best_params": bp, "threshold": round(thr, 2),
           "overall": {"n": int(atk.sum()), "node_f1": f1(P[atk], Y[atk]), "swf1": f1(P[atk], Y[atk], sample=True)}, "per_family": {}}
    print(f"TEST overall: node-F1 {res['overall']['node_f1']:.3f}  swF1 {res['overall']['swf1']:.3f}")
    for k, name in FAMILIES.items():
        m = Fm == k
        if m.any():
            r = {"n": int(m.sum()), "node_f1": f1(P[m], Y[m]), "swf1": f1(P[m], Y[m], sample=True)}; res["per_family"][name] = r
            print(f"  {name:6s} node-F1 {r['node_f1']:.3f}  swF1 {r['swf1']:.3f}")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2); print("wrote", args.out)


if __name__ == "__main__":
    main()
