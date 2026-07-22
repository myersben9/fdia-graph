#!/usr/bin/env python
"""PG-DGAT — Physics-Guided Dynamic Graph Attention Network for FDIA localization/detection AND
attack-resilient state estimation. Faithful reimplementation of a scanned IEEE PES paper (Eqs 5-27),
adapted from its radial DistFlow feeders (IEEE 13/123) to our MESHED transmission grids (IEEE 14/118/300).

Runs ONE system and ONE task per invocation:
    python _pgdgat.py --system 14  --task loc   # localization + grid-level detection  -> results/pgdgat_14.json
    python _pgdgat.py --system 14  --task se    # attack-resilient V/theta estimation   -> results/pgdgat_se_14.json

============================================================================================================
ARCHITECTURE (three PG-DGAT parts, shared backbone; only the output head + loss change between tasks)
------------------------------------------------------------------------------------------------------------
1) ADAPTIVE EDGE WEIGHTS (Eqs 7-12).  The paper derives nodal voltage sensitivities from a RADIAL DistFlow
   model (V_i = V_1 - sum Z_ij S_j*/V_j*), which is only valid on radial distribution feeders. Our systems
   are meshed transmission networks, so we instead derive the sensitivities dV/dP, dV/dQ, dtheta/dP,
   dtheta/dQ from the base-case power-flow JACOBIAN: J = d(P,Q)/d(V,theta) evaluated once at a constant
   linearization point (the pool-mean operating state), inverted via a rank-truncated pseudo-inverse (the
   slack/angle-reference makes J singular). This is the "constant-sensitivity" first version the task blesses.
     - Predicted voltage (Eq 7):  Vhat_i = V0_i + sum_j [ dV_i/dP_j * dP_j + dV_i/dQ_j * dQ_j ]
       with dP,dQ = measured injection minus base injection (per sample; unmetered buses contribute 0).
     - Base weight (Eq 8):        w_base_ij = |Y_ij| + |Vhat_i - V0_i|
     - Consistency residual (Eq 11): delta_i = | Shat_i - S_i | where Shat = h(Vhat,thhat) (physics-op
       injection from the PREDICTED state) and S = measured injection; spikes at attacked buses.
     - Adaptive weight (Eq 12):   w_phy_ij = w_base_ij * (1 + norm(delta_j))   (norm = per-sample min-max)
2) PHYSICS-GUIDED MESSAGE PASSING (Eqs 13-20), L layers, multi-head GAT whose attention logit is SCALED by
   w_phy (Eq 14), followed by a GRU node update (Eqs 17-20).
3) COMPOSITE LOSS (loc task, Eqs 21-26):  L = 0.7 L_focal + 0.15 L_grad + 0.15 L_consistency  (fixed split).
     - L_focal (Eq 21): class-balanced focal loss, gamma=2, mu = pos_weight (matches the imbalance handling
       the ARMA baseline uses).
     - L_grad  (Eq 22): trusted-node physical-monotonicity penalty. NOTE: with constant (precomputed)
       sensitivities the bracket term is a per-bus constant, so this term degenerates to a fixed soft bias
       on trusted buses; kept faithful but down-weighted (lambda_g=0.15). See DESIGN NOTE below.
     - L_consist (Eq 25): credibility-weighted (b_i = 1 - p_i) measurement-reconstruction residual using the
       LINEARIZED SE prediction Vhat (NOT a per-step weighted-WLS solve). This is the documented efficient
       simplification of the SE-in-the-loop term — it captures the intent (down-weight residuals at
       low-credibility/attacked buses, push trusted buses to be physically consistent) at O(matmul) cost.
   SE task: regression head -> per-bus (|V| p.u., theta rad); loss = smooth_l1 on the TRUE pre-attack state
   (pool state the record was built from) + 0.2 * branch-flow physics term — identical recipe to our PI-SE
   so the comparison to results/se_{C}.json (PINN) / se_{C}_nophys.json (NN) / se_{C}_wls.json (WLS) is clean.

Data/conventions match _train_baselines.py / train_arma.py / _se_pinn_v040.py exactly: release_v0.4.1 shards,
units="pu", same edge_index, same chronological splits, same swF1 / DR-FA-detF1 / V-theta-MAE metric code.
Env: KMP_DUPLICATE_LIB_OK=TRUE, PYTHONPATH=../src.  seed_everything(123).
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandapower as pp, pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from torch_geometric.utils import softmax as pyg_softmax, scatter
from fdia_graph.dataset import FdiaGraph, FAMILIES

HERE = os.path.dirname(os.path.abspath(__file__))
SHARDS = os.path.join(HERE, "release_v0.4.1"); RES = os.path.join(HERE, "results")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(s=123):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ metrics (verbatim from train_arma.py)
def f1(pred, tgt, sample=False):
    tp = (pred * tgt).sum(-1); fp = (pred * (1 - tgt)).sum(-1); fn = ((1 - pred) * tgt).sum(-1)
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); fv = 2 * p * r / (p + r + 1e-9)
    return fv.mean().item() if sample else (2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum() + 1e-9)).item()


# ============================================================ PHYSICS: Ybus, AC operator, sensitivities
class Physics:
    """Constant base-case physics for one system: Ybus (bus-ordered), differentiable AC operator, and the
    precomputed voltage-sensitivity blocks used to build the adaptive edge weights (Eqs 7-12, 22)."""
    def __init__(self, C, edge_index):
        net = {14: pn.case14, 118: pn.case118, 300: pn.case300}[C]()
        pp.runpp(net)
        ppc = net._ppc
        Yb, Yf, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
        self.bMVA = float(ppc["baseMVA"]); self.N = C
        lut = net._pd2ppc_lookups["bus"][:C].astype(np.int64)          # pandapower bus -> ppc index
        self.nppc = ppc["bus"].shape[0]
        self.LUT = torch.as_tensor(lut, device=DEV)
        self.FB = torch.as_tensor(ppc["branch"][:, 0].real.astype(np.int64), device=DEV)
        self.Ybus = torch.as_tensor(np.asarray(Yb.todense()), dtype=torch.complex64, device=DEV)   # ppc-ordered
        self.Yf = torch.as_tensor(np.asarray(Yf.todense()), dtype=torch.complex64, device=DEV)
        # bus-ordered dense Ybus (pandapower bus index 0..N-1) for per-edge |Y_ij|
        Yb_bus = self.Ybus[self.LUT][:, self.LUT]                       # [N,N] complex
        # linearization point t_0 = pool-mean operating state (representative of the data distribution)
        pool = np.load(os.path.join(SHARDS, f"pool_ieee{C}.npz"))["X"].astype(np.float32)  # [T,N,4]=[P,Q,|V|,th_deg]
        self.V0 = torch.as_tensor(pool[:, :, 2].mean(0), device=DEV)                        # [N] p.u.
        self.th0 = torch.deg2rad(torch.as_tensor(pool[:, :, 3].mean(0), device=DEV))        # [N] rad
        with torch.no_grad():
            P0, Q0, _, _ = self.ac(self.V0[None], self.th0[None])
        self.P0 = P0[0]; self.Q0 = Q0[0]                                                    # [N] p.u. base injections
        # ---- Jacobian sensitivities via autograd on the AC operator (one-time precompute) ----
        def PQ(vt):
            V = vt[:C]; th = vt[C:]
            Pb, Qb, _, _ = self.ac(V[None], th[None])
            return torch.cat([Pb[0], Qb[0]])
        J = torch.autograd.functional.jacobian(PQ, torch.cat([self.V0, self.th0]))          # [2N,2N]=d(P,Q)/d(V,th)
        Jinv = torch.linalg.pinv(J, rtol=1e-4)                                              # rank-truncated (slack singular)
        self.SVP = Jinv[:C, :C].contiguous()      # dV/dP
        self.SVQ = Jinv[:C, C:].contiguous()      # dV/dQ
        self.STP = Jinv[C:, :C].contiguous()      # dtheta/dP
        self.STQ = Jinv[C:, C:].contiguous()      # dtheta/dQ
        # Eq-22 per-bus constant: max(0,-dV_i/dP_i) + max(0, dtheta_i/dQ_i)
        self.c_grad = (F.relu(-self.SVP.diag()) + F.relu(self.STQ.diag())).detach()          # [N] >= 0
        # per-directed-edge admittance magnitude for the (bidirectional) message-passing graph
        ei_bi = torch.cat([edge_index, edge_index.flip(0)], 1).to(DEV)                        # [2,2E]
        self.ei_bi = ei_bi
        s, d = ei_bi[0], ei_bi[1]
        self.Ymag = Yb_bus[d, s].abs().to(DEV)                                                # |Y_ij| per directed edge

    def ac(self, Vmag, theta):
        """state (|V| p.u., theta rad) -> (bus P,Q [.,N], branch Pf,Qf [.,E]) per-unit. Same op as _se_pinn."""
        b = Vmag.shape[0]
        Vc_pp = torch.polar(Vmag, theta)
        Vc = torch.zeros(b, self.nppc, dtype=torch.complex64, device=Vmag.device)
        Vc[:, self.LUT] = Vc_pp
        Sbus = Vc * torch.conj(Vc @ self.Ybus.T)
        Sf = Vc[:, self.FB] * torch.conj(Vc @ self.Yf.T)
        return Sbus.real[:, self.LUT], Sbus.imag[:, self.LUT], Sf.real, Sf.imag

    def edge_weights(self, nx_pu, nm):
        """Eqs 7-12. nx_pu [B,N,4]=[V,P,Q,th], nm [B,N,4] mask -> (w_flat [B*2E], phys tuple for consistency).
        Physics uses RAW per-unit injections (not the standardized model input)."""
        P, Q = nx_pu[..., 1], nx_pu[..., 2]; mP, mQ = nm[..., 1] > 0.5, nm[..., 2] > 0.5
        Pu = torch.where(mP, P, self.P0); Qu = torch.where(mQ, Q, self.Q0)                    # base-fill unmetered
        dP = Pu - self.P0; dQ = Qu - self.Q0                                                  # [B,N]
        Vhat = self.V0 + dP @ self.SVP.T + dQ @ self.SVQ.T                                    # Eq 7 (magnitude)
        thhat = self.th0 + dP @ self.STP.T + dQ @ self.STQ.T
        Phat, Qhat, _, _ = self.ac(Vhat, thhat)                                              # Eq 10 injection from Vhat
        delta = torch.sqrt((Phat - Pu) ** 2 + (Qhat - Qu) ** 2 + 1e-12)                       # Eq 11 |Shat - S|
        dmin = delta.min(1, keepdim=True).values; dmax = delta.max(1, keepdim=True).values
        dnorm = (delta - dmin) / (dmax - dmin + 1e-9)                                         # per-sample min-max norm
        Vdev = (Vhat - self.V0).abs()                                                        # [B,N]
        s, d = self.ei_bi[0], self.ei_bi[1]
        wbase = self.Ymag[None] + Vdev[:, d]                                                  # Eq 8  [B,2E]
        wphy = wbase * (1.0 + dnorm[:, s])                                                    # Eq 12
        wphy = wphy / wphy.mean(1, keepdim=True).clamp(min=1e-6)                              # O(1) scale for attention
        return wphy.reshape(-1), (Phat, Qhat, Pu, Qu, mP.float(), mQ.float(), delta)


# ============================================================ PG-DGAT backbone (Eqs 13-20)
class PGDGAT(nn.Module):
    def __init__(self, fin, hidden=128, heads=4, layers=2, out_dim=1, dropout=0.1):
        super().__init__()
        self.H, self.d, self.L = heads, hidden // heads, layers
        self.enc = nn.Linear(fin, hidden)
        self.Wproj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])        # Eq 13 z=W^k h
        self.att = nn.ParameterList([nn.Parameter(torch.empty(heads, 2 * self.d)) for _ in range(layers)])  # a^k
        for a in self.att: nn.init.xavier_uniform_(a)
        self.gr = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])       # Eq 17 W_r
        self.gz = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])       # Eq 18 W_z
        self.gh = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])       # Eq 19 W_h
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x, ei, w, nnode):
        h = F.relu(self.enc(x))
        s, d = ei[0], ei[1]
        for l in range(self.L):
            z = self.Wproj[l](h).view(-1, self.H, self.d)                                     # [Nn,H,d]
            zc = torch.cat([z[d], z[s]], dim=-1)                                              # Eq 14 [z_i || z_j]
            e = F.leaky_relu((zc * self.att[l]).sum(-1), 0.2)                                 # [E,H]
            e = e * w.unsqueeze(-1)                                                           # Eq 14 scale by w_phy
            alpha = self.drop(pyg_softmax(e, d, num_nodes=nnode))                             # Eq 15
            msg = (alpha.unsqueeze(-1) * z[s])                                                # Eq 16 alpha * W^k h_j
            M = scatter(msg, d, dim=0, dim_size=nnode, reduce="sum").reshape(nnode, -1)       # Concat heads
            hM = torch.cat([h, M], -1)
            r = torch.sigmoid(self.gr[l](hM)); zt = torch.sigmoid(self.gz[l](hM))             # Eqs 17-18
            hb = torch.tanh(self.gh[l](torch.cat([M, r * h], -1)))                            # Eq 19
            h = (1 - zt) * h + zt * hb                                                         # Eq 20
        return self.head(h)


# ============================================================ shared data loading
def load_split(C, split, want_state=False):
    ds = FdiaGraph(os.path.join(SHARDS, f"ml_only_ieee{C}.h5"), split=split, units="pu")
    a = ds.to_numpy()
    g = {k: torch.as_tensor(a[k], device=DEV, dtype=torch.float32) for k in ("node_x", "node_m", "edge_x", "edge_m", "y")}
    g["family"] = torch.as_tensor(a["family"], device=DEV)
    if want_state:
        pool = np.load(os.path.join(SHARDS, f"pool_ieee{C}.npz"))["X"].astype(np.float32)     # [T,N,4]=[P,Q,|V|,th_deg]
        Xtrue = pool[a["timestep"].astype(np.int64)]                                          # [n,N,4]
        g["Vtrue"] = torch.as_tensor(Xtrue[:, :, 2], device=DEV)                              # p.u.
        g["THtrue_deg"] = torch.as_tensor(Xtrue[:, :, 3], device=DEV)                         # deg
    return g, ds


def standardize(trG, others, mu=None, sd=None):
    """train-metered feature standardization on node_x (matches train_arma / _se_pinn std2)."""
    if mu is None:
        w = trG["node_m"].sum((0, 1)).clamp(min=1.0)
        mu = (trG["node_x"] * trG["node_m"]).sum((0, 1)) / w
        sd = (((trG["node_x"] - mu) ** 2 * trG["node_m"]).sum((0, 1)) / w).sqrt().clamp(min=1e-3)
    for g in [trG] + others:
        g["node_xn"] = (g["node_x"] - mu) / sd * g["node_m"]
    return mu, sd


# ============================================================ LOCALIZATION / DETECTION task
def run_loc(C, epochs, seed):
    seed_everything(seed)
    trG, ds = load_split(C, "train"); vaG, _ = load_split(C, "val"); teG, _ = load_split(C, "test")
    N, E = ds.N, ds.E
    phys = Physics(C, ds.edge_index)
    standardize(trG, [vaG, teG])
    ei_bi = phys.ei_bi                                                                        # [2,2E]

    def batched_ei(B):
        off = (torch.arange(B, device=DEV) * N).repeat_interleave(ei_bi.shape[1])
        return ei_bi.repeat(1, B) + off.unsqueeze(0)

    model = PGDGAT(fin=8, hidden=128, heads=4, layers=2, out_dim=1, dropout=0.1).to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pos = float(trG["y"].sum()); mu_pos = min(max((trG["y"].numel() - pos) / max(pos, 1), 1.0), 30.0)
    n = trG["y"].shape[0]; B = 256; gamma = 2.0
    lam_d, lam_g, lam_c = 0.7, 0.15, 0.15
    print(f"[loc ieee{C}] N={N} E={E} train {n:,}  PG-DGAT(h128,heads4,L2)  mu_pos={mu_pos:.1f} batch={B} epochs={epochs} [{DEV}]", flush=True)

    def batch_forward(g, idx):
        b = len(idx)
        nx_pu = g["node_x"][idx]; nm = g["node_m"][idx]
        w, phy = phys.edge_weights(nx_pu, nm)
        x = torch.cat([g["node_xn"][idx], nm], -1).reshape(b * N, -1)
        logit = model(x, batched_ei(b), w, b * N).reshape(b, N)
        return logit, phy

    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEV); tot = 0.0
        for i in range(0, n, B):
            idx = perm[i:i + B]; b = len(idx)
            logit, (Phat, Qhat, Pu, Qu, mP, mQ, delta) = batch_forward(trG, idx)
            y = trG["y"][idx]; p = torch.sigmoid(logit)
            logp = -F.softplus(-logit); log1mp = -F.softplus(logit)                           # stable log p / log(1-p)
            Ld = -(mu_pos * y * (1 - p) ** gamma * logp + (1 - y) * p ** gamma * log1mp).mean()  # Eq 21
            trusted = torch.sigmoid(10.0 * (0.5 - p))                                          # soft (1-sign(p-.5))/2
            Lg = (trusted * phys.c_grad[None]).mean()                                          # Eq 22
            b_cred = 1 - p                                                                     # Eq 23 credibility
            resid = (Phat - Pu) ** 2 * mP + (Qhat - Qu) ** 2 * mQ                              # metered-only residual
            Lc = (b_cred * resid).sum() / (mP + mQ).sum().clamp(min=1.0)                       # Eq 25 (linearized)
            loss = lam_d * Ld + lam_g * Lg + lam_c * Lc
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * b
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{epochs} loss {tot/n:.4f} (Ld {Ld.item():.3f} Lg {Lg.item():.4f} Lc {Lc.item():.4f})", flush=True)

    @torch.no_grad()
    def collect(g):
        model.eval(); LG = []
        for i in range(0, g["y"].shape[0], B):
            idx = torch.arange(i, min(i + B, g["y"].shape[0]), device=DEV)
            LG.append(batch_forward(g, idx)[0].float().cpu())
        return torch.cat(LG), g["y"].cpu(), g["family"].cpu()

    vL, vY, vFm = collect(vaG); vatk = vFm > 0
    thr = float(max(torch.linspace(-2, 3, 26), key=lambda t: f1((vL[vatk] > t).float(), vY[vatk])))
    L, Y, Fm = collect(teG); P = (L > thr).float(); atk = Fm > 0
    res = {"system": f"ieee{C}", "N": int(N), "E": int(E), "model": "PG-DGAT", "threshold": round(thr, 2),
           "overall": {"n": int(atk.sum()), "node_f1": f1(P[atk], Y[atk]), "swf1": f1(P[atk], Y[atk], sample=True)},
           "per_family": {}}
    print(f"[loc ieee{C}] tuned thr {thr:.2f}  overall node-F1 {res['overall']['node_f1']:.3f} swF1 {res['overall']['swf1']:.3f}", flush=True)
    for k, name in FAMILIES.items():
        m = Fm == k
        if m.any():
            r = {"n": int(m.sum()), "node_f1": f1(P[m], Y[m]), "swf1": f1(P[m], Y[m], sample=True)}
            res["per_family"][name] = r
            print(f"    {name:6s} n={r['n']:5d} node-F1 {r['node_f1']:.3f} swF1 {r['swf1']:.3f}", flush=True)

    def dscore(logits): return torch.sigmoid(logits).max(1).values
    def detf1(S, lab, t):
        pr = (S > t).float(); tp = (pr * lab).sum(); fp = (pr * (1 - lab)).sum(); fn = ((1 - pr) * lab).sum()
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); return (2 * p * r / (p + r + 1e-9)).item()
    vS = dscore(vL); vlab = (vFm > 0).float()
    dthr = float(max(torch.linspace(0.05, 0.95, 19), key=lambda t: detf1(vS, vlab, float(t))))
    tS = dscore(L); tlab = (Fm > 0).float(); pr = (tS > dthr).float()
    DR = (pr * tlab).sum().item() / tlab.sum().item(); FA = (pr * (1 - tlab)).sum().item() / (1 - tlab).sum().item()
    res["detection"] = {"DR": round(DR, 4), "FA": round(FA, 4), "det_f1": round(detf1(tS, tlab, dthr), 4),
                        "threshold": round(dthr, 3), "per_family_DR": {}}
    print(f"[loc ieee{C}] DETECTION DR {DR:.3f} FA {FA:.3f} det-F1 {res['detection']['det_f1']:.3f} (thr {dthr:.2f})", flush=True)
    for k, name in FAMILIES.items():
        m = Fm == k
        if k and m.any():
            res["detection"]["per_family_DR"][name] = round(pr[m].mean().item(), 3)
    os.makedirs(RES, exist_ok=True)
    outp = os.path.join(RES, f"pgdgat_{C}.json"); json.dump(res, open(outp, "w"), indent=2); print("wrote", outp, flush=True)


# ============================================================ STATE-ESTIMATION task
def run_se(C, epochs, seed, w_phys=0.2):
    seed_everything(seed)
    trG, ds = load_split(C, "train", want_state=True)
    vaG, _ = load_split(C, "val", want_state=True); teG, _ = load_split(C, "test", want_state=True)
    N, E = ds.N, ds.E
    phys = Physics(C, ds.edge_index); ei_bi = phys.ei_bi
    standardize(trG, [vaG, teG])
    # target normalization (matches _se_pinn: standardize V/theta targets by train stats)
    THtrue_rad = {g: torch.deg2rad(d["THtrue_deg"]) for g, d in (("tr", trG), ("va", vaG), ("te", teG))}
    vmu, vsd = trG["Vtrue"].mean(), trG["Vtrue"].std().clamp(min=1e-3)
    tmu, tsd = THtrue_rad["tr"].mean(), THtrue_rad["tr"].std().clamp(min=1e-3)

    def batched_ei(B):
        off = (torch.arange(B, device=DEV) * N).repeat_interleave(ei_bi.shape[1])
        return ei_bi.repeat(1, B) + off.unsqueeze(0)

    HID = 256
    model = PGDGAT(fin=8, hidden=HID, heads=4, layers=2, out_dim=2, dropout=0.05).to(DEV)
    opt = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = trG["y"].shape[0]; B = 128 if C >= 300 else 256
    print(f"[se ieee{C}] N={N} E={E} train {n:,}  PG-DGAT-SE(h{HID},heads4,L2) w_phys={w_phys} batch={B} epochs={epochs} [{DEV}]", flush=True)

    def fwd(g, idx):
        b = len(idx); nx_pu = g["node_x"][idx]; nm = g["node_m"][idx]
        w, _ = phys.edge_weights(nx_pu, nm)
        x = torch.cat([g["node_xn"][idx], nm], -1).reshape(b * N, -1)
        out = model(x, batched_ei(b), w, b * N).reshape(b, N, 2)
        return out

    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, B):
            idx = perm[i:i + B]
            out = fwd(trG, idx)
            Vh = out[..., 0] * vsd + vmu; THh = out[..., 1] * tsd + tmu                        # de-standardize (rad)
            Ls = F.smooth_l1_loss((trG["Vtrue"][idx] - vmu) / vsd, out[..., 0]) \
               + F.smooth_l1_loss((THtrue_rad["tr"][idx] - tmu) / tsd, out[..., 1])
            _, _, Pf, Qf = phys.ac(Vh, THh)
            _, _, cPf, cQf = phys.ac(trG["Vtrue"][idx], THtrue_rad["tr"][idx])                 # physics from TRUE state
            Lp = F.smooth_l1_loss(Pf, cPf) + F.smooth_l1_loss(Qf, cQf)
            loss = Ls + w_phys * Lp
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{epochs} L_state {Ls.item():.4f} L_phys {Lp.item():.4f}", flush=True)

    @torch.no_grad()
    def estimate(g):
        model.eval(); V, T = [], []
        for i in range(0, g["y"].shape[0], 512):
            idx = torch.arange(i, min(i + 512, g["y"].shape[0]), device=DEV)
            out = fwd(g, idx); V.append((out[..., 0] * vsd + vmu).cpu()); T.append((out[..., 1] * tsd + tmu).cpu())
        return torch.cat(V), torch.cat(T)

    Vh, THh_rad = estimate(teG); THh = torch.rad2deg(THh_rad)
    tF = teG["family"].cpu()
    Vtr = teG["Vtrue"].cpu(); THtr = teG["THtrue_deg"].cpu()
    Vmeas = teG["node_x"][:, :, 0].cpu(); THmeas = torch.rad2deg(teG["node_x"][:, :, 3]).cpu()  # meter readings (V pu, th deg)
    mV = teG["node_m"][:, :, 0].cpu().bool(); mTH = teG["node_m"][:, :, 3].cpu().bool()

    def mae(a, b, mask=None):
        e = (a - b).abs()
        return (e[mask].mean().item() if mask is not None and mask.any() else e.mean().item())

    res = {"system": f"ieee{C}", "model": "PG-DGAT-SE", "w_phys": w_phys, "seed": seed, "hid": HID,
           "epochs": epochs, "units": "pu", "release": "v0.4.1", "per_family": {}}
    print(f"[se ieee{C}] === STATE-ESTIMATION ERROR on TEST (|V| p.u., theta deg) ===", flush=True)
    for k, name in FAMILIES.items():
        m = (tF == k).numpy()
        if not m.any(): continue
        mv = mV[m]; mt = mTH[m]
        vAll = mae(Vh[m], Vtr[m]); vseM = mae(Vh[m][mv], Vtr[m][mv]); vmeM = mae(Vmeas[m][mv], Vtr[m][mv])
        tseM = mae(THh[m][mt], THtr[m][mt]); tmeM = mae(THmeas[m][mt], THtr[m][mt])
        res["per_family"][name] = dict(V_mae_all=round(vAll, 4), V_mae_se_metered=round(vseM, 4),
            V_mae_meter=round(vmeM, 4), th_mae_se_metered=round(tseM, 3), th_mae_meter=round(tmeM, 3),
            V_mae_meter_attacked=round(vmeM, 4), V_mae_se_metered_attacked=round(vseM, 4),
            th_mae_meter_attacked=round(tmeM, 3), th_mae_se_metered_attacked=round(tseM, 3))
        print(f"    {name:6s} |V|SE {vseM:.4f} (meter {vmeM:.4f})  thSE {tseM:.3f} (meter {tmeM:.3f})", flush=True)
    atk = (tF > 0).numpy(); mvA = mV[atk]; mtA = mTH[atk]
    res["overall"] = dict(V_mae_all=round(mae(Vh, Vtr), 4),
        V_mae_se_metered_attacked=round(mae(Vh[atk][mvA], Vtr[atk][mvA]), 4),
        V_mae_meter_attacked=round(mae(Vmeas[atk][mvA], Vtr[atk][mvA]), 4),
        th_mae_se_metered_attacked=round(mae(THh[atk][mtA], THtr[atk][mtA]), 3),
        th_mae_meter_attacked=round(mae(THmeas[atk][mtA], THtr[atk][mtA]), 3))
    o = res["overall"]
    print(f"[se ieee{C}] ATTACKED metered: |V| SE {o['V_mae_se_metered_attacked']:.4f} (meter {o['V_mae_meter_attacked']:.4f}) | "
          f"theta SE {o['th_mae_se_metered_attacked']:.3f} (meter {o['th_mae_meter_attacked']:.3f}) deg", flush=True)
    os.makedirs(RES, exist_ok=True)
    outp = os.path.join(RES, f"pgdgat_se_{C}.json"); json.dump(res, open(outp, "w"), indent=2); print("wrote", outp, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", type=int, required=True, choices=[14, 118, 300])
    ap.add_argument("--task", required=True, choices=["loc", "se"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    if args.task == "loc":
        run_loc(args.system, args.epochs or 40, args.seed)
    else:
        run_se(args.system, args.epochs or 80, args.seed)
