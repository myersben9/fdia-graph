#!/usr/bin/env python
"""Diagnostic for the accuracy-class noise knob (coordinator red-flag check).
For IEEE-14, seed 123, loops c in {0.1,0.2,0.5,1.0}% x wp in {0,0.3}:
  (1) prints the EMPIRICAL std of (noised_meas - clean_meas) per channel -> must fall ~10x as c falls 10x.
  (2) prints benign + attacked SE angle MAE vs METER angle MAE -> is the SE tracking the meter noise down,
      or floored by model capacity?
Trains the identical SE model as _se_pinn_ac.py (80 epochs). Loads the clean shard once. No files written."""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, h5py
import pandapower as pp, pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from torch_geometric.nn import ARMAConv
from fdia_graph.dataset import FAMILIES

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
C = int(os.environ.get("CASE", "14")); SEED = 123; EPOCHS = 80; HID = 256
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_v0.4.1")
H5 = os.path.join(REL, f"ml_only_ieee{C}_clean.h5"); POOL = os.path.join(REL, f"pool_ieee{C}.npz")

NET = {14: pn.case14, 118: pn.case118, 300: pn.case300}[C]
base = NET(); pp.runpp(base)
_ppc = base._ppc; _Ybus, _Yf, _Yt = makeYbus(_ppc["baseMVA"], _ppc["bus"], _ppc["branch"]); bMVA = _ppc["baseMVA"]
lut = base._pd2ppc_lookups["bus"][:C].astype(np.int64); fb = _ppc["branch"][:, 0].real.astype(np.int64); nppc = _ppc["bus"].shape[0]
Ybus = torch.as_tensor(np.asarray(_Ybus.todense()), dtype=torch.complex64, device=DEV)
Yf = torch.as_tensor(np.asarray(_Yf.todense()), dtype=torch.complex64, device=DEV)
LUT = torch.as_tensor(lut, device=DEV); FB = torch.as_tensor(fb, device=DEV)

def ac_from_state(Vmag, theta_rad):
    b = Vmag.shape[0]; Vc = torch.zeros(b, nppc, dtype=torch.complex64, device=Vmag.device)
    Vc[:, LUT] = torch.polar(Vmag, theta_rad)
    Sbus = Vc * torch.conj(Vc @ Ybus.T); Sf = Vc[:, FB] * torch.conj(Vc @ Yf.T)
    return Sbus.real[:, LUT], Sbus.imag[:, LUT], Sf.real, Sf.imag

with h5py.File(H5, "r") as f:
    d = f["data"]; A0 = {k: d[k][:] for k in ("node_x", "node_m", "edge_x", "edge_m", "y", "family", "timestep", "split")}
    baseMVA = float(f.attrs.get("baseMVA", bMVA))
Nrec = len(A0["node_x"]); N = A0["node_x"].shape[1]; E = A0["edge_x"].shape[1]
POOLX = np.load(POOL)["X"].astype(np.float32); Xtrue = POOLX[A0["timestep"].astype(np.int64)]
sp = A0["split"]; tr = np.where(sp == 0)[0]; teI = np.where(sp == 2)[0]
fam = torch.as_tensor(A0["family"].astype(np.int64))
Vtrue = torch.as_tensor(Xtrue[:, :, 2], device=DEV); THtrue_deg = torch.as_tensor(Xtrue[:, :, 3], device=DEV); THtrue = torch.deg2rad(THtrue_deg)
ei0 = torch.stack([torch.as_tensor(np.r_[base.line["from_bus"].values, base.trafo["hv_bus"].values], dtype=torch.long),
                   torch.as_tensor(np.r_[base.line["to_bus"].values, base.trafo["lv_bus"].values], dtype=torch.long)], 0).to(DEV)
ei_bi = torch.cat([ei0, ei0.flip(0)], 1)
nm = torch.as_tensor(A0["node_m"], device=DEV, dtype=torch.float32); em = torch.as_tensor(A0["edge_m"], device=DEV, dtype=torch.float32)
with torch.no_grad():
    cPf, cQf = [], []
    for i in range(0, Nrec, 4096):
        _, _, c, dd = ac_from_state(Vtrue[i:i+4096], THtrue[i:i+4096]); cPf.append(c); cQf.append(dd)
    cPf = torch.cat(cPf); cQf = torch.cat(cQf)

class SE(nn.Module):
    def __init__(self, hid=HID):
        super().__init__(); self.nenc = nn.Linear(8, hid); self.eenc = nn.Linear(4, hid)
        self.b1 = ARMAConv(hid, hid, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.05, act=F.relu)
        self.b2 = ARMAConv(hid, hid, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.05, act=F.relu)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))
    def forward(self, x, e, ei, b):
        h = F.relu(self.nenc(x)); he = F.relu(self.eenc(e)); h = h + torch.zeros_like(h).index_add_(0, ei[1], he)
        h = self.b1(h, ei); h = self.b2(h, ei); return self.head(h).reshape(b, N, 2)

def run(AC, W_PHYS):
    _nrng = np.random.default_rng(1_000_000 * SEED + int(round(AC * 1000)))
    nxc = A0["node_x"].astype(np.float64); exc = A0["edge_x"].astype(np.float64); r = (AC / 100.0) / 3.0
    sig_node = np.stack([r*np.abs(nxc[...,0]), r*np.abs(nxc[...,1])+1e-3, r*np.abs(nxc[...,2])+1e-3, np.full_like(nxc[...,3], AC*0.096)], -1)
    sig_edge = np.stack([r*np.abs(exc[...,0])+1e-3, r*np.abs(exc[...,1])+1e-3], -1)
    node_noise = sig_node * _nrng.standard_normal(nxc.shape) * A0["node_m"]
    edge_noise = sig_edge * _nrng.standard_normal(exc.shape) * A0["edge_m"]
    nx_phys_np = (nxc + node_noise).astype(np.float32); ex_np = (exc + edge_noise).astype(np.float32)
    # empirical injected-noise std per channel (metered only), benign records
    ben = A0["family"] == 0; mn = A0["node_m"].astype(bool)
    estd = [float(node_noise[ben][..., ch][mn[ben][..., ch]].std()) for ch in range(4)]
    nx_phys = torch.as_tensor(nx_phys_np, device=DEV)
    nx = nx_phys.clone(); nx[..., 1] /= baseMVA; nx[..., 2] /= baseMVA; nx[..., 3] = torch.deg2rad(nx[..., 3])
    ex = torch.as_tensor(ex_np, device=DEV) / baseMVA
    def std2(X, M, idx):
        w = M[idx].sum((0,1)).clamp(min=1); mu = (X[idx]*M[idx]).sum((0,1))/w
        sd = (((X[idx]-mu)**2*M[idx]).sum((0,1))/w).sqrt().clamp(min=1e-3); return (X-mu)/sd*M
    NXn = std2(nx, nm, tr); EXn = std2(ex, em, tr)
    vmu, vsd = Vtrue[tr].mean(), Vtrue[tr].std().clamp(min=1e-3); tmu, tsd = THtrue[tr].mean(), THtrue[tr].std().clamp(min=1e-3)
    def feed(idx):
        b = len(idx); x = torch.cat([NXn[idx], nm[idx]], -1).reshape(b*N, -1)
        e2 = torch.cat([EXn[idx], em[idx]], -1); e = torch.cat([e2, e2], 1).reshape(b*2*E, -1)
        off = (torch.arange(b, device=DEV)*N).repeat_interleave(ei_bi.shape[1]); ei = ei_bi.repeat(1, b)+off.unsqueeze(0)
        return x, e, ei, b
    torch.manual_seed(SEED); model = SE().to(DEV); opt = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS); BS = 256
    for ep in range(EPOCHS):
        model.train(); perm = tr[torch.randperm(len(tr)).numpy()]
        for i in range(0, len(perm), BS):
            idx = torch.as_tensor(perm[i:i+BS], device=DEV); x, e, ei, b = feed(idx)
            out = model(x, e, ei, b); Vh = out[...,0]*vsd+vmu; THh = out[...,1]*tsd+tmu
            Ls = F.smooth_l1_loss((Vtrue[idx]-vmu)/vsd, out[...,0]) + F.smooth_l1_loss((THtrue[idx]-tmu)/tsd, out[...,1])
            _,_,Pf,Qf = ac_from_state(Vh, THh); Lp = F.smooth_l1_loss(Pf, cPf[idx]) + F.smooth_l1_loss(Qf, cQf[idx])
            (Ls + W_PHYS*Lp).backward(); opt.step(); opt.zero_grad()
        sched.step()
    model.eval(); V, T = [], []
    with torch.no_grad():
        for i in range(0, len(teI), 512):
            j = teI[i:i+512]; x, e, ei, b = feed(torch.as_tensor(j, device=DEV)); out = model(x, e, ei, b)
            V.append((out[...,0]*vsd+vmu).cpu()); T.append(torch.rad2deg(out[...,1]*tsd+tmu).cpu())
    Vh = torch.cat(V); THh = torch.cat(T); tF = fam[teI]
    Vtr = Vtrue[teI].cpu(); THtr = THtrue_deg[teI].cpu()
    Vmeas = nx_phys[teI,:,0].cpu(); THmeas = nx_phys[teI,:,3].cpu()
    mV = nm[teI,:,0].cpu().bool(); mTH = nm[teI,:,3].cpu().bool()
    def mae(a,b,m): return (a-b).abs()[m].mean().item()
    bm = (tF==0).numpy(); atk = (tF>0).numpy()
    return dict(estd=estd,
        b_thse=mae(THh[bm], THtr[bm], mTH[bm]), b_thme=mae(THmeas[bm], THtr[bm], mTH[bm]),
        b_vse=mae(Vh[bm], Vtr[bm], mV[bm]), b_vme=mae(Vmeas[bm], Vtr[bm], mV[bm]),
        a_thse=mae(THh[atk], THtr[atk], mTH[atk]), a_thme=mae(THmeas[atk], THtr[atk], mTH[atk]))

print(f"IEEE-{C} diagnostic (seed {SEED}, {EPOCHS} ep). Empirical injected-noise std per channel [V pu, P MW, Q MVAr, th deg]:")
print(f"{'c%':>5} {'wp':>4} | {'noiseStd(V,P,Q,th)':>34} | {'benign thSE':>11} {'thMeter':>8} {'benign VSE':>10} {'Vmeter':>8} | {'atk thSE':>9} {'atk thMe':>9}")
for AC in [1.0, 0.5, 0.2, 0.1]:
    for wp in [0.0, 0.3]:
        R = run(AC, wp)
        es = ",".join(f"{x:.4f}" for x in R["estd"])
        print(f"{AC:>5} {wp:>4} | [{es:>32}] | {R['b_thse']:>11.4f} {R['b_thme']:>8.4f} {R['b_vse']:>10.5f} {R['b_vme']:>8.5f} | {R['a_thse']:>9.4f} {R['a_thme']:>9.4f}", flush=True)
