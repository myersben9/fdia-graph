#!/usr/bin/env python
"""Part B experiment harness: identical to _se_pinn_v040.py EXCEPT the SE architecture / training budget
are parametrized via env so we can diagnose the 300 benign-angle floor without touching the frozen script.
Extra env: HID (128), ARMA_LAYERS (2), BLOCKS (2), COSINE (0/1), TAG (exp). Writes results/se_{C}_exp_{TAG}.json.
Everything else (data source, true-state, metric, seed 123) is byte-identical to _se_pinn_v040.py."""
import os, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, h5py
import pandapower as pp, pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from torch_geometric.nn import ARMAConv
from fdia_graph.dataset import FAMILIES

DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
C = int(os.environ.get("CASE", "300")); W_PHYS = float(os.environ.get("W_PHYS", "0.2"))
EPOCHS = int(os.environ.get("EPOCHS", "60")); SMOKE = os.environ.get("SMOKE", "0") == "1"
HID = int(os.environ.get("HID", "128")); ARMA_LAYERS = int(os.environ.get("ARMA_LAYERS", "2"))
BLOCKS = int(os.environ.get("BLOCKS", "2")); COSINE = os.environ.get("COSINE", "0") == "1"
TAG = os.environ.get("TAG", "exp")
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_v0.4.0")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
H5 = os.path.join(REL, f"ml_only_ieee{C}.h5"); POOL = os.path.join(REL, f"pool_ieee{C}.npz")

NET = {14: pn.case14, 118: pn.case118, 300: pn.case300}[C]
base = NET(); pp.runpp(base)
_ppc = base._ppc; _Ybus, _Yf, _Yt = makeYbus(_ppc["baseMVA"], _ppc["bus"], _ppc["branch"]); bMVA = _ppc["baseMVA"]
lut = base._pd2ppc_lookups["bus"][:C].astype(np.int64)
fb = _ppc["branch"][:, 0].real.astype(np.int64); nppc = _ppc["bus"].shape[0]
Ybus = torch.as_tensor(np.asarray(_Ybus.todense()), dtype=torch.complex64, device=DEV)
Yf = torch.as_tensor(np.asarray(_Yf.todense()), dtype=torch.complex64, device=DEV)
LUT = torch.as_tensor(lut, device=DEV); FB = torch.as_tensor(fb, device=DEV)


def ac_from_state(Vmag, theta_deg):
    b = Vmag.shape[0]; Vc_pp = torch.polar(Vmag, torch.deg2rad(theta_deg))
    Vc = torch.zeros(b, nppc, dtype=torch.complex64, device=Vmag.device); Vc[:, LUT] = Vc_pp
    Sbus = Vc * torch.conj(Vc @ Ybus.T) * bMVA
    Sf = Vc[:, FB] * torch.conj(Vc @ Yf.T) * bMVA
    return Sbus.real[:, LUT], Sbus.imag[:, LUT], Sf.real, Sf.imag


with h5py.File(H5, "r") as f:
    d = f["data"]; A = {k: d[k][:] for k in ("node_x", "node_m", "edge_x", "edge_m", "y", "family", "timestep", "split")}
Nrec = len(A["node_x"]); N = A["node_x"].shape[1]; E = A["edge_x"].shape[1]
POOLX = np.load(POOL)["X"].astype(np.float32)
Xtrue = POOLX[A["timestep"].astype(np.int64)]
print(f"IEEE-{C}: {Nrec} recs N={N} E={E} | HID={HID} ARMA_LAYERS={ARMA_LAYERS} BLOCKS={BLOCKS} "
      f"EPOCHS={EPOCHS} w_phys={W_PHYS} cosine={COSINE} tag={TAG}", flush=True)

sp = A["split"]
nx = torch.as_tensor(A["node_x"], device=DEV, dtype=torch.float32); nm = torch.as_tensor(A["node_m"], device=DEV, dtype=torch.float32)
ex = torch.as_tensor(A["edge_x"], device=DEV, dtype=torch.float32); em = torch.as_tensor(A["edge_m"], device=DEV, dtype=torch.float32)
Vtrue = torch.as_tensor(Xtrue[:, :, 2], device=DEV); THtrue = torch.as_tensor(Xtrue[:, :, 3], device=DEV)
fam = torch.as_tensor(A["family"].astype(np.int64))
ei0 = torch.stack([
    torch.as_tensor(np.r_[base.line["from_bus"].values, base.trafo["hv_bus"].values], dtype=torch.long),
    torch.as_tensor(np.r_[base.line["to_bus"].values, base.trafo["lv_bus"].values], dtype=torch.long)], 0).to(DEV)
ei_bi = torch.cat([ei0, ei0.flip(0)], 1)
tr = np.where(sp == 0)[0]; vaI = np.where(sp == 1)[0]; teI = np.where(sp == 2)[0]
if SMOKE: tr = tr[:1024]


def std2(X, M, idx):
    w = M[idx].sum((0, 1)).clamp(min=1); mu = (X[idx] * M[idx]).sum((0, 1)) / w
    sd = (((X[idx] - mu) ** 2 * M[idx]).sum((0, 1)) / w).sqrt().clamp(min=1e-3); return (X - mu) / sd * M, mu, sd
NXn, _, _ = std2(nx, nm, tr); EXn, _, _ = std2(ex, em, tr)
vmu, vsd = Vtrue[tr].mean(), Vtrue[tr].std().clamp(min=1e-3); tmu, tsd = THtrue[tr].mean(), THtrue[tr].std().clamp(min=1e-3)

with torch.no_grad():
    cPb, cQb, cPf, cQf = [], [], [], []
    for i in range(0, Nrec, 2048):
        a, bb, c, dd = ac_from_state(Vtrue[i:i+2048], THtrue[i:i+2048])
        cPb.append(a); cQb.append(bb); cPf.append(c); cQf.append(dd)
    cPb = torch.cat(cPb); cQb = torch.cat(cQb); cPf = torch.cat(cPf); cQf = torch.cat(cQf)


def feed(idx):
    b = len(idx); x = torch.cat([NXn[idx], nm[idx]], -1).reshape(b * N, -1)
    e2 = torch.cat([EXn[idx], em[idx]], -1); e = torch.cat([e2, e2], 1).reshape(b * 2 * E, -1)
    off = (torch.arange(b, device=DEV) * N).repeat_interleave(ei_bi.shape[1]); ei = ei_bi.repeat(1, b) + off.unsqueeze(0)
    return x, e, ei, b


class SE(nn.Module):
    def __init__(self, hid=HID, arma_layers=ARMA_LAYERS, blocks=BLOCKS):
        super().__init__(); self.nenc = nn.Linear(8, hid); self.eenc = nn.Linear(4, hid)
        self.blocks = nn.ModuleList([ARMAConv(hid, hid, num_stacks=3, num_layers=arma_layers,
                                              shared_weights=True, dropout=0.05, act=F.relu) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))
    def forward(self, x, e, ei, b):
        h = F.relu(self.nenc(x)); he = F.relu(self.eenc(e)); h = h + torch.zeros_like(h).index_add_(0, ei[1], he)
        for blk in self.blocks: h = blk(h, ei)
        return self.head(h).reshape(b, N, 2)


torch.manual_seed(123); model = SE().to(DEV); opt = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS) if COSINE else None
BS = 128 if C >= 300 else 256
print(f"train {len(tr)}  params {sum(p.numel() for p in model.parameters()):,}", flush=True)
t0 = time.time()
for ep in range(EPOCHS if not SMOKE else 2):
    te0 = time.time(); model.train(); perm = tr[torch.randperm(len(tr)).numpy()]
    for i in range(0, len(perm), BS):
        idx = torch.as_tensor(perm[i:i + BS], device=DEV); x, e, ei, b = feed(idx)
        out = model(x, e, ei, b); Vh = out[..., 0] * vsd + vmu; THh = out[..., 1] * tsd + tmu
        Ls = F.smooth_l1_loss((Vtrue[idx] - vmu) / vsd, out[..., 0]) + F.smooth_l1_loss((THtrue[idx] - tmu) / tsd, out[..., 1])
        Pb, Qb, Pf, Qf = ac_from_state(Vh, THh)
        Lp = (F.smooth_l1_loss(Pf, cPf[idx]) + F.smooth_l1_loss(Qf, cQf[idx])) / bMVA
        loss = Ls + W_PHYS * Lp
        opt.zero_grad(); loss.backward(); opt.step()
    if sched: sched.step()
    if ep == 0 or (ep + 1) % 10 == 0 or SMOKE:
        # quick val angle at metered buses (benign+all) to watch plateau
        print(f"  epoch {ep+1}/{EPOCHS} L_state {Ls.item():.4f} L_phys {Lp.item():.4f} "
              f"({time.time()-te0:.1f}s/ep)", flush=True)
print(f"train wall {time.time()-t0:.1f}s", flush=True)


@torch.no_grad()
def estimate(idx):
    model.eval(); V, T = [], []
    for i in range(0, len(idx), 512):
        j = idx[i:i + 512]; x, e, ei, b = feed(torch.as_tensor(j, device=DEV))
        out = model(x, e, ei, b); V.append((out[..., 0] * vsd + vmu).cpu()); T.append((out[..., 1] * tsd + tmu).cpu())
    return torch.cat(V), torch.cat(T)


Vh, THh = estimate(teI); tF = fam[teI]
Vtr = Vtrue[teI].cpu(); THtr = THtrue[teI].cpu()
Vmeas = nx[teI, :, 0].cpu(); THmeas = nx[teI, :, 3].cpu()
mV = nm[teI, :, 0].cpu().bool(); mTH = nm[teI, :, 3].cpu().bool()


def mae(a, b, mask=None):
    e = (a - b).abs()
    return (e[mask].mean().item() if mask is not None and mask.any() else e.mean().item())


res = {"system": f"ieee{C}", "w_phys": W_PHYS, "seed": 123, "hid": HID, "arma_layers": ARMA_LAYERS,
       "blocks": BLOCKS, "epochs": EPOCHS, "cosine": COSINE, "tag": TAG, "per_family": {}}
print(f"\n{'family':8s} | {'|V| SE':>7s} {'|V| meter':>9s} | {'th SE':>6s} {'th meter':>8s}")
for k, name in FAMILIES.items():
    m = (tF == k).numpy()
    if not m.any(): continue
    mv = mV[m]; mt = mTH[m]
    vseM = mae(Vh[m][mv], Vtr[m][mv]); vmeM = mae(Vmeas[m][mv], Vtr[m][mv])
    tseM = mae(THh[m][mt], THtr[m][mt]); tmeM = mae(THmeas[m][mt], THtr[m][mt])
    res["per_family"][name] = dict(V_mae_se_metered=round(vseM, 4), V_mae_meter=round(vmeM, 4),
                                   th_mae_se_metered=round(tseM, 3), th_mae_meter=round(tmeM, 3))
    print(f"{name:8s} | {vseM:7.4f} {vmeM:9.4f} | {tseM:6.3f} {tmeM:8.3f}")
atk = (tF > 0).numpy(); mvA = mV[atk]; mtA = mTH[atk]
res["overall"] = dict(
    V_mae_se_metered_attacked=round(mae(Vh[atk][mvA], Vtr[atk][mvA]), 4),
    V_mae_meter_attacked=round(mae(Vmeas[atk][mvA], Vtr[atk][mvA]), 4),
    th_mae_se_metered_attacked=round(mae(THh[atk][mtA], THtr[atk][mtA]), 3),
    th_mae_meter_attacked=round(mae(THmeas[atk][mtA], THtr[atk][mtA]), 3))
b = res["per_family"]["benign"]
print(f"\n>>> BENIGN angle: SE {b['th_mae_se_metered']:.3f} deg vs meter {b['th_mae_meter']:.3f} deg  "
      f"(target: beat meter {b['th_mae_meter']:.3f}, ideally WLS 0.027)")
if not SMOKE:
    os.makedirs(RES, exist_ok=True)
    outp = os.path.join(RES, f"se_{C}_exp_{TAG}.json")
    json.dump(res, open(outp, "w"), indent=2); print(f"wrote {outp}")
