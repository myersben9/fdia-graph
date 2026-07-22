#!/usr/bin/env python
"""Our LOCALIZATION-WINNER recipe applied to the SE regression task.

The v0.4.1 SE model (_se_pinn_v040.py) is plain ARMA (2 blocks + edge fusion, measurements+mask only). Our
localization WINNER is ARMA + a gated GATv2 attention branch + engineered features (KCL power-balance
residual, temporal delta, windowed swing). Those extra ingredients are what pushed localization ahead of the
GAT and the plain MLP, and none of them have been tried on state estimation. This script grafts the full
winner recipe onto the SE regression head (2 outputs, |V| and theta) to test whether the attention branch and
the physics/temporal features rescue the state estimate, especially the IEEE-300 angle channel where the plain
ARMA-SE stalls (~0.65 deg).

Everything else is held identical to _se_pinn_v040.py so the result is directly comparable: same v0.4.1 shards,
same true-state target (pool X[timestep]), same per-unit convention, same physics-flow loss, same metrics, same
seed. Output: results/se_{C}_armaattn.json (schema matches se_{C}.json).
Env: CASE(118) W_PHYS(0.2) EPOCHS(80) HID(256) SMOKE(0/1). Seed 123."""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, h5py
import pandapower as pp, pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from torch_geometric.nn import ARMAConv, GATv2Conv
from fdia_graph.dataset import FAMILIES

DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
C = int(os.environ.get("CASE", "118")); W_PHYS = float(os.environ.get("W_PHYS", "0.2"))
EPOCHS = int(os.environ.get("EPOCHS", "80")); SMOKE = os.environ.get("SMOKE", "0") == "1"
HID = int(os.environ.get("HID", "256")); COSINE = os.environ.get("COSINE", "1") == "1"
FULL = os.environ.get("FULL_AVAIL", "0") == "1"   # diagnostic: meter every channel at every bus (all masks=1)
TEMPORAL = os.environ.get("TEMPORAL", "0") == "1" # feed the previous scan's measurements as a prior (forecasting-aided SE)
REFINE = os.environ.get("REFINE", "0") == "1"     # test-time physics refinement (robust-WLS polish from the learned init)
REF_STEPS = int(os.environ.get("REF_STEPS", "50")); REF_LR = float(os.environ.get("REF_LR", "1.0"))  # LBFGS max_iter / step
REF_C = float(os.environ.get("REF_C", "3.0"))     # Cauchy robustness scale (in sigmas); bad meters beyond ~c get rejected
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_v0.4.1")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
H5 = os.path.join(REL, f"ml_only_ieee{C}.h5"); POOL = os.path.join(REL, f"pool_ieee{C}.npz")
print(f"[data] reading {H5}", flush=True)

# ---- physics setup (identical to _se_pinn_v040.py) ----
NET = {14: pn.case14, 118: pn.case118, 300: pn.case300}[C]
base = NET(); pp.runpp(base)
_ppc = base._ppc; _Ybus, _Yf, _Yt = makeYbus(_ppc["baseMVA"], _ppc["bus"], _ppc["branch"]); bMVA = _ppc["baseMVA"]
lut = base._pd2ppc_lookups["bus"][:C].astype(np.int64)
fb = _ppc["branch"][:, 0].real.astype(np.int64)
nppc = _ppc["bus"].shape[0]
Ybus = torch.as_tensor(np.asarray(_Ybus.todense()), dtype=torch.complex64, device=DEV)
Yf = torch.as_tensor(np.asarray(_Yf.todense()), dtype=torch.complex64, device=DEV)
LUT = torch.as_tensor(lut, device=DEV); FB = torch.as_tensor(fb, device=DEV)


def ac_from_state(Vmag, theta_rad):
    b = Vmag.shape[0]
    Vc_pp = torch.polar(Vmag, theta_rad)
    Vc = torch.zeros(b, nppc, dtype=torch.complex64, device=Vmag.device)
    Vc[:, LUT] = Vc_pp
    Sbus = Vc * torch.conj(Vc @ Ybus.T)
    Sf = Vc[:, FB] * torch.conj(Vc @ Yf.T)
    return Sbus.real[:, LUT], Sbus.imag[:, LUT], Sf.real, Sf.imag


def kcl_residual(node_x, edge_x, ei, N):
    """Per-bus real/reactive power-balance residual from the measurements (inflow - injection). The winner's
    physics feature: ~0 where the state is consistent, spikes at buses whose meters were corrupted."""
    n = node_x.shape[0]; fr, to = ei[0], ei[1]
    inP = torch.zeros(n, N, device=node_x.device); inQ = torch.zeros(n, N, device=node_x.device)
    inP.index_add_(1, to, edge_x[:, :, 0]); inP.index_add_(1, fr, -edge_x[:, :, 0])
    inQ.index_add_(1, to, edge_x[:, :, 1]); inQ.index_add_(1, fr, -edge_x[:, :, 1])
    return torch.stack([inP - node_x[:, :, 1], inQ - node_x[:, :, 2]], -1)      # [n,N,2] (P,Q) residual


def refine_state(V0, T0, meas, masks, sig, steps, lr, c):
    """Test-time physics refinement: a robust-WLS polish initialized at the network estimate (V0,T0). We
    minimize sum_i w_i * rho_i^2 over the state, where rho_i is the noise-normalized measurement residual and
    w_i is a Cauchy IRLS weight that down-weights meters the estimate cannot fit (bad-data rejection). On
    benign data every residual is ~1 sigma so w~1 and it converges to the WLS optimum. Under attack the
    corrupted meters get large residuals, w->0, and the resilient learned init keeps it out of the bad-data
    basin. Uses the differentiable AC operator, so gradients flow through the physics."""
    Vm, Pm, Qm, Tm, Pfm, Qfm = meas; mV, mP, mQ, mT, mPf, mQf = masks; sV, sP, sQ, sT, sPf, sQf = sig
    V = V0.clone().detach().requires_grad_(True); T = T0.clone().detach().requires_grad_(True)
    # LBFGS with a strong-Wolfe line search: quasi-Newton, so it takes well-scaled steps on this stiff
    # nonlinear least-squares (WLS) objective instead of overshooting like first-order Adam.
    opt = torch.optim.LBFGS([V, T], lr=lr, max_iter=steps, line_search_fn="strong_wolfe")

    def term(r, m, s):
        rho = r / s
        with torch.no_grad():
            w = m / (1.0 + (rho / c) ** 2)                                     # Cauchy weight (robust, bad-data rejection)
        return (w * rho ** 2).sum()

    def closure():
        opt.zero_grad()
        Pb, Qb, Pf, Qf = ac_from_state(V, T)
        J = (term(V - Vm, mV, sV) + term(T - Tm, mT, sT) + term(Pb - Pm, mP, sP)
             + term(Qb - Qm, mQ, sQ) + term(Pf - Pfm, mPf, sPf) + term(Qf - Qfm, mQf, sQf))
        J.backward(); return J
    opt.step(closure)
    return V.detach(), T.detach()


# ---- load dataset + true states (v0.4.1); now also swing + temporal_delta (winner features) ----
with h5py.File(H5, "r") as f:
    d = f["data"]
    A = {k: d[k][:] for k in ("node_x", "node_m", "edge_x", "edge_m", "y", "family",
                              "timestep", "split", "temporal_delta", "swing")}
    baseMVA = float(f.attrs.get("baseMVA", bMVA))
Nrec = len(A["node_x"]); N = A["node_x"].shape[1]; E = A["edge_x"].shape[1]
assert abs(baseMVA - bMVA) < 1e-6, f"h5 baseMVA {baseMVA} != net baseMVA {bMVA}"
POOLX = np.load(POOL)["X"].astype(np.float32)
Xtrue = POOLX[A["timestep"].astype(np.int64)]                                 # [Nrec,C,4]=[Pinj,Qinj,|V|,theta(deg)]
print(f"IEEE-{C}: {Nrec} records, N={N} E={E}, w_phys={W_PHYS}, hid={HID}, epochs={EPOCHS}, arch=ARMA+attn+feats")

# ---- per-unit tensors (identical convention to _se_pinn_v040.py) ----
sp = A["split"]
nx_phys = torch.as_tensor(A["node_x"], device=DEV, dtype=torch.float32)
nx = nx_phys.clone(); nx[..., 1] /= baseMVA; nx[..., 2] /= baseMVA; nx[..., 3] = torch.deg2rad(nx[..., 3])
nm = torch.as_tensor(A["node_m"], device=DEV, dtype=torch.float32)
ex = torch.as_tensor(A["edge_x"], device=DEV, dtype=torch.float32) / baseMVA
em = torch.as_tensor(A["edge_m"], device=DEV, dtype=torch.float32)
temporal = torch.as_tensor(A["temporal_delta"], device=DEV, dtype=torch.float32)   # [Nrec,N,2]
swing = torch.as_tensor(A["swing"], device=DEV, dtype=torch.float32)               # [Nrec,N,2]
Vtrue = torch.as_tensor(Xtrue[:, :, 2], device=DEV)
THtrue_deg = torch.as_tensor(Xtrue[:, :, 3], device=DEV); THtrue = torch.deg2rad(THtrue_deg)

if FULL:
    # FULL-AVAILABILITY DIAGNOSTIC: meter every channel at every bus. We rebuild the measurement tensors from
    # the TRUE state plus Gaussian noise whose per-channel sigma is measured from the dataset's own benign
    # metered residuals, so the noise level matches the realistic model and only the AVAILABILITY changes.
    torch.manual_seed(123)
    Pinj_pu = torch.as_tensor(Xtrue[:, :, 0], device=DEV) / baseMVA
    Qinj_pu = torch.as_tensor(Xtrue[:, :, 1], device=DEV) / baseMVA
    Pf_t, Qf_t = [], []
    with torch.no_grad():
        for i in range(0, Nrec, 2048):
            _, _, pf, qf = ac_from_state(Vtrue[i:i + 2048], THtrue[i:i + 2048]); Pf_t.append(pf); Qf_t.append(qf)
    Pf_t = torch.cat(Pf_t); Qf_t = torch.cat(Qf_t)
    benm = torch.as_tensor(A["family"] == 0, device=DEV)                       # benign records for noise calibration

    def sig(meas, true, mask):                                                # meter-noise std from benign metered entries
        m = mask.bool() & benm[:, None]; d = (meas - true)[m]
        return d.std().clamp(min=1e-6).item() if d.numel() > 10 else 0.01
    sV, sP, sQ, sT = sig(nx[..., 0], Vtrue, nm[..., 0]), sig(nx[..., 1], Pinj_pu, nm[..., 1]), \
        sig(nx[..., 2], Qinj_pu, nm[..., 2]), sig(nx[..., 3], THtrue, nm[..., 3])
    sPf = sig(ex[..., 0], Pf_t, em[..., 0]); sQf = sig(ex[..., 1], Qf_t, em[..., 1])
    nx = torch.stack([Vtrue + torch.randn_like(Vtrue) * sV, Pinj_pu + torch.randn_like(Pinj_pu) * sP,
                      Qinj_pu + torch.randn_like(Qinj_pu) * sQ, THtrue + torch.randn_like(THtrue) * sT], -1)
    nm = torch.ones_like(nm)
    ex = torch.stack([Pf_t + torch.randn_like(Pf_t) * sPf, Qf_t + torch.randn_like(Qf_t) * sQf], -1)
    em = torch.ones_like(em)
    nx_phys = torch.stack([nx[..., 0], nx[..., 1] * baseMVA, nx[..., 2] * baseMVA, torch.rad2deg(nx[..., 3])], -1)
    print(f"[FULL_AVAIL] every channel metered at every bus. noise sigma  V={sV:.4f}pu P={sP:.4f} Q={sQ:.4f} "
          f"th={np.rad2deg(sT):.3f}deg  flow(P/Q)={sPf:.4f}/{sQf:.4f}", flush=True)

fam = torch.as_tensor(A["family"].astype(np.int64))
ei0 = torch.stack([
    torch.as_tensor(np.r_[base.line["from_bus"].values, base.trafo["hv_bus"].values], dtype=torch.long),
    torch.as_tensor(np.r_[base.line["to_bus"].values, base.trafo["lv_bus"].values], dtype=torch.long)], 0).to(DEV)
ei_bi = torch.cat([ei0, ei0.flip(0)], 1)
tr = np.where(sp == 0)[0]; vaI = np.where(sp == 1)[0]; teI = np.where(sp == 2)[0]
if SMOKE: tr = tr[:1024]

# precompute clean AC targets for the physics loss (per-unit branch flows), same as _se_pinn_v040.py
with torch.no_grad():
    cPf, cQf = [], []
    for i in range(0, Nrec, 2048):
        _, _, c, dd = ac_from_state(Vtrue[i:i + 2048], THtrue[i:i + 2048]); cPf.append(c); cQf.append(dd)
    cPf = torch.cat(cPf); cQf = torch.cat(cQf)

SIG = None
if REFINE or TEMPORAL:
    # per-channel meter-noise sigma, calibrated from benign metered residuals (the accuracy-class constants a
    # real operator knows); used as the 1/sigma^2 WLS weights (refine) and the previous-scan noise (temporal).
    Pinj_pu_t = torch.as_tensor(Xtrue[:, :, 0], device=DEV) / baseMVA
    Qinj_pu_t = torch.as_tensor(Xtrue[:, :, 1], device=DEV) / baseMVA
    _benm = torch.as_tensor(A["family"] == 0, device=DEV)

    def _sig(meas, true, mask):
        m = mask.bool() & _benm[:, None]; dd = (meas - true)[m]
        return float(dd.std().clamp(min=1e-6)) if dd.numel() > 10 else 0.01
    SIG = (_sig(nx[..., 0], Vtrue, nm[..., 0]), _sig(nx[..., 1], Pinj_pu_t, nm[..., 1]),
           _sig(nx[..., 2], Qinj_pu_t, nm[..., 2]), _sig(nx[..., 3], THtrue, nm[..., 3]),
           _sig(ex[..., 0], cPf, em[..., 0]), _sig(ex[..., 1], cQf, em[..., 1]))
    print(f"[REFINE] noise sigma V={SIG[0]:.4f} P={SIG[1]:.4f} Q={SIG[2]:.4f} th={np.rad2deg(SIG[3]):.3f}deg "
          f"Pf={SIG[4]:.4f} Qf={SIG[5]:.4f}  steps={REF_STEPS} lr={REF_LR} c={REF_C}", flush=True)

# ---- build the extended node-feature block: [V,P,Q,th, kcl(2), temporal(2), swing(2)] then standardize ----
kcl = kcl_residual(nx, ex, ei0, N)                                             # [Nrec,N,2] on the pu measurements
feats_list = [nx, kcl, temporal, swing]; mask_list = [nm, torch.ones(Nrec, N, 6, device=DEV)]
if TEMPORAL:
    # previous scan = pool operating point at timestep-1 (the year is a continuous 15-min series). We feed its
    # measurements as an honest second observation (matched meter noise added, not the clean true state), so
    # the model can denoise across time and pin the unobserved angles from the previous field -- information a
    # single-snapshot WLS never uses. This is the lever to BEAT (not just tie) WLS on benign.
    torch.manual_seed(123)
    prev_ts = torch.clamp(torch.as_tensor(A["timestep"].astype("int64")) - 1, min=0).numpy()
    Xp = POOLX[prev_ts]                                                        # [Nrec,N,4]=[Pinj,Qinj,V,thetadeg]
    pv = torch.stack([torch.as_tensor(Xp[:, :, 2], device=DEV),               # V pu
                      torch.as_tensor(Xp[:, :, 0], device=DEV) / baseMVA,      # P pu
                      torch.as_tensor(Xp[:, :, 1], device=DEV) / baseMVA,      # Q pu
                      torch.deg2rad(torch.as_tensor(Xp[:, :, 3], device=DEV))], -1)   # theta rad
    pv = pv + torch.stack([torch.randn_like(pv[..., 0]) * SIG[0], torch.randn_like(pv[..., 1]) * SIG[1],
                           torch.randn_like(pv[..., 2]) * SIG[2], torch.randn_like(pv[..., 3]) * SIG[3]], -1)
    feats_list.append(pv); mask_list.append(nm)                               # prev scan metered at the same buses
    print(f"[TEMPORAL] previous-scan prior added ({pv.shape[-1]} channels), matched noise", flush=True)
NXfull = torch.cat(feats_list, -1)
NMfull = torch.cat(mask_list, -1)


def std2(X, M, idx):
    w = M[idx].sum((0, 1)).clamp(min=1); mu = (X[idx] * M[idx]).sum((0, 1)) / w
    sd = (((X[idx] - mu) ** 2 * M[idx]).sum((0, 1)) / w).sqrt().clamp(min=1e-3); return (X - mu) / sd * M
NXn = std2(NXfull, NMfull, tr); EXn = std2(ex, em, tr)
vmu, vsd = Vtrue[tr].mean(), Vtrue[tr].std().clamp(min=1e-3)
tmu, tsd = THtrue[tr].mean(), THtrue[tr].std().clamp(min=1e-3)
NODE_IN = NXn.shape[-1] + nm.shape[-1]                                         # 10 + 4 = 14
EDGE_IN = EXn.shape[-1] + em.shape[-1]                                         # 2 + 2 = 4


def feed(idx):
    b = len(idx)
    x = torch.cat([NXn[idx], nm[idx]], -1).reshape(b * N, -1)                  # [b*N, 14]
    e2 = torch.cat([EXn[idx], em[idx]], -1)                                    # [b,E,4]
    e = torch.cat([e2, e2], 1).reshape(b * 2 * E, -1)                          # bidirectional [b*2E,4]
    off = (torch.arange(b, device=DEV) * N).repeat_interleave(ei_bi.shape[1])
    ei = ei_bi.repeat(1, b) + off.unsqueeze(0)
    return x, e, ei, b


class SEArmaAttn(nn.Module):
    """The winner backbone with a regression head: node/edge encode + edge fusion, 2 ARMA blocks, a gated
    GATv2 attention branch (starts closed, opens only if it helps), then a 2-way (|V|, theta) head."""
    def __init__(self, node_in, edge_in, hid=HID):
        super().__init__()
        self.nenc = nn.Linear(node_in, hid); self.eenc = nn.Linear(edge_in, hid)
        self.b1 = ARMAConv(hid, hid, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.05, act=F.relu)
        self.b2 = ARMAConv(hid, hid, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.05, act=F.relu)
        self.gat = GATv2Conv(hid, hid // 4, heads=4, edge_dim=edge_in, dropout=0.05, add_self_loops=False)
        self.gate = nn.Parameter(torch.zeros(1))
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))

    def forward(self, x, e, ei, b):
        h = F.relu(self.nenc(x)); he = F.relu(self.eenc(e))
        h = h + torch.zeros_like(h).index_add_(0, ei[1], he)
        h = self.b1(h, ei); h = self.b2(h, ei)
        h = h + torch.tanh(self.gate) * self.gat(h, ei, edge_attr=e)
        return self.head(h).reshape(b, N, 2)


torch.manual_seed(123); model = SEArmaAttn(NODE_IN, EDGE_IN).to(DEV)
opt = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS) if COSINE else None
BS = 128 if C >= 300 else 256
print(f"train {len(tr)}  node_in={NODE_IN} edge_in={EDGE_IN}  w_phys={W_PHYS}  epochs={EPOCHS}  hid={HID}  "
      f"batch={BS}  params {sum(p.numel() for p in model.parameters()):,}")
for ep in range(EPOCHS if not SMOKE else 3):
    model.train(); perm = tr[torch.randperm(len(tr)).numpy()]
    for i in range(0, len(perm), BS):
        idx = torch.as_tensor(perm[i:i + BS], device=DEV); x, e, ei, b = feed(idx)
        out = model(x, e, ei, b); Vh = out[..., 0] * vsd + vmu; THh = out[..., 1] * tsd + tmu
        Ls = F.smooth_l1_loss((Vtrue[idx] - vmu) / vsd, out[..., 0]) + F.smooth_l1_loss((THtrue[idx] - tmu) / tsd, out[..., 1])
        Pb, Qb, Pf, Qf = ac_from_state(Vh, THh)
        Lp = F.smooth_l1_loss(Pf, cPf[idx]) + F.smooth_l1_loss(Qf, cQf[idx])
        loss = Ls + W_PHYS * Lp
        opt.zero_grad(); loss.backward(); opt.step()
    if sched: sched.step()
    if (ep + 1) % 10 == 0 or SMOKE: print(f"  epoch {ep+1} L_state {Ls.item():.4f} L_phys {Lp.item():.4f} gate {torch.tanh(model.gate).item():+.3f}")


@torch.no_grad()
def estimate(idx):
    model.eval(); V, T = [], []
    for i in range(0, len(idx), 512):
        j = idx[i:i + 512]; x, e, ei, b = feed(torch.as_tensor(j, device=DEV))
        out = model(x, e, ei, b); V.append((out[..., 0] * vsd + vmu).cpu()); T.append((out[..., 1] * tsd + tmu).cpu())
    return torch.cat(V), torch.cat(T)


Vh, THh_rad = estimate(teI); tF = fam[teI]
THh = torch.rad2deg(THh_rad)
Vtr = Vtrue[teI].cpu(); THtr = THtrue_deg[teI].cpu()
Vmeas = nx_phys[teI, :, 0].cpu(); THmeas = nx_phys[teI, :, 3].cpu()
mV = nm[teI, :, 0].cpu().bool(); mTH = nm[teI, :, 3].cpu().bool()


def mae(a, b, mask=None):
    e = (a - b).abs()
    return (e[mask].mean().item() if mask is not None and mask.any() else e.mean().item())


print(f"\n=== STATE-ESTIMATION ERROR on TEST (|V| p.u., theta deg) — ARMA+attn+feats ===")
res = {"system": f"ieee{C}", "w_phys": W_PHYS, "seed": 123, "hid": HID, "epochs": EPOCHS, "units": "pu",
       "release": "v0.4.1", "model": "ARMA+attn+feats", "per_family": {}}
for k, name in FAMILIES.items():
    m = (tF == k).numpy()
    if not m.any(): continue
    vAll = mae(Vh[m], Vtr[m]); mv = mV[m]; mt = mTH[m]
    vseM = mae(Vh[m][mv], Vtr[m][mv]); vmeM = mae(Vmeas[m][mv], Vtr[m][mv])
    tseM = mae(THh[m][mt], THtr[m][mt]); tmeM = mae(THmeas[m][mt], THtr[m][mt])
    res["per_family"][name] = dict(V_mae_all=round(vAll, 4), V_mae_se_metered=round(vseM, 4),
                                   V_mae_meter=round(vmeM, 4), th_mae_se_metered=round(tseM, 3), th_mae_meter=round(tmeM, 3),
                                   V_mae_meter_attacked=round(vmeM, 4), V_mae_se_metered_attacked=round(vseM, 4),
                                   th_mae_meter_attacked=round(tmeM, 3), th_mae_se_metered_attacked=round(tseM, 3))
    print(f"{name:8s} | V(all) {vAll:.4f} | V_se {vseM:.4f} | th_se {tseM:.3f} deg")
atk = (tF > 0).numpy(); mvA = mV[atk]; mtA = mTH[atk]
res["overall"] = dict(
    V_mae_all=round(mae(Vh, Vtr), 4),
    V_mae_se_metered_attacked=round(mae(Vh[atk][mvA], Vtr[atk][mvA]), 4),
    V_mae_meter_attacked=round(mae(Vmeas[atk][mvA], Vtr[atk][mvA]), 4),
    th_mae_se_metered_attacked=round(mae(THh[atk][mtA], THtr[atk][mtA]), 3),
    th_mae_meter_attacked=round(mae(THmeas[atk][mtA], THtr[atk][mtA]), 3))
# FULL REGIME (every test sample, benign+attacked) and BENIGN-only, at metered buses. This is the honest,
# whole-dataset comparison to WLS, not just the attacked-sample resilience view.
bmask = (tF == 0).numpy(); mvB = mV[bmask]; mtB = mTH[bmask]
res["overall_all"] = dict(
    V_mae_se_metered=round(mae(Vh[mV], Vtr[mV]), 4),
    th_mae_se_metered=round(mae(THh[mTH], THtr[mTH]), 3),
    V_mae_meter=round(mae(Vmeas[mV], Vtr[mV]), 4),
    th_mae_meter=round(mae(THmeas[mTH], THtr[mTH]), 3))
res["overall_benign"] = dict(
    V_mae_se_metered=round(mae(Vh[bmask][mvB], Vtr[bmask][mvB]), 4),
    th_mae_se_metered=round(mae(THh[bmask][mtB], THtr[bmask][mtB]), 3))
o = res["overall"]; oa = res["overall_all"]; ob = res["overall_benign"]
print(f"\noverall (attacked): V_se {o['V_mae_se_metered_attacked']:.4f}  th_se {o['th_mae_se_metered_attacked']:.3f} deg")
print(f"overall (ALL/full): V_se {oa['V_mae_se_metered']:.4f}  th_se {oa['th_mae_se_metered']:.3f} deg   "
      f"(benign: V_se {ob['V_mae_se_metered']:.4f}  th_se {ob['th_mae_se_metered']:.3f} deg)")

# UNMETERED buses — the buses that have no direct meter, i.e. what state estimation actually exists to
# recover. WLS has no measurement here and must propagate through noisy flows (near-singular gain where the
# grid is under-observed); a learned prior over the state manifold does not degrade the same way. Same buses,
# same truth, apples-to-apples with the WLS script's new unmetered block. This is the tier-2 test.
umV = ~mV; umTH = ~mTH
def _um(selmask, chan_mask, Hh, Tt):
    s = None if selmask is None else selmask
    cm = chan_mask if s is None else chan_mask[s]
    Hh_ = Hh if s is None else Hh[s]; Tt_ = Tt if s is None else Tt[s]
    return mae(Hh_[cm], Tt_[cm]) if cm.any() else None
res["overall_all_unmetered"] = dict(
    V_mae_se=round(_um(None, umV, Vh, Vtr), 4), th_mae_se=round(_um(None, umTH, THh, THtr), 3))
res["overall_benign_unmetered"] = dict(
    V_mae_se=round(_um(bmask, umV, Vh, Vtr), 4), th_mae_se=round(_um(bmask, umTH, THh, THtr), 3))
res["overall_attacked_unmetered"] = dict(
    V_mae_se=round(_um(atk, umV, Vh, Vtr), 4), th_mae_se=round(_um(atk, umTH, THh, THtr), 3))
ou = res["overall_all_unmetered"]; obu = res["overall_benign_unmetered"]; oau = res["overall_attacked_unmetered"]
nU = int(umTH[0].sum()); nM = int(mTH[0].sum())
print(f"UNMETERED buses (n_um={nU}/{nU+nM}): full th {ou['th_mae_se']:.3f}  benign th {obu['th_mae_se']:.3f}  "
      f"attacked th {oau['th_mae_se']:.3f} deg  |  benign V {obu['V_mae_se']:.4f}  attacked V {oau['V_mae_se']:.4f}")

if REFINE and SIG is not None:
    # polish every test estimate with the robust-WLS refinement, initialized at the network output, then
    # re-score per regime. This is the run that should TIE WLS on benign while keeping the attacked win.
    Vr, Tr = [], []
    for i in range(0, len(teI), 512):
        jj = torch.as_tensor(teI[i:i + 512], device=DEV)
        with torch.no_grad():
            x, e, ei, b = feed(jj); out = model(x, e, ei, b)
            V0 = (out[..., 0] * vsd + vmu).detach(); T0 = (out[..., 1] * tsd + tmu).detach()
        meas = (nx[jj, :, 0], nx[jj, :, 1], nx[jj, :, 2], nx[jj, :, 3], ex[jj, :, 0], ex[jj, :, 1])
        masks = (nm[jj, :, 0], nm[jj, :, 1], nm[jj, :, 2], nm[jj, :, 3], em[jj, :, 0], em[jj, :, 1])
        Vf, Tf = refine_state(V0, T0, meas, masks, SIG, REF_STEPS, REF_LR, REF_C)
        Vr.append(Vf.cpu()); Tr.append(Tf.cpu())
    Vr = torch.cat(Vr); Trd = torch.rad2deg(torch.cat(Tr))

    def rr(sel):
        s = None if sel is None else torch.as_tensor(sel)
        mv = mV if s is None else mV[s]; mt = mTH if s is None else mTH[s]
        Vh_ = Vr if s is None else Vr[s]; Th_ = Trd if s is None else Trd[s]
        Vt_ = Vtr if s is None else Vtr[s]; Tt_ = THtr if s is None else THtr[s]
        return round(mae(Vh_[mv], Vt_[mv]), 4), round(mae(Th_[mt], Tt_[mt]), 3)
    res["refined_all"] = dict(zip(("V_mae_se_metered", "th_mae_se_metered"), rr(None)))
    res["refined_benign"] = dict(zip(("V_mae_se_metered", "th_mae_se_metered"), rr(bmask)))
    res["refined_attacked"] = dict(zip(("V_mae_se_metered", "th_mae_se_metered"), rr(atk)))
    rb = res["refined_benign"]; rk = res["refined_attacked"]; ral = res["refined_all"]
    print(f"REFINED (test-time WLS polish): benign th {rb['th_mae_se_metered']:.3f}  attacked th {rk['th_mae_se_metered']:.3f}  "
          f"full th {ral['th_mae_se_metered']:.3f} deg  |  benign V {rb['V_mae_se_metered']:.4f}  attacked V {rk['V_mae_se_metered']:.4f}")
if not SMOKE:
    os.makedirs(RES, exist_ok=True)
    tag = '_temporal' if TEMPORAL else '_full' if FULL else ''
    outp = os.path.join(RES, f"se_{C}_armaattn{tag}.json")
    json.dump(res, open(outp, "w"), indent=2); print(f"wrote {outp}")
