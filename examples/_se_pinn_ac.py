#!/usr/bin/env python
"""Accuracy-class noise variant of _se_pinn_v040.py.

IDENTICAL model / w_phys / metric definitions / training to _se_pinn_v040.py. The ONLY changes:
  (1) reads the NOISE-FREE regenerated shard release_v0.4.1/ml_only_ieee{C}_clean.h5 (built by _gen_clean.py:
      exact h(x)/h(Xsolved) signal + attacks, zero meter noise), and
  (2) adds ACCURACY-CLASS heteroscedastic meter noise at load time (the noise KNOB), per Asprou/Falas
      instrument accuracy class AC (percent):
        - P_inj, Q_inj, P_flow, Q_flow : sigma = (AC/100)*|magnitude|/3  (+1e-3 floor, matches generator)
        - |V|                          : sigma = (AC/100)*|V|/3           (relative; |V|~1 -> ~AC/300 pu)
        - theta                        : sigma = AC * 0.096 deg           (absolute IEC/PMU angle class,
                                         anchored so AC=1.0% reproduces the flat-shard 0.096 deg angle noise)
      Noise is applied only at metered positions (node_m / edge_m). This lets us sweep the noise level
      WITHOUT re-solving power flows; the attack signal is baked into the clean shard and preserved.

Everything downstream (per-unit conversion, standardization, the physics operator, the metrics, seeds) is
byte-for-byte the flat-sweep pipeline. Output schema matches results/sweep_pinn/ exactly.
Env: CASE, W_PHYS, AC_CLASS (percent), SEED, EPOCHS (80), SMOKE. Output:
results/sweep_pinn_ac/se_{C}_ac{AC}_wp{W_PHYS}_s{SEED}.json."""
import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, h5py
import pandapower as pp, pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from torch_geometric.nn import ARMAConv
from fdia_graph.dataset import FAMILIES

DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
C = int(os.environ.get("CASE", "14")); W_PHYS = float(os.environ.get("W_PHYS", "0.2"))
SEED = int(os.environ.get("SEED", "123")); AC = float(os.environ.get("AC_CLASS", "1.0"))
EPOCHS = int(os.environ.get("EPOCHS", "80")); SMOKE = os.environ.get("SMOKE", "0") == "1"
HID = int(os.environ.get("HID", "256")); COSINE = os.environ.get("COSINE", "1") == "1"
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_v0.4.1")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
H5 = os.path.join(REL, f"ml_only_ieee{C}_clean.h5"); POOL = os.path.join(REL, f"pool_ieee{C}.npz")
print(f"[data] reading {H5}  (accuracy class AC={AC}%)", flush=True)

# ---- physics setup (replicates the generator's Ybus machinery; deterministic from the case) ----
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
    Pb = Sbus.real[:, LUT]; Qb = Sbus.imag[:, LUT]
    return Pb, Qb, Sf.real, Sf.imag


# ---- load dataset + true states (v0.4.1) ----
with h5py.File(H5, "r") as f:
    d = f["data"]; A = {k: d[k][:] for k in ("node_x", "node_m", "edge_x", "edge_m", "y", "family", "timestep", "split")}
    baseMVA = float(f.attrs.get("baseMVA", bMVA))
Nrec = len(A["node_x"]); N = A["node_x"].shape[1]; E = A["edge_x"].shape[1]
assert abs(baseMVA - bMVA) < 1e-6, f"h5 baseMVA {baseMVA} != pandapower net baseMVA {bMVA} -- case mismatch"

# ---- ACCURACY-CLASS NOISE KNOB: re-noise the clean (noise-free) measurements at instrument class AC% ----
# Seeded independently of the model init so the noise realization is reproducible and identical across w_phys.
_nrng = np.random.default_rng(1_000_000 * SEED + int(round(AC * 1000)))
nxc = A["node_x"].astype(np.float64); exc = A["edge_x"].astype(np.float64)
r = (AC / 100.0) / 3.0
sig_node = np.stack([
    r * np.abs(nxc[..., 0]),                    # |V| : relative
    r * np.abs(nxc[..., 1]) + 1e-3,             # P_inj : relative + MW floor
    r * np.abs(nxc[..., 2]) + 1e-3,             # Q_inj : relative + MVAr floor
    np.full_like(nxc[..., 3], AC * 0.096),      # theta : absolute IEC/PMU class (deg), AC=1% -> 0.096 deg
], -1)
sig_edge = np.stack([r * np.abs(exc[..., 0]) + 1e-3, r * np.abs(exc[..., 1]) + 1e-3], -1)
A["node_x"] = (nxc + sig_node * _nrng.standard_normal(nxc.shape) * A["node_m"]).astype(np.float32)
A["edge_x"] = (exc + sig_edge * _nrng.standard_normal(exc.shape) * A["edge_m"]).astype(np.float32)

POOLX = np.load(POOL)["X"].astype(np.float32)
Xtrue = POOLX[A["timestep"].astype(np.int64)]
print(f"IEEE-{C}: {Nrec} records, N={N} E={E}, pool T={len(POOLX)}, w_phys={W_PHYS}, AC={AC}%, hid={HID}, "
      f"epochs={EPOCHS}, cosine={COSINE}, baseMVA={baseMVA}, units=pu")

# ---- physics operator validation (per-unit): h(x_true) must reproduce benign measurements within meter noise ----
ben = np.where((A["family"] == 0))[0][:512]
with torch.no_grad():
    Vt = torch.as_tensor(Xtrue[ben, :, 2], device=DEV); TH = torch.deg2rad(torch.as_tensor(Xtrue[ben, :, 3], device=DEV))
    Pb, Qb, Pf, Qf = ac_from_state(Vt, TH)
    mPi = torch.as_tensor(A["node_m"][ben, :, 1], device=DEV).bool()
    meas_Pi_pu = torch.as_tensor(A["node_x"][ben, :, 1], device=DEV) / baseMVA
    err_inj = (Pb - meas_Pi_pu)[mPi].abs().mean().item()
    mF = torch.as_tensor(A["edge_m"][ben, :, 0], device=DEV).bool()
    meas_Pf_pu = torch.as_tensor(A["edge_x"][ben, :, 0], device=DEV) / baseMVA
    err_flow = (Pf - meas_Pf_pu)[mF].abs().mean().item()
print(f"physics-op validation (benign, p.u.): inj sign-agnostic; flow MAE {err_flow:.5f}  (should be << 1 p.u.)")

# ---- tensors (PER-UNIT: node_x/edge_x P,Q divided by baseMVA; theta channels in RADIANS) ----
sp = A["split"]
nx_phys = torch.as_tensor(A["node_x"], device=DEV, dtype=torch.float32)
nx = nx_phys.clone(); nx[..., 1] /= baseMVA; nx[..., 2] /= baseMVA; nx[..., 3] = torch.deg2rad(nx[..., 3])
nm = torch.as_tensor(A["node_m"], device=DEV, dtype=torch.float32)
ex = torch.as_tensor(A["edge_x"], device=DEV, dtype=torch.float32) / baseMVA
em = torch.as_tensor(A["edge_m"], device=DEV, dtype=torch.float32)
Vtrue = torch.as_tensor(Xtrue[:, :, 2], device=DEV)
THtrue_deg = torch.as_tensor(Xtrue[:, :, 3], device=DEV)
THtrue = torch.deg2rad(THtrue_deg)
fam = torch.as_tensor(A["family"].astype(np.int64))
ei0 = torch.stack([
    torch.as_tensor(np.r_[base.line["from_bus"].values, base.trafo["hv_bus"].values], dtype=torch.long),
    torch.as_tensor(np.r_[base.line["to_bus"].values, base.trafo["lv_bus"].values], dtype=torch.long)], 0).to(DEV)
ei_bi = torch.cat([ei0, ei0.flip(0)], 1)
tr = np.where(sp == 0)[0]; vaI = np.where(sp == 1)[0]; teI = np.where(sp == 2)[0]
if SMOKE: tr = tr[:1024]

with torch.no_grad():
    cPb, cQb, cPf, cQf = [], [], [], []
    for i in range(0, Nrec, 2048):
        a, bb, c, dd = ac_from_state(Vtrue[i:i+2048], THtrue[i:i+2048])
        cPb.append(a); cQb.append(bb); cPf.append(c); cQf.append(dd)
    cPb = torch.cat(cPb); cQb = torch.cat(cQb); cPf = torch.cat(cPf); cQf = torch.cat(cQf)


def std2(X, M, idx):
    w = M[idx].sum((0, 1)).clamp(min=1); mu = (X[idx] * M[idx]).sum((0, 1)) / w
    sd = (((X[idx] - mu) ** 2 * M[idx]).sum((0, 1)) / w).sqrt().clamp(min=1e-3); return (X - mu) / sd * M, mu, sd
NXn, _, _ = std2(nx, nm, tr); EXn, _, _ = std2(ex, em, tr)
vmu, vsd = Vtrue[tr].mean(), Vtrue[tr].std().clamp(min=1e-3); tmu, tsd = THtrue[tr].mean(), THtrue[tr].std().clamp(min=1e-3)


def feed(idx):
    b = len(idx); x = torch.cat([NXn[idx], nm[idx]], -1).reshape(b * N, -1)
    e2 = torch.cat([EXn[idx], em[idx]], -1); e = torch.cat([e2, e2], 1).reshape(b * 2 * E, -1)
    off = (torch.arange(b, device=DEV) * N).repeat_interleave(ei_bi.shape[1]); ei = ei_bi.repeat(1, b) + off.unsqueeze(0)
    return x, e, ei, b


class SE(nn.Module):
    def __init__(self, hid=HID):
        super().__init__(); self.nenc = nn.Linear(8, hid); self.eenc = nn.Linear(4, hid)
        self.b1 = ARMAConv(hid, hid, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.05, act=F.relu)
        self.b2 = ARMAConv(hid, hid, num_stacks=3, num_layers=2, shared_weights=True, dropout=0.05, act=F.relu)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 2))
    def forward(self, x, e, ei, b):
        h = F.relu(self.nenc(x)); he = F.relu(self.eenc(e)); h = h + torch.zeros_like(h).index_add_(0, ei[1], he)
        h = self.b1(h, ei); h = self.b2(h, ei); return self.head(h).reshape(b, N, 2)


torch.manual_seed(SEED); model = SE().to(DEV); opt = torch.optim.Adam(model.parameters(), 2e-3, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS) if COSINE else None
BS = 128 if C >= 300 else 256
print(f"train {len(tr)}  w_phys={W_PHYS}  AC={AC}%  epochs={EPOCHS}  hid={HID}  cosine={COSINE}  batch={BS}  params {sum(p.numel() for p in model.parameters()):,}")
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
    if (ep + 1) % 10 == 0 or SMOKE: print(f"  epoch {ep+1} L_state {Ls.item():.4f} L_phys {Lp.item():.4f}")


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


print(f"\n=== STATE-ESTIMATION ERROR on TEST (|V| p.u., theta deg) AC={AC}% ===")
res = {"system": f"ieee{C}", "w_phys": W_PHYS, "seed": SEED, "hid": HID, "epochs": EPOCHS, "units": "pu",
       "release": "v0.4.1", "ac_class_pct": AC, "per_family": {}}
for k, name in FAMILIES.items():
    m = (tF == k).numpy()
    if not m.any(): continue
    vAll = mae(Vh[m], Vtr[m])
    mv = mV[m]; mt = mTH[m]
    vseM = mae(Vh[m][mv], Vtr[m][mv]); vmeM = mae(Vmeas[m][mv], Vtr[m][mv])
    tseM = mae(THh[m][mt], THtr[m][mt]); tmeM = mae(THmeas[m][mt], THtr[m][mt])
    res["per_family"][name] = dict(V_mae_all=round(vAll, 4), V_mae_se_metered=round(vseM, 4),
                                   V_mae_meter=round(vmeM, 4), th_mae_se_metered=round(tseM, 3), th_mae_meter=round(tmeM, 3),
                                   V_mae_meter_attacked=round(vmeM, 4), V_mae_se_metered_attacked=round(vseM, 4),
                                   th_mae_meter_attacked=round(tmeM, 3), th_mae_se_metered_attacked=round(tseM, 3))
    print(f"{name:8s} | Vse {vseM:.4f} Vmeter {vmeM:.4f} | thse {tseM:.3f} thmeter {tmeM:.3f}")
atk = (tF > 0).numpy(); mvA = mV[atk]; mtA = mTH[atk]
res["overall"] = dict(
    V_mae_all=round(mae(Vh, Vtr), 4),
    V_mae_se_metered_attacked=round(mae(Vh[atk][mvA], Vtr[atk][mvA]), 4),
    V_mae_meter_attacked=round(mae(Vmeas[atk][mvA], Vtr[atk][mvA]), 4),
    th_mae_se_metered_attacked=round(mae(THh[atk][mtA], THtr[atk][mtA]), 3),
    th_mae_meter_attacked=round(mae(THmeas[atk][mtA], THtr[atk][mtA]), 3))
o = res["overall"]
print(f"\nATTACKED metered: |V| SE {o['V_mae_se_metered_attacked']:.4f} vs meter {o['V_mae_meter_attacked']:.4f} p.u. | "
      f"theta SE {o['th_mae_se_metered_attacked']:.3f} vs meter {o['th_mae_meter_attacked']:.3f} deg")
if not SMOKE:
    outp = os.path.join(RES, "sweep_pinn_ac", f"se_{C}_ac{AC}_wp{W_PHYS}_s{SEED}.json")
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    json.dump(res, open(outp, "w"), indent=2); print(f"wrote {outp}")
