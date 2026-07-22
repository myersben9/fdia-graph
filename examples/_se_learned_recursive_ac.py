#!/usr/bin/env python
"""FULL-AC seconds-cadence recursive state estimation (companion to the DC study).

The DC script `_se_learned_recursive_case.py` runs Static-WLS / classical Kalman / learned-GRU over a
DC-LINEAR angle-only state at 2 s cadence. This script lifts that SAME comparison to the FULL AC model so
the recursive/win-both result matches the single-snapshot part (which is already full AC, see
`_se_pinn_v040.py`). Nothing in the DC script is overwritten; results go to a new file.

WHAT CHANGES vs the DC version
  - State is (theta, |V|) at every non-slack bus (dim 2*NS), not angle-only. Slack angle and slack |V| are
    fixed to the operating point (known reference), exactly as in a real AC SE.
  - The measurement model is the FULL AC operator h(V,theta) reused verbatim from _se_pinn_v040.py
    (makeYbus, complex Ybus with shunts / line charging / taps): bus P,Q injections (all buses), branch
    P,Q flows (from-side, all branches), and |V| at every non-slack bus. Per-unit throughout.
  - The classical recursive baseline becomes an EXTENDED KALMAN FILTER: predict by state persistence
    (random-walk, x_pred = x_prev, P_pred = P + Q), update by linearising h through its makeYbus Jacobian
    and applying the Kalman correction on the AC innovation z - h(x_pred). The static AC-WLS (memoryless
    Gauss-Newton per scan from a flat start) is kept as the weak baseline.
  - The learned filter is the identical fused-GRU residual-on-the-recursive-estimate design as DC, now
    over the AC state: GRU([ekf, wls, wls-ekf]) -> correction added to the EKF estimate.

HONEST MODELLING NOTE (Jacobian linearisation point). At 2 s cadence the state moves < ~2 deg / < ~0.01
p.u. between scans and the attack ramp is <= 1.8 deg, so the AC measurement Jacobian is essentially
constant over a sequence. We therefore linearise h ONCE per sequence at that sequence's nominal operating
point and hold H fixed across its L steps (a constant-Jacobian / linearised-KF, a standard EKF variant),
which is what makes IEEE-118 tractable at seconds cadence. The measurement INNOVATION z - h(x) is always
the full NONLINEAR AC residual (h recomputed every step); only the gain's linearisation is frozen. Both
cases use the identical method so the 14-vs-118 comparison is apples-to-apples, and the DC-vs-AC
comparison differs only in (nonlinear AC h + 2-channel state) vs (linear h + angle-only).

Report: angle MAE (deg) and voltage MAE (p.u.) for Static AC-WLS, AC-EKF, Learned-AC, benign and under
the stealthy temporal ramp, IEEE-14 (primary) and IEEE-118 (if feasible), seeds 123/124/125.
Output: results/se_learned_recursive_ac_ieee{case}.json.
Env: SE_CASE (14|118, default 14), SE_SEEDS (123,124,125), SMOKE (0/1), CPU (0/1)."""
import os, sys, json, time, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
from scipy.interpolate import CubicSpline
import torch, torch.nn as nn
from torch.func import jacrev, vmap

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
REL = os.path.join(HERE, "release_v0.4.1")
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
CASE = int(os.environ.get("SE_CASE", "14"))
SEEDS = tuple(int(s) for s in os.environ.get("SE_SEEDS", "123,124,125").split(","))
SMOKE = os.environ.get("SMOKE", "0") == "1"
torch.manual_seed(123); np.random.seed(123)
TO_DEG = 180.0 / np.pi

# ---------------- AC physics setup (replicates _se_pinn_v040.py's makeYbus machinery) ----------------
NET = {14: pn.case14, 118: pn.case118, 300: pn.case300}[CASE]
base = NET(); pp.runpp(base)
_ppc = base._ppc; _Ybus, _Yf, _Yt = makeYbus(_ppc["baseMVA"], _ppc["bus"], _ppc["branch"])
bMVA = float(_ppc["baseMVA"]); nppc = _ppc["bus"].shape[0]
C = CASE
lut = base._pd2ppc_lookups["bus"][:C].astype(np.int64)                 # pp bus -> ppc index
fb = _ppc["branch"][:, 0].real.astype(np.int64)                        # ppc from-bus per branch
# Physics + classical-estimator linear algebra run in float64 / complex128: the AC measurement Jacobian
# has condition number ~1e5 on IEEE-118, and squared inside the EKF innovation covariance that exceeds
# float32 precision (solve reports "singular"). The learned GRU keeps float32 (features cast on the way in).
DT = torch.float64
Ybus = torch.as_tensor(np.asarray(_Ybus.todense()), dtype=torch.complex128, device=DEV)
Yf = torch.as_tensor(np.asarray(_Yf.todense()), dtype=torch.complex128, device=DEV)
LUT = torch.as_tensor(lut, device=DEV); FB = torch.as_tensor(fb, device=DEV)
E = int(Yf.shape[0])
slack = int(base.ext_grid.bus.values[0])                               # pp slack bus
keep = [i for i in range(C) if i != slack]; NS = len(keep)
KEEP = torch.as_tensor(keep, device=DEV, dtype=torch.long)
SD = 2 * NS                                                            # state dim: [theta(keep), V(keep)]


def ac_meas_batch(Vmag, theta_rad):
    """Full AC operator (identical math to _se_pinn_v040.ac_from_state), per-unit.
    (Vmag,theta)[B,C] -> (Pb,Qb)[B,C] bus inj, (Pf,Qf)[B,E] from-side branch flows."""
    b = Vmag.shape[0]
    Vc_pp = torch.polar(Vmag, theta_rad)
    idxL = LUT.unsqueeze(0).expand(b, C)                                        # out-of-place scatter (vmap-safe)
    Vr = torch.zeros(b, nppc, device=Vmag.device, dtype=Vmag.dtype).scatter(1, idxL, Vc_pp.real)
    Vi = torch.zeros(b, nppc, device=Vmag.device, dtype=Vmag.dtype).scatter(1, idxL, Vc_pp.imag)
    Vc = torch.complex(Vr, Vi)
    Sbus = Vc * torch.conj(Vc @ Ybus.T)
    Sf = Vc[:, FB] * torch.conj(Vc @ Yf.T)
    return Sbus.real[:, LUT], Sbus.imag[:, LUT], Sf.real, Sf.imag


# measurement layout: [Pinj(C), Qinj(C), Pf(E), Qf(E), |V|(NS)] ; M = 2C+2E+NS
M = 2 * C + 2 * E + NS
sig = np.concatenate([np.full(C, 0.02), np.full(C, 0.02), np.full(E, 0.02), np.full(E, 0.02), np.full(NS, 0.005)]).astype(np.float32)
SIG = torch.as_tensor(sig, device=DEV, dtype=DT)
Wdiag = (1.0 / SIG ** 2)                                               # WLS weights
Rdiag = SIG ** 2                                                       # EKF measurement noise cov (diag)


def state_to_full(x, Vsl, thsl):
    """x[B,2NS] + per-seq fixed slack (Vsl[B], thsl[B]) -> full (V,theta)[B,C] in pp order.
    Built out-of-place with scatter so it is safe under torch.func.vmap (in-place indexed writes are not)."""
    B = x.shape[0]
    idx = KEEP.unsqueeze(0).expand(B, NS)
    sidx = torch.full((B, 1), slack, device=x.device, dtype=torch.long)
    th = torch.zeros(B, C, device=x.device, dtype=x.dtype).scatter(1, idx, x[:, :NS]).scatter(1, sidx, thsl.reshape(B, 1))
    V = torch.zeros(B, C, device=x.device, dtype=x.dtype).scatter(1, idx, x[:, NS:]).scatter(1, sidx, Vsl.reshape(B, 1))
    return V, th


def h_batch(x, Vsl, thsl):
    """AC measurement prediction for a batch of states. x[B,2NS] -> zhat[B,M]."""
    V, th = state_to_full(x, Vsl, thsl)
    Pb, Qb, Pf, Qf = ac_meas_batch(V, th)
    return torch.cat([Pb, Qb, Pf, Qf, V[:, KEEP]], dim=1)


def jac_at(xnom, Vsl, thsl):
    """Per-sample AC measurement Jacobian dh/dx at xnom, via makeYbus autograd. -> H[B,M,2NS].
    Reverse-mode jacrev (cost ~ M backward passes) is fine for IEEE-14; for 118 M is larger but this
    is called ONCE per sequence (constant-Jacobian EKF), not once per timestep, so it stays cheap."""
    def h1(xi, vi, ti):
        return h_batch(xi[None], vi[None], ti[None])[0]
    return vmap(jacrev(h1))(xnom, Vsl, thsl)


# ---------------- DC susceptance block (order-consistent) for the fast-load angle response ----------------
# The P-injection-vs-theta block of the AC Jacobian at the flat point (V=1,theta=0) IS the DC bus
# susceptance reduced to the free buses. We pull it straight from the AC Jacobian so bus ordering is
# guaranteed consistent with the AC operator (avoids the ppc-vs-pp reindex footgun).
with torch.no_grad():
    _x0 = torch.zeros(1, SD, device=DEV, dtype=DT); _x0[:, NS:] = 1.0
    _J0 = jac_at(_x0, torch.ones(1, device=DEV, dtype=DT), torch.zeros(1, device=DEV, dtype=DT))[0]   # [M,2NS]
A_dc = _J0[:C, :NS][KEEP].cpu().numpy().astype(np.float64)                        # dPinj_keep/dtheta_keep
POOLX = np.load(os.path.join(REL, f"pool_ieee{CASE}.npz"))["X"].astype(np.float32)  # [T,C,4]=[P,Q,|V|,th deg]
load_pp = base.load.bus.values.astype(int)
base_load_pu = base.load.p_mw.values / bMVA
ok = np.abs(base_load_pu) > 0                                                     # drop reactive-only loads
load_pp, base_load_pu = load_pp[ok], base_load_pu[ok]
keep_pos = {b: i for i, b in enumerate(keep)}

CAD, ANCHOR_DT = 2.0, 300.0
SIG_F, TAU = 0.004, 30.0; PHI = np.exp(-CAD / TAU); SF = SIG_F * np.sqrt(1 - PHI ** 2)   # calibrated fast-load OU
SIG_VF = 0.0005                                                                  # small fast |V| band (p.u.)
NA = 6                                                                           # anchors per sequence (~25 min)
L = 256 if not SMOKE else 64
print(f"IEEE-{CASE}: NS={NS} free buses, state dim {SD}, M={M} measurements ({2*C+2*E} AC + {NS} |V|), "
      f"E={E} branches, device {DEV}", flush=True)


def make_sequence(rs, attack):
    """One benign or ramp-attacked 2 s AC sequence.
    Returns x_true[L,2NS], z[L,M] (AC measurements with meter noise), Vsl, thsl (fixed slack refs)."""
    t0 = int(rs.integers(0, POOLX.shape[0] - NA - 1))
    thA = np.deg2rad(POOLX[t0:t0 + NA, keep, 3].astype(float))                   # anchor angles (rad) at free buses
    VA = POOLX[t0:t0 + NA, keep, 2].astype(float)                                # anchor |V| at free buses
    ta = np.arange(NA) * ANCHOR_DT; tt = np.arange(L) * CAD
    th_slow = np.stack([CubicSpline(ta, thA[:, j])(tt) for j in range(NS)], 1)
    V_slow = np.stack([CubicSpline(ta, VA[:, j])(tt) for j in range(NS)], 1)
    # fast load OU -> angle response through the DC sensitivity (order-consistent A_dc); small |V| OU
    oul = np.zeros((L, len(base_load_pu)))
    for t in range(1, L): oul[t] = PHI * oul[t - 1] + rs.normal(0, SF, len(base_load_pu))
    dP = np.zeros((L, NS))
    for li, b in enumerate(load_pp):
        if b in keep_pos: dP[:, keep_pos[b]] += -oul[:, li] * base_load_pu[li]
    th_fast = np.linalg.solve(A_dc, dP.T).T
    ouv = np.zeros((L, NS))
    for t in range(1, L): ouv[t] = PHI * ouv[t - 1] + rs.normal(0, SIG_VF * np.sqrt(1 - PHI ** 2), NS)
    th_true = th_slow + th_fast; V_true = V_slow + ouv
    x_true = np.concatenate([th_true, V_true], axis=1).astype(np.float32)        # [L,2NS]
    meas_th = th_true.copy()
    if attack:                                                                   # stealthy angle ramp on 1-3 buses
        nb = int(rs.integers(1, 4)); ab = rs.choice(NS, nb, replace=False)
        a0 = int(rs.integers(int(L * 0.3), int(L * 0.6))); dur = int(rs.integers(int(30 / CAD), int(120 / CAD)))
        mag = rs.uniform(np.deg2rad(0.6), np.deg2rad(1.8))
        prog = np.clip((np.arange(L) - a0) / dur, 0, 1)
        for b in ab: meas_th[:, b] += prog * mag
    Vsl = float(POOLX[t0, slack, 2]); thsl = float(np.deg2rad(POOLX[t0, slack, 3]))
    # build AC measurements from the (attacked-in-angle) meter state
    xt = torch.as_tensor(np.concatenate([meas_th, V_true], 1), dtype=DT, device=DEV)
    with torch.no_grad():
        zc = h_batch(xt, torch.full((L,), Vsl, device=DEV, dtype=DT), torch.full((L,), thsl, device=DEV, dtype=DT)).cpu().numpy()
    z = zc + rs.normal(0, sig, (L, M))
    # nominal operating point (the operator's base case = first-scan slow state); the EKF/WLS Jacobian is
    # linearised here and the memoryless WLS is warm-started here. Fixed across the sequence -> not temporal.
    xnom = np.concatenate([th_slow[0], V_slow[0]]).astype(np.float64)
    return x_true.astype(np.float32), z.astype(np.float32), Vsl, thsl, xnom.astype(np.float32)


# ---------------- static AC-WLS (memoryless Gauss-Newton per scan, constant Jacobian) ----------------
def wls_batch(z, Vsl, thsl, Hs, xnom, iters=8):
    """z[B,L,M] -> xhat[B,L,2NS]. Solve the nonlinear AC-WLS per scan, warm-started at the per-sequence
    nominal xnom[B,2NS]; Jacobian held at that nominal Hs[B,M,2NS] (chord-Newton). Scans are independent,
    so vectorise over L. Memoryless: the same fixed nominal seeds every scan (no cross-scan filtering)."""
    B = z.shape[0]
    Gs = torch.linalg.solve((Hs.transpose(1, 2) * Wdiag) @ Hs, Hs.transpose(1, 2) * Wdiag)   # [B,2NS,M] gain
    zf = z.reshape(B * L, M)
    Vsf = Vsl.repeat_interleave(L); thf = thsl.repeat_interleave(L)
    x = xnom.repeat_interleave(L, dim=0).clone()
    Gf = Gs.repeat_interleave(L, dim=0)                                          # [B*L,2NS,M]
    for _ in range(iters):
        r = zf - h_batch(x, Vsf, thf)
        x = x + torch.bmm(Gf, r.unsqueeze(-1)).squeeze(-1)
    return x.reshape(B, L, SD)


# ---------------- classical AC-EKF (random-walk predict, AC innovation, constant-Jacobian gain) --------
def ekf_batch(z, Vsl, thsl, Hs, xnom, Qdiag):
    """z[B,L,M] -> xhat[B,L,2NS]. Extended Kalman filter over the AC state: predict x_pred=x_prev,
    P_pred=P+Q; update K = P_pred H^T (H P_pred H^T + R)^-1 on the nonlinear innovation z-h(x_pred).
    Initialised at the per-sequence nominal xnom (the base case); Jacobian linearised there."""
    B = z.shape[0]
    Ht = Hs.transpose(1, 2)
    I = torch.eye(SD, device=DEV, dtype=DT).expand(B, SD, SD).contiguous()
    Rm = torch.diag(Rdiag).expand(B, M, M)
    P = torch.diag(torch.full((SD,), 1.0, device=DEV, dtype=DT)).expand(B, SD, SD).contiguous()
    x = xnom.clone()
    out = torch.empty(B, L, SD, device=DEV, dtype=DT)
    Qm = torch.diag(Qdiag)
    for t in range(L):
        P = P + Qm                                                              # predict (random walk)
        S = Hs @ P @ Ht + Rm
        K = torch.linalg.solve(S, Hs @ P).transpose(1, 2)                       # [B,2NS,M]
        r = z[:, t, :] - h_batch(x, Vsl, thsl)                                  # nonlinear AC innovation
        x = x + torch.bmm(K, r.unsqueeze(-1)).squeeze(-1)
        P = (I - K @ Hs) @ P
        out[:, t, :] = x
    return out


# ---------------- learned residual-on-EKF filter (same fused-GRU design as the DC version) --------------
class LearnedFilter(nn.Module):
    def __init__(self, n, hid=96):
        super().__init__()
        self.gru = nn.GRU(3 * n, hid, batch_first=True)
        self.res = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, n))

    def forward(self, ekf_n, wls_n):                                            # normalised [B,L,2NS]
        x = torch.cat([ekf_n, wls_n, wls_n - ekf_n], -1)
        h, _ = self.gru(x)
        return self.res(h)                                                      # correction in normalised state units


def build_features(seqs):
    """seqs: list of (x_true,z,Vsl,thsl,xnom). -> tensors X_true,EKF,WLS [n,L,2NS] on DEV (float32 for NN).
    Classical estimators run in float64 (ill-conditioned AC Jacobian); features cast to float32 on return."""
    n = len(seqs)
    Xt = torch.as_tensor(np.stack([s[0] for s in seqs]), device=DEV, dtype=torch.float32)
    Z = torch.as_tensor(np.stack([s[1] for s in seqs]), device=DEV, dtype=DT)
    Vsl = torch.as_tensor(np.array([s[2] for s in seqs]), device=DEV, dtype=DT)
    thsl = torch.as_tensor(np.array([s[3] for s in seqs]), device=DEV, dtype=DT)
    xnom = torch.as_tensor(np.stack([s[4] for s in seqs]), device=DEV, dtype=DT)
    Hs = jac_at(xnom, Vsl, thsl)                                              # linearise at the base case
    EKF = torch.empty(n, L, SD, device=DEV, dtype=DT); WLS = torch.empty(n, L, SD, device=DEV, dtype=DT)
    Qd = torch.as_tensor(QDIAG, device=DEV, dtype=DT)
    CH = 64 if CASE < 100 else 16                                              # chunk sequences to bound memory
    for i in range(0, n, CH):
        sl = slice(i, i + CH)
        WLS[sl] = wls_batch(Z[sl], Vsl[sl], thsl[sl], Hs[sl], xnom[sl])
        EKF[sl] = ekf_batch(Z[sl], Vsl[sl], thsl[sl], Hs[sl], xnom[sl], Qd)
    return Xt, EKF.float(), WLS.float()


def make_seqs(rs, n, mode="mixed"):
    """mode: 'mixed' (half attacked, for training), True (all attacked), False (all benign, for eval)."""
    if mode == "mixed":
        return [make_sequence(rs, attack=(i % 2 == 0)) for i in range(n)]
    return [make_sequence(rs, attack=bool(mode)) for _ in range(n)]


# calibrate EKF process-noise Q from benign state increments (per channel), like the DC Kalman calibration
_rc = np.random.default_rng(555)
_cal = [make_sequence(_rc, attack=False) for _ in range(24)]
_incr = np.concatenate([np.diff(s[0], axis=0) for s in _cal], 0)
QDIAG = np.maximum(_incr.var(0), 1e-10).astype(np.float32)                      # [2NS]
STATE_STD = np.concatenate([s[0] for s in _cal], 0).std(0).clip(1e-4).astype(np.float32)   # per-channel scale
SS = torch.as_tensor(STATE_STD, device=DEV)


def to_norm(x):  return x / SS
def from_norm(x): return x * SS


def eval_seqs(model, seqs, window=False):
    Xt, EKF, WLS = build_features(seqs)
    with torch.no_grad():
        corr = from_norm(model(to_norm(EKF), to_norm(WLS)))
        LN = EKF + corr
    if window:
        s = slice(int(L * 0.7), L); Xt, EKF, WLS, LN = Xt[:, s], EKF[:, s], WLS[:, s], LN[:, s]

    def chan(est):
        d = (est - Xt).abs()
        th = d[..., :NS].mean().item() * TO_DEG                                 # angle MAE deg
        vv = d[..., NS:].mean().item()                                          # voltage MAE p.u.
        return round(th, 4), round(vv, 5)
    (tw, vw) = chan(WLS); (te, ve) = chan(EKF); (tl, vl) = chan(LN)
    return dict(th_wls=tw, th_ekf=te, th_learned=tl, v_wls=vw, v_ekf=ve, v_learned=vl)


def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ntr, nva = (512, 96) if not SMOKE else (48, 24)
    tr_seqs = make_seqs(np.random.default_rng(seed), ntr)
    va_seqs = make_seqs(np.random.default_rng(seed + 876), nva)
    TXt, TE, TW = build_features(tr_seqs); VXt, VE, VW = build_features(va_seqs)
    TXt_n, TE_n, TW_n = to_norm(TXt), to_norm(TE), to_norm(TW)
    VE_n, VW_n = to_norm(VE), to_norm(VW)
    model = LearnedFilter(SD).to(DEV); opt = torch.optim.Adam(model.parameters(), 2e-3)
    EP = 600 if not SMOKE else 60
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP)
    print(f"[seed {seed}] {sum(p.numel() for p in model.parameters()):,} params, train {ntr} seqs, {EP} epochs", flush=True)
    for ep in range(EP):
        perm = torch.randperm(TXt.shape[0], device=DEV); model.train(); tot = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            out = TE_n[idx] + model(TE_n[idx], TW_n[idx])                        # normalised EKF + correction
            loss = ((out - TXt_n[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if (ep + 1) % 150 == 0:
            with torch.no_grad():
                vout = from_norm(VE_n + model(VE_n, VW_n)); vmae = (vout - VXt).abs()[..., :NS].mean().item() * TO_DEG
            print(f"  [seed {seed}] ep {ep+1} train MSE {tot/(len(perm)//64+1):.3e} val angle MAE {vmae:.4f} deg", flush=True)
    return model


per_seed = []
for _seed in SEEDS:
    t0 = time.time(); m = train_one(_seed)
    be = eval_seqs(m, make_seqs(np.random.default_rng(2024), 120, mode=False))
    _atk_seqs = make_seqs(np.random.default_rng(2025), 120, mode=True)
    at = eval_seqs(m, _atk_seqs)
    atw = eval_seqs(m, _atk_seqs, window=True)
    rec = dict(seed=_seed,
               benign=dict(th_wls=be["th_wls"], th_ekf=be["th_ekf"], th_learned=be["th_learned"],
                           v_wls=be["v_wls"], v_ekf=be["v_ekf"], v_learned=be["v_learned"]),
               attacked=dict(th_wls=at["th_wls"], th_ekf=at["th_ekf"], th_learned=at["th_learned"],
                             v_wls=at["v_wls"], v_ekf=at["v_ekf"], v_learned=at["v_learned"],
                             atk_th_wls=atw["th_wls"], atk_th_ekf=atw["th_ekf"], atk_th_learned=atw["th_learned"],
                             atk_v_wls=atw["v_wls"], atk_v_ekf=atw["v_ekf"], atk_v_learned=atw["v_learned"]))
    per_seed.append(rec)
    print(f"[seed {_seed}] ({time.time()-t0:.0f}s) {json.dumps(rec)}", flush=True)


def agg(grp, k):
    vals = [d[grp][k] for d in per_seed if k in d[grp]]
    return dict(mean=round(float(np.mean(vals)), 5), std=round(float(np.std(vals)), 5), n=len(vals))


mk = ([("benign", k) for k in ("th_wls", "th_ekf", "th_learned", "v_wls", "v_ekf", "v_learned")] +
      [("attacked", k) for k in ("th_wls", "th_ekf", "th_learned", "v_wls", "v_ekf", "v_learned",
                                 "atk_th_wls", "atk_th_ekf", "atk_th_learned", "atk_v_wls", "atk_v_ekf", "atk_v_learned")])
res = dict(system=f"ieee{CASE}", model="AC-SE (EKF + learned residual)", cadence_s=CAD, seeds=list(SEEDS),
           device=DEV, n_free_buses=NS, state_dim=SD, n_meas=M, classical_recursive="AC-EKF (constant-Jacobian)",
           per_seed=per_seed, aggregate={f"{g}.{k}": agg(g, k) for (g, k) in mk},
           benign=per_seed[0]["benign"], attacked=per_seed[0]["attacked"], seed=SEEDS[0])
outp = os.path.join(RES, f"se_learned_recursive_ac_ieee{CASE}.json")
json.dump(res, open(outp, "w"), indent=2)
print(json.dumps(res["aggregate"], indent=2), flush=True)
print(f"wrote {outp}", flush=True)
