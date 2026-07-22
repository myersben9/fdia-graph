#!/usr/bin/env python
"""Case-parameterized copy of _se_learned_recursive.py so the seconds-cadence learned recursive
estimator can be evaluated beyond IEEE-14 (reviewer ask: "no evidence the recursive estimator
scales beyond IEEE-14"). Pick the system with SE_CASE=14|118 (default 118). Everything else --
the DC measurement model, the calibrated seconds-cadence fast-load OU, the classical steady-state
Kalman baseline, the learned residual-on-Kalman GRU filter, the multi-seed train/eval -- is IDENTICAL
in structure to the IEEE-14 script; only the network, the free-angle count and the pool file change.

The DC model build mirrors _se_resolution_sensitivity.py's build() (already proven on 14/118/300):
full injections + full branch flows + 65% PMU angle meters, accuracy-class-ish fixed sigmas.

Output: results/se_learned_recursive_ieee{case}.json (+ per-seed, aggregate with error bars).
The IEEE-14 numbers stay in the original se_learned_recursive.json; this writes a case-suffixed file
so nothing is overwritten. Seeds 123,124,125."""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from scipy.interpolate import CubicSpline
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
CASE = int(os.environ.get("SE_CASE", "118"))
NETS = {14: pn.case14, 118: pn.case118, 300: pn.case300}
SEEDS = tuple(int(s) for s in os.environ.get("SE_SEEDS", "123,124,125").split(","))
torch.manual_seed(123); np.random.seed(123)

# ---------------- DC measurement model (same recipe as the resolution sweep's build()) ----------------
net = NETS[CASE](); pp.rundcpp(net); ppc = net._ppc; br = ppc["branch"]; bus = ppc["bus"]
NBp = bus.shape[0]; fb = br[:, 0].real.astype(int); tb = br[:, 1].real.astype(int); bl = 1.0 / br[:, 3].real
B = np.zeros((NBp, NBp))
for f, t, b in zip(fb, tb, bl):
    B[f, f] += b; B[t, t] += b; B[f, t] -= b; B[t, f] -= b
slack = int(np.where(bus[:, 1] == 3)[0][0]); keep = [i for i in range(NBp) if i != slack]; NS = len(keep)
pos = {b: i for i, b in enumerate(keep)}
rows, kinds = [], []
for i in range(NBp): rows.append(B[i, keep]); kinds.append("inj")
for (f, t, b) in zip(fb, tb, bl):
    r = np.zeros(NS)
    if f in keep: r[pos[f]] += b
    if t in keep: r[pos[t]] -= b
    rows.append(r); kinds.append("flow")
pmu = sorted(np.random.default_rng(7).choice(keep, int(round(0.65 * NS)), replace=False))
for i in pmu:
    r = np.zeros(NS); r[pos[i]] = 1.0; rows.append(r); kinds.append("pmu")
H = np.array(rows); sig = np.array([0.02 if k != "pmu" else 0.005 for k in kinds])
GinvHtW = np.linalg.solve((H.T * (1 / sig ** 2)) @ H, H.T * (1 / sig ** 2))
Bred = B[np.ix_(keep, keep)]
POOLX = np.load(os.path.join(HERE, "release_v0.4.1", f"pool_ieee{CASE}.npz"))["X"].astype(np.float32)
baseMVA = float(ppc["baseMVA"]); load_ppc = net._pd2ppc_lookups["bus"][net.load.bus.values].astype(int)
base_load_pu = net.load.p_mw.values / baseMVA
ok = np.abs(base_load_pu) > 0                                   # drop reactive-only / zero-P load buses
load_ppc, base_load_pu = load_ppc[ok], base_load_pu[ok]
CAD, ANCHOR_DT = 2.0, 300.0
SIG_F, TAU = 0.004, 30.0; PHI = np.exp(-CAD / TAU); SF = SIG_F * np.sqrt(1 - PHI ** 2)   # calibrated fast OU
NA = 6                                                          # anchors per sequence (~25 min)
L = 256                                                         # steps per sequence (~8.5 min)
print(f"IEEE-{CASE}: {NS} free angles, {H.shape[0]} measurements, {len(base_load_pu)} load buses, device {DEV}", flush=True)


def make_sequence(rs, attack):
    """One benign or ramp-attacked seconds-cadence sequence. Returns theta_true[L,NS], theta_wls[L,NS]."""
    t0 = rs.integers(0, POOLX.shape[0] - NA - 1)
    th_anchor = np.deg2rad(POOLX[t0:t0 + NA, keep, 3].astype(float))
    ta = np.arange(NA) * ANCHOR_DT; tt = np.arange(L) * CAD
    th_slow = np.stack([CubicSpline(ta, th_anchor[:, j])(tt) for j in range(NS)], 1)
    oul = np.zeros((L, len(base_load_pu)))
    for t in range(1, L): oul[t] = PHI * oul[t - 1] + rs.normal(0, SF, len(base_load_pu))
    dP = np.zeros((L, NBp))
    for li, b in enumerate(load_ppc): dP[:, b] += -oul[:, li] * base_load_pu[li]
    th_true = th_slow + np.linalg.solve(Bred, dP[:, keep].T).T
    meas_state = th_true.copy()
    if attack:                                                 # stealthy ramp on 1-3 buses
        nb = rs.integers(1, 4); ab = rs.choice(NS, nb, replace=False)
        a0 = rs.integers(int(L * 0.3), int(L * 0.6)); dur = rs.integers(int(30 / CAD), int(120 / CAD))
        mag = rs.uniform(np.deg2rad(0.6), np.deg2rad(1.8))
        prog = np.clip((np.arange(L) - a0) / dur, 0, 1)
        for b in ab: meas_state[:, b] += prog * mag
    z = meas_state @ H.T + rs.normal(0, sig, (L, H.shape[0]))
    th_wls = z @ GinvHtW.T
    return th_true.astype(np.float32), th_wls.astype(np.float32)


# ---------------- fixed global steady-state Kalman (the benign optimum baseline the learned model builds on) --------
def kalman_gain(q, r):
    K = np.zeros(NS)
    for i in range(NS):
        P = r[i]
        for _ in range(400): Pm = P + q[i]; K[i] = Pm / (Pm + r[i]); P = (1 - K[i]) * Pm
    return K

_rc = np.random.default_rng(555); _ct, _cw = [], []
for _ in range(48):
    tr, wl = make_sequence(_rc, attack=False); _ct.append(tr); _cw.append(wl)
Q = np.mean([np.diff(t, axis=0).var(axis=0) for t in _ct], 0)
Rn = np.mean([np.maximum((np.diff(w, axis=0).var(axis=0) - np.diff(t, axis=0).var(axis=0)) / 2, 1e-12) for t, w in zip(_ct, _cw)], 0)
Kg = kalman_gain(Q, Rn)


def kf_np(wl):
    out = wl.copy()
    for t in range(1, len(wl)): out[t] = out[t - 1] + Kg * (wl[t] - out[t - 1])
    return out


# ---------------- learned RESIDUAL on the Kalman ----------------
# Same architecture as the IEEE-14 sandbox (a causal GRU producing a per-step hidden state, then a
# residual head added to the Kalman estimate) but expressed with the fused cuDNN nn.GRU instead of a
# per-timestep GRUCell python loop. nn.GRU stacks the identical GRU update equations, so the model is
# mathematically the same recurrent filter; it is ~120x faster and makes the 117-angle IEEE-118 run
# tractable. Both cases here (14 and 118) use this SAME implementation so the scale comparison is
# apples-to-apples; the original GRUCell IEEE-14 numbers remain untouched in se_learned_recursive.json.
class LearnedFilter(nn.Module):
    def __init__(self, n, hid=96):
        super().__init__()
        self.gru = nn.GRU(3 * n, hid, batch_first=True)
        self.res = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, n))

    def forward(self, z, kf):                                  # z, kf [B,L,n]
        x = torch.cat([kf, z, z - kf], -1)                     # kalman estimate, measurement, their gap
        h, _ = self.gru(x)                                     # [B,L,hid] causal hidden states
        return kf + self.res(h)                                # kalman + learned attack-rejection correction


def make_set(n, seed):
    rs = np.random.default_rng(seed); T, W, F = [], [], []
    for i in range(n):
        tr, wl = make_sequence(rs, attack=(i % 2 == 0))
        T.append(tr); W.append(wl); F.append(kf_np(wl))
    return (torch.tensor(np.array(T)), torch.tensor(np.array(W)), torch.tensor(np.array(F)))


def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    TRt, TRw, TRf = [x.to(DEV) for x in make_set(512, seed)]
    VAt, VAw, VAf = [x.to(DEV) for x in make_set(96, seed + 876)]
    model = LearnedFilter(NS).to(DEV); opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
    print(f"[seed {seed}] device {DEV}; residual-on-Kalman filter ({sum(p.numel() for p in model.parameters()):,} params), train {TRt.shape[0]} seqs", flush=True)
    for ep in range(600):
        perm = torch.randperm(TRt.shape[0], device=DEV)
        model.train(); tot = 0.0
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]
            out = model(TRw[idx], TRf[idx]); loss = ((out - TRt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        sched.step()
        if (ep + 1) % 150 == 0:
            with torch.no_grad():
                vmae = (model(VAw, VAf) - VAt).abs().mean().item() * 180 / np.pi
            print(f"  [seed {seed}] epoch {ep+1}  train MSE {tot/(len(perm)//64+1):.2e}  val MAE {vmae:.4f} deg", flush=True)
    return model


def eval_set(model, attack, nseq=120):
    rs = np.random.default_rng(2024 + int(attack)); to_deg = 180 / np.pi
    e_wls, e_kf, e_ln, p_wls, p_kf, p_ln = [], [], [], [], [], []
    for _ in range(nseq):
        tr, wl = make_sequence(rs, attack=attack); kf = kf_np(wl)
        with torch.no_grad():
            ln = model(torch.tensor(wl[None]).to(DEV), torch.tensor(kf[None]).to(DEV)).cpu().numpy()[0]
        e_wls.append(np.abs(wl - tr).mean() * to_deg); e_kf.append(np.abs(kf - tr).mean() * to_deg); e_ln.append(np.abs(ln - tr).mean() * to_deg)
        if attack:
            s = slice(int(L * 0.7), L)
            p_wls.append(np.abs(wl[s] - tr[s]).mean() * to_deg); p_kf.append(np.abs(kf[s] - tr[s]).mean() * to_deg); p_ln.append(np.abs(ln[s] - tr[s]).mean() * to_deg)
    d = dict(th_wls=round(float(np.mean(e_wls)), 4), th_kalman=round(float(np.mean(e_kf)), 4), th_learned=round(float(np.mean(e_ln)), 4))
    if attack: d.update(atk_wls=round(float(np.mean(p_wls)), 4), atk_kalman=round(float(np.mean(p_kf)), 4), atk_learned=round(float(np.mean(p_ln)), 4))
    return d


per_seed = []
for _seed in SEEDS:
    m = train_one(_seed)
    rs_d = dict(seed=_seed, benign=eval_set(m, False), attacked=eval_set(m, True))
    per_seed.append(rs_d)
    print(f"[seed {_seed}] {json.dumps(rs_d)}", flush=True)


def agg(key_path):
    grp, k = key_path
    vals = [d[grp][k] for d in per_seed if k in d[grp]]
    return dict(mean=round(float(np.mean(vals)), 4), std=round(float(np.std(vals)), 4), n=len(vals))


metric_keys = [("benign", "th_wls"), ("benign", "th_kalman"), ("benign", "th_learned"),
               ("attacked", "th_wls"), ("attacked", "th_kalman"), ("attacked", "th_learned"),
               ("attacked", "atk_wls"), ("attacked", "atk_kalman"), ("attacked", "atk_learned")]
res = dict(system=f"ieee{CASE}", model="DC-SE (BLUE)", cadence_s=CAD, seeds=list(SEEDS), device=DEV,
           n_free_angles=NS, n_meas=int(H.shape[0]),
           per_seed=per_seed, aggregate={f"{g}.{k}": agg((g, k)) for (g, k) in metric_keys},
           benign=per_seed[0]["benign"], attacked=per_seed[0]["attacked"], seed=SEEDS[0])
json.dump(res, open(os.path.join(RES, f"se_learned_recursive_ieee{CASE}.json"), "w"), indent=2)
print(json.dumps(res["aggregate"], indent=2), flush=True)
print(f"wrote results/se_learned_recursive_ieee{CASE}.json", flush=True)
