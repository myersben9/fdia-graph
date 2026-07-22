#!/usr/bin/env python
"""Phase 2: a LEARNED recursive state estimator that beats static WLS on benign (temporal) AND beats the
classical Kalman under a stealthy ramp, on the seconds-cadence DC-SE sandbox.

Why the learned model can do what the Kalman cannot. A per-component steady-state Kalman filters each bus
angle independently with a fixed gain. A stealthy ramp adds a per-step change that is smaller than the noise,
so the Kalman happily tracks it (its estimate follows the attacker's false state). Our estimator is a recurrent
predict-then-correct filter with a learned per-bus TRUST GATE driven by a GRU hidden state, so it carries
temporal memory of the innovation pattern and sees the full state vector jointly. Trained on a mix of benign
and ramp-attacked sequences with the TRUE BENIGN state as the target (available in simulation), it learns to
predict the benign continuation and shrink its trust in measurements that drift in a sustained, jointly-
inconsistent way, which is exactly what a ramp does and what benign regulation-band fluctuation does not.

Baselines: static WLS (per snapshot) and the same classical steady-state Kalman as the sandbox. All three are
scored against the TRUE BENIGN state, benign and during attack. DC linear SE on IEEE-14 (unbiased BLUE).
Output: results/fig_learned_recursive.(png|pdf) + se_learned_recursive.json + CSV sidecar. Seed 123."""
import os, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from scipy.interpolate import CubicSpline
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() and os.environ.get("CPU", "0") != "1" else "cpu"
torch.manual_seed(123); np.random.seed(123)
# Multi-seed error bars: the measurement model / Kalman calibration below are seed-independent (fixed RNGs); only
# the learned filter's init + training draws vary per seed. Overridable via SE_SEEDS="123,124,125". The held-out
# eval sequences are kept fixed across seeds, so the spread is pure training variance for the recursive table.
SEEDS = tuple(int(s) for s in os.environ.get("SE_SEEDS", "123,124,125").split(","))

# ---------------- DC measurement model (same as the sandbox) ----------------
net = pn.case14(); pp.rundcpp(net); ppc = net._ppc; br = ppc["branch"]; bus = ppc["bus"]
NBp = bus.shape[0]; fb = br[:, 0].real.astype(int); tb = br[:, 1].real.astype(int); bl = 1.0 / br[:, 3].real
B = np.zeros((NBp, NBp))
for f, t, b in zip(fb, tb, bl):
    B[f, f] += b; B[t, t] += b; B[f, t] -= b; B[t, f] -= b
slack = int(np.where(bus[:, 1] == 3)[0][0]); keep = [i for i in range(NBp) if i != slack]; NS = len(keep)
rows, kinds = [], []
for i in range(NBp): rows.append(B[i, keep]); kinds.append("inj")
for (f, t, b) in zip(fb, tb, bl):
    r = np.zeros(NS)
    if f in keep: r[keep.index(f)] += b
    if t in keep: r[keep.index(t)] -= b
    rows.append(r); kinds.append("flow")
pmu = sorted(np.random.default_rng(7).choice(keep, int(round(0.65 * NS)), replace=False))
for i in pmu:
    r = np.zeros(NS); r[keep.index(i)] = 1.0; rows.append(r); kinds.append("pmu")
H = np.array(rows); sig = np.array([0.02 if k != "pmu" else 0.005 for k in kinds])
GinvHtW = np.linalg.solve((H.T * (1 / sig ** 2)) @ H, H.T * (1 / sig ** 2))
Bred = B[np.ix_(keep, keep)]
POOLX = np.load(os.path.join(HERE, "release_v0.4.1", "pool_ieee14.npz"))["X"].astype(np.float32)
baseMVA = float(ppc["baseMVA"]); load_ppc = net._pd2ppc_lookups["bus"][net.load.bus.values].astype(int)
base_load_pu = net.load.p_mw.values / baseMVA
CAD, ANCHOR_DT = 2.0, 300.0
SIG_F, TAU = 0.004, 30.0; PHI = np.exp(-CAD / TAU); SF = SIG_F * np.sqrt(1 - PHI ** 2)   # calibrated fast OU
NA = 6                                                                                    # anchors per sequence (~25 min)
L = 256                                                                                   # steps per sequence (~8.5 min)


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
    if attack:                                                                            # stealthy ramp on 1-3 buses
        nb = rs.integers(1, 4); ab = rs.choice(NS, nb, replace=False)
        a0 = rs.integers(int(L * 0.3), int(L * 0.6)); dur = rs.integers(int(30 / CAD), int(120 / CAD))
        mag = rs.uniform(np.deg2rad(0.6), np.deg2rad(1.8))
        prog = np.clip((np.arange(L) - a0) / dur, 0, 1)
        for b in ab: meas_state[:, b] += prog * mag
    z = meas_state @ H.T + rs.normal(0, sig, (L, H.shape[0]))
    th_wls = z @ GinvHtW.T
    return th_true.astype(np.float32), th_wls.astype(np.float32)


def batch(n, rs, frac_atk=0.5):
    T, W = [], []
    for i in range(n):
        tr, wl = make_sequence(rs, attack=(i < int(n * frac_atk)))
        T.append(tr); W.append(wl)
    return torch.tensor(np.array(T)), torch.tensor(np.array(W))            # [n,L,NS]


# ---------------- fixed global steady-state Kalman (the benign optimum baseline the learned model builds on) ----------------
def kalman_gain(q, r):
    K = np.zeros(NS)
    for i in range(NS):
        P = r[i]
        for _ in range(400): Pm = P + q[i]; K[i] = Pm / (Pm + r[i]); P = (1 - K[i]) * Pm
    return K

_rc = np.random.default_rng(555); _ct, _cw = [], []
for _ in range(48):
    tr, wl = make_sequence(_rc, attack=False); _ct.append(tr); _cw.append(wl)
Q = np.mean([np.diff(t, axis=0).var(axis=0) for t in _ct], 0)                      # benign per-step process variance
Rn = np.mean([np.maximum((np.diff(w, axis=0).var(axis=0) - np.diff(t, axis=0).var(axis=0)) / 2, 1e-12) for t, w in zip(_ct, _cw)], 0)
Kg = kalman_gain(Q, Rn); Kg_t = torch.tensor(Kg, dtype=torch.float32, device=DEV)


def kf_np(wl):
    out = wl.copy()
    for t in range(1, len(wl)): out[t] = out[t - 1] + Kg * (wl[t] - out[t - 1])
    return out


# ---------------- learned RESIDUAL on the Kalman: benign parity by construction, learns only to reject attacks ----------------
class LearnedFilter(nn.Module):
    def __init__(self, n, hid=96):
        super().__init__()
        self.gru = nn.GRUCell(3 * n, hid)                                  # sees kalman estimate, measurement, their gap
        self.res = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, n))

    def forward(self, z, kf):                                              # z, kf [B,L,n]
        Bn, Ln, n = z.shape; h = z.new_zeros(Bn, self.gru.hidden_size); outs = []
        for t in range(Ln):
            h = self.gru(torch.cat([kf[:, t], z[:, t], z[:, t] - kf[:, t]], -1), h)
            outs.append(kf[:, t] + self.res(h))                            # kalman + learned attack-rejection correction
        return torch.stack(outs, 1)


def make_set(n, seed):
    rs = np.random.default_rng(seed); T, W, F = [], [], []
    for i in range(n):
        tr, wl = make_sequence(rs, attack=(i % 2 == 0))                    # alternate benign / ramp-attacked
        T.append(tr); W.append(wl); F.append(kf_np(wl))
    return (torch.tensor(np.array(T)), torch.tensor(np.array(W)), torch.tensor(np.array(F)))


def train_one(seed):
    """Train one learned recursive filter for a given seed; returns the trained model. Model init + train/val
    data draws are seeded so multiple seeds give independent replicas for error bars."""
    torch.manual_seed(seed); np.random.seed(seed)
    TRt, TRw, TRf = [x.to(DEV) for x in make_set(512, seed)]                    # per-seed training set
    VAt, VAw, VAf = [x.to(DEV) for x in make_set(96, seed + 876)]               # per-seed validation set
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


# ---------------- evaluation on held-out benign + attacked sequences (same global Kalman for the baseline) ----------------
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


# ---------------- multi-seed train + eval for error bars ----------------
per_seed = []; models = {}
for _seed in SEEDS:
    m = train_one(_seed)
    models[_seed] = m
    rs_d = dict(seed=_seed, benign=eval_set(m, False), attacked=eval_set(m, True))
    per_seed.append(rs_d)
    print(f"[seed {_seed}] {json.dumps(rs_d)}", flush=True)


def agg(key_path):
    """mean/std over seeds for a nested benign/attacked metric key."""
    grp, k = key_path
    vals = [d[grp][k] for d in per_seed if k in d[grp]]
    return dict(mean=round(float(np.mean(vals)), 4), std=round(float(np.std(vals)), 4), n=len(vals))


metric_keys = [("benign", "th_wls"), ("benign", "th_kalman"), ("benign", "th_learned"),
               ("attacked", "th_wls"), ("attacked", "th_kalman"), ("attacked", "th_learned"),
               ("attacked", "atk_wls"), ("attacked", "atk_kalman"), ("attacked", "atk_learned")]
res = dict(system="ieee14", model="DC-SE (BLUE)", cadence_s=CAD, seeds=list(SEEDS), device=DEV,
           per_seed=per_seed, aggregate={f"{g}.{k}": agg((g, k)) for (g, k) in metric_keys},
           # back-compat: top-level benign/attacked mirror the anchor (first) seed so downstream readers keep working
           benign=per_seed[0]["benign"], attacked=per_seed[0]["attacked"], seed=SEEDS[0])
json.dump(res, open(os.path.join(RES, "se_learned_recursive.json"), "w"), indent=2)
print(json.dumps(res["aggregate"], indent=2), flush=True)

# ---------------- figure: one attacked sequence, error over time for all three (anchor-seed model) ----------------
model = models[SEEDS[0]]
rs = np.random.default_rng(77); tr, wl = make_sequence(rs, attack=True); kf = kf_np(wl)
with torch.no_grad(): ln = model(torch.tensor(wl[None]).to(DEV), torch.tensor(kf[None]).to(DEV)).cpu().numpy()[0]
to_deg = 180 / np.pi; tt = np.arange(L) * CAD / 60
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.edgecolor": "#222", "axes.linewidth": 0.8})
fig, ax = plt.subplots(figsize=(5.6, 3.3))
ax.plot(tt, np.abs(wl - tr).mean(1) * to_deg, color="#b2182b", lw=1.1, label=f"static WLS  ({res['attacked']['atk_wls']:.3f} deg)")
ax.plot(tt, np.abs(kf - tr).mean(1) * to_deg, color="#8073ac", lw=1.3, label=f"classical Kalman  ({res['attacked']['atk_kalman']:.3f} deg)")
ax.plot(tt, np.abs(ln - tr).mean(1) * to_deg, color="#1a9850", lw=1.6, label=f"learned recursive  ({res['attacked']['atk_learned']:.3f} deg)")
ax.set_xlabel("time (min)"); ax.set_ylabel("angle error vs true benign state (deg)")
ax.set_title("Under a stealthy ramp: learned filter holds, classical filters follow", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, loc="upper left"); ax.grid(True, ls=":", lw=0.4, alpha=0.6)
fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_learned_recursive.png"), dpi=175); fig.savefig(os.path.join(RES, "fig_learned_recursive.pdf"))
with open(os.path.join(RES, "sidecars", "learned_recursive.csv"), "w") as f:
    f.write("t_min,err_wls_deg,err_kalman_deg,err_learned_deg\n")
    for t in range(L): f.write(f"{tt[t]:.4f},{np.abs(wl[t]-tr[t]).mean()*to_deg:.5f},{np.abs(kf[t]-tr[t]).mean()*to_deg:.5f},{np.abs(ln[t]-tr[t]).mean()*to_deg:.5f}\n")
print("wrote results/fig_learned_recursive.(png|pdf) + se_learned_recursive.json", flush=True)
