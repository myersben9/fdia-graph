#!/usr/bin/env python
"""Seconds-cadence SE sandbox (Phase 1): realistic 2-second state trajectory anchored on the REAL 5-minute
NYISO-driven pool, plus a stealthy ramp attack, evaluated with static WLS and a classical Kalman filter.

Two things to show here.
1. We CAN produce a realistic seconds-resolution state trajectory. The slow trend is real (the pool's own
   5-minute angle trajectory, NYISO-driven), spline-interpolated down to 2 s, with a CALIBRATED fast fluctuation
   added on top: the OU/AR(1) regulation-band load process (RMS 0.4% of load, tau = 30 s), verified in
   se_fast_calibration.json (fast-band RMS 0.40% of load; its 5-min increment 0.56% sits just below the real
   NYISO 5-min increment 0.60% and inside the 1% reserve). This is the asset that lets temporal SE actually
   help, which 5-minute data cannot.
2. At 2 s cadence a classical Kalman filter beats static WLS on BENIGN state estimation (temporal win), but a
   stealthy RAMP attack fools BOTH classical estimators. The Kalman's per-component linear model happily tracks
   a smooth ramp, so its estimate follows the attacker's false state. That is the gap a LEARNED spatiotemporal
   estimator is meant to close (Phase 2), because the ramp violates the joint cross-bus dynamics even while each
   bus looks locally plausible.

DC linear SE on IEEE-14 (unbiased BLUE, so the test is not confounded by AC-solver modeling bias). Output:
results/fig_seconds_sandbox.(png|pdf) + se_seconds_sandbox.json + CSV sidecar. CPU only, seed 123."""
import os, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
rng = np.random.default_rng(123)

# ---- DC measurement model from real IEEE-14 topology (same construction as the recursive PoC) ----
net = pn.case14(); pp.rundcpp(net)
ppc = net._ppc; br = ppc["branch"]; bus = ppc["bus"]
NBp = bus.shape[0]
fb = br[:, 0].real.astype(int); tb = br[:, 1].real.astype(int); bl = 1.0 / br[:, 3].real
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
H = np.array(rows)
sig = np.array([0.02 if k == "inj" else 0.02 if k == "flow" else 0.005 for k in kinds])   # pu, pu, rad
GinvHtW = np.linalg.solve((H.T * (1 / sig ** 2)) @ H, H.T * (1 / sig ** 2))                # BLUE operator

# ---- real 5-min anchors from the pool, spline-interpolated to 2 s + a small fast fluctuation ----
POOLX = np.load(os.path.join(HERE, "release_v0.4.1", "pool_ieee14.npz"))["X"].astype(np.float32)  # [T,N,4]
CAD = 2.0                          # seconds per SE step
ANCHOR_DT = 300.0                  # pool spacing = 5 min (NYISO real-time actual load)
NA = 25                            # anchors -> 2 h window
t0 = 1000
th_anchor = np.deg2rad(POOLX[t0:t0 + NA, keep, 3].astype(float))                 # [NA, NS] real 5-min angle anchors
ta = np.arange(NA) * ANCHOR_DT
tt = np.arange(0, (NA - 1) * ANCHOR_DT, CAD)                                     # 2 s grid
th_slow = np.stack([CubicSpline(ta, th_anchor[:, j])(tt) for j in range(NS)], 1) # [Tt, NS] smooth real trend
# calibrated fast component: the Hirst-Kirby REGULATION band, modeled as a uPMU-validated OU/AR(1) process on
# the LOADS (Roberts 2016, Milano 2013), RMS = 0.4% of load (bounded by NYISO's ~1% regulation reserve, which
# covers ~2.5 sigma), correlation time tau = 30 s. It is a PHYSICAL load signal, distinct from meter noise. We
# drive it on the loads and propagate to the angle state through the DC map dtheta = Bred^-1 dP.
Tt = len(tt)
baseMVA = float(ppc["baseMVA"])
load_ppc = net._pd2ppc_lookups["bus"][net.load.bus.values].astype(int)            # ppc bus index per load
base_load_pu = net.load.p_mw.values / baseMVA
SIG_F, TAU = 0.004, 30.0                                                          # 0.4% of load RMS, 30 s corr time
phi = np.exp(-CAD / TAU); sf = SIG_F * np.sqrt(1 - phi ** 2)
oul = np.zeros((Tt, len(base_load_pu)))
for t in range(1, Tt):
    oul[t] = phi * oul[t - 1] + rng.normal(0, sf, len(base_load_pu))              # per-load OU scale fluctuation
dP = np.zeros((Tt, NBp))                                                          # injection delta (pu) = -load delta
for li, b in enumerate(load_ppc):
    dP[:, b] += -oul[:, li] * base_load_pu[li]
Bred = B[np.ix_(keep, keep)]
dtheta = np.linalg.solve(Bred, dP[:, keep].T).T                                   # DC map: load fluctuation -> angle
th_true = th_slow + dtheta                                                        # [Tt, NS] realistic 2 s benign truth
print(f"calibrated fast component: 0.4% load OU (tau=30s) -> angle RMS {np.rad2deg(np.std(dtheta)):.4f} deg", flush=True)

# ---- stealthy ramp attack on a few buses over a ~2 min window (creep below any single-scan jump) ----
atk_buses = [keep.index(b) for b in [pmu[0], pmu[1], pmu[2]]]                     # attack a few metered buses
A0 = int(Tt * 0.55); A1 = A0 + int(120 / CAD)                                     # ramp over 120 s
ramp_mag = np.deg2rad(1.2)                                                        # final false shift (~1.2 deg), plausible
false = np.zeros_like(th_true)
prog = np.clip((np.arange(Tt) - A0) / (A1 - A0), 0, 1)                            # 0 before, ramps to 1, holds
for b in atk_buses: false[:, b] = prog * ramp_mag
th_meas_state = th_true + false                                                  # state the (spoofed) measurements encode

# ---- estimators ----
def wls_stream(state):
    return np.stack([GinvHtW @ (H @ state[t] + rng.normal(0, sig)) for t in range(Tt)])

hat = wls_stream(th_meas_state)                                                  # static WLS per snapshot (sees spoof during attack)
# classical Kalman: per-component steady-state gain from benign drift q and WLS noise r (estimated online, benign part)
q = np.diff(th_true[:A0], axis=0).var(axis=0)                                     # benign per-step process variance (slow + fast)
benign_hat = wls_stream(th_true)                                                 # a benign reference stream to calibrate r
r = np.maximum((np.diff(benign_hat[:A0], axis=0).var(axis=0) - q) / 2.0, 1e-12)
K = np.zeros(NS)
for i in range(NS):
    P = r[i]
    for _ in range(300):
        Pm = P + q[i]; K[i] = Pm / (Pm + r[i]); P = (1 - K[i]) * Pm
kf = hat.copy()
for t in range(1, Tt):
    kf[t] = kf[t - 1] + K * (hat[t] - kf[t - 1])                                 # predict (persistence) + robust-free correct

to_deg = 180 / np.pi
err_wls = np.abs(hat - th_true).mean(1) * to_deg                                 # error vs BENIGN true state
err_kf = np.abs(kf - th_true).mean(1) * to_deg
ben = slice(0, A0 - 5)
res = dict(system="ieee14", model="DC-SE (BLUE)", cadence_s=CAD, anchor_dt_s=ANCHOR_DT, seed=123,
           benign=dict(th_wls=round(float(err_wls[ben].mean()), 4), th_kalman=round(float(err_kf[ben].mean()), 4)),
           attack_peak=dict(th_wls=round(float(err_wls[A1]), 4), th_kalman=round(float(err_kf[A1]), 4),
                            ramp_mag_deg=round(float(np.rad2deg(ramp_mag)), 2)))
res["benign"]["kalman_gain_pct"] = round(100 * (1 - res["benign"]["th_kalman"] / res["benign"]["th_wls"]), 1)
json.dump(res, open(os.path.join(RES, "se_seconds_sandbox.json"), "w"), indent=2)
print(json.dumps(res, indent=2), flush=True)

# ---- figure: (A) one bus's real-anchored 2 s trajectory, (B) benign+attack error for WLS vs Kalman ----
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
                     "mathtext.fontset": "dejavuserif", "axes.edgecolor": "#222", "axes.linewidth": 0.8})
fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.0))
jb = atk_buses[0]
ax[0].plot(ta / 60, np.rad2deg(th_anchor[:, jb]), "o", color="#b2182b", ms=5, label="real 5-min anchors (NYISO)")
ax[0].plot(tt / 60, np.rad2deg(th_true[:, jb]), "-", color="#2166ac", lw=1.0, label="generated 2-s trajectory")
ax[0].set_xlabel("time (min)"); ax[0].set_ylabel("bus angle (deg)"); ax[0].legend(fontsize=7.3, frameon=False)
ax[0].set_title("Real-anchored seconds-resolution state", fontsize=9.5)
ax[1].axvspan(A0 * CAD / 60, A1 * CAD / 60, color="#b2182b", alpha=0.08)
ax[1].plot(tt / 60, err_wls, color="#b2182b", lw=1.2, label="static WLS")
ax[1].plot(tt / 60, err_kf, color="#2166ac", lw=1.2, label="classical Kalman")
ax[1].text((A0 + A1) / 2 * CAD / 60, ax[1].get_ylim()[1] * 0.9, "stealthy\nramp", ha="center", fontsize=7.3, color="#b2182b")
ax[1].set_xlabel("time (min)"); ax[1].set_ylabel("angle error vs true state (deg)"); ax[1].legend(fontsize=7.8, frameon=False, loc="upper left")
ax[1].set_title("Benign win, then both fooled by the ramp", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_seconds_sandbox.png"), dpi=175); fig.savefig(os.path.join(RES, "fig_seconds_sandbox.pdf"))
with open(os.path.join(RES, "sidecars", "seconds_sandbox.csv"), "w") as f:
    f.write("t_min,err_wls_deg,err_kalman_deg\n")
    for t in range(Tt): f.write(f"{tt[t]/60:.4f},{err_wls[t]:.5f},{err_kf[t]:.5f}\n")
print("wrote results/fig_seconds_sandbox.(png|pdf) + se_seconds_sandbox.json", flush=True)
