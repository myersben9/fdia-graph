#!/usr/bin/env python
"""Task 3 — RESOLUTION SENSITIVITY (the paper figure): how much does a recursive estimator beat static WLS on
BENIGN state estimation, and how does that temporal win change with measurement cadence, across IEEE-14/118/300.

This extends the DC-14 recursive PoC to all three systems and grounds the trajectory in the REAL 5-min NYISO
anchors instead of a synthetic OU. For each system and each cadence in {300 s, 60 s, 15 s, 2 s} we:
  1. draw hour-long windows of the real 5-min angle anchors from the pool,
  2. cubic-spline the slow trend to the target cadence and add the calibrated OU fast load band
     (phi = exp(-cad/tau), stationary RMS = 0.4% of load; the SAME physical band, only its per-step
     correlation changes with cadence),
  3. form accuracy-class noisy measurements, solve static WLS (BLUE) per snapshot, and run an online
     steady-state Kalman over the WLS stream (process var from the true per-step move, measurement var
     estimated online from successive differences),
  4. report benign angle MAE for both and the temporal win gain% = 100 (1 - MAE_kalman / MAE_wls).

The physical story: at 5-min cadence consecutive states are the raw anchors (a full 5-min move apart) so the
filter has almost nothing to average and the win is ~0; as cadence shortens the state barely moves between
scans, consecutive scans are strongly correlated, and the recursive filter's variance reduction grows. We
report the REAL per-system curve, honest either way. DC linear SE (unbiased BLUE, unconfounded). Seed 123.

Output: results/se_resolution_sensitivity.json + results/fig_resolution_sensitivity.(png|pdf) + CSV sidecar."""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from scipy.interpolate import CubicSpline
sys.path.insert(0, r"C:/Users/bm539044/desktop/fedpig")
from handmade.figures.paper_style import use_paper_style, save_figure_data, DBL_WIDTH
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
NETS = {14: pn.case14, 118: pn.case118, 300: pn.case300}
ANCHOR_DT = 300.0                    # real pool spacing = 5 min
SIG_F, TAU = 0.004, 30.0             # calibrated fast OU: 0.4% of load, 30 s correlation time
CADENCES = [300, 60, 15, 2]          # seconds per SE step
N_ANCHORS = 13                       # 12 intervals -> 1 h window
WINDOW_S = (N_ANCHORS - 1) * ANCHOR_DT
NWIN = 40                            # windows averaged per (system, cadence)
SIG_INJ, SIG_FLOW, SIG_PMU = 0.02, 0.02, 0.005
# Multi-seed error bars on the cadence table. Each seed re-runs the full per-system window draw with its own RNG
# (kept shared across cadences within a seed, exactly as the single-seed version, so seed 123 alone reproduces the
# old numbers). Overridable via SE_SEEDS="123,124,125"; reported scalars become the seed mean, with _std added.
SEEDS = tuple(int(s) for s in os.environ.get("SE_SEEDS", "123,124,125").split(","))


def build(c):
    net = NETS[c](); pp.rundcpp(net); ppc = net._ppc; br = ppc["branch"]; bus = ppc["bus"]
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
    H = np.array(rows)
    sig = np.array([SIG_INJ if k == "inj" else SIG_FLOW if k == "flow" else SIG_PMU for k in kinds])
    GinvHtW = np.linalg.solve((H.T * (1 / sig ** 2)) @ H, H.T * (1 / sig ** 2))
    Bred = B[np.ix_(keep, keep)]
    baseMVA = float(ppc["baseMVA"]); load_ppc = net._pd2ppc_lookups["bus"][net.load.bus.values].astype(int)
    base_load_pu = net.load.p_mw.values / baseMVA
    ok = np.abs(base_load_pu) > 0
    load_ppc, base_load_pu = load_ppc[ok], base_load_pu[ok]
    POOLX = np.load(os.path.join(HERE, "release_v0.4.1", f"pool_ieee{c}.npz"))["X"].astype(np.float64)
    return dict(c=c, NS=NS, NBp=NBp, keep=keep, H=H, sig=sig, GinvHtW=GinvHtW, Bred=Bred,
                load_ppc=load_ppc, base_load_pu=base_load_pu, POOLX=POOLX, n_meas=H.shape[0], n_pmu=len(pmu))


def kalman_gain(q, r):
    K = np.zeros_like(q)
    for i in range(len(q)):
        P = r[i]
        for _ in range(400):
            Pm = P + q[i]; K[i] = Pm / (Pm + r[i]); P = (1 - K[i]) * Pm
    return K


def eval_cadence(d, cad, rng):
    """Benign angle MAE for static WLS and an online Kalman at one cadence, averaged over NWIN real windows."""
    NS, keep, NBp = d["NS"], d["keep"], d["NBp"]
    phi = np.exp(-cad / TAU); sf = SIG_F * np.sqrt(1 - phi ** 2)
    nL = len(d["base_load_pu"])
    tt = np.arange(0, WINDOW_S + 1e-6, cad)                                    # cadence grid over the 1 h window
    Ln = len(tt); ta = np.arange(N_ANCHORS) * ANCHOR_DT
    warm = min(20, Ln // 3)                                                    # discard filter warm-up
    e_wls, e_kf = [], []
    for _ in range(NWIN):
        t0 = int(rng.integers(0, d["POOLX"].shape[0] - N_ANCHORS - 1))
        th_anchor = np.deg2rad(d["POOLX"][t0:t0 + N_ANCHORS, keep, 3])        # [N_ANCHORS, NS] real 5-min anchors
        slow = np.stack([CubicSpline(ta, th_anchor[:, j])(tt) for j in range(NS)], 1)
        oul = np.zeros((Ln, nL))
        for t in range(1, Ln): oul[t] = phi * oul[t - 1] + rng.normal(0, sf, nL)
        dP = np.zeros((Ln, NBp))
        for li, b in enumerate(d["load_ppc"]): dP[:, b] += -oul[:, li] * d["base_load_pu"][li]
        th_true = slow + np.linalg.solve(d["Bred"], dP[:, keep].T).T          # [Ln, NS] realistic benign truth
        z = th_true @ d["H"].T + rng.normal(0, d["sig"], (Ln, d["n_meas"]))
        hat = z @ d["GinvHtW"].T                                              # static WLS (BLUE) per snapshot
        if Ln < 3:
            e_wls.append(np.abs(hat - th_true).mean()); e_kf.append(np.abs(hat - th_true).mean()); continue
        q = np.diff(th_true, axis=0).var(0)                                  # true per-step process variance
        r = np.maximum((np.diff(hat, axis=0).var(0) - q) / 2.0, 1e-12)       # measurement var estimated online
        K = kalman_gain(q, r)
        kf = hat.copy()
        for t in range(1, Ln): kf[t] = kf[t - 1] + K * (hat[t] - kf[t - 1])
        sl = slice(warm, None)
        e_wls.append(np.abs(hat[sl] - th_true[sl]).mean())
        e_kf.append(np.abs(kf[sl] - th_true[sl]).mean())
    to_deg = 180.0 / np.pi
    mw = float(np.mean(e_wls)) * to_deg; mk = float(np.mean(e_kf)) * to_deg
    return dict(th_wls_deg=round(mw, 5), th_recursive_deg=round(mk, 5), gain_pct=round(100 * (1 - mk / mw), 1),
                n_steps=Ln)


out = {"model": "DC-SE (BLUE), real-anchored + calibrated OU", "seeds": list(SEEDS), "seed": SEEDS[0],
       "anchor_dt_s": ANCHOR_DT, "sigma_f_pct": SIG_F * 100, "tau_s": TAU, "n_windows": NWIN,
       "window_s": WINDOW_S, "cadences_s": CADENCES, "per_system": {}}
for c in (14, 118, 300):
    d = build(c)
    # per seed: one shared RNG walked across cadences (matches the single-seed structure exactly)
    per_seed_rows = []
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        per_seed_rows.append({str(cad): eval_cadence(d, cad, rng) for cad in CADENCES})
    row = {}
    for cad in CADENCES:
        vals = [ps[str(cad)] for ps in per_seed_rows]
        cell = {"n_steps": vals[0]["n_steps"], "per_seed": vals}
        for key in ("th_wls_deg", "th_recursive_deg", "gain_pct"):
            arr = [v[key] for v in vals]
            cell[key] = round(float(np.mean(arr)), 5)
            cell[key + "_std"] = round(float(np.std(arr)), 5)
        row[str(cad)] = cell
    out["per_system"][str(c)] = dict(n_meas=d["n_meas"], n_free_angles=d["NS"], per_cadence=row)
    parts = "  ".join(f"{cad}s:{row[str(cad)]['gain_pct']:+.0f}%" for cad in CADENCES)
    print(f"IEEE-{c:<3d} ({d['NS']} angles, {d['n_meas']} meas)  temporal win (mean over {len(SEEDS)} seeds)  {parts}", flush=True)
json.dump(out, open(os.path.join(RES, "se_resolution_sensitivity.json"), "w"), indent=2)

# ---- figure (gsp spec): (A) benign angle MAE WLS vs Kalman across cadence, (B) temporal-win gain% ----
use_paper_style(8.0)
COLORS = {14: "#2166ac", 118: "#1a9850", 300: "#b2182b"}
fig, ax = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.5))
for c in (14, 118, 300):
    pc = out["per_system"][str(c)]["per_cadence"]
    xs = CADENCES
    yw = [pc[str(k)]["th_wls_deg"] for k in xs]; yk = [pc[str(k)]["th_recursive_deg"] for k in xs]
    yg = [pc[str(k)]["gain_pct"] for k in xs]
    ax[0].plot(xs, yw, "o--", color=COLORS[c], lw=1.0, ms=3.5, mfc="white", label=f"IEEE-{c} WLS")
    ax[0].plot(xs, yk, "o-", color=COLORS[c], lw=1.3, ms=3.5, label=f"IEEE-{c} Kalman")
    ax[1].plot(xs, yg, "o-", color=COLORS[c], lw=1.4, ms=4, label=f"IEEE-{c}")
for a in ax:
    a.set_xscale("log"); a.set_xticks(CADENCES); a.set_xticklabels([str(k) for k in CADENCES])
    a.set_xlabel("measurement cadence (s, log scale)"); a.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    a.axvspan(2, 4, color="#2166ac", alpha=0.06)
ax[0].set_ylabel("benign angle MAE (deg)"); ax[0].legend(fontsize=6.0, frameon=False, ncol=3, loc="upper left")
ax[1].set_ylabel("temporal win (%)"); ax[1].axhline(0, color="#888", lw=0.5)
ax[1].legend(fontsize=7.0, frameon=False, loc="lower left")
ax[1].annotate("win grows as\ncadence shortens", xy=(2, 60), xytext=(30, 30), fontsize=6.3, color="#444",
               ha="center", arrowprops=dict(arrowstyle="->", color="#444", lw=0.6))
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_resolution_sensitivity.png")); fig.savefig(os.path.join(RES, "fig_resolution_sensitivity.pdf"))
cols = {"cadence_s": CADENCES}
for c in (14, 118, 300):
    pc = out["per_system"][str(c)]["per_cadence"]
    cols[f"ieee{c}_wls_deg"] = [pc[str(k)]["th_wls_deg"] for k in CADENCES]
    cols[f"ieee{c}_kalman_deg"] = [pc[str(k)]["th_recursive_deg"] for k in CADENCES]
    cols[f"ieee{c}_gain_pct"] = [pc[str(k)]["gain_pct"] for k in CADENCES]
save_figure_data(os.path.join(RES, "sidecars", "resolution_sensitivity.csv"), cols)
print("wrote results/se_resolution_sensitivity.json + fig_resolution_sensitivity.(png|pdf)", flush=True)
