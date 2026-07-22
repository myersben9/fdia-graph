#!/usr/bin/env python
"""Task 1 — CALIBRATION CHECK for the seconds-cadence fast load component.

The seconds generator adds a zero-mean fast fluctuation on top of the spline-interpolated real 5-min NYISO
anchors, driven on the LOADS (a physical signal, not meter noise). Per fastload_cites.md the process is an
Ornstein-Uhlenbeck / AR(1) load process (uPMU-validated, Roberts 2016; Milano 2013):

    f_{k+1} = phi f_k + eps,   phi = exp(-dt/tau),   eps ~ N(0, sigma_f^2 (1 - phi^2))

with sigma_f = 0.4% of instantaneous load (range 0.3-0.5%, bounded by NYISO's ~1%-of-load regulation reserve
which covers ~2.5 sigma) and correlation time tau = 30 s. At dt = 2 s that gives phi = exp(-2/30) = 0.9355.

This script VERIFIES that construction against the real anchors, two honest checks per system:
  (A) 2-second fast-band RMS = 0.4% of load  (the stationary amplitude of the OU is what we asked for).
  (B) the fast band's 5-min increment is CONSISTENT with the real NYISO 5-min increment observed in the pool
      (i.e. the unresolved sub-5-min fluctuation is a modest fraction of, and does not overwhelm, the resolved
      5-min load variability, and stays inside the 1% regulation reserve). tau << 300 s so two consecutive
      anchors' fast components are nearly independent, giving a fast-band 5-min increment ~= sqrt(2)*sigma_f.

Real either way: we report the measured real 5-min increment and let it stand. DC map (Bred^-1 dP) also gives
the angle-state RMS the fast band induces. Output: results/se_fast_calibration.json + a small figure. CPU, seed 123."""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
sys.path.insert(0, r"C:/Users/bm539044/desktop/fedpig")
from handmade.figures.paper_style import use_paper_style, save_figure_data, COL_WIDTH, DBL_WIDTH
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
os.makedirs(os.path.join(RES, "sidecars"), exist_ok=True)
NETS = {14: pn.case14, 118: pn.case118, 300: pn.case300}
CAD, ANCHOR_DT = 2.0, 300.0          # 2 s SE cadence, 5-min real anchor spacing
SIG_F, TAU = 0.004, 30.0             # 0.4% of load RMS, 30 s correlation time (fastload_cites.md)
PHI = np.exp(-CAD / TAU)             # AR(1) coefficient at 2 s = 0.9355
SF = SIG_F * np.sqrt(1 - PHI ** 2)   # innovation std so the stationary std stays at SIG_F
STEP_PER_ANCHOR = int(round(ANCHOR_DT / CAD))   # 150 two-second steps per 5-min anchor


def build(c):
    """DC susceptance model + load bookkeeping for one IEEE case (ppc bus order)."""
    net = NETS[c](); pp.rundcpp(net); ppc = net._ppc; br = ppc["branch"]; bus = ppc["bus"]
    NBp = bus.shape[0]; fb = br[:, 0].real.astype(int); tb = br[:, 1].real.astype(int); bl = 1.0 / br[:, 3].real
    B = np.zeros((NBp, NBp))
    for f, t, b in zip(fb, tb, bl):
        B[f, f] += b; B[t, t] += b; B[f, t] -= b; B[t, f] -= b
    slack = int(np.where(bus[:, 1] == 3)[0][0]); keep = [i for i in range(NBp) if i != slack]
    Bred = B[np.ix_(keep, keep)]
    baseMVA = float(ppc["baseMVA"])
    load_ppc = net._pd2ppc_lookups["bus"][net.load.bus.values].astype(int)
    base_load_pu = net.load.p_mw.values / baseMVA
    ok = np.abs(base_load_pu) > 0                                   # skip reactive-only P=0 load rows (300-bus gotcha)
    load_ppc, base_load_pu = load_ppc[ok], base_load_pu[ok]
    gen_ppc = set(net._pd2ppc_lookups["bus"][net.gen.bus.values].astype(int).tolist()) if len(net.gen) else set()
    gen_ppc |= set(net._pd2ppc_lookups["bus"][net.ext_grid.bus.values].astype(int).tolist())
    pure_load = np.array(sorted({b for b in load_ppc.tolist() if b not in gen_ppc}))   # Pinj == -Pload here
    POOLX = np.load(os.path.join(HERE, "release_v0.4.1", f"pool_ieee{c}.npz"))["X"].astype(np.float64)  # [T,N,4] phys units
    return dict(c=c, keep=keep, Bred=Bred, load_ppc=load_ppc, base_load_pu=base_load_pu,
                pure_load=pure_load, POOLX=POOLX, NBp=NBp, baseMVA=baseMVA)


def check(c):
    d = build(c); rng = np.random.default_rng(123)
    nL = len(d["base_load_pu"])
    H = 9000                                                        # 5 h of 2-s steps for a stable OU estimate
    oul = np.zeros((H, nL))
    for t in range(1, H):
        oul[t] = PHI * oul[t - 1] + rng.normal(0, SF, nL)          # per-load fractional OU fluctuation
    # (A) 2-s fast-band RMS as a fraction of load (stationary std of the OU, after a short warm-up)
    fast_rms_pct = float(np.mean(oul[500:].std(0))) * 100.0
    # (B1) SYNTHESIZED 5-min increment of the fast band: subsample every 150 steps (=300 s), take differences
    anc = oul[::STEP_PER_ANCHOR]                                    # fast-band value at each 5-min mark
    synth_5min_incr_pct = float(np.mean(np.diff(anc, axis=0).std(0))) * 100.0
    # (B2) REAL 5-min increment from the pool: relative std of 5-min Pinj differences on pure-load buses
    P = d["POOLX"][:, d["pure_load"], 0]                            # [T, nPureLoad] Pinj (MW), == -Pload on these buses
    lvl = np.abs(P).mean(0); lvl[lvl < 1e-6] = np.nan
    real_5min_incr_pct = float(np.nanmedian(np.diff(P, axis=0).std(0) / lvl)) * 100.0
    # angle-state RMS that the fast band induces through the DC map (dtheta = Bred^-1 dP), for context
    dP = np.zeros((H, d["NBp"]))
    for li, b in enumerate(d["load_ppc"]): dP[:, b] += -oul[:, li] * d["base_load_pu"][li]
    dth = np.linalg.solve(d["Bred"], dP[:, d["keep"]].T).T
    angle_rms_deg = float(np.rad2deg(dth[500:].std()))
    return dict(system=f"ieee{c}", n_loads=int(nL), n_pure_load=int(len(d["pure_load"])),
                fast_rms_pct_of_load=round(fast_rms_pct, 4), target_rms_pct=SIG_F * 100,
                synth_5min_incr_pct=round(synth_5min_incr_pct, 4),
                real_5min_incr_pct=round(real_5min_incr_pct, 4),
                fast_within_real_5min=bool(synth_5min_incr_pct < real_5min_incr_pct),
                fast_within_1pct_reserve=bool(synth_5min_incr_pct < 1.0),
                angle_state_rms_deg=round(angle_rms_deg, 5),
                example_fast_trace=oul[500:1400, 0].round(6).tolist())   # one load's 30-min fast trace for the figure


out = {"process": "OU/AR(1) fast load component", "cadence_s": CAD, "anchor_dt_s": ANCHOR_DT,
       "sigma_f_pct": SIG_F * 100, "tau_s": TAU, "phi": round(float(PHI), 5), "seed": 123, "per_system": {}}
for c in (14, 118, 300):
    r = check(c); out["per_system"][str(c)] = r
    print(f"IEEE-{c:<3d}: fast-band RMS {r['fast_rms_pct_of_load']:.3f}% of load (target {SIG_F*100:.1f}%) | "
          f"5-min incr synth {r['synth_5min_incr_pct']:.3f}% vs REAL {r['real_5min_incr_pct']:.3f}% "
          f"({'consistent' if r['fast_within_real_5min'] else 'EXCEEDS real'}; "
          f"{'<1% reserve' if r['fast_within_1pct_reserve'] else '>1% reserve'}) | "
          f"angle RMS {r['angle_state_rms_deg']:.4f} deg", flush=True)

traces = {str(c): out["per_system"][str(c)].pop("example_fast_trace") for c in (14, 118, 300)}
json.dump(out, open(os.path.join(RES, "se_fast_calibration.json"), "w"), indent=2)

# ---- figure: (A) a 2-s fast-band trace, (B) synth vs real 5-min increment per system ----
use_paper_style(8.0)
fig, ax = plt.subplots(1, 2, figsize=(DBL_WIDTH, 2.35))
tmin = np.arange(len(traces["14"])) * CAD / 60.0
ax[0].axhspan(-SIG_F * 100, SIG_F * 100, color="#2166ac", alpha=0.10)
ax[0].plot(tmin, np.array(traces["14"]) * 100, color="#2166ac", lw=0.8)
ax[0].axhline(0, color="#888", lw=0.5)
ax[0].set_xlabel("time (min)"); ax[0].set_ylabel("fast load fluctuation (% of load)")
ax[0].text(0.02, 0.96, r"OU/AR(1), $\sigma_f{=}0.4\%$, $\tau{=}30$ s", transform=ax[0].transAxes,
           va="top", fontsize=7.2, color="#2166ac")
syst = [14, 118, 300]; x = np.arange(len(syst)); w = 0.36
synth = [out["per_system"][str(c)]["synth_5min_incr_pct"] for c in syst]
real = [out["per_system"][str(c)]["real_5min_incr_pct"] for c in syst]
ax[1].bar(x - w / 2, synth, w, color="#2166ac", label="fast band (synth)")
ax[1].bar(x + w / 2, real, w, color="#b2182b", label="real NYISO anchors")
ax[1].axhline(1.0, color="#444", ls="--", lw=0.7); ax[1].text(2.35, 1.02, "1% reserve", fontsize=6.8, color="#444", ha="right")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"IEEE-{c}" for c in syst])
ax[1].set_ylabel("5-min increment std (% of load)"); ax[1].legend(fontsize=7.0, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(RES, "fig_fast_calibration.png")); fig.savefig(os.path.join(RES, "fig_fast_calibration.pdf"))
save_figure_data(os.path.join(RES, "sidecars", "fast_calibration.csv"),
                 {"system": syst, "synth_5min_incr_pct": synth, "real_5min_incr_pct": real,
                  "fast_rms_pct": [out["per_system"][str(c)]["fast_rms_pct_of_load"] for c in syst]})
print("wrote results/se_fast_calibration.json + fig_fast_calibration.(png|pdf)", flush=True)
