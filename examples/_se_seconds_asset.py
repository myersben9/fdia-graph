#!/usr/bin/env python
"""Task 2 — packaged SECONDS-CADENCE state-trajectory ASSET for IEEE-14/118/300.

Each asset is a realistic 2-second benign state trajectory built by interpolating the REAL 5-min AC-solved pool
anchors (release_v0.4.1/pool_ieee{C}.npz, key X = [Pinj, Qinj, Vm, theta], already AC-solved) to 2 s and adding
the CALIBRATED OU fast load band (verified in se_fast_calibration.json: 0.4% of load RMS, tau = 30 s). Because
the anchors are already AC-consistent, we do NOT re-solve AC per 2-second step: the slow trend is spline-
interpolated real AC state and the fast fluctuation is a physically-calibrated load signal propagated to the
angle state through the DC map (dtheta = Bred^-1 dP). Voltage magnitude is carried as slow-interpolated AC
context (ground truth the attack never touches). A stealthy temporal RAMP attack segment is included exactly as
in the sandbox (a sub-single-scan angle creep on a few buses).

Saved per system: results/seconds_assets/seconds_ieee{C}.npz (arrays + metadata) and a CSV sidecar. Seed 123,
CPU only. This is the streamable realistic-operator-data asset the seconds-cadence contribution is built on."""
import os, sys, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
ASSET = os.path.join(RES, "seconds_assets"); os.makedirs(ASSET, exist_ok=True)
NETS = {14: pn.case14, 118: pn.case118, 300: pn.case300}
CAD, ANCHOR_DT = 2.0, 300.0
SIG_F, TAU = 0.004, 30.0
PHI = np.exp(-CAD / TAU); SF = SIG_F * np.sqrt(1 - PHI ** 2)
NA = 25                              # 25 anchors -> 2 h window at 5-min spacing
T0 = 1000                            # fixed pool start index (deterministic asset)


def build(c):
    net = NETS[c](); pp.rundcpp(net); ppc = net._ppc; br = ppc["branch"]; bus = ppc["bus"]
    NBp = bus.shape[0]; fb = br[:, 0].real.astype(int); tb = br[:, 1].real.astype(int); bl = 1.0 / br[:, 3].real
    B = np.zeros((NBp, NBp))
    for f, t, b in zip(fb, tb, bl):
        B[f, f] += b; B[t, t] += b; B[f, t] -= b; B[t, f] -= b
    slack = int(np.where(bus[:, 1] == 3)[0][0]); keep = [i for i in range(NBp) if i != slack]; NS = len(keep)
    Bred = B[np.ix_(keep, keep)]
    baseMVA = float(ppc["baseMVA"]); load_ppc = net._pd2ppc_lookups["bus"][net.load.bus.values].astype(int)
    base_load_pu = net.load.p_mw.values / baseMVA
    ok = np.abs(base_load_pu) > 0
    load_ppc, base_load_pu = load_ppc[ok], base_load_pu[ok]
    POOLX = np.load(os.path.join(HERE, "release_v0.4.1", f"pool_ieee{c}.npz"))["X"].astype(np.float64)
    return dict(c=c, keep=np.array(keep), NS=NS, NBp=NBp, Bred=Bred, load_ppc=load_ppc,
                base_load_pu=base_load_pu, POOLX=POOLX)


def make_asset(c):
    d = build(c); rng = np.random.default_rng(123)
    keep = d["keep"]; NS = d["NS"]
    th_anchor = np.deg2rad(d["POOLX"][T0:T0 + NA, keep, 3])                    # [NA, NS] real 5-min angle anchors
    vm_anchor = d["POOLX"][T0:T0 + NA, keep, 2]                                # [NA, NS] real 5-min |V| anchors (pu)
    ta = np.arange(NA) * ANCHOR_DT
    tt = np.arange(0, (NA - 1) * ANCHOR_DT, CAD); Tt = len(tt)
    th_slow = np.stack([CubicSpline(ta, th_anchor[:, j])(tt) for j in range(NS)], 1)
    vm_slow = np.stack([CubicSpline(ta, vm_anchor[:, j])(tt) for j in range(NS)], 1)  # slow AC voltage context
    # calibrated OU fast load band -> angle state via the DC map
    nL = len(d["base_load_pu"]); oul = np.zeros((Tt, nL))
    for t in range(1, Tt): oul[t] = PHI * oul[t - 1] + rng.normal(0, SF, nL)
    dP = np.zeros((Tt, d["NBp"]))
    for li, b in enumerate(d["load_ppc"]): dP[:, b] += -oul[:, li] * d["base_load_pu"][li]
    dtheta = np.linalg.solve(d["Bred"], dP[:, keep].T).T
    th_true = th_slow + dtheta                                                # [Tt, NS] realistic 2-s benign truth
    # stealthy ramp attack on a few load-adjacent buses over a ~2 min window (sub-single-scan creep)
    ar = np.random.default_rng(1234 + c)
    atk_buses = np.sort(ar.choice(NS, size=min(3, NS), replace=False))
    a0 = int(Tt * 0.55); a1 = a0 + int(120 / CAD); ramp_mag = np.deg2rad(1.2)
    prog = np.clip((np.arange(Tt) - a0) / (a1 - a0), 0, 1)
    th_attacked = th_true.copy()
    for b in atk_buses: th_attacked[:, b] += prog * ramp_mag                  # state the spoofed measurements encode
    meta = dict(system=f"ieee{c}", cadence_s=CAD, anchor_dt_s=ANCHOR_DT, n_anchors=NA, pool_t0=T0,
                sigma_f_pct=SIG_F * 100, tau_s=TAU, phi=round(float(PHI), 5), seed=123,
                fast_angle_rms_deg=round(float(np.rad2deg(dtheta.std())), 5),
                attacked_buses=[int(b) for b in atk_buses], attack_start_step=a0, attack_end_step=a1,
                ramp_mag_deg=round(float(np.rad2deg(ramp_mag)), 3), n_steps=Tt, n_free_angles=NS,
                keep_bus_ppc=[int(b) for b in keep.tolist()],
                columns="theta_true_deg,theta_attacked_deg,vm_true_pu are [n_steps, n_free_angles]; slack excluded")
    npz_path = os.path.join(ASSET, f"seconds_ieee{c}.npz")
    np.savez_compressed(npz_path, t_s=tt.astype(np.float32),
                        theta_true_deg=np.rad2deg(th_true).astype(np.float32),
                        theta_attacked_deg=np.rad2deg(th_attacked).astype(np.float32),
                        vm_true_pu=vm_slow.astype(np.float32),
                        attacked_buses=np.array(meta["attacked_buses"], dtype=np.int32),
                        keep_bus_ppc=keep.astype(np.int32), meta=json.dumps(meta))
    # small CSV sidecar: time, mean benign angle, and the first attacked bus benign vs attacked angle
    jb = int(atk_buses[0])
    csv_path = os.path.join(ASSET, f"seconds_ieee{c}.csv")
    with open(csv_path, "w") as f:
        f.write(f"# ieee{c} seconds-cadence asset; attacked_buses={meta['attacked_buses']}; "
                f"attack steps {a0}-{a1}; ramp {meta['ramp_mag_deg']} deg\n")
        f.write(f"t_s,mean_theta_true_deg,bus{jb}_true_deg,bus{jb}_attacked_deg\n")
        mt = np.rad2deg(th_true).mean(1); bt = np.rad2deg(th_true[:, jb]); ba = np.rad2deg(th_attacked[:, jb])
        for t in range(Tt): f.write(f"{tt[t]:.1f},{mt[t]:.5f},{bt[t]:.5f},{ba[t]:.5f}\n")
    kb = os.path.getsize(npz_path) / 1024
    print(f"IEEE-{c:<3d}: {Tt} steps x {NS} angles, fast angle RMS {meta['fast_angle_rms_deg']:.4f} deg, "
          f"attacked buses {meta['attacked_buses']}, npz {kb:.0f} KB -> {os.path.basename(npz_path)}", flush=True)
    return meta


summary = {str(c): make_asset(c) for c in (14, 118, 300)}
json.dump(summary, open(os.path.join(ASSET, "seconds_assets_manifest.json"), "w"), indent=2)
print(f"wrote {ASSET}/seconds_ieee(14|118|300).npz + .csv + seconds_assets_manifest.json", flush=True)
