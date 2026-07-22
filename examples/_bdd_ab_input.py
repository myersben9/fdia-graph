# -*- coding: utf-8 -*-
"""THROWAWAY A/B test for the Boyaci critique: does feeding the STATE-ESTIMATE (PSSE output) as the BDD
input, instead of the raw noisy measurement, drive benign false-alarm (FA) toward ~100%?

We reproduce the OLD FDIA_localization pipeline exactly on fresh IEEE-118 benign timesteps:
  input A  z_noisy   : raw measurement = h(x_true) + 1% meter noise            (init_dataset.py L264-267, the "Z_t")
  input B  x_hat     : WLS state estimate net.res_bus_est [p,q,v,va]           (init_dataset.py run_psse(), the OLD "X_t")
  input A0 z_clean   : noise-free h(x_true) (control: noise-free BUT consistent)
  input B' x_hat_R0  : x_hat fed with R matched to its OWN (tiny) deviation, to isolate the R-normalization effect

Everything else (network, R=1% convention, WLS init='flat', chi2 + LNR test) is IDENTICAL across inputs, so the
only variable is WHAT is fed as z. We report chi2 detect% and LNR detect% at tau=3.0 and at Boyaci's tuned
tau_bdd(118)=2.37. FA = fraction of benign samples flagged.

Run:  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=../src ../../venv/python.exe _bdd_ab_input.py
"""
import os, warnings, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); warnings.filterwarnings("ignore")
import pandas as pd
pd.options.mode.chained_assignment = None
import pandapower as pp, pandapower.networks as pn
from pandapower.create import create_measurement
from pandapower.estimation import estimate, chi2_analysis
from pandapower.pypower.makeYbus import makeYbus

N       = int(os.environ.get("N_SAMPLES", "150"))
TAU_HI  = 3.0            # standard 3-sigma LNR threshold
TAU_BOY = 2.37          # Boyaci Table II tuned tau_bdd for IEEE-118
V_MIN, V_MAX = 0.9, 1.1
k, sigma_s = 0.1, 0.03
np.random.seed(123)

net0 = pn.case118(); pp.runpp(net0)
nodelist = sorted(net0.bus.index.tolist())
pos = {int(b): i for i, b in enumerate(nodelist)}
base_load_p = net0.load["p_mw"].values.copy(); base_load_q = net0.load["q_mvar"].values.copy()
base_gen_p  = net0.gen["p_mw"].values.copy()
load_buses  = net0.load["bus"].values; gen_buses = net0.gen["bus"].values
all_buses   = np.unique(np.concatenate([load_buses, gen_buses]))

# Ybus (properly reindexed ppc->pandapower) so h(x_true) is computed with the SAME operator the WLS uses.
ppc = net0._ppc; Ybus, _, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
lut = net0._pd2ppc_lookups["bus"][:net0.bus.shape[0]]
Ybus_pd = np.asarray(Ybus.todense())[lut][:, lut]; bMVA = ppc["baseMVA"]

def safe_std(v, rel, floor=1e-3):
    return max(abs(float(v)) * rel, floor)

def scale_and_solve():
    """One benign timestep: scale loads/gens, run AC PF, return SE-consistent z_true [N,4]=[p,q,v,va] or None."""
    net = pn.case118()
    sf = np.clip(np.random.normal(1 + k * np.random.uniform(-1, 1), sigma_s, size=len(all_buses)), 0.7, 1.3)
    b2s = dict(zip(all_buses, sf))
    net.load["p_mw"]  = base_load_p * [b2s[b] for b in load_buses]
    net.load["q_mvar"] = base_load_q * [b2s[b] for b in load_buses]
    net.gen["p_mw"]   = base_gen_p  * [b2s[b] for b in gen_buses]
    try:
        pp.runpp(net, init="flat", max_iteration=50, tolerance_mva=1e-6)
    except Exception:
        return None
    attrs = ["p_mw", "q_mvar", "vm_pu", "va_degree"]
    z = net.res_bus.reindex(nodelist)[attrs].values.copy()
    # SE-consistent injection: remove shunt draw (pandapower WLS h() excludes net.shunt)
    if len(net.res_shunt):
        for b, ps, qs in zip(net.shunt.bus.values, net.res_shunt.p_mw.values, net.res_shunt.q_mvar.values):
            if int(b) in pos:
                z[pos[int(b)], 0] -= ps; z[pos[int(b)], 1] -= qs
    return z

def bdd(meas_vals, rel_override=None):
    """Run pandapower WLS + BDD on a [N,4] measurement array. Returns (chi2_detected, lnr_max, x_hat[N,4]) or None.
    rel_override: dict to change the R (std) convention; default 1% P/Q, 2e-4 V, 57e-4 va (init_dataset.py)."""
    rel = {"p": 0.01, "q": 0.01, "v": 2e-4, "va": 57e-4}
    if rel_override: rel.update(rel_override)
    net = pn.case118()
    net.measurement = net.measurement.iloc[0:0]
    for idx, bus in enumerate(nodelist):
        p, q, vm, va = meas_vals[idx]
        create_measurement(net, "p",  "bus", p,  safe_std(p,  rel["p"]),  element=bus)
        create_measurement(net, "q",  "bus", q,  safe_std(q,  rel["q"]),  element=bus)
        create_measurement(net, "v",  "bus", vm, safe_std(vm, rel["v"]),  element=bus)
        create_measurement(net, "va", "bus", va, safe_std(va, rel["va"]), element=bus)
    try:
        detected = bool(chi2_analysis(net, init="flat"))
    except Exception:
        return None
    if not hasattr(net, "res_bus_est") or net.res_bus_est is None:
        return None
    est = net.res_bus_est.reindex(nodelist)
    xhat = est[["p_mw", "q_mvar", "vm_pu", "va_degree"]].values.copy()
    # LNR (approx): |z - h(x_hat)| / sigma over the 4 bus meas types. h(x_hat) reconstructed from est V,theta.
    Vh = est["vm_pu"].values; Th = np.deg2rad(est["va_degree"].values)
    Vc = Vh * np.exp(1j * Th)
    S = Vc * np.conj(Ybus_pd @ Vc) * bMVA         # complex injection at each bus (SE-consistent, no shunt)
    Ph, Qh = S.real, S.imag
    rN = []
    for idx in range(len(nodelist)):
        p, q, vm, va = meas_vals[idx]
        rN.append(abs(p  - Ph[idx])  / safe_std(p,  rel["p"]))
        rN.append(abs(q  - Qh[idx])  / safe_std(q,  rel["q"]))
        rN.append(abs(vm - Vh[idx])  / safe_std(vm, rel["v"]))
    return detected, float(np.max(rN)), xhat

conds = {"A_znoisy": [], "A0_zclean": [], "B_xhat": [], "Bp_xhat_R0": []}
lnr   = {c: [] for c in conds}
nconv = 0
for t in range(N * 3):
    if sum(len(v) for v in conds.values()) >= N * 4:
        break
    z_true = scale_and_solve()
    if z_true is None:
        continue
    z_noisy = z_true.copy()
    z_noisy[:, 0] = np.random.normal(z_true[:, 0], np.maximum(np.abs(z_true[:, 0]) * 0.01, 1e-3))
    z_noisy[:, 1] = np.random.normal(z_true[:, 1], np.maximum(np.abs(z_true[:, 1]) * 0.01, 1e-3))
    z_noisy[:, 2] = np.clip(np.random.normal(z_true[:, 2], np.maximum(np.abs(z_true[:, 2]) * 2e-4, 1e-3)), V_MIN, V_MAX)
    z_noisy[:, 3] = np.random.normal(z_true[:, 3], np.maximum(np.abs(z_true[:, 3]) * 57e-4, 1e-3))

    rA = bdd(z_noisy)                    # input A: raw noisy measurement (the "Z_t")
    if rA is None:
        continue
    nconv += 1
    conds["A_znoisy"].append(rA[0]); lnr["A_znoisy"].append(rA[1])
    x_hat = rA[2]                        # the WLS estimate = OLD "X_t" (run_psse output)

    r0 = bdd(z_true)                     # control: noise-free but model-consistent
    if r0: conds["A0_zclean"].append(r0[0]); lnr["A0_zclean"].append(r0[1])

    rB = bdd(x_hat)                      # input B: STATE ESTIMATE fed as measurement, R still = 1% of x_hat
    if rB: conds["B_xhat"].append(rB[0]); lnr["B_xhat"].append(rB[1])

    rBp = bdd(x_hat, rel_override={"p": 1e-4, "q": 1e-4, "v": 1e-5, "va": 1e-4})  # isolate R-normalization
    if rBp: conds["Bp_xhat_R0"].append(rBp[0]); lnr["Bp_xhat_R0"].append(rBp[1])

print(f"\nIEEE-118 benign, {nconv} converged samples. FA = fraction flagged.\n")
hdr = f"{'condition':14s} {'n':>4s} {'chi2 FA%':>9s} {'LNR FA% @tau=3.0':>17s} {'LNR FA% @tau=2.37':>18s} {'median LNRmax':>14s}"
print(hdr); print("-" * len(hdr))
for c in ["A_znoisy", "A0_zclean", "B_xhat", "Bp_xhat_R0"]:
    n = len(conds[c])
    if n == 0:
        print(f"{c:14s} {0:4d}  (no data)"); continue
    chi2fa = 100 * np.mean(conds[c])
    L = np.array(lnr[c])
    fa_hi = 100 * np.mean(L > TAU_HI); fa_boy = 100 * np.mean(L > TAU_BOY)
    print(f"{c:14s} {n:4d} {chi2fa:9.1f} {fa_hi:17.1f} {fa_boy:18.1f} {np.median(L):14.2f}")
print("\nLegend: A_znoisy=raw noisy meas (proper BDD input) | A0_zclean=noise-free consistent |"
      " B_xhat=WLS estimate as input (old X_t) | Bp_xhat_R0=B with R shrunk to its own deviation")
