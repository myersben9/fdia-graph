# -*- coding: utf-8 -*-
"""Refined A/B (chi2 only, pandapower's own correct-H test) to isolate the DOMINANT cause of the benign
false-alarm blow-up when the STATE ESTIMATE is fed to the BDD instead of the raw measurement.

Conditions (identical net, R=1% convention, WLS init='flat'), reported as benign chi2 detect% = FA:
  A_meas_noshunt   raw noisy meas, shunt-corrected (proper Z_t)               -> baseline noise floor
  B_xhat_raw       WLS estimate res_bus_est as-is (literal OLD X_t)           -> the pipeline had this
  B_xhat_noshunt   WLS estimate WITH shunt draw subtracted (same conv as A)   -> isolates estimate-vs-shunt
  A_meas_shunt     raw noisy meas WITHOUT shunt correction                    -> isolates shunt alone (on a measurement)

If B_xhat_noshunt falls back to the noise floor, the 100% FA is the SHUNT-CONVENTION mismatch the estimate
carries, not 'estimate vs measurement' per se. If it stays ~100%, the estimate itself is the driver.
"""
import os, warnings, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); warnings.filterwarnings("ignore")
import pandas as pd; pd.options.mode.chained_assignment = None
import pandapower as pp, pandapower.networks as pn
from pandapower.create import create_measurement
from pandapower.estimation import chi2_analysis

N = int(os.environ.get("N_SAMPLES", "120"))
V_MIN, V_MAX = 0.9, 1.1; k, sigma_s = 0.1, 0.03
np.random.seed(123)

net0 = pn.case118(); pp.runpp(net0)
nodelist = sorted(net0.bus.index.tolist()); pos = {int(b): i for i, b in enumerate(nodelist)}
base_load_p = net0.load["p_mw"].values.copy(); base_load_q = net0.load["q_mvar"].values.copy()
base_gen_p = net0.gen["p_mw"].values.copy()
load_buses = net0.load["bus"].values; gen_buses = net0.gen["bus"].values
all_buses = np.unique(np.concatenate([load_buses, gen_buses]))

def safe_std(v, rel, floor=1e-3): return max(abs(float(v)) * rel, floor)

def solve():
    net = pn.case118()
    sf = np.clip(np.random.normal(1 + k * np.random.uniform(-1, 1), sigma_s, size=len(all_buses)), 0.7, 1.3)
    b2s = dict(zip(all_buses, sf))
    net.load["p_mw"] = base_load_p * [b2s[b] for b in load_buses]
    net.load["q_mvar"] = base_load_q * [b2s[b] for b in load_buses]
    net.gen["p_mw"] = base_gen_p * [b2s[b] for b in gen_buses]
    try: pp.runpp(net, init="flat", max_iteration=50, tolerance_mva=1e-6)
    except Exception: return None
    z = net.res_bus.reindex(nodelist)[["p_mw", "q_mvar", "vm_pu", "va_degree"]].values.copy()
    shunt = np.zeros((len(nodelist), 2))
    if len(net.res_shunt):
        for b, ps, qs in zip(net.shunt.bus.values, net.res_shunt.p_mw.values, net.res_shunt.q_mvar.values):
            if int(b) in pos: shunt[pos[int(b)]] = [ps, qs]
    return z, shunt

def bdd_chi2(meas):
    net = pn.case118(); net.measurement = net.measurement.iloc[0:0]
    for idx, bus in enumerate(nodelist):
        p, q, vm, va = meas[idx]
        create_measurement(net, "p", "bus", p, safe_std(p, 0.01), element=bus)
        create_measurement(net, "q", "bus", q, safe_std(q, 0.01), element=bus)
        create_measurement(net, "v", "bus", vm, safe_std(vm, 2e-4), element=bus)
        create_measurement(net, "va", "bus", va, safe_std(va, 57e-4), element=bus)
    try:
        det = bool(chi2_analysis(net, init="flat"))
    except Exception:
        return None, None
    est = net.res_bus_est.reindex(nodelist)[["p_mw", "q_mvar", "vm_pu", "va_degree"]].values.copy()
    return det, est

R = {c: [] for c in ["A_meas_noshunt", "B_xhat_raw", "B_xhat_noshunt", "A_meas_shunt"]}
nconv = 0
for _ in range(N * 3):
    if nconv >= N: break
    s = solve()
    if s is None: continue
    z_true, shunt = s
    z_ns = z_true.copy(); z_ns[:, 0] -= shunt[:, 0]; z_ns[:, 1] -= shunt[:, 1]   # shunt-corrected true
    zn = z_ns.copy()                                                              # noisy proper meas
    zn[:, 0] = np.random.normal(z_ns[:, 0], np.maximum(np.abs(z_ns[:, 0]) * 0.01, 1e-3))
    zn[:, 1] = np.random.normal(z_ns[:, 1], np.maximum(np.abs(z_ns[:, 1]) * 0.01, 1e-3))
    zn[:, 2] = np.clip(np.random.normal(z_ns[:, 2], np.maximum(np.abs(z_ns[:, 2]) * 2e-4, 1e-3)), V_MIN, V_MAX)
    zn[:, 3] = np.random.normal(z_ns[:, 3], np.maximum(np.abs(z_ns[:, 3]) * 57e-4, 1e-3))

    dA, xhat = bdd_chi2(zn)                     # A proper; xhat = WLS estimate (old X_t)
    if dA is None or xhat is None: continue
    nconv += 1
    R["A_meas_noshunt"].append(dA)
    # B raw: estimate as-is
    dB, _ = bdd_chi2(xhat); R["B_xhat_raw"].append(dB if dB is not None else True)
    # B noshunt: estimate with shunt subtracted (match A's convention)
    xh2 = xhat.copy(); xh2[:, 0] -= shunt[:, 0]; xh2[:, 1] -= shunt[:, 1]
    dB2, _ = bdd_chi2(xh2); R["B_xhat_noshunt"].append(dB2 if dB2 is not None else True)
    # A with shunt (measurement, but shunt NOT corrected) -> isolate shunt effect alone
    zs = zn.copy(); zs[:, 0] += shunt[:, 0]; zs[:, 1] += shunt[:, 1]
    dAs, _ = bdd_chi2(zs); R["A_meas_shunt"].append(dAs if dAs is not None else True)

print(f"\nIEEE-118 benign, {nconv} samples. chi2 FA = fraction of benign samples flagged.\n")
for c in ["A_meas_noshunt", "A_meas_shunt", "B_xhat_raw", "B_xhat_noshunt"]:
    v = np.array(R[c], float)
    print(f"  {c:16s} n={len(v):4d}   chi2 FA = {100*np.mean(v):5.1f}%")
