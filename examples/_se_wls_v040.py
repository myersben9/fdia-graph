#!/usr/bin/env python
"""Classical WLS state-estimation baseline (pandapower) — v0.4.1 retarget.

Same measurement construction / metric as the original _se_wls.py; only the data source and true-state source
are retargeted to the v0.4.1 release shards + pool. Reports |V|,theta MAE vs the TRUE (pool) state at metered
buses, per family, on a stratified TEST subset. Under a stealthy FDIA WLS fits the corrupted measurements ->
returns the attacker's FALSE state (the sanity check: WLS error on Aq must exceed its error on benign).
Note on units: pandapower's `create_measurement` API requires PHYSICAL engineering units (MW/MVAr/p.u./deg)
regardless of how the NN-based estimators are trained -- this is an external-library interface constraint,
not a modeling choice, so this script intentionally stays in physical units (the SE PINN scripts train in
per-unit internally but this classical baseline talks to pandapower directly).
Env: CASE (118), PER_FAM (150). CPU only. Seed 123."""
import os, json, numpy as np, h5py
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from pandapower.estimation import estimate
from pandapower.create import create_measurement

C = int(os.environ.get("CASE", "118")); PER_FAM = int(os.environ.get("PER_FAM", "150"))
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_v0.4.1")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
H5 = os.path.join(REL, f"ml_only_ieee{C}.h5"); POOL = os.path.join(REL, f"pool_ieee{C}.npz")
print(f"[data] reading {H5}", flush=True)
NET = {14: pn.case14, 118: pn.case118, 300: pn.case300}[C]
base = NET(); pp.runpp(base); nl = len(base.line); ntr = len(base.trafo)
SD = dict(pf=0.02, qf=0.02, v=0.005, pi=0.03, qi=0.03, va=0.005)
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
rng = np.random.default_rng(123)

with h5py.File(H5, "r") as f:
    d = f["data"]; A = {k: d[k][:] for k in ("node_x", "node_m", "edge_x", "edge_m", "family", "timestep", "split")}
N = A["node_x"].shape[1]
POOLX = np.load(POOL)["X"].astype(np.float32)                    # [T,N,4]=[Pinj,Qinj,|V|,theta]
Xtrue = POOLX[A["timestep"].astype(np.int64)]                    # [Nrec,N,4]

te = np.where(A["split"] == 2)[0]; teF = A["family"][te]
pick = []
for k in FAMILIES:
    idx = te[teF == k]
    if len(idx): pick.extend(rng.choice(idx, min(PER_FAM, len(idx)), replace=False).tolist())
pick = np.array(sorted(pick))
print(f"IEEE-{C}: WLS on {len(pick)} test records ({PER_FAM}/family)")


def wls_state(nx, nm, ex, em):
    est = NET()
    for e in range(nl):
        if em[e, 0]: create_measurement(est, "p", "line", ex[e, 0], max(abs(ex[e, 0]) * SD["pf"], 1e-3), element=e, side="from")
        if em[e, 1]: create_measurement(est, "q", "line", ex[e, 1], max(abs(ex[e, 1]) * SD["qf"], 1e-3), element=e, side="from")
    for ti in range(ntr):
        e = nl + ti
        if em[e, 0]: create_measurement(est, "p", "trafo", ex[e, 0], max(abs(ex[e, 0]) * SD["pf"], 1e-3), element=ti, side="hv")
        if em[e, 1]: create_measurement(est, "q", "trafo", ex[e, 1], max(abs(ex[e, 1]) * SD["qf"], 1e-3), element=ti, side="hv")
    for b in range(C):
        if nm[b, 1]: create_measurement(est, "p", "bus", nx[b, 1], max(abs(nx[b, 1]) * SD["pi"], 1e-3), element=b)
        if nm[b, 2]: create_measurement(est, "q", "bus", nx[b, 2], max(abs(nx[b, 2]) * SD["qi"], 1e-3), element=b)
        if nm[b, 0]: create_measurement(est, "v", "bus", nx[b, 0], SD["v"], element=b)
        if nm[b, 3]: create_measurement(est, "va", "bus", nx[b, 3], np.degrees(SD["va"]), element=b)
    try:
        if estimate(est, init="flat"):
            return est.res_bus_est.vm_pu.values.copy(), est.res_bus_est.va_degree.values.copy()
    except Exception:
        pass
    return None


accV = {k: [] for k in FAMILIES}; accT = {k: [] for k in FAMILIES}; nconv = {k: 0 for k in FAMILIES}; nfail = 0
for c, i in enumerate(pick):
    r = wls_state(A["node_x"][i], A["node_m"][i], A["edge_x"][i], A["edge_m"][i])
    k = int(A["family"][i])
    if r is None: nfail += 1; continue
    vm, va = r
    mv = A["node_m"][i, :, 0].astype(bool); mt = A["node_m"][i, :, 3].astype(bool)
    if mv.any(): accV[k].append(np.abs(vm[mv] - Xtrue[i, mv, 2]).mean())
    if mt.any(): accT[k].append(np.abs(va[mt] - Xtrue[i, mt, 3]).mean())
    nconv[k] += 1
    if (c + 1) % 200 == 0: print(f"  {c+1}/{len(pick)}  (fails {nfail})", flush=True)

res = {"system": f"ieee{C}", "seed": 123, "per_family": {}, "n_fail": nfail}
print(f"\n{'family':8s} {'|V| WLS':>10s} {'th WLS':>9s} {'n_conv':>7s}")
allV, allT = [], []
for k, name in FAMILIES.items():
    if not accV[k] and not accT[k]: continue
    v = float(np.mean(accV[k])) if accV[k] else None; t = float(np.mean(accT[k])) if accT[k] else None
    res["per_family"][name] = dict(V_mae_wls=round(v, 4) if v is not None else None,
                                   th_mae_wls=round(t, 3) if t is not None else None, n_conv=nconv[k])
    if k > 0:
        allV += accV[k]; allT += accT[k]
    print(f"{name:8s} {v if v is None else round(v,4):>10} {t if t is None else round(t,3):>9} {nconv[k]:>7d}")
res["overall_attacked"] = dict(V_mae_wls=round(float(np.mean(allV)), 4) if allV else None,
                               th_mae_wls=round(float(np.mean(allT)), 3) if allT else None)
print(f"\nATTACKED (all families): |V| WLS {res['overall_attacked']['V_mae_wls']}  theta WLS {res['overall_attacked']['th_mae_wls']} deg")
os.makedirs(RES, exist_ok=True)
outp = os.path.join(RES, f"se_{C}_wls.json")
json.dump(res, open(outp, "w"), indent=2)
print(f"wrote {outp}  (convergence failures: {nfail}/{len(pick)})")
