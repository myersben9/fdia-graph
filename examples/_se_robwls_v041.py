#!/usr/bin/env python
"""Oracle robust-WLS ceiling test.

Question: if an ORACLE tells WLS exactly which buses are attacked and we drop those buses' bus-injection /
voltage / angle measurements before solving, how much of the true state does classical WLS recover -- per
attack family? This is the ceiling for a GNN-guided robust WLS (localizer flags -> drop -> WLS): the real
pipeline can only do as well as the oracle, and only on the families where dropping bad data actually helps.

Expectation (the whole point of measuring it): dropping bad data recovers the CRUDE, BDD-inconsistent families
(Ad/As/Ar) but NOT the STEALTHY ones (Aq/At/Al) -- a stealthy re-solved attack is a valid power-flow solution,
so the surviving measurements are self-consistent and WLS re-fits the FALSE state even with the attacked buses
dropped. That division of labor (robust-WLS for crude, learned prior for stealthy) is the result we want on
paper.

Runs plain WLS and oracle-robust WLS on the same samples; reports |V|/theta MAE vs the TRUE (pool) state at
metered and unmetered buses, per family. Physical units (pandapower interface). CPU only. Seed 123.
Env: CASE (14), PER_FAM (100), WLS_MAXIT (20)."""
import os, json, numpy as np, h5py
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import warnings; warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from pandapower.estimation import estimate
from pandapower.create import create_measurement

C = int(os.environ.get("CASE", "14")); PER_FAM = int(os.environ.get("PER_FAM", "100"))
MAXIT = int(os.environ.get("WLS_MAXIT", "20"))
REL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_v0.4.1")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
H5 = os.path.join(REL, f"ml_only_ieee{C}.h5"); POOL = os.path.join(REL, f"pool_ieee{C}.npz")
print(f"[data] reading {H5}", flush=True)
NET = {14: pn.case14, 118: pn.case118, 300: pn.case300}[C]
base = NET(); pp.runpp(base); nl = len(base.line); ntr = len(base.trafo)
SD = dict(pf=0.02, qf=0.02, v=0.005, pi=0.03, qi=0.03, va=0.005)
FAMILIES = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}
STEALTHY = {1, 5, 6}
rng = np.random.default_rng(123)

with h5py.File(H5, "r") as f:
    d = f["data"]; A = {k: d[k][:] for k in ("node_x", "node_m", "edge_x", "edge_m", "family", "timestep", "split", "y")}
N = A["node_x"].shape[1]
POOLX = np.load(POOL)["X"].astype(np.float32)                    # [T,N,4]=[Pinj,Qinj,|V|,theta]
Xtrue = POOLX[A["timestep"].astype(np.int64)]                    # [Nrec,N,4]

te = np.where(A["split"] == 2)[0]; teF = A["family"][te]
pick = []
for k in FAMILIES:
    idx = te[teF == k]
    if len(idx): pick.extend(rng.choice(idx, min(PER_FAM, len(idx)), replace=False).tolist())
pick = np.array(sorted(pick))
print(f"IEEE-{C}: robust-WLS ceiling on {len(pick)} test records ({PER_FAM}/family)", flush=True)


def wls_state(nx, nm, ex, em):
    """WLS from the given measurement masks. Zeroing nm rows at attacked buses = dropping their measurements."""
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
        if estimate(est, init="flat", maximum_iterations=MAXIT):
            return est.res_bus_est.vm_pu.values.copy(), est.res_bus_est.va_degree.values.copy()
    except Exception:
        pass
    return None


def maes(vm, va, i, mask):
    """MAE of (vm,va) vs true state over the given bus mask; returns (V_mae, th_mae) or (None,None)."""
    if not mask.any(): return None, None
    return (float(np.abs(vm[mask] - Xtrue[i, mask, 2]).mean()),
            float(np.abs(va[mask] - Xtrue[i, mask, 3]).mean()))


# accumulators: [family] -> list, for plain/robust x metered/unmetered x V/th
acc = {tag: {k: {"Vm": [], "Tm": [], "Vu": [], "Tu": []} for k in FAMILIES} for tag in ("plain", "robust")}
nfail = {"plain": 0, "robust": 0}
for c, i in enumerate(pick):
    k = int(A["family"][i]); nm = A["node_m"][i]; y = A["y"][i].astype(bool)
    mvt = nm[:, 0].astype(bool); mtt = nm[:, 3].astype(bool)      # metered masks for V and theta
    for tag in ("plain", "robust"):
        nm_use = nm.copy()
        if tag == "robust":
            nm_use[y] = 0                                          # drop attacked buses' bus measurements
        r = wls_state(A["node_x"][i], nm_use, A["edge_x"][i], A["edge_m"][i])
        if r is None: nfail[tag] += 1; continue
        vm, va = r
        Vm, _ = maes(vm, va, i, mvt); Vu, _ = maes(vm, va, i, ~mvt)   # V error on V-metered / V-unmetered buses
        _, Tm2 = maes(vm, va, i, mtt); _, Tu2 = maes(vm, va, i, ~mtt)  # th error on th-metered / th-unmetered buses
        if Vm is not None: acc[tag][k]["Vm"].append(Vm)
        if Vu is not None: acc[tag][k]["Vu"].append(Vu)
        if Tm2 is not None: acc[tag][k]["Tm"].append(Tm2)
        if Tu2 is not None: acc[tag][k]["Tu"].append(Tu2)
    if (c + 1) % 100 == 0: print(f"  {c+1}/{len(pick)}", flush=True)


def m(x): return round(float(np.mean(x)), 4) if x else None
def m3(x): return round(float(np.mean(x)), 3) if x else None
res = {"system": f"ieee{C}", "seed": 123, "maxit": MAXIT, "per_family": {}}
print(f"\n{'family':8s} | {'plain th_met':>12s} {'robust th_met':>13s} | {'plain th_unmet':>14s} {'robust th_unmet':>15s}  (deg)")
for k, name in FAMILIES.items():
    p = acc["plain"][k]; r = acc["robust"][k]
    row = dict(
        plain=dict(V_met=m(p["Vm"]), th_met=m3(p["Tm"]), V_unmet=m(p["Vu"]), th_unmet=m3(p["Tu"])),
        robust=dict(V_met=m(r["Vm"]), th_met=m3(r["Tm"]), V_unmet=m(r["Vu"]), th_unmet=m3(r["Tu"])),
        stealthy=(k in STEALTHY))
    res["per_family"][name] = row
    print(f"{name:8s} | {str(row['plain']['th_met']):>12s} {str(row['robust']['th_met']):>13s} | "
          f"{str(row['plain']['th_unmet']):>14s} {str(row['robust']['th_unmet']):>15s}   {'STEALTHY' if k in STEALTHY else ''}")
res["fails"] = nfail
os.makedirs(RES, exist_ok=True)
outp = os.path.join(RES, f"se_{C}_robwls_oracle.json")
json.dump(res, open(outp, "w"), indent=2)
print(f"\nfails: {nfail}\nwrote {outp}", flush=True)
