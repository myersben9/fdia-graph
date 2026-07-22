# -*- coding: utf-8 -*-
"""Prototype the MIMICRY attack (Am) and test it against BDD + a temporal-delta threshold.

Am idea: replace a target bus's load with a DIFFERENT but plausible value it genuinely had at another time
(so it is individually realistic), then re-solve the power flow (so it is spatially consistent). It should
evade BDD (spatial consistency) AND a per-bus temporal threshold (the value is in-range, blended), while
breaking the JOINT spatiotemporal correlation — the part only a learned model can catch.

This prototype tests the two CLASSICAL detectors (BDD, temporal threshold). If both are blind, that is the
green light to build Am as a family and run the learned localizer. Env: CASE, N (samples/arm).
"""
import os, warnings, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdia_graph._core import FdiaGenerator
import pandapower as pp
from pandapower.create import create_measurement
from pandapower.estimation import chi2_analysis

C = int(os.environ.get("CASE", "118")); N = int(os.environ.get("N", "150"))
g = FdiaGenerator(C, seed=123)
if os.environ.get("LOWNOISE"):        # accuracy-class realistic meter noise (~0.5-0.7%) instead of the inflated 3%
    g.SD = dict(pf=0.005, qf=0.005, v=0.002, pi=0.007, qi=0.007, va=0.002)
print(f"[noise] SD = {g.SD}", flush=True)
X = np.load(os.path.join(os.path.dirname(__file__), "release_v0.4.1", f"pool_ieee{C}.npz"))["X"]
nT = len(X); SD = g.SD; DAY = 1440
rng = np.random.default_rng(0)
nl, ntr = len(g.base.line), len(g.base.trafo)

def bdd_flag(nx, nm, ex, em):
    est = g.NET()
    for e in range(nl):
        if em[e,0]: create_measurement(est,"p","line",ex[e,0],max(abs(ex[e,0])*SD["pf"],1e-3),element=e,side="from")
        if em[e,1]: create_measurement(est,"q","line",ex[e,1],max(abs(ex[e,1])*SD["qf"],1e-3),element=e,side="from")
    for ti in range(ntr):
        e=nl+ti
        if em[e,0]: create_measurement(est,"p","trafo",ex[e,0],max(abs(ex[e,0])*SD["pf"],1e-3),element=ti,side="hv")
        if em[e,1]: create_measurement(est,"q","trafo",ex[e,1],max(abs(ex[e,1])*SD["qf"],1e-3),element=ti,side="hv")
    for b in range(g.C):
        if nm[b,1]: create_measurement(est,"p","bus",nx[b,1],max(abs(nx[b,1])*SD["pi"],1e-3),element=b)
        if nm[b,2]: create_measurement(est,"q","bus",nx[b,2],max(abs(nx[b,2])*SD["qi"],1e-3),element=b)
        if nm[b,0]: create_measurement(est,"v","bus",nx[b,0],SD["v"],element=b)
        if nm[b,3]: create_measurement(est,"va","bus",nx[b,3],np.degrees(SD["va"]),element=b)
    try: return bool(chi2_analysis(est, init="flat"))
    except Exception: return True

def temporal_stat(nx, Xprev, nm):
    # SAMPLE-level temporal statistic: the largest per-bus injection jump vs the previous scan, in units of
    # that bus's own benign step-change scale (a z-score). Using a z-score (not raw MW) stops big buses from
    # dominating, and taking the max is the natural "any bus moved anomalously" detector.
    mP = nm[:,1] > 0
    return float(np.max(np.abs(nx[mP,1] - Xprev[mP,0]) / SIG[mP]))

# per-bus benign step-change scale sigma_b (std of consecutive-scan |dP| on benign), for the z-score
D = np.abs(np.diff(X[:,:,0], axis=0)); SIG = np.maximum(D.std(0), 1.0)     # [N], floor 1 MW
# calibrate the SAMPLE-level threshold so benign false-alarm rate = 5% (the honest, multiple-comparisons-free way):
cal = []
for t in rng.integers(1, nT, 400):
    nx,nm,_,_ = g.emit_from_state(X[t]); cal.append(temporal_stat(nx, X[t-1], nm))
THR = np.percentile(cal, 95)                                               # 5% benign FA BY CONSTRUCTION
print(f"[cal] temporal sample-level z threshold (5% benign FA) = {THR:.2f}", flush=True)

res = {"benign": [0,0,0], "Am": [0,0,0]}   # [n, bdd_flagged, temporal_flagged]
for _ in range(N):
    t = int(rng.integers(1, nT))
    # ---- benign arm ----
    nx,nm,ex,em = g.emit_from_state(X[t])
    res["benign"][0]+=1; res["benign"][1]+=int(bdd_flag(nx,nm,ex,em)); res["benign"][2]+=int(temporal_stat(nx,X[t-1],nm)>THR)
    # ---- Am arm: MIMICRY. Splice in the target buses' load from a DIFFERENT day, but BLEND from the true
    # value at t-1 to the mimic value at t (a smooth 1-scan transition, no discontinuity), then re-solve.
    # The result is an individually-plausible load that is jointly out-of-sync with the rest of the grid.
    a = rng.choice(len(g.load_bus), rng.integers(1,4), replace=False)   # 1-3 target load buses
    Lp = X[t, g.load_bus, 0] + g.load_genP; Lq = X[t, g.load_bus, 1].copy(); Lp_true = Lp.copy()
    tprime = (t + int(rng.choice([-1,1]))*int(rng.integers(3,12))*DAY) % nT   # a plausible load this bus had, other day
    mimic = X[tprime, g.load_bus[a], 0] + g.load_genP[a]
    prevL = X[t-1, g.load_bus[a], 0] + g.load_genP[a]
    Lp = Lp.copy(); Lp[a] = 0.5*prevL + 0.5*mimic     # blended onset from the true previous scan toward the mimic
    net = g.solve(Lp, Lq, Xt=X[t], Lp_true=Lp_true)
    if net is None: continue
    nx2,nm2,ex2,em2 = g.emit(net)
    res["Am"][0]+=1; res["Am"][1]+=int(bdd_flag(nx2,nm2,ex2,em2)); res["Am"][2]+=int(temporal_stat(nx2,X[t-1],nm2)>THR)

print("\n            n   BDD-flagged   temporal-flagged", flush=True)
for k,(n,bd,td) in res.items():
    if n: print(f"  {k:7s} {n:4d}   {100*bd/n:6.1f}%       {100*td/n:6.1f}%", flush=True)
print("\n(Am is stealthy vs a detector if its flag rate ~= benign's. Both blind -> build Am + test the learned model.)", flush=True)
