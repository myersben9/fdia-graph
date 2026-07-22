# -*- coding: utf-8 -*-
"""Bad-data detection (chi-square + largest-normalized-residual) on the NEW release shards, on the ACTUAL
stored sparse measurements. Run pandapower WLS to get x_hat, form r = z - h(x_hat) with the validated AC
operator (Ybus/Yf) on FLOWS + |V| + theta (bus P/Q injection excluded: the stored injection uses a
shunt-corrected convention that differs from the Ybus injection by a fixed per-bus offset). J = sum(r/sig)^2
is the chi-square statistic; max|r/sig| is the LNR. Writes results/bdd_summary_{C}.json + bdd_resid_{C}.npz.

Env: CASE (14/118/300), N_PER (samples/family), SHARD_DIR (default release_v0.4.1), SEED (default 123;
multi-seed error-bar runs: SEED=124/125 read ml_only_ieee{C}_s{SEED}.h5 instead of the unsuffixed canonical
seed-123 shard, and write bdd_summary_{C}_s{SEED}.json so per-seed results don't clobber each other)."""
import os, json, numpy as np, h5py, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); warnings.filterwarnings("ignore")
import pandapower as pp, pandapower.networks as pn
from pandapower.create import create_measurement
from pandapower.estimation import estimate, chi2_analysis
from pandapower.pypower.makeYbus import makeYbus

HERE = os.path.dirname(os.path.abspath(__file__))
C = int(os.environ.get("CASE", "118")); N_PER = int(os.environ.get("N_PER", "120"))
SHARD_DIR = os.environ.get("SHARD_DIR", os.path.join(HERE, "release_v0.4.1"))
SEED = int(os.environ.get("SEED", "123"))
SEED_SUF = "" if SEED == 123 else f"_s{SEED}"        # seed123 == the canonical unsuffixed shard
OUT_TAG = f"{C}" if SEED == 123 else f"{C}_s{SEED}"  # keeps existing bdd_summary_{C}.json filename for seed123
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
NET = {14: pn.case14, 118: pn.case118, 300: pn.case300}[C]
base = NET(); pp.runpp(base); nl = len(base.line); ntr = len(base.trafo)
SD = dict(pf=0.017, qf=0.017, v=0.0012, pi=0.017, qi=0.017, va=0.00168)
FAM = {0: "benign", 1: "Aq", 2: "Ad", 3: "As", 4: "Ar", 5: "At", 6: "Al"}   # new A-taxonomy names
LNR_THR = 3.0; rng = np.random.default_rng(123)

ppc = base._ppc; Ybus, Yf, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"]); bMVA = ppc["baseMVA"]
Ybus = np.asarray(Ybus.todense()); Yf = np.asarray(Yf.todense())
lut = base._pd2ppc_lookups["bus"][:base.bus.shape[0]]; fb = ppc["branch"][:, 0].real.astype(int); nppc = ppc["bus"].shape[0]

def h_of_state(Vmag, th_deg):
    Vc = np.zeros(nppc, complex); Vc[lut] = Vmag * np.exp(1j * np.deg2rad(th_deg))
    Sf = Vc[fb] * np.conj(Yf @ Vc) * bMVA
    return Sf.real, Sf.imag

with h5py.File(os.path.join(SHARD_DIR, f"ml_only_ieee{C}{SEED_SUF}.h5"), "r") as f:
    d = f["data"]; A = {k: d[k][:] for k in ("node_x", "node_m", "edge_x", "edge_m", "family")}
N = A["node_x"].shape[1]; E = A["edge_x"].shape[1]

def build_meas(nx, nm, ex, em):
    est = NET()
    for e in range(nl):
        if em[e, 0]: create_measurement(est, "p", "line", ex[e, 0], max(abs(ex[e, 0]) * SD["pf"], 1e-3), element=e, side="from")
        if em[e, 1]: create_measurement(est, "q", "line", ex[e, 1], max(abs(ex[e, 1]) * SD["qf"], 1e-3), element=e, side="from")
    for ti in range(ntr):
        e = nl + ti
        if em[e, 0]: create_measurement(est, "p", "trafo", ex[e, 0], max(abs(ex[e, 0]) * SD["pf"], 1e-3), element=ti, side="hv")
        if em[e, 1]: create_measurement(est, "q", "trafo", ex[e, 1], max(abs(ex[e, 1]) * SD["qf"], 1e-3), element=ti, side="hv")
    for b in range(N):
        if nm[b, 1]: create_measurement(est, "p", "bus", nx[b, 1], max(abs(nx[b, 1]) * SD["pi"], 1e-3), element=b)
        if nm[b, 2]: create_measurement(est, "q", "bus", nx[b, 2], max(abs(nx[b, 2]) * SD["qi"], 1e-3), element=b)
        if nm[b, 0]: create_measurement(est, "v", "bus", nx[b, 0], SD["v"], element=b)
        if nm[b, 3]: create_measurement(est, "va", "bus", nx[b, 3], np.degrees(SD["va"]), element=b)
    return est

def run_bdd(nx, nm, ex, em):
    est = build_meas(nx, nm, ex, em)
    try:
        detected = bool(chi2_analysis(est, init="flat"))
    except Exception:
        return True, True, np.nan, np.nan, False
    if not hasattr(est, "res_bus_est") or est.res_bus_est is None:
        return True, True, np.nan, np.nan, False
    Vh = est.res_bus_est.vm_pu.values; Th = est.res_bus_est.va_degree.values
    Pfh, Qfh = h_of_state(Vh, Th)
    rs = []
    for b in range(N):
        if nm[b, 0]: rs.append((nx[b, 0] - Vh[b]) / SD["v"])
        if nm[b, 3]: rs.append((nx[b, 3] - Th[b]) / np.degrees(SD["va"]))
    for e in range(E):
        if em[e, 0]: rs.append((ex[e, 0] - Pfh[e]) / max(abs(ex[e, 0]) * SD["pf"], 1e-3))
        if em[e, 1]: rs.append((ex[e, 1] - Qfh[e]) / max(abs(ex[e, 1]) * SD["qf"], 1e-3))
    r = np.array(rs); return detected, float(np.max(np.abs(r))) > LNR_THR, float(np.sum(r ** 2)), float(np.max(np.abs(r))), True

summary = {"system": f"ieee{C}", "families": {}}; resid = {}
for k, name in FAM.items():
    ids = np.where(A["family"] == k)[0]
    if len(ids) == 0: continue
    ids = rng.choice(ids, min(N_PER, len(ids)), replace=False)
    c2, lnrd, cf, Js, Ls = 0, 0, 0, [], []
    for i in ids:
        cd, ld, J, lnr, ok = run_bdd(A["node_x"][i], A["node_m"][i], A["edge_x"][i], A["edge_m"][i])
        c2 += int(cd); lnrd += int(ld)
        if ok: Js.append(J); Ls.append(lnr)
        else: cf += 1
    n = len(ids)
    summary["families"][name] = dict(n=int(n), chi2_detect_pct=round(100 * c2 / n, 1),
        lnr_detect_pct=round(100 * lnrd / n, 1), conv_fail_pct=round(100 * cf / n, 1),
        median_J=round(float(np.median(Js)), 1) if Js else None, median_lnr=round(float(np.median(Ls)), 2) if Ls else None)
    resid[f"ieee{C}_{name}_J"] = np.array(Js); resid[f"ieee{C}_{name}_lnr"] = np.array(Ls)
    s = summary["families"][name]
    print(f"[ieee{C}/{name:6s}] chi2 {s['chi2_detect_pct']:5.1f}%  medJ {s['median_J']}  med_lnr {s['median_lnr']}", flush=True)

json.dump(summary, open(os.path.join(RES, f"bdd_summary_{OUT_TAG}.json"), "w"), indent=2)
np.savez_compressed(os.path.join(RES, f"bdd_resid_{OUT_TAG}.npz"), **resid)
print(f"[done] results/bdd_summary_{OUT_TAG}.json + bdd_resid_{OUT_TAG}.npz")
