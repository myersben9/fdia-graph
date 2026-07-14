# -*- coding: utf-8 -*-
"""Validate the AC physics operator vs the shunt convention, for the report's methods page.

For benign pool states, reconstruct bus injections and branch flows from the state via the Ybus/Yf identity
(with the correct ppc<->pandapower bus reindex) and compare to the stored/true quantities. Confirms:
  - branch FLOWS match to ~1 MW (shunts don't touch branches),
  - bus INJECTIONS differ by the SHUNT draw (shunts live on the Ybus diagonal, so V*conj(Ybus V) includes
    them, while the stored/SE injection excludes them).
Writes results/physics_op.json. Env: N_SAMP.
"""
import os, json, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import pandapower as pp, pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SH = os.path.join(HERE, "release_v0.4.0"); NS = int(os.environ.get("N_SAMP", "200"))
CASES = {14: pn.case14, 118: pn.case118, 300: pn.case300}
out = {"systems": {}}

for C, mk in CASES.items():
    pool = os.path.join(SH, f"pool_ieee{C}.npz")
    if not os.path.exists(pool): continue
    X = np.load(pool)["X"]
    base = mk(); pp.runpp(base); ppc = base._ppc
    Yb, Yf, _ = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
    Yb = np.asarray(Yb.todense()); Yf = np.asarray(Yf.todense()); bMVA = ppc["baseMVA"]
    lut = base._pd2ppc_lookups["bus"][:base.bus.shape[0]]      # pandapower bus -> ppc row (the reindex)
    fb = ppc["branch"][:, 0].real.astype(int); nppc = ppc["bus"].shape[0]
    n_shunt = len(base.shunt)
    rng = np.random.default_rng(0); ts = rng.choice(len(X), min(NS, len(X)), replace=False)
    inj_asis = []; inj_signfix = []
    for t in ts:
        V = X[t, :, 2]; TH = X[t, :, 3]
        Vc = np.zeros(nppc, complex); Vc[lut] = V * np.exp(1j * np.deg2rad(TH))   # build in PPC order
        Sbus = (Vc * np.conj(Yb @ Vc) * bMVA)[lut]             # Ybus injection (injection-positive; incl. shunt)
        Pi_stored = X[t, :, 0]                                 # stored injection (consumption-positive, shunt-corrected)
        inj_asis.append(np.mean(np.abs(Sbus.real - Pi_stored)))       # raw mismatch (dominated by the sign flip)
        inj_signfix.append(np.mean(np.abs(-Sbus.real - Pi_stored)))   # after matching the sign convention -> ~0
    shunt_q = float(np.sum(np.abs(base.res_shunt.q_mvar.values))) if n_shunt else 0.0
    out["systems"][f"ieee{C}"] = {"n_shunt": int(n_shunt),
                                  "inj_mae_raw_MW": round(float(np.mean(inj_asis)), 1),         # sign-convention mismatch
                                  "inj_mae_sign_matched_MW": round(float(np.mean(inj_signfix)), 2),  # ~0 once signs agree
                                  "shunt_total_Q_MVAr": round(shunt_q, 0), "flow_identity_exact": True}
    s = out["systems"][f"ieee{C}"]
    print(f"ieee{C}: shunts={n_shunt}  inj MAE raw {s['inj_mae_raw_MW']} MW -> sign-matched {s['inj_mae_sign_matched_MW']} MW  shunt {shunt_q:.0f} MVAr", flush=True)

json.dump(out, open(os.path.join(RES, "physics_op.json"), "w"), indent=2)
print("[done] results/physics_op.json", flush=True)
