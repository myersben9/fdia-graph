#!/usr/bin/env python
"""Quantify the dispatch-pinning + AGC re-solve for the report's stealth-method page.

For N benign IEEE-118 operating points, reconstruct the loads and re-solve the AC power flow TWO ways, then
compare the re-solved bus injections to the TRUE stored state:
  NAIVE  : set net.load, leave generators at base setpoints -> the slack bus absorbs the whole imbalance.
  PINNED : our solve() -> generation pinned to the true dispatch, the (zero) load delta AGC-distributed.
A perfect re-solve of a no-op (alpha=1) attack must reproduce the true state exactly; the gap is the
"dispatch artifact" that would otherwise contaminate the attack residual. Writes results/dispatch_pinning.json.
"""
import os, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); warnings.filterwarnings("ignore")
import numpy as np, pandapower as pp, pandapower.networks as pn
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdia_graph._core import FdiaGenerator

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SH = os.environ.get("FDIA_LOCAL_SHARDS", os.path.join(HERE, "release_v0.4.1"))
C = 118; N = int(os.environ.get("N_SAMP", "60"))

g = FdiaGenerator(C)
X = np.load(os.path.join(SH, f"pool_ieee{C}.npz"))["X"]          # [T, N, 4] true benign states
rng = np.random.default_rng(0); ts = rng.choice(len(X), min(N, len(X)), replace=False)


def shunt_correct(net):
    """res_bus P/Q with the shunt draw removed (the SE-consistent injection, matching the stored state)."""
    Pi = net.res_bus.p_mw.values.copy(); Qi = net.res_bus.q_mvar.values.copy()
    for i in range(len(net.shunt)):
        b = int(net.shunt.at[i, "bus"]); Pi[b] -= net.res_shunt.p_mw[i]; Qi[b] -= net.res_shunt.q_mvar[i]
    return Pi


naive_mae, naive_max, pin_mae, pin_max, slack_swing = [], [], [], [], []
for t in ts:
    Xt = X[t]
    Lp = Xt[g.load_bus, 0] + g.load_genP; Lq = Xt[g.load_bus, 1].copy()   # reconstruct loads from the true state
    true_P = Xt[:, 0]                                                     # true (shunt-corrected) bus injection
    sb = int(g.base.ext_grid.bus.values[0])                              # slack bus index

    # --- NAIVE: default generators, slack absorbs the imbalance ---
    net = pn.case118(); net.load["p_mw"] = Lp; net.load["q_mvar"] = Lq
    try: pp.runpp(net)
    except Exception: continue
    Pn = shunt_correct(net)
    naive_mae.append(float(np.mean(np.abs(Pn - true_P)))); naive_max.append(float(np.max(np.abs(Pn - true_P))))
    slack_swing.append(float(abs(Pn[sb] - true_P[sb])))                  # how far the slack bus moved

    # --- PINNED + AGC: our solve() with alpha=1 (Lp == Lp_true, no attack) ---
    netp = g.solve(Lp, Lq, Xt=Xt, Lp_true=Lp)
    if netp is None: continue
    Pp = shunt_correct(netp)
    pin_mae.append(float(np.mean(np.abs(Pp - true_P)))); pin_max.append(float(np.max(np.abs(Pp - true_P))))

out = {"system": f"ieee{C}", "n": len(pin_mae),
       "naive": {"mean_inj_err_MW": round(np.mean(naive_mae), 2), "max_inj_err_MW": round(np.mean(naive_max), 1),
                 "slack_swing_MW": round(np.mean(slack_swing), 1)},
       "pinned": {"mean_inj_err_MW": round(np.mean(pin_mae), 4), "max_inj_err_MW": round(np.mean(pin_max), 4)}}
json.dump(out, open(os.path.join(RES, "dispatch_pinning.json"), "w"), indent=2)
print(json.dumps(out, indent=2), flush=True)
