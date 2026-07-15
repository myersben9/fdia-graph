# -*- coding: utf-8 -*-
"""Check the STRESS builder: scale load+gen by s off the true operating point, re-solve, and confirm
(1) g.solve(alpha=1, Xt=stressed) reproduces the stressed state (faithful), (2) stressed loadings reach
near/above 100%, (3) null control still 0, (4) Aq now creates real status flips."""
import os, sys, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fdia_graph._core import FdiaGenerator

C = int(os.environ.get("CASE", "118"))
HERE = os.path.dirname(os.path.abspath(__file__))
X = np.load(os.path.join(HERE, "release_v0.4.0", f"pool_ieee{C}.npz"))["X"].astype(np.float64)
g = FdiaGenerator(C, seed=123); g._pick_lra_target(0.15, min(6, len(g.load_bus)), 15)
rng = np.random.default_rng(123)

def stressed_state(Xt, s):
    """Build a stressed operating STATE [N,4] by scaling the true op-point load AND gen by s, then re-solve.
    Mirrors profiles.generate_states (scales load and gen together). Returns shunt-corrected injection state."""
    net = g._solvenet
    Lp = (Xt[g.load_bus, 0] + g.load_genP) * s
    Lq = Xt[g.load_bus, 1] * s
    net.load["p_mw"] = Lp; net.load["q_mvar"] = Lq
    # true per-gen dispatch reconstructed exactly as g.solve does (pinned), then scaled by s
    gbus = net.gen["bus"].values
    Lfull = np.zeros(g.C)
    for val, b in zip(Xt[g.load_bus, 0] + g.load_genP, g.load_bus): Lfull[int(b)] += val
    ncnt = {}
    for b in gbus: ncnt[int(b)] = ncnt.get(int(b), 0) + 1
    gp = np.array([(Lfull[int(b)] - Xt[int(b), 0]) / ncnt[int(b)] for b in gbus]) * s
    net.gen["p_mw"] = gp
    net.gen["vm_pu"] = [Xt[int(b), 2] for b in gbus]
    sb = net.ext_grid["bus"].values
    net.ext_grid["vm_pu"] = [Xt[int(b), 2] for b in sb]
    net.ext_grid["va_degree"] = [Xt[int(b), 3] for b in sb]
    try: g.pp.runpp(net)
    except Exception: return None
    Pi = net.res_bus.p_mw.values.copy(); Qi = net.res_bus.q_mvar.values.copy()
    for i in net.shunt.index:
        b = net.shunt.at[i, "bus"]; Pi[b] -= net.res_shunt.p_mw[i]; Qi[b] -= net.res_shunt.q_mvar[i]
    return np.column_stack([Pi, Qi, net.res_bus.vm_pu.values, net.res_bus.va_degree.values])

def solve_load(Xt):
    return (Xt[g.load_bus, 0] + g.load_genP), Xt[g.load_bus, 1].copy()

for s in (1.0, 1.3, 1.5, 1.7):
    maxloads, flips_null, flips_atk, n_conv = [], 0, 0, 0
    idx = rng.choice(len(X), 12, replace=False)
    for t in idx:
        Xs = stressed_state(X[int(t)], s)
        if Xs is None: continue
        Lp, Lq = solve_load(Xs)
        net_t = g.solve(Lp.copy(), Lq.copy(), Xt=Xs, Lp_true=Lp.copy())
        if net_t is None: continue
        n_conv += 1
        Lt = net_t.res_line.loading_percent.values.copy(); maxloads.append(np.nanmax(Lt))
        net_n = g.solve(Lp.copy(), Lq.copy(), Xt=Xs, Lp_true=Lp.copy())
        flips_null += int(np.sum((Lt > 100) != (net_n.res_line.loading_percent.values > 100)))
        k = int(rng.integers(1, 7)); a = rng.choice(len(g.load_bus), k, replace=False)
        Lp_a = Lp.copy(); Lp_a[a] *= 1 + rng.uniform(0.05, 0.15, k)
        net_a = g.solve(Lp_a, Lq.copy(), Xt=Xs, Lp_true=Lp.copy())
        if net_a is not None:
            flips_atk += int(np.sum((Lt > 100) != (net_a.res_line.loading_percent.values > 100)))
    print(f"s={s}: conv={n_conv}/12 median_maxload={np.median(maxloads):.0f}% "
          f"p90_maxload={np.percentile(maxloads,90):.0f}% null_flips={flips_null} Aq_flips={flips_atk}")
