# -*- coding: utf-8 -*-
"""VALIDATE the operator-impact setup on ~10 samples before the full sweep.
Checks: (a) alpha=1 re-solve reproduces the TRUE stored state (V/theta close) -> faithfulness,
        (b) null-attack (mult=1.0) gives EXACTLY 0 line-loading misjudgments -> plumbing control,
        (c) a real Aq attack actually MOVES line loadings.
Also confirms res_line.loading_percent is populated (needs max_i_ka on the case)."""
import os, sys, numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from fdia_graph._core import FdiaGenerator

C = int(os.environ.get("CASE", "118"))
HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "release_v0.4.0", f"pool_ieee{C}.npz")
X = np.load(POOL)["X"].astype(np.float64)
print(f"[pool] ieee{C} X={X.shape}")

g = FdiaGenerator(C, seed=123)
g._pick_lra_target(0.15, min(6, len(g.load_bus)), n_targets=15)
rng = np.random.default_rng(123)
nl = g.nl

def loadings(net):
    return net.res_line.loading_percent.values.copy()

def solve_true(Xt):
    Lp = Xt[g.load_bus, 0] + g.load_genP; Lq = Xt[g.load_bus, 1].copy()
    net = g.solve(Lp.copy(), Lq.copy(), Xt=Xt, Lp_true=Lp.copy())
    return net, Lp, Lq

# check max_i_ka present
base = g.NET(); g.pp.runpp(base)
print(f"[case] lines={nl} max_i_ka set: {base.line.max_i_ka.notna().all()} "
      f"true base max loading%={np.nanmax(base.res_line.loading_percent.values):.1f}")

idx = rng.choice(len(X), 10, replace=False)
vth_err, null_flips, atk_moved = [], [], []
for t in idx:
    Xt = X[int(t)]
    net_true, Lp, Lq = solve_true(Xt)
    if net_true is None:
        print(f"  t={t} true solve FAILED"); continue
    Ltrue = loadings(net_true)
    # (a) faithfulness: compare solved V/theta to stored
    v_err = np.abs(net_true.res_bus.vm_pu.values - Xt[:, 2]).max()
    th_err = np.abs(net_true.res_bus.va_degree.values - Xt[:, 3]).max()
    vth_err.append((v_err, th_err))
    # (b) null attack
    net_null = g.solve(Lp.copy(), Lq.copy(), Xt=Xt, Lp_true=Lp.copy())
    Lnull = loadings(net_null)
    null_flips.append(np.sum((Ltrue > 100) != (Lnull > 100)))
    # (c) real Aq
    k = int(rng.integers(1, min(6, len(g.load_bus)) + 1))
    a = rng.choice(len(g.load_bus), k, replace=False)
    mult = 1 + rng.uniform(0.05, 0.15, size=k)
    Lp_atk = Lp.copy(); Lp_atk[a] *= mult
    net_atk = g.solve(Lp_atk, Lq.copy(), Xt=Xt, Lp_true=Lp.copy())
    if net_atk is None:
        print(f"  t={t} Aq solve failed"); continue
    Latk = loadings(net_atk)
    atk_moved.append(np.abs(Latk - Ltrue).max())
    print(f"  t={t:6d} Verr={v_err:.2e} THerr={th_err:.2e} null_flip={null_flips[-1]} "
          f"Aq k={k} max|dLoad|={atk_moved[-1]:.3f}%pt  base_maxload={np.nanmax(Ltrue):.1f}%")

vth = np.array(vth_err)
print("\n=== VALIDATION SUMMARY ===")
print(f"faithfulness: max V err={vth[:,0].max():.2e} pu, max theta err={vth[:,1].max():.2e} deg (want ~0)")
print(f"null-attack control: total status flips across 10 samples = {int(np.sum(null_flips))} (want 0)")
print(f"Aq moves loadings: median max|dLoad|={np.median(atk_moved):.3f}%pt, max={np.max(atk_moved):.3f}%pt (want >0)")
