#!/usr/bin/env python
"""Sensitivity analysis for Aq (stealthy load-scaling): at what attack magnitude does the attack become
indistinguishable from meter noise, so NO detector (ML included) can catch it?

For a grid of load-scale multipliers m (1.005 ... 1.5), we apply Aq to k random active-load buses on real
benign operating points, re-solve the AC power flow (pinned dispatch + AGC, the stealthy construction), and add
accuracy-class meter noise. Detectability is the ROC-AUC of separating the ATTACKED buses from the (noise-only)
un-attacked buses using the injection-deviation |measured P - true P| — the exact signal a per-bus detector sees.
We also report the SNR = attack shift / meter-noise sigma. When AUC -> 0.5 (SNR -> ~1), the attack IS noise:
an information-theoretic detectability floor, independent of model. Writes results/aq_sensitivity.json.

Env: SEED (default 123; multi-seed error-bar runs point at pool_ieee{C}_s{SEED}.npz and write
aq_sensitivity_s{SEED}.json). Note this script draws its own operating-point states straight from the
per-seed POOL (not the attack shard), so SEED here varies the underlying AC operating points the Aq scan
is run against -- a genuine independent replicate, not just a different RNG draw on the same states.
"""
import os, json, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE"); warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fdia_graph._core import FdiaGenerator

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
SH = os.environ.get("FDIA_LOCAL_SHARDS", os.path.join(HERE, "release_v0.4.1"))
SIG_P = 0.017          # accuracy-class relative meter noise on P (Asprou class-0.2 IT, 1.7%)
TIERS = [1.005, 1.01, 1.02, 1.03, 1.05, 1.08, 1.12, 1.20, 1.35, 1.50]
NS = int(os.environ.get("N_SAMP", "70")); K = 3   # attacked buses per record
SEED = int(os.environ.get("SEED", "123"))
SEED_SUF = "" if SEED == 123 else f"_s{SEED}"
OUT_NAME = "aq_sensitivity.json" if SEED == 123 else f"aq_sensitivity_s{SEED}.json"


def shunt_correct(net):
    Pi = net.res_bus.p_mw.values.copy()
    for i in range(len(net.shunt)):
        b = int(net.shunt.at[i, "bus"]); Pi[b] -= net.res_shunt.p_mw[i]
    return Pi


def run_system(C):
    g = FdiaGenerator(C, seed=SEED)                    # default seed=123 -> identical to the original hardcoded call
    X = np.load(os.path.join(SH, f"pool_ieee{C}{SEED_SUF}.npz"))["X"]
    # sample-draw rng: keep the ORIGINAL fixed seed 0 for the seed123 canonical run (bit-identical to the
    # pre-multi-seed baseline); for the 124/125 replicates key it off SEED so the draw is independent too.
    rng = np.random.default_rng(0 if SEED == 123 else SEED)
    ts = rng.choice(len(X), min(NS, len(X)), replace=False)
    apos = g.attackable_pos
    out = []
    for m in TIERS:
        atk_dev, ben_dev, snrs = [], [], []
        for t in ts:
            Xt = X[t]
            Lp = Xt[g.load_bus, 0] + g.load_genP; Lq = Xt[g.load_bus, 1].copy()
            a = rng.choice(apos, min(K, len(apos)), replace=False)
            Lp_atk = Lp.copy(); Lp_atk[a] *= m                        # scale the targeted loads
            net = g.solve(Lp_atk, Lq, Xt=Xt, Lp_true=Lp)              # stealthy re-solve (pinned + AGC)
            if net is None: continue
            true_P = Xt[:, 0]; atk_P = shunt_correct(net)             # true vs attacked injection
            noise = rng.normal(0, SIG_P * np.abs(atk_P))              # accuracy-class meter noise
            meas = atk_P + noise
            dev = np.abs(meas - true_P)                               # |attack shift + noise| the detector sees
            abus = set(int(b) for b in g.load_bus[a])
            for b in range(len(dev)):
                if b in abus: atk_dev.append(dev[b])
                elif dev[b] > 0 and np.abs(true_P[b]) > 1: ben_dev.append(dev[b])   # metered load buses, noise-only
            for b in g.load_bus[a]:
                s = np.abs(atk_P[int(b)] - true_P[int(b)])            # pure attack shift
                snrs.append(s / max(SIG_P * np.abs(atk_P[int(b)]), 1e-6))
        y = [1] * len(atk_dev) + [0] * len(ben_dev)
        s = atk_dev + ben_dev
        auc = float(roc_auc_score(y, s)) if (len(atk_dev) > 10 and len(ben_dev) > 10) else float("nan")
        out.append({"scale_pct": round((m - 1) * 100, 1), "auc": round(auc, 3),
                    "median_snr": round(float(np.median(snrs)), 2)})
        print(f"ieee{C}  +{(m-1)*100:5.1f}% load  ->  AUC {auc:.3f}   SNR {np.median(snrs):.2f}", flush=True)
    return out


res = {"note": "Aq detectability vs attack magnitude; AUC=separability of attacked vs noise-only buses (|inj dev|); "
               "SNR=attack shift / 1.7% meter noise. AUC~0.5 / SNR~1 => indistinguishable from noise.",
       "sigma_P_pct": SIG_P * 100, "systems": {}}
for C in (14, 118, 300):
    if os.path.exists(os.path.join(SH, f"pool_ieee{C}{SEED_SUF}.npz")):
        res["systems"][f"ieee{C}"] = run_system(C)
json.dump(res, open(os.path.join(RES, OUT_NAME), "w"), indent=2)
print(f"[done] results/{OUT_NAME}", flush=True)
