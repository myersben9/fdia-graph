"""Fit the fdia_graph.localization methods on one system and write results/loc_<system>.json.

Two protocols, one results file:

1. Common protocol: every method fits on the unfiltered train split and is calibrated on benign
   records at the same false-alarm budget, then scores the full test split. The learned arms train
   on all six families here, so the slow ramp At is in-distribution.
2. Papers' zero-shot protocol: train and val hold benign + Aq + Ad only, test adds As and Ar, the
   threshold is the validation-best global tau. This is the protocol behind the federated
   localization paper's headline numbers.

Set FG_SYSTEM (default ieee14; the README covers ieee14, ieee118, ieee300). Tables and figures come
from make_report.py, which reads the JSON, so plots can be restyled without re-running.
ResidualLocalizer and the Jacobian feature arms need the [se] extra; BusCNN/BusMLP need [torch] and
use the GPU when visible.

Each arm's per-bus test scores and calibrated thresholds are cached in results/cache/ as soon as it
finishes (the residual arm's state-estimation solves and the learned arms' training are the slow
parts), so a re-run scores from the cache in seconds. Delete a cache file to recompute that arm.
"""

import json
import os
import time

import numpy as np

import fdia_graph as fg
from fdia_graph.localization import BusCNN, BusMLP, DeltaThreshold, ResidualLocalizer, SwingThreshold

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)


def run(protocol, name, m, train, test, val=None):
    """Fit (or restore thresholds from the cache), then score the test split from cached scores."""
    t0 = time.time()
    f = os.path.join(CACHE, f"loc_{SYSTEM}_{protocol}_{name}.npz")
    if os.path.exists(f):
        with np.load(f) as z:
            m.thr, s = z["thr"], z["scores"]
            m.tau = float(z["tau"]) if "tau" in z.files and not np.isnan(z["tau"]) else None
        how = "cached"
    else:
        m.fit(train, val=val) if val is not None else m.fit(train)
        s = m.scores(test)
        np.savez_compressed(f, scores=s, thr=m.thr, tau=np.nan if getattr(m, "tau", None) is None else m.tau)
        how = "fitted"
    r = m.score(test, scores=s)
    if getattr(m, "tau", None) is not None:
        r = {"tau": m.tau, **r}
    print(
        f"  {protocol:9s} {name:8s} {how:7s} {time.time() - t0:6.0f}s  macro-F1 {r['all']['macro_f1']:.4f}  FR {r['all']['macro_fr']:.4f}"
    )
    return r


# ---- 1. common protocol: same train, same FA budget, full test ------------------------------
train = fg.load(SYSTEM, split="train")
test = fg.load(SYSTEM, split="test")
methods = {
    "swing": SwingThreshold(),
    "delta": DeltaThreshold(),
    "residual": ResidualLocalizer(),
    "mlp": BusMLP(),
    "cnn": BusCNN(),
    "cnn+jac": BusCNN(features="full14+jac"),  # the digest's Model C, every family in-distribution
}
report = {name: run("common", name, m, train, test) for name, m in methods.items()}

# ---- 2. papers' zero-shot protocol: benign + Aq + Ad seen, As + Ar unseen, val-tuned tau -----
zs = dict(families=[0, 1, 2])
ztr, zva = fg.load(SYSTEM, split="train", **zs), fg.load(SYSTEM, split="val", **zs)
zte = fg.load(SYSTEM, split="test", families=[0, 1, 2, 3, 4])
# The Jacobian-informed digest's ablation on the best encoder: A measurements only, B the papers'
# 14-dim vector (the existing "cnn" row), C = B + the 8 Jacobian features, D = Jacobian features only.
zs_methods = {
    "mlp": BusMLP(),
    "cnn": BusCNN(),
    "cnn_meas": BusCNN(features="meas"),
    "cnn+jac": BusCNN(features="full14+jac"),
    "cnn_jac": BusCNN(features="jac"),
}
zero_shot = {name: run("zero_shot", name, m, ztr, zte, val=zva) for name, m in zs_methods.items()}
zero_shot["swing"] = run("zero_shot", "swing", SwingThreshold(), ztr, zte)  # the feature alone, FA-calibrated
report["zero_shot"] = zero_shot

with open(os.path.join(OUT, f"loc_{SYSTEM}.json"), "w") as fh:
    json.dump(report, fh, indent=1)
print(f"[ok] wrote loc_{SYSTEM}.json to {OUT}; run make_report.py for tables and figures")
