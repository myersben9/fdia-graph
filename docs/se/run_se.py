"""Fit the fdia_graph.se estimators on one system and write results/se_<system>.json.

Set FG_SYSTEM (default ieee14; the README covers ieee14, ieee118, ieee300). Hyperparameters are
the estimation paper's validation-selected values per system. Tables and figures come from
make_report.py, which reads the JSON, so plots can be restyled without re-running. Needs the [se]
extra (torch + pandapower).

The expensive part is estimate() over the test split (one chord-Newton solve per record, hours on
IEEE-300 for the robust arms), so each estimator's estimates are cached in results/cache/ as soon
as it finishes. Re-running scores from the cache in seconds; delete a cache file to recompute that
estimator. fit() is cheap and always runs, so hyperparameter changes still take effect on the
uncached arms.
"""

import json
import os
import time

import numpy as np

import fdia_graph as fg
from fdia_graph.se import WLS, AdaptiveWeighting, ResidualRemoval, SubspacePrior

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
# Validation-selected hyperparameters per system: (huber c, prior rank fraction, removal threshold).
HP = {"ieee14": (1.5, 0.20, 4.0), "ieee118": (2.5, 0.50, 5.0), "ieee300": (6.0, 0.50, 5.0)}
c, rank, thr = HP.get(SYSTEM, (1.5, 0.5, 4.0))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)

train = fg.load(SYSTEM, split="train")
test = fg.load(SYSTEM, split="test")
methods = {
    "wls": WLS(),
    "removal": ResidualRemoval(threshold=thr),
    "huber": AdaptiveWeighting(c=c),
    "prior+huber": SubspacePrior(rank_frac=rank, reweight="huber", c=c),
}
report = {}
for name, m in methods.items():
    t0 = time.time()
    m.fit(train)
    f = os.path.join(CACHE, f"se_{SYSTEM}_{name}.npz")
    if os.path.exists(f):
        with np.load(f) as z:
            xhat = z["xhat"]
        how = "cached estimates"
    else:
        xhat = m.estimate(test)
        np.savez_compressed(f, xhat=xhat)
        how = "estimated"
    report[name] = m.score(test, xhat=xhat)
    print(
        f"  {name:12s} {how:17s} {time.time() - t0:6.0f}s  geo angle MAE {report[name]['geo']['angle_mae_deg']:.4f} deg"
    )

with open(os.path.join(OUT, f"se_{SYSTEM}.json"), "w") as fh:
    json.dump(report, fh, indent=1)
print(f"[ok] wrote se_{SYSTEM}.json to {OUT}; run make_report.py for tables and figures")
