"""Fit the fdia_graph.se estimators on one system and write a run of `se.estimators` to the
results store (`results/`).

Set FG_SYSTEM (default ieee14; the README covers ieee14, ieee118, ieee300). Hyperparameters are
the estimation paper's validation-selected values per system. The README's tables are results
blocks (`tools/results_docs.py --write`) and make_report.py draws the figures, both from the store,
so nothing is re-run to restyle them. Needs the [se]
extra (torch + pandapower).

The expensive part is estimate() over the test split (one chord-Newton solve per record, hours on
IEEE-300 for the robust arms), so each estimator's estimates are cached in results/cache/ as soon
as it finishes. Re-running scores from the cache in seconds; delete a cache file to recompute that
estimator. fit() is cheap and always runs, so hyperparameter changes still take effect on the
uncached arms.
"""

import os
import time

import numpy as np

import fdia_graph as fg
from fdia_graph.localization import BusCNN
from fdia_graph.results import Run, Store
from fdia_graph.se import (
    WLS,
    AdaptiveWeighting,
    GatedPrior,
    JacobianWeighting,
    ResidualRemoval,
    SubspacePrior,
)

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
# Validation-selected hyperparameters per system: (huber c, prior rank fraction, removal threshold).
HP = {"ieee14": (1.5, 0.20, 4.0), "ieee118": (2.5, 0.50, 5.0), "ieee300": (6.0, 0.50, 5.0)}
c, rank, thr = HP.get(SYSTEM, (1.5, 0.5, 4.0))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
CACHE = os.path.join(OUT, "cache")
os.makedirs(CACHE, exist_ok=True)
STORE = Store(os.path.join(HERE, "..", "..", "results"))  # the repository's results store

train = fg.load(SYSTEM, split="train")
test = fg.load(SYSTEM, split="test")
methods = {
    "wls": WLS(),
    "removal": ResidualRemoval(threshold=thr),
    "huber": AdaptiveWeighting(c=c),
    "prior+huber": SubspacePrior(rank_frac=rank, reweight="huber", c=c),
    "jacobian": JacobianWeighting(
        c=3.0
    ),  # weights from the unexplained temporal residual (Abdulin & Narimani)
}
# Localization-gated arms: the proposed estimator with a localizer down-weighting the meters of the
# buses it flags. The CNN gate is the papers' localizer trained on this train split (all families,
# the common protocol); the oracle gate uses the true labels and is the ceiling for any gate.
gate = BusCNN().fit(train)
methods["prior+huber+gate"] = GatedPrior(gate=gate, rank_frac=rank, reweight="huber", c=c)
methods["prior+huber+oracle"] = GatedPrior(gate="oracle", rank_frac=rank, reweight="huber", c=c)
# FG_SKIP=removal,... leaves arms out of a run. Residual removal is the slow arm on IEEE-300 (about
# three hours, its per-record observability guard) and does not beat WLS there; the published column
# ran it anyway so every column carries every arm.
for name in [a.strip() for a in os.environ.get("FG_SKIP", "").split(",") if a.strip()]:
    methods.pop(name, None)
report = {}
for name, m in methods.items():
    t0 = time.time()
    m.fit(train)
    f = os.path.join(CACHE, f"se_{os.path.splitext(os.path.basename(test.path))[0]}_{name}.npz")
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

settings = {"huber_c": c, "rank_frac": rank, "removal_threshold": thr, "arms": list(methods)}
release = fg.resolve(SYSTEM).release or ""
with Run("se.estimators", system=SYSTEM, settings=settings, data_release=release, store=STORE) as out:
    n = out.add_tree(report, levels=("method", "family"))
print(
    f"[ok] wrote {n} records of se.estimators for {SYSTEM}; run make_report.py and tools/results_docs.py --write"
)
