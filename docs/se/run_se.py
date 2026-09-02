"""Fit the fdia_graph.se estimators on one system and write results/se_<system>.json.

Set FG_SYSTEM (default ieee14; the README covers ieee14, ieee118, ieee300). Hyperparameters are
the estimation paper's validation-selected values per system. Tables and figures come from
make_report.py, which reads the JSON, so plots can be restyled without re-running. Needs the [se]
extra (torch + pandapower).
"""

import json
import os

import fdia_graph as fg
from fdia_graph.se import WLS, AdaptiveWeighting, ResidualRemoval, SubspacePrior

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
# Validation-selected hyperparameters per system: (huber c, prior rank fraction, removal threshold).
HP = {"ieee14": (1.5, 0.20, 4.0), "ieee118": (2.5, 0.50, 5.0), "ieee300": (6.0, 0.50, 5.0)}
c, rank, thr = HP.get(SYSTEM, (1.5, 0.5, 4.0))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

train = fg.load(SYSTEM, split="train")
test = fg.load(SYSTEM, split="test")
methods = {
    "wls": WLS(),
    "removal": ResidualRemoval(threshold=thr),
    "huber": AdaptiveWeighting(c=c),
    "prior+huber": SubspacePrior(rank_frac=rank, reweight="huber", c=c),
}
report = {name: m.fit(train).score(test) for name, m in methods.items()}
with open(os.path.join(OUT, f"se_{SYSTEM}.json"), "w") as f:
    json.dump(report, f, indent=1)
print(f"[ok] wrote se_{SYSTEM}.json to {OUT}; run make_report.py for tables and figures")
for name, r in report.items():
    print(
        f"  {name:12s} geo angle MAE {r['geo']['angle_mae_deg']:.4f} deg  voltage {r['geo']['voltage_mae_pu'] * 1e3:.3f} e-3 pu"
    )
