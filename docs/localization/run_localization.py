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
ResidualLocalizer needs the [se] extra; BusCNN/BusMLP need [torch] and use the GPU when visible.
"""

import json
import os

import fdia_graph as fg
from fdia_graph.localization import BusCNN, BusMLP, DeltaThreshold, ResidualLocalizer, SwingThreshold

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

# ---- 1. common protocol: same train, same FA budget, full test ------------------------------
train = fg.load(SYSTEM, split="train")
test = fg.load(SYSTEM, split="test")
methods = {
    "swing": SwingThreshold(),
    "delta": DeltaThreshold(),
    "residual": ResidualLocalizer(),
    "mlp": BusMLP(),
    "cnn": BusCNN(),
}
report = {name: m.fit(train).score(test) for name, m in methods.items()}

# ---- 2. papers' zero-shot protocol: benign + Aq + Ad seen, As + Ar unseen, val-tuned tau -----
zs = dict(families=[0, 1, 2])
ztr, zva = fg.load(SYSTEM, split="train", **zs), fg.load(SYSTEM, split="val", **zs)
zte = fg.load(SYSTEM, split="test", families=[0, 1, 2, 3, 4])
zero_shot = {}
for name, m in {"mlp": BusMLP(), "cnn": BusCNN()}.items():
    zero_shot[name] = {"tau": m.fit(ztr, val=zva).tau, **m.score(zte)}
zero_shot["swing"] = SwingThreshold().fit(ztr).score(zte)  # the feature alone, FA-calibrated
report["zero_shot"] = zero_shot

with open(os.path.join(OUT, f"loc_{SYSTEM}.json"), "w") as f:
    json.dump(report, f, indent=1)
print(f"[ok] wrote loc_{SYSTEM}.json to {OUT}; run make_report.py for tables and figures")
for name, r in zero_shot.items():
    print(f"  zero-shot {name:8s} macro-F1 {r['all']['macro_f1']:.4f}  FR {r['all']['macro_fr']:.4f}")
