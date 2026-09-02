"""Fit the fdia_graph.localization methods and write results/ (metrics JSON + figure + CSV).

Two protocols, one results file:

1. Common protocol (the heatmap): every method fits on the unfiltered train split and is
   calibrated on benign records at the same false-alarm budget, then scores the full test split.
   The learned arms train on all six families here, so the slow ramp At is in-distribution.
2. Papers' zero-shot protocol (the macro-F1 table): train and val hold benign + Aq + Ad only,
   test adds As and Ar, the threshold is the validation-best global tau. This is the protocol
   behind the federated localization paper's headline numbers.

Runs on IEEE-14 by default so it finishes in minutes; set FG_SYSTEM=ieee118 (or any system in the
ladder) to scale up. ResidualLocalizer needs the [se] extra; BusCNN/BusMLP need [torch] and use
the GPU when one is visible.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    r = m.fit(ztr, val=zva).score(zte)
    zero_shot[name] = {"tau": m.tau, **r}
r = SwingThreshold().fit(ztr).score(zte)  # the feature alone, FA-calibrated, same splits
zero_shot["swing"] = r
report["zero_shot"] = zero_shot

with open(os.path.join(OUT, f"loc_{SYSTEM}.json"), "w") as f:
    json.dump(report, f, indent=1)

# Figure: node-F1 heatmap, methods x attacked families (common protocol), cell values printed so
# no legend is needed. Row labels carry each method's benign record-level false-alarm rate, since
# a detection rate is only meaningful next to its false alarms.
fams = [k for k in report["swing"] if k not in ("benign", "all")]
rows = list(methods)
F1 = np.array([[report[m][f]["node_f1"] for f in fams] for m in rows])
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.2, 2.6))
ax.imshow(F1, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
for i in range(F1.shape[0]):
    for j in range(F1.shape[1]):
        ax.text(j, i, f"{F1[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_xticks(range(len(fams)), [f"$A_{f[1:]}$" for f in fams])
ax.set_yticks(range(len(rows)), [f"{m} (FA {report[m]['benign']['false_alarm_rate']:.2f})" for m in rows])
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"fig_loc_{SYSTEM}.png"), dpi=200, bbox_inches="tight")

# Data sidecar so the figure can be restyled without re-running.
with open(os.path.join(OUT, f"fig_loc_{SYSTEM}_data.csv"), "w") as f:
    f.write("method,benign_fa," + ",".join(f"{fam}_node_f1" for fam in fams) + "\n")
    for i, m in enumerate(rows):
        fa = report[m]["benign"]["false_alarm_rate"]
        f.write(f"{m},{fa:.4f}," + ",".join(f"{v:.4f}" for v in F1[i]) + "\n")

print(f"[ok] wrote loc_{SYSTEM}.json, fig_loc_{SYSTEM}.png (+ CSV sidecar) to {OUT}")
for name, r in zero_shot.items():
    print(
        f"  zero-shot {name:8s} macro-F1 {r['all']['macro_f1']:.4f}  benign FA {r['benign']['false_alarm_rate']:.4f}"
    )
