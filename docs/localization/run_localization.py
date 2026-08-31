"""Fit the three fdia_graph.localization methods and write results/ (metrics JSON + figure + CSV).

Runs on IEEE-14 by default so it finishes in minutes on a laptop; set FG_SYSTEM=ieee118 (or any
system in the ladder) to scale up. ResidualLocalizer needs the [se] extra (torch + pandapower).
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fdia_graph as fg
from fdia_graph.localization import DeltaThreshold, ResidualLocalizer, SwingThreshold

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

train = fg.load(SYSTEM, split="train")
test = fg.load(SYSTEM, split="test")
methods = {
    "swing": SwingThreshold(),
    "delta": DeltaThreshold(),
    "residual": ResidualLocalizer(),
}
report = {name: m.fit(train).score(test) for name, m in methods.items()}
with open(os.path.join(OUT, f"loc_{SYSTEM}.json"), "w") as f:
    json.dump(report, f, indent=1)

# Figure: node-F1 heatmap, methods x attacked families, cell values printed so no legend is needed.
# Row labels carry each method's benign record-level false-alarm rate — a detection rate is only
# meaningful next to its false alarms.
fams = [k for k in report["swing"] if k != "benign"]
rows = list(methods)
F1 = np.array([[report[m][f]["node_f1"] for f in fams] for m in rows])
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.2, 1.9))
im = ax.imshow(F1, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
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
