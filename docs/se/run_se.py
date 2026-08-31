"""Fit three fdia_graph.se estimators and write results/ (metrics JSON + figure + CSV sidecar).

Runs on IEEE-14 by default so it finishes in minutes on a laptop; set FG_SYSTEM=ieee118 (or any
system in the ladder) to scale up. Needs the [se] extra (torch + pandapower). Hyperparameters are
the paper's validation-selected values per system.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import fdia_graph as fg
from fdia_graph.se import WLS, AdaptiveWeighting, SubspacePrior

SYSTEM = os.environ.get("FG_SYSTEM", "ieee14")
# Validation-selected hyperparameters per system: (huber c, prior rank fraction).
HP = {"ieee14": (1.5, 0.20), "ieee118": (2.5, 0.50), "ieee300": (6.0, 0.50)}
c, rank = HP.get(SYSTEM, (1.5, 0.5))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

train = fg.load(SYSTEM, split="train")
test = fg.load(SYSTEM, split="test")
methods = {
    "wls": WLS(),
    "huber": AdaptiveWeighting(c=c),
    "prior+huber": SubspacePrior(rank_frac=rank, reweight="huber", c=c),
}
report = {name: m.fit(train).score(test) for name, m in methods.items()}
with open(os.path.join(OUT, f"se_{SYSTEM}.json"), "w") as f:
    json.dump(report, f, indent=1)

# Figure: angle-MAE heatmap (degrees), methods x families, cell values printed so no legend is
# needed. Lower is better; the geometric mean over families ('geo') is the summary column.
fams = [k for k in report["wls"] if k != "geo"] + ["geo"]
rows = list(methods)
A = np.array([[report[m][f]["angle_mae_deg"] for f in fams] for m in rows])
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.6, 1.9))
ax.imshow(A, cmap="RdYlGn_r", vmin=0.0, vmax=max(A.max(), 1e-9), aspect="auto")
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=8)
labels = [f"$A_{f[1:]}$" if f.startswith("A") else f for f in fams]
ax.set_xticks(range(len(fams)), labels)
ax.set_yticks(range(len(rows)), rows)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"fig_se_{SYSTEM}.png"), dpi=200, bbox_inches="tight")

# Data sidecar so the figure can be restyled without re-running.
with open(os.path.join(OUT, f"fig_se_{SYSTEM}_data.csv"), "w") as f:
    f.write("method," + ",".join(f"{fam}_angle_mae_deg" for fam in fams) + "\n")
    for i, m in enumerate(rows):
        f.write(f"{m}," + ",".join(f"{v:.5f}" for v in A[i]) + "\n")
print(f"[ok] wrote se_{SYSTEM}.json, fig_se_{SYSTEM}.png (+ CSV sidecar) to {OUT}")
