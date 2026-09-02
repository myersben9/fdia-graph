"""Build the README tables and figures from results/loc_*.json, in the localization paper's layout.

Tables: rows are methods, columns are F1 / DR / FR per system, all three per-bus macro scores over
the attackable buses (F1 and DR accumulate over every test record, FR is the per-bus false-positive
rate on benign records). One table per protocol, printed as markdown.

Figures: one per-family per-bus F1 heatmap per system (the paper shows per-family F1 as a figure),
row labels carrying each method's FR, plus a CSV sidecar. Nothing here re-runs a model.
"""

import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SYSTEMS = ["ieee14", "ieee118", "ieee300"]
COMMON = [
    ("swing", "Swing threshold"),
    ("delta", "Delta threshold"),
    ("residual", "Residual (LNR)"),
    ("mlp", "Per-bus MLP"),
    ("cnn", "**1D CNN**"),
]
ZERO_SHOT = [("mlp", "Per-bus MLP"), ("cnn", "**1D CNN**"), ("swing", "Swing threshold")]

res = {}
for path in glob.glob(os.path.join(OUT, "loc_*.json")):
    res[re.sub(r"^loc_|\.json$", "", os.path.basename(path))] = json.load(open(path))
systems = [s for s in SYSTEMS if s in res]


def table(rows, pick):
    print("| Method | " + " | ".join(f"F1 {s[4:]} | DR {s[4:]} | FR {s[4:]}" for s in systems) + " |")
    print("|---|" + "---:|" * (3 * len(systems)))
    for k, lab in rows:
        cells = []
        for s in systems:
            a = pick(res[s], k)
            cells += (
                [f"{a['macro_f1']:.4f}", f"{a['macro_dr']:.4f}", f"{a['macro_fr']:.4f}"]
                if a
                else ["", "", ""]
            )
        print("| " + " | ".join([lab] + cells) + " |")
    print()


print("Common protocol\n")
table(COMMON, lambda r, k: r[k]["all"] if k in r else None)
print("Zero-shot protocol\n")
table(ZERO_SHOT, lambda r, k: r["zero_shot"][k]["all"] if k in r.get("zero_shot", {}) else None)

# Per-family per-bus F1 heatmap per system, cell values printed so no legend is needed. Row labels
# carry the method's per-bus false-positive rate on benign records (FR), the paper's companion
# number to any detection score.
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
for s in systems:
    r = res[s]
    rows = [k for k, _ in COMMON if k in r]
    fams = [k for k in r["swing"] if k not in ("benign", "all")]
    F1 = np.array([[r[m][f]["macro_f1"] for f in fams] for m in rows])
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.imshow(F1, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(F1.shape[0]):
        for j in range(F1.shape[1]):
            ax.text(j, i, f"{F1[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(fams)), [f"$A_{f[1:]}$" for f in fams])
    ax.set_yticks(range(len(rows)), [f"{m} (FR {r[m]['all']['macro_fr']:.3f})" for m in rows])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_loc_{s}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(OUT, f"fig_loc_{s}_data.csv"), "w") as f:
        f.write("method,fr," + ",".join(f"{fam}_macro_f1" for fam in fams) + "\n")
        for i, m in enumerate(rows):
            f.write(f"{m},{r[m]['all']['macro_fr']:.5f}," + ",".join(f"{v:.4f}" for v in F1[i]) + "\n")
    print(f"[ok] fig_loc_{s}.png + CSV sidecar")
