"""Build the README tables and figures from results/fed_*.json, in the federated paper's layout.

Tables: rows are model and client count, columns are F1 / DR / FR per system, the papers' per-bus
macro scores over the attackable buses as mean +/- standard deviation over three seeds. FR is the
paper's Table IV false-alarm rate over every test record; a second table gives it over benign
records only, the budget the localization guide reports.

Figures: one per-family per-bus F1 heatmap per system (rows model and K, row labels carrying FR),
plus a CSV sidecar. Nothing here re-runs a model.
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
ROWS = [
    (f"{m}_K{k}", f"{lab}, K = {k}")
    for m, lab in (("cnn", "1D CNN"), ("mlp", "Per-bus MLP"))
    for k in (1, 2, 3)
]
FAMS = ["Aq", "Ad", "As", "Ar"]

res = {}
for path in glob.glob(os.path.join(OUT, "fed_*.json")):
    with open(path) as fh:
        res[re.sub(r"^fed_|\.json$", "", os.path.basename(path))] = json.load(fh)
systems = [s for s in SYSTEMS if s in res]


def cell(x, digits):
    return f"{x['mean']:.{digits}f} ± {x['std']:.{digits}f}"


def table(fr_over, keys, digits):
    print("| Model | " + " | ".join(f"{k.upper()} {s[4:]}" for s in systems for k, _ in keys) + " |")
    print("|---|" + "---:|" * (len(keys) * len(systems)))
    for row, lab in ROWS:
        cells = [
            cell(res[s][row][fr_over]["all"][m], d) if row in res[s] else ""
            for s in systems
            for (_, m), d in zip(keys, digits)
        ]
        print("| " + " | ".join([lab] + cells) + " |")
    print()


print("Table IV layout, FR over every test record\n")
table("all", [("f1", "macro_f1"), ("dr", "macro_dr"), ("fr", "macro_fr")], [3, 3, 4])
print("FR over benign records only\n")
table("benign", [("fr", "macro_fr")], [5])
print("Per-family node F1 (that family + benign), mean over seeds\n")
print("| Model | " + " | ".join(f"{f} {s[4:]}" for s in systems for f in FAMS) + " |")
print("|---|" + "---:|" * (len(FAMS) * len(systems)))
for row, lab in ROWS:
    cells = [
        f"{res[s][row]['all'][f]['macro_f1']['mean']:.3f}" if row in res[s] else ""
        for s in systems
        for f in FAMS
    ]
    print("| " + " | ".join([lab] + cells) + " |")
print()

# Per-family per-bus F1 heatmap per system, cell values printed so no legend is needed; row labels
# carry the row's FR over every record, the paper's companion number to any detection score.
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
for s in systems:
    r = res[s]
    rows = [(k, lab) for k, lab in ROWS if k in r]
    F1 = np.array([[r[k]["all"][f]["macro_f1"]["mean"] for f in FAMS] for k, _ in rows])
    fr = [r[k]["all"]["all"]["macro_fr"]["mean"] for k, _ in rows]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.imshow(F1, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(F1.shape[0]):
        for j in range(F1.shape[1]):
            ax.text(j, i, f"{F1[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(FAMS)), [f"$A_{f[1:]}$" for f in FAMS])
    ax.set_yticks(range(len(rows)), [f"{lab} (FR {v:.4f})" for (_, lab), v in zip(rows, fr)])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_fed_{s}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(OUT, f"fig_fed_{s}_data.csv"), "w") as f:
        f.write("model,fr," + ",".join(f"{fam}_macro_f1" for fam in FAMS) + "\n")
        for (k, _), v, row in zip(rows, fr, F1):
            f.write(f"{k},{v:.6f}," + ",".join(f"{x:.4f}" for x in row) + "\n")
    print(f"[ok] fig_fed_{s}.png + CSV sidecar")
