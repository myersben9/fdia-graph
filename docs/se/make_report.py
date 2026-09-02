"""Build the README tables and figures from results/se_*.json, in the estimation paper's layout.

Table 1: estimator x system, an angle-MAE block and a voltage-MAE block (10^-3 pu), each closed by
the proposed estimator's error reduction over WLS. Table 2: family x system for the proposed
estimator, baseline error and percent reduction. Summary cells are the geometric mean over the
seven record classes ("geo"), as the paper aggregates them.

Figures: one angle-MAE heatmap per system (estimators x families) plus a CSV sidecar. Nothing here
re-runs an estimator.
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
    ("wls", "WLS baseline"),
    ("removal", "Residual removal"),
    ("huber", "Adaptive weighting"),
    ("prior+huber", "**Prior + Huber (proposed)**"),
]
FAMS = [
    ("benign", "Benign"),
    ("Ad", "Bias (Ad)"),
    ("As", "Scaling (As)"),
    ("Ar", "Replay (Ar)"),
    ("Aq", "Stealthy re-solve (Aq)"),
    ("At", "Slow ramp (At)"),
    ("Al", "Load redistribution (Al)"),
]

res = {}
for path in glob.glob(os.path.join(OUT, "se_*.json")):
    with open(path) as fh:
        res[re.sub(r"^se_|\.json$", "", os.path.basename(path))] = json.load(fh)
systems = [s for s in SYSTEMS if s in res]
head = (
    "| "
    + " | ".join(["Estimator"] + [f"IEEE {s[4:]}" for s in systems])
    + " |\n|---|"
    + "---:|" * len(systems)
)


def red(sys: str, fam: str, key: str) -> float:
    base, prop = res[sys]["wls"][fam][key], res[sys]["prior+huber"][fam][key]
    return 100.0 * (1.0 - prop / base)


print("Estimator comparison\n")
for key, label, scale, fmt in [
    ("angle_mae_deg", "Angle MAE (deg)", 1.0, "{:.3f}"),
    ("voltage_mae_pu", "Voltage MAE (10^-3 pu)", 1e3, "{:.3f}"),
]:
    print(head)
    print(f"| *{label}* |" + " |" * len(systems))
    for k, lab in ROWS:
        cells = [fmt.format(res[s][k]["geo"][key] * scale) if k in res[s] else "" for s in systems]
        if k == "prior+huber":
            cells = [f"**{c}**" for c in cells]
        print("| " + " | ".join([lab] + cells) + " |")
    print("| WLS error reduction | " + " | ".join(f"{red(s, 'geo', key):.0f}%" for s in systems) + " |")
    print()

print("Per-family results of the proposed estimator\n")
blocks = [
    ("Base angle (deg)", "angle_mae_deg", 1.0, "{:.3f}"),
    ("Base volt (10^-3)", "voltage_mae_pu", 1e3, "{:.2f}"),
    ("Angle red. (%)", "angle_mae_deg", None, "{:.0f}"),
    ("Volt red. (%)", "voltage_mae_pu", None, "{:.0f}"),
]
print("| Family | " + " | ".join(f"{b} {s[4:]}" for b, _, _, _ in blocks for s in systems) + " |")
print("|---|" + "---:|" * (len(blocks) * len(systems)))
for fam, lab in FAMS:
    cells = []
    for _, key, scale, fmt in blocks:
        for s in systems:
            cells.append(fmt.format(res[s]["wls"][fam][key] * scale if scale else red(s, fam, key)))
    print("| " + " | ".join([lab] + cells) + " |")
print()

# Angle-MAE heatmap per system (degrees), estimators x families, cell values printed so no legend
# is needed. Lower is better; the geometric mean over families ('geo') is the summary column.
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
for s in systems:
    r = res[s]
    rows = [k for k, _ in ROWS if k in r]
    fams = [k for k in r["wls"] if k != "geo"] + ["geo"]
    A = np.array([[r[m][f]["angle_mae_deg"] for f in fams] for m in rows])
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.imshow(A, cmap="RdYlGn_r", vmin=0.0, vmax=max(A.max(), 1e-9), aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(fams)), [f"$A_{f[1:]}$" if f.startswith("A") else f for f in fams])
    ax.set_yticks(range(len(rows)), rows)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_se_{s}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(OUT, f"fig_se_{s}_data.csv"), "w") as f:
        f.write("method," + ",".join(f"{fam}_angle_mae_deg" for fam in fams) + "\n")
        for i, m in enumerate(rows):
            f.write(f"{m}," + ",".join(f"{v:.5f}" for v in A[i]) + "\n")
    print(f"[ok] fig_se_{s}.png + CSV sidecar")
