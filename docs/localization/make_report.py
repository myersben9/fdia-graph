"""Draw the README figures from the results store (experiment `localization.common`).

One per-family per-bus F1 heatmap per system, row labels carrying each method's false-positive
rate (FR), and its CSV sidecar, both read from the store; nothing here re-runs a model, and the
README's tables are results blocks that `tools/results_docs.py --write` fills from the same store.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fdia_graph.results import Store, figure_data

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SYSTEMS = ["ieee14", "ieee118", "ieee300"]
METHODS = ["swing", "delta", "residual", "mlp", "cnn", "cnn+jac", "cnn+prev", "cnn+prev+jac"]
FAMS = ["Aq", "Ad", "As", "Ar", "At", "Al", "Am"]

store = Store(os.path.join(HERE, "..", "..", "results"))
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
for s in SYSTEMS:
    f1 = {(r.method, r.family): r for r in store.latest("localization.common", system=s, metric="macro_f1")}
    fr = {r.method: r for r in store.latest("localization.common", system=s, metric="macro_fr", family="all")}
    rows = [m for m in METHODS if m in fr]
    fams = [f for f in FAMS if any((m, f) in f1 for m in rows)]
    if not rows:
        continue
    F1 = np.array([[f1[(m, f)].value for f in fams] for m in rows])
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ax.imshow(F1, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(F1.shape[0]):
        for j in range(F1.shape[1]):
            ax.text(j, i, f"{F1[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(fams)), [f"$A_{f[1:]}$" for f in fams])
    ax.set_yticks(range(len(rows)), [f"{m} (FR {fr[m].value:.3f})" for m in rows])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_loc_{s}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    ordered = [r for m in rows for r in [fr[m], *(f1[(m, f)] for f in fams)]]
    figure_data(ordered, os.path.join(OUT, f"fig_loc_{s}_data.csv"), row="method", col="metric+family")
    print(f"[ok] fig_loc_{s}.png + CSV sidecar")
