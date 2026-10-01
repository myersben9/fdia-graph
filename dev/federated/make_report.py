"""Draw the README figures from the results store (experiment `federated.localization`).

One per-family per-bus F1 heatmap per system (rows model and K, row labels carrying FR over every
record) and its CSV sidecar, both read from the store; nothing here re-runs a model, and the
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
ROWS = [
    (m, k, f"{lab}, K = {k}")
    for m, lab in (("cnn", "1D CNN"), ("mlp", "Per-bus MLP"))
    for k in ("1", "2", "3")
]
FAMS = ["Aq", "Ad", "As", "Ar"]

store = Store(os.path.join(HERE, "..", "..", "results"))
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
for s in SYSTEMS:
    recs = store.latest("federated.localization", system=s, pool="all")
    f1 = {(r.method, r.tag("clients"), r.family): r for r in recs if r.metric == "macro_f1"}
    fr = {(r.method, r.tag("clients")): r for r in recs if r.metric == "macro_fr" and r.family == "all"}
    rows = [(m, k, lab) for m, k, lab in ROWS if (m, k) in fr]
    if not rows:
        continue
    F1 = np.array([[f1[(m, k, f)].value for f in FAMS] for m, k, _ in rows])
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    ax.imshow(F1, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(F1.shape[0]):
        for j in range(F1.shape[1]):
            ax.text(j, i, f"{F1[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(FAMS)), [f"$A_{f[1:]}$" for f in FAMS])
    ax.set_yticks(range(len(rows)), [f"{lab} (FR {fr[(m, k)].value:.4f})" for m, k, lab in rows])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_fed_{s}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    ordered = [r for m, k, _ in rows for r in [fr[(m, k)], *(f1[(m, k, f)] for f in FAMS)]]
    figure_data(
        ordered, os.path.join(OUT, f"fig_fed_{s}_data.csv"), row="method+clients", col="metric+family"
    )
    print(f"[ok] fig_fed_{s}.png + CSV sidecar")
