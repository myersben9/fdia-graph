"""Draw the README figures from the results store (experiment `se.estimators`).

One angle-MAE heatmap per system (estimators x families) and its CSV sidecar, both read from the
store; nothing here re-runs an estimator, and the README's tables are results blocks that
`tools/results_docs.py --write` fills from the same store.
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
METHODS = ["wls", "removal", "huber", "prior+huber", "jacobian", "prior+huber+gate", "prior+huber+oracle"]
FAMS = ["benign", "Aq", "Ad", "As", "Ar", "At", "Al", "Am", "geo"]

store = Store(os.path.join(HERE, "..", "..", "results"))
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
for s in SYSTEMS:
    have = {(r.method, r.family): r for r in store.latest("se.estimators", system=s, metric="angle_mae_deg")}
    rows = [m for m in METHODS if any((m, f) in have for f in FAMS)]
    fams = [f for f in FAMS if any((m, f) in have for m in rows)]
    if not rows:
        continue
    A = np.array([[have[(m, f)].value if (m, f) in have else np.nan for f in fams] for m in rows])
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.imshow(A, cmap="RdYlGn_r", vmin=0.0, vmax=max(np.nanmax(A), 1e-9), aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(fams)), [f"$A_{f[1:]}$" if f.startswith("A") else f for f in fams])
    ax.set_yticks(range(len(rows)), rows)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_se_{s}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    ordered = [have[(m, f)] for m in rows for f in fams if (m, f) in have]
    figure_data(ordered, os.path.join(OUT, f"fig_se_{s}_data.csv"), row="method", col="family")
    print(f"[ok] fig_se_{s}.png + CSV sidecar")
