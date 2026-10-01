"""Measured results as data: typed records with their provenance, one tidy store, and the renderers
that put the numbers into docs and figures, so no measured number is typed by hand.

    from fdia_graph.results import Run, Store

    with Run("se.estimators", system="ieee14", settings=hp, data_release="v0.9.0") as run:
        run.add_tree(est.score(test), method="wls", levels=("family",))   # family -> metric -> value

    Store().latest("se.estimators", system="ieee14", metric="angle_mae_deg", family="geo")

The store is `results/` in the repository: one CSV per experiment (`results/<experiment>.csv`) and
the runs (`results/runs.csv`). A document shows a number through a results block that
`tools/results_docs.py` fills from the store and checks in CI (`results.docs`).
"""

from ..models.results import METRICS, Better, Metric, Provenance, Record
from .docs import QUERIES, fill, query, stale
from .render import cell, figure_data, pivot, table
from .run import Run
from .store import Store

__all__ = [
    "METRICS",
    "QUERIES",
    "Better",
    "Metric",
    "Provenance",
    "Record",
    "Run",
    "Store",
    "cell",
    "figure_data",
    "fill",
    "pivot",
    "query",
    "stale",
    "table",
]
