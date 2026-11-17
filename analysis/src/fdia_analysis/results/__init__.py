"""Measured results as data: typed records with their provenance, one tidy store, and the renderers
that put the numbers into docs and figures, so no measured number is typed by hand.

    from fdia_analysis.results.run import Run
    from fdia_analysis.results.store import Store

    with Run("se.estimators", system="ieee14", settings=hp, data_release="v0.9.0") as run:
        run.add_tree(est.score(test), method="wls", levels=("family",))   # family -> metric -> value

    Store().latest("se.estimators", system="ieee14", metric="angle_mae_deg", family="geo")

The store is `results/` in the repository: one CSV per experiment (`results/<experiment>.csv`) and
the runs (`results/runs.csv`). A document shows a number through a results block that
`tools/results_docs.py` fills from the store and checks in CI (`results.docs`).
"""
