"""fdia-analysis: what is measured on an fdia-graph timeline: state estimation, attack localization,
federated training, trusted-meter selection, and the store the measured results are kept in.

It reads the files the generation toolkit `fdia_graph` writes (through `fdia_graph.dataset`) and
builds on its formulas and records; `fdia_graph` never imports this package.

    import fdia_graph as fg
    from fdia_analysis.se import WLS

    train, test = fg.load("ieee118", split="train"), fg.load("ieee118", split="test")
    print(WLS().fit(train).score(test))

Each subpackage is imported by name: `se` (the estimators, and the alarm rules in `se.detect`),
`localization`, `federated`, `trust`, `results`; their settings and score tables are in `models`,
their mathematics in `formulas`.
"""

__version__ = "0.21.0"
