"""Every data model of the analysis toolkit in one place: the ones only the estimators, the localizers,
the federated trainers, the trust selectors and the results store use. A model the generation
toolkit also uses is in `fdia_graph.models`, and is imported from there.

Grouped by what the data is: `config` (the settings every consumer accepts, and `inputs`, what the
formulas accept, both checked by the engine in `fdia_graph.models.validation`, with the fixed sets in
`choices`; `inputs` also holds the one rule only they declare), `scores` (the score tables), `data`
(the truth an estimator is scored against), `grid` (the estimator's pseudo-measurement records),
`federated` (a partition and a round's log), `training` (what a learned localizer's fitting pieces
share), `results` (the records of the results store). The package's own modules import only numpy,
the standard library and `fdia_graph.models`, so no model depends on a producer.

A model is imported from the module that defines it (`from fdia_analysis.models.config import
PriorConfig`). The bundles a user receives from the public API are the score tables `scores` defines;
the data dictionary lists those.
"""
