"""Every data model of the generation toolkit, one module per subject; import a model from its module.

By subject: `fields` (the fields the record bundles share), `grid` (the static system), `frames` (what
the generators pass per scan), `plan` (a timeline's plan and design records), `config` (the settings
the loader and the generators accept), `inputs` (what the formulas and parsers accept), `validation`
(the one engine that checks them), `choices` (the fixed sets), `data` (what a user gets back) and
`assets` (how files are found). These modules import only numpy, the standard library and
`fdia_graph.errors`, so no model depends on a producer and no import cycle is possible. The analysis
toolkit keeps its own models in `fdia_analysis.models`.
"""
