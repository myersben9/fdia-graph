"""Every data model of the generation toolkit, one module per subject; import a model from its module.

Grouped by what the data is: `fields` (the fields the record bundles share), `grid` (the static system), `frames` (what the generators pass
around per scan), `plan` (the records of a timeline's plan and design stages), `config` (the settings the loader and the generators
accept, and `inputs`, what the formulas and parsers accept, checked by the one engine in `validation`, with the fixed sets in `choices`), `data` (what a user gets back:
the bundles the data dictionary lists), `assets` (how files
are found). The package's own modules import only numpy, the standard library and `fdia_graph.errors`, so no model
depends on a producer and no import cycle is possible (importing it still runs the parent
package, loader and h5py included). The models of the estimators, the localizers, the trust
selectors and the results store are the analysis toolkit's (`fdia_analysis.models`).
"""
