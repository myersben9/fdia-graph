"""The generation engine — the math/physics/theory half of the SDK, behind fg.generate().

FdiaGenerator (core.py) holds an IEEE case as a Case with its Network (fdia_graph.systems) and its meters
(fdia_graph.sensing) and
composes three concern mixins: measurement.py (a scan's readings as a file's layers), physics.py
(a scan's load and generation), attacks/ (At and Am).
Everything user-facing (loading, registry, streams) lives one level up.
"""
