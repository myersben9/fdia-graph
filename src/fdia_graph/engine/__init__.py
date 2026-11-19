"""The generation engine — the math/physics/theory half of the SDK, behind fg.generate().

FdiaGenerator (core.py) holds an IEEE case as a Case with its Network (fdia_graph.systems) and its meters
(fdia_graph.sensing) and
holds one object per job (`base.Part`): measurement.py (a scan's readings as a file's layers),
physics.py (a scan's load and generation) and the parts of attacks/ (At and Am).
Everything user-facing (loading, registry, streams) lives one level up.
"""
