"""The generation engine — the math/physics/theory half of the SDK, behind fg.generate().

FdiaGenerator (core.py) composes three concern mixins over a pandapower grid:
measurement.py (meters + noise), physics.py (a scan's load and generation), attacks/ (At and Am).
Everything user-facing (loading, registry, streams) lives one level up.
"""

from .core import FAM_ID, FdiaGenerator

__all__ = ["FdiaGenerator", "FAM_ID"]
