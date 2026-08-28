"""The generation engine — the math/physics/theory half of the SDK, behind fg.generate().

FdiaGenerator (core.py) composes three concern mixins over a pandapower grid:
measurement.py (meters + noise), physics.py (AC solves), attacks.py (the six families).
Everything user-facing (loading, registry, streams) lives one level up.
"""

from .core import _CASE, FAM_ID, FdiaGenerator, line_outage_candidates

__all__ = ["FdiaGenerator", "FAM_ID", "line_outage_candidates"]
