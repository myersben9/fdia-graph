"""The meters: what each client measures, and how wrong each meter is.

`plan_meters` places SCADA meters (injections, branch flows, voltage magnitudes) and PMUs on the New York buses and
gives each to the client that owns its bus: the `MeterPlan`, a table beside the flat measurement vector `z`, so a
client's readings are `z[:, plan.owner == c]`. `Measurement` evaluates h(x) and its Jacobian for a state, and
`NoiseModel` gives each meter its accuracy class split into a constant bias and a per-scan jitter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph.nyfed._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from .measure import Measurement
    from .meters import JITTER_FRAC, KINDS, MeterPlan, plan_meters
    from .noise import NoiseModel

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "MeterPlan": ".meters",
    "plan_meters": ".meters",
    "KINDS": ".meters",
    "JITTER_FRAC": ".meters",
    "Measurement": ".measure",
    "NoiseModel": ".noise",
}

__all__ = [
    "JITTER_FRAC",
    "KINDS",
    "Measurement",
    "MeterPlan",
    "NoiseModel",
    "plan_meters",
]
__getattr__, __dir__ = attach(__name__, _PUBLIC)
