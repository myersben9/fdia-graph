"""The attacker: false data that the operator's state estimator accepts.

An attack changes the readings by h(x_false) - h(x_true) for a false state that differs from the true one only on a
local subnetwork (`Attacker`), so the residual test cannot see it. A family (`AttackFamily`) designs one `Episode`
for a window of true states, or finds none: `MsFdia`, `OpfFdia` and `InterfaceFdia` drive a line or a NYISO
interface toward its limit within every plausibility bound (load cap, basepoint band, ramps, voltage limits), and
`Ramp` moves loads slowly.

When a design finds no attack, `design_explained` says why (`MISSES`: no target, infeasible, or a solver failure).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph.nyfed._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from .attacker import (
        Attacker,
        generator_band,
        generator_q_capability,
        generator_ramps,
        natural_steps,
        wrap_angles,
    )
    from .episode import MISSES, AttackFamily, Episode, Window
    from .interface_attack import InterfaceFdia
    from .msfdia import MsFdia
    from .opf_attack import OpfFdia
    from .ramp import Ramp

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "Episode": ".episode",
    "AttackFamily": ".episode",
    "Window": ".episode",
    "MISSES": ".episode",
    "Attacker": ".attacker",
    "natural_steps": ".attacker",
    "generator_ramps": ".attacker",
    "generator_band": ".attacker",
    "generator_q_capability": ".attacker",
    "wrap_angles": ".attacker",
    "MsFdia": ".msfdia",
    "OpfFdia": ".opf_attack",
    "InterfaceFdia": ".interface_attack",
    "Ramp": ".ramp",
}

__all__ = [
    "AttackFamily",
    "Attacker",
    "Episode",
    "InterfaceFdia",
    "MISSES",
    "MsFdia",
    "OpfFdia",
    "Ramp",
    "Window",
    "generator_band",
    "generator_q_capability",
    "generator_ramps",
    "natural_steps",
    "wrap_angles",
]
__getattr__, __dir__ = attach(__name__, _PUBLIC)
