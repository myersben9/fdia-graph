"""Detection: what the operator's checks see, as building blocks.

The residual test fits each scan by AC weighted least squares (`wls`) and tests the residual sum; an attack on a
consistent false state leaves it unchanged until the operator acts and moves the true state away from the one the
attack was designed on. `PersistenceRule` turns a statistic per scan into a first alarm (k of the last n past a
threshold fitted on benign days, `PersistenceRule.fit`); any object with `first_alarm(stat)` is an `AlarmRule` and
plugs in instead. `with_fleet` gives the meters another realization of their constant biases, to check that a
detector's rates hold beyond the fleet it was fitted on. The true state after the operator acts comes from the
operator package (`TrueGrid.run_state`); the interchange check against the neighbours' meters lives there too
(`tie_gap`). A fit that does not converge says so (`WlsFit.converged`); a scan that cannot be evaluated is NaN,
which the persistence rule counts as below its threshold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fdia_graph.nyfed._lazy import attach

if TYPE_CHECKING:  # the names below, with their real signatures for type checkers
    from .estimate import WlsFit, exact_state, reference_bus, wls
    from .fleets import FLEET_STREAM, fleet_bias, with_fleet
    from .rules import AlarmRule, PersistenceRule

# each public name and the module that defines it, imported on first use
_PUBLIC = {
    "WlsFit": ".estimate",
    "wls": ".estimate",
    "exact_state": ".estimate",
    "reference_bus": ".estimate",
    "AlarmRule": ".rules",
    "PersistenceRule": ".rules",
    "FLEET_STREAM": ".fleets",
    "fleet_bias": ".fleets",
    "with_fleet": ".fleets",
}

__all__ = [
    "FLEET_STREAM",
    "AlarmRule",
    "PersistenceRule",
    "WlsFit",
    "exact_state",
    "fleet_bias",
    "reference_bus",
    "with_fleet",
    "wls",
]
__getattr__, __dir__ = attach(__name__, _PUBLIC)
