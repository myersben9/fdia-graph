"""Other meter fleets: the same meters and accuracy classes with fresh constant biases.

One realization of the meters' constant biases is one fleet. A detector calibrated and tested on one fleet says
little about the next, so the residual test is checked over further fleets drawn here (N79, N81). The draws use
their own random stream, keyed by the draw number; the interchange check's fleets draw
only the tie meters from another stream, so draw d is not the same fleet in the two studies.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from fdia_graph.nyfed.sensing.meters import JITTER_FRAC, MeterPlan

__all__ = ["FLEET_STREAM", "fleet_bias", "with_fleet"]

FLEET_STREAM = (20190101, 4219)  # the residual test's fleets: draw d is the stream (*FLEET_STREAM, d)


def fleet_bias(plan: MeterPlan, draw: int) -> np.ndarray:
    """`[M]` the constant biases of fleet `draw`: each meter's bias share of its accuracy class (a gain on relative
    meters, an offset on absolute ones), drawn afresh."""
    rel = plan.sigma_rel > 0
    scale = np.sqrt(1.0 - JITTER_FRAC**2) * np.where(rel, plan.sigma_rel, plan.sigma_abs)
    return np.random.default_rng([*FLEET_STREAM, draw]).normal(0.0, scale)


def with_fleet(plan: MeterPlan, draw: int) -> MeterPlan:
    """`plan` with the biases of fleet `draw` (draw 0 is the plan's own fleet, returned as it is)."""
    return plan if draw == 0 else dataclasses.replace(plan, bias=fleet_bias(plan, draw))
