"""The operator's state estimator: AC weighted least squares on one scan, and its residual test statistic.

The residual sum J of the fit is what bad-data detection tests. An attack built as h(x_false) - h(x_true) on a
consistent false state leaves J as it was; anything else raises it. `wls` is the estimator the residual test runs,
`exact_state` the noise-free fit that recovers the state a set of readings describes (for an attack vector, its
false state).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fdia_graph.engine.attacks.attacker import wrap_angles
from fdia_graph.engine.sensing.measure import Measurement
from fdia_graph.generation.nygrid.case import REF, Case

__all__ = ["WlsFit", "exact_state", "reference_bus", "wls"]


def reference_bus(case: Case) -> int:
    """The position of the case's reference (slack) bus, whose angle every fit holds fixed."""
    return int(np.flatnonzero(case.bus_type == REF)[0])


@dataclass(frozen=True)
class WlsFit:
    """One scan's weighted least-squares fit: the residual sum `j` (the test statistic), the `rank` of the weighted
    Jacobian (the degrees of freedom are the readings less the rank), the state reached (`vm` p.u., `va` degrees),
    and whether the Gauss-Newton steps converged within the iterations allowed."""

    j: float
    rank: int
    vm: np.ndarray
    va: np.ndarray
    converged: bool
    iterations: int


def wls(
    meas: Measurement,
    z: np.ndarray,
    sigma: np.ndarray,
    vm: np.ndarray,
    va: np.ndarray,
    ref: int,
    *,
    max_iter: int = 12,
    tol: float = 1e-7,
) -> WlsFit:
    """Fit the scan `z [M]` with meter standard deviations `sigma [M]` by Gauss-Newton from `(vm, va)`, the angle of
    bus `ref` fixed. A small damping keeps the buses no meter sees from making a step singular. A fit that does not
    converge still returns its last residual, with `converged` False."""
    kind = meas.plan.kind
    keep = np.ones(2 * len(vm), bool)
    keep[len(vm) + ref] = False
    w = 1.0 / sigma
    vm, va = vm.copy(), va.copy()
    converged, it, jac = False, 0, np.zeros((0, 0))
    for it in range(1, max_iter + 1):
        r = wrap_angles(z - meas.h(vm, va), kind) * w
        jac = meas.jacobian(vm, va)[:, keep] * w[:, None]
        step = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(keep.sum()), jac.T @ r)
        dx = np.zeros(2 * len(vm))
        dx[keep] = step
        vm, va = vm + dx[: len(vm)], va + dx[len(vm) :]
        if np.abs(step).max() < tol:
            converged = True
            break
    r = wrap_angles(z - meas.h(vm, va), kind) * w
    return WlsFit(float(r @ r), int(np.linalg.matrix_rank(jac, tol=1e-6)), vm, va, converged, it)


def exact_state(
    meas: Measurement,
    z: np.ndarray,
    vm: np.ndarray,
    va: np.ndarray,
    ref: int,
    *,
    max_iter: int = 20,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float]:
    """The state whose noise-free readings are `z`, by Gauss-Newton from `(vm, va)` with the angle of `ref` fixed,
    and the largest reading it still misses (near zero when `z` is consistent with some state, as an attacked scan
    without noise is). Returns `(vm, va, mismatch)`."""
    kind = meas.plan.kind
    keep = np.ones(2 * len(vm), bool)
    keep[len(vm) + ref] = False
    for _ in range(max_iter):
        r = wrap_angles(z - meas.h(vm, va), kind)
        step = np.linalg.lstsq(meas.jacobian(vm, va)[:, keep], r, rcond=None)[0]
        dx = np.zeros(2 * len(vm))
        dx[keep] = step
        vm, va = vm + dx[: len(vm)], va + dx[len(vm) :]
        if np.abs(step).max() < tol:
            break
    return vm, va, float(np.abs(wrap_angles(z - meas.h(vm, va), kind)).max())
