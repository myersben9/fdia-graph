"""Jacobian-informed features (Abdulin & Narimani): the measurement Jacobian as a physics transform
of the scan-to-scan measurement change, not as raw model input.

For a record with measurements z_t, the change dz = z_t - h(x_hat_{t-1}) is taken against the
measurement prediction of the previous frame's state estimate: the fitted estimator's plain solve
of the frame emitted just before (the dataset's prev_node_x / prev_edge_x, read whatever split or
family that frame belongs to, attacked or not). Only what an operator holds enters the feature;
the true state is never read, apart from the slack angle that fixes the angle reference of every
estimate in fdia_graph.se. A timeline is required (a record shard's rows are not consecutive
frames). The chord Jacobian H at the benign
mean state, the meter weights W and the measurement mask all come from a fitted fdia_graph.se
estimator, so the physics here is the estimator's physics.

Global features per record (the paper's phi_t):
    dx_hat = H_W^+ dz      implied state change (weighted pseudo-inverse, the WLS step)
    q_perp, q_par, R       unexplained / explained energy of dz (weighted) and their ratio
    kappa(H)               condition number of W^1/2 H (constant for the chord Jacobian)
    alpha = U^T W^1/2 dz   direction coefficients; alpha_weak = energy in the weakest directions

Per-bus features [n, N, 8] for localization, each feature aggregated to a bus over its own
meters and incident branch flows:
    0 |d theta_hat|, 1 |dV_hat|          implied state move at the bus
    2 unexplained energy, 3 explained energy   sum over incident meters of (r/sigma)^2
    4 consistency ratio                  sqrt(2 / 3)
    5 leverage-weighted change           max over incident meters of l_k |dz_k| / sigma_k
    6 sensitivity-normalised change      max over incident meters of |dz_k| / s_k
    7 weak-direction move                implied move projected on the n_weak weakest directions
Needs the [se] extra (pandapower + scipy) and a timeline: the previous frame's readings, and its
clean layer for the slack angle reference alone. A v0.7.2 record shard is refused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..formulas.projection import (
    bus_incidence as _bus_incidence,
)
from ..formulas.projection import (
    direction_coefficients,
    explained_unexplained,
    leverage,
    meters_to_buses,
    weak_directions,
    weak_move,
    weighted_pseudoinverse,
)
from ..models.scores import JacobianOutputs  # noqa: F401  re-exported: defined here before the models package
from .base import require_physical

if TYPE_CHECKING:
    from ..dataset import FdiaGraph
    from .base import SEBase

BUS_FEATURE_NAMES = [
    "dtheta_hat",
    "dv_hat",
    "unexplained",
    "explained",
    "ratio",
    "leverage_change",
    "sensitivity_change",
    "weak_move",
]
GLOBAL_FEATURE_NAMES = ["q_perp", "q_par", "ratio", "alpha_weak"]


def bus_incidence(est: SEBase, edge_index: np.ndarray) -> list[np.ndarray]:
    """Masked-measurement indices touching each bus: its own V/P/Q/theta channels plus the flows
    of every incident branch (a flow meter reacts to both endpoints). Same map as ResidualLocalizer."""
    return _bus_incidence(est.N, est.E, edge_index, est.mask)


class JacobianFeatures:
    """Fit on the train split (any fdia_graph.se estimator supplies the physics), then transform
    any split of the same timeline into per-bus and global Jacobian-informed features, each frame's
    change taken against the previous frame's estimate (a v0.7.2 record shard is refused: its rows
    are not consecutive frames).

    n_weak: how many of the weakest observable state directions define the "weak" subspace
    (default: 10 percent of the state dimension, at least 2).
    """

    def __init__(self, estimator: Optional[SEBase] = None, n_weak: Optional[int] = None) -> None:
        self.estimator = estimator
        self.n_weak = n_weak

    def fit(self, ds: FdiaGraph) -> JacobianFeatures:
        from .base import SEBase  # noqa: F401  (typing aid)
        from .methods import WLS

        require_physical(ds)
        self.est = self.estimator if self.estimator is not None else WLS()
        if not self.est.is_fitted:
            self.est.fit(ds)
        est = self.est
        if not ds.is_timeline or ds._clean_np is None:
            raise ValueError(
                "Jacobian features need a timeline: the previous frame's readings and its angle reference"
            )
        self._pool: np.ndarray = ds._clean_np  # read for the slack angle reference only
        self._inc = bus_incidence(est, ds.edge_index_np)
        sw = np.sqrt(est.Wk)  # W^1/2 as a vector
        Hw = sw[:, None] * est.H  # [m, SD], the whitened Jacobian
        self._Hw, self._sw = Hw, sw
        # weighted pseudo-inverse H_W^+ = (H^T W H)^-1 H^T W, as the [SD, m] map dz -> dx_hat
        self._pinv = weighted_pseudoinverse(est.H, est.Wk, est._Ai)
        # projection onto the column space of the whitened Jacobian: leverage on its diagonal
        self.leverage = leverage(Hw, est._Ai)
        self.sensitivity = np.linalg.norm(Hw, axis=1)  # s_i, whitened row norms
        self.observability = np.linalg.norm(Hw, axis=0)  # o_j, column norms
        k = self.n_weak if self.n_weak is not None else max(2, est.SD // 10)
        U, S, Vweak, weak_rows = weak_directions(Hw, k)
        self.singular_values = S
        self.kappa = float(S[0] / max(S[-1], 1e-300))
        self._U = U  # [m, SD]
        self._Vweak = Vweak  # [SD, k] weakest right-singular directions
        self._weak_rows = weak_rows
        self.N, self.ns = est.N, len(est.keep)
        return self

    # ---- the measurement change against the previous frame's estimate ---------------------
    def previous_estimate(self, d: dict[str, np.ndarray], chunk: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """The estimator's plain solve of the previous frame's readings, x_hat_{t-1} [n, SD], and
        the slack angle it is referenced to [n]."""
        est = self.est
        zp = est._z_of(d["prev_node_x"], d["prev_edge_x"])
        # the one truth read: the slack angle, the reference frame every fdia_graph.se estimate is
        # expressed in (the scored estimators pin it the same way); no other part of the state
        thsl = est._truth_of(self._pool[d["prev_timestep"].astype(int)])["thsl"]
        parts = [est._solve_plain(zp[i : i + chunk], thsl[i : i + chunk]) for i in range(0, len(zp), chunk)]
        return (np.concatenate(parts) if parts else np.zeros((0, est.SD))), thsl

    def delta_z(self, d: dict[str, np.ndarray]) -> np.ndarray:
        """dz = z_t - h(x_hat_{t-1}), the reading change the previous estimate does not predict."""
        z = self.est._z_of(d["node_x"], d["edge_x"])
        x, thsl = self.previous_estimate(d)
        return z - self.est._h(x, thsl)

    # ---- features ------------------------------------------------------------------------
    def transform(self, d: dict[str, np.ndarray]) -> JacobianOutputs:
        """d must carry node_x, edge_x, prev_node_x, prev_edge_x and prev_timestep (as a timeline's
        export returns them, physical units). Returns {"bus": [n, N, 8], "global": [n, 4], "dx_hat": [n, SD], "r_perp": [n, m]}."""
        est = self.est
        dz = self.delta_z(d)  # [n, m]
        dx, r_par, r_perp = explained_unexplained(
            dz, est.H, self._pinv
        )  # implied state change, its two parts
        u_perp = r_perp * self._sw  # whitened (r / sigma)
        u_par = r_par * self._sw
        q_perp = np.linalg.norm(u_perp, axis=1)
        q_par = np.linalg.norm(u_par, axis=1)
        ratio = q_perp / (q_par + 1e-9)
        alpha = direction_coefficients(dz * self._sw, self._U)  # [n, SD]
        alpha_weak = np.linalg.norm(alpha[:, self._weak_rows], axis=1)
        glob = np.stack([q_perp, q_par, ratio, alpha_weak], axis=1)

        n, N, ns = dz.shape[0], self.N, self.ns
        dth = np.zeros((n, N))
        dth[:, est.keep] = np.abs(dx[:, :ns])
        dv = np.abs(dx[:, ns:])
        weak = weak_move(dx, self._Vweak)  # implied move restricted to the weak subspace
        wth = np.zeros((n, N))
        wth[:, est.keep] = weak[:, :ns]
        weak_bus = np.sqrt(wth**2 + weak[:, ns:] ** 2)  # per-bus size of the weak-subspace move
        lev_change = np.abs(dz) * self._sw * self.leverage[None, :]
        sens_change = np.abs(dz) / np.maximum(self.sensitivity[None, :], 1e-9)
        unexp = meters_to_buses(u_perp**2, self._inc, "sum")
        expl = meters_to_buses(u_par**2, self._inc, "sum")
        lev = meters_to_buses(lev_change, self._inc, "max")
        sens = meters_to_buses(sens_change, self._inc, "max")
        bus_ratio = np.sqrt(unexp) / (np.sqrt(expl) + 1e-9)
        bus = np.stack([dth, dv, unexp, expl, bus_ratio, lev, sens, weak_bus], axis=2)
        return JacobianOutputs(bus=bus, global_=glob, dx_hat=dx, r_perp=r_perp)
