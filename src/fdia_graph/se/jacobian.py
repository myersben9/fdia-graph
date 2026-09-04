"""Jacobian-informed features (Abdulin & Narimani): the measurement Jacobian as a physics transform
of the scan-to-scan measurement change, not as raw model input.

For a record with measurements z_t, the change dz = z_t - h(x_{t-1}) is taken against the exact
measurement prediction of the previous clean state (the shard stores the clean pool per timestep;
this is the same construction as the shard's temporal_delta). The chord Jacobian H at the benign
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
Needs the [se] extra (torch + pandapower) and a v0.7.2+ shard (the clean layer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

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


def bus_incidence(est: "SEBase", edge_index: np.ndarray) -> List[np.ndarray]:
    """Masked-measurement indices touching each bus: its own V/P/Q/theta channels plus the flows
    of every incident branch (a flow meter reacts to both endpoints). Same map as ResidualLocalizer."""
    N, E = est.N, est.E
    inc = np.zeros((N, 4 * N + 2 * E), bool)
    for c in range(4):
        inc[np.arange(N), c * N + np.arange(N)] = True
    for c in range(2):
        cols = 4 * N + c * E + np.arange(E)
        inc[edge_index[0], cols] = True
        inc[edge_index[1], cols] = True
    incm = inc[:, est.mask]
    return [np.where(incm[b])[0] for b in range(N)]


class JacobianFeatures:
    """Fit on the train split (any fdia_graph.se estimator supplies the physics), then transform
    any split of the same shard into per-bus and global Jacobian-informed features.

    n_weak: how many of the weakest observable state directions define the "weak" subspace
    (default: 10 percent of the state dimension, at least 2).
    """

    def __init__(self, estimator: Optional["SEBase"] = None, n_weak: Optional[int] = None) -> None:
        self.estimator = estimator
        self.n_weak = n_weak

    def fit(self, ds: "FdiaGraph") -> "JacobianFeatures":
        from .base import SEBase  # noqa: F401  (typing aid)
        from .methods import WLS

        self.est = self.estimator if self.estimator is not None else WLS()
        if not hasattr(self.est, "H"):
            self.est.fit(ds)
        est = self.est
        if ds._clean_np is None:
            raise ValueError("Jacobian features need the clean layer (a v0.7.2+ shard)")
        self._pool: np.ndarray = ds._clean_np  # clean state per pool timestep, shared by every split
        self._inc = bus_incidence(est, ds.edge_index_np)
        sw = np.sqrt(est.Wk)  # W^1/2 as a vector
        Hw = sw[:, None] * est.H  # [m, SD], the whitened Jacobian
        self._Hw, self._sw = Hw, sw
        # weighted pseudo-inverse H_W^+ = (H^T W H)^-1 H^T W, as the [SD, m] map dz -> dx_hat
        self._pinv = est._Ai @ (est.H.T * est.Wk[None, :])
        # projection onto the column space of the whitened Jacobian: leverage on its diagonal
        P = Hw @ est._Ai @ Hw.T
        self.leverage = np.clip(np.diag(P), 0.0, 1.0)
        self.sensitivity = np.linalg.norm(Hw, axis=1)  # s_i, whitened row norms
        self.observability = np.linalg.norm(Hw, axis=0)  # o_j, column norms
        U, S, Vt = np.linalg.svd(Hw, full_matrices=False)
        self.singular_values = S
        self.kappa = float(S[0] / max(S[-1], 1e-300))
        k = self.n_weak if self.n_weak is not None else max(2, est.SD // 10)
        self._U = U  # [m, SD]
        self._Vweak = Vt[-k:].T  # [SD, k] weakest right-singular directions
        self._weak_rows = np.arange(len(S) - k, len(S))
        self.N, self.ns = est.N, len(est.keep)
        return self

    # ---- the measurement change against the previous clean state --------------------------
    def delta_z(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        est = self.est
        z = est._z_of(d["node_x"], d["edge_x"])
        t = d["timestep"].astype(int)
        prev = self._pool[np.maximum(t - 1, 0)]  # first pool step: dz is the noise alone
        tr = est._truth_of(prev)
        return z - est._h(tr["x"], tr["thsl"])

    # ---- features ------------------------------------------------------------------------
    def transform(self, d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """d must carry node_x, edge_x, timestep (as fdia_graph's to_numpy returns them, physical
        units). Returns {"bus": [n, N, 8], "global": [n, 4], "dx_hat": [n, SD], "r_perp": [n, m]}."""
        est = self.est
        dz = self.delta_z(d)  # [n, m]
        dx = dz @ self._pinv.T  # [n, SD] implied state change
        r_par = dx @ est.H.T  # explained part
        r_perp = dz - r_par  # unexplained part
        u_perp = r_perp * self._sw  # whitened (r / sigma)
        u_par = r_par * self._sw
        q_perp = np.linalg.norm(u_perp, axis=1)
        q_par = np.linalg.norm(u_par, axis=1)
        ratio = q_perp / (q_par + 1e-9)
        alpha = (dz * self._sw) @ self._U  # [n, SD] direction coefficients
        alpha_weak = np.linalg.norm(alpha[:, self._weak_rows], axis=1)
        glob = np.stack([q_perp, q_par, ratio, alpha_weak], axis=1)

        n, N, ns = dz.shape[0], self.N, self.ns
        dth = np.zeros((n, N))
        dth[:, est.keep] = np.abs(dx[:, :ns])
        dv = np.abs(dx[:, ns:])
        weak = (dx @ self._Vweak) @ self._Vweak.T  # implied move restricted to the weak subspace
        wth = np.zeros((n, N))
        wth[:, est.keep] = weak[:, :ns]
        weak_move = np.sqrt(wth**2 + weak[:, ns:] ** 2)
        lev_change = np.abs(dz) * self._sw * self.leverage[None, :]
        sens_change = np.abs(dz) / np.maximum(self.sensitivity[None, :], 1e-9)
        unexp = np.zeros((n, N))
        expl = np.zeros((n, N))
        lev = np.zeros((n, N))
        sens = np.zeros((n, N))
        for b, ix in enumerate(self._inc):
            if len(ix):
                unexp[:, b] = (u_perp[:, ix] ** 2).sum(axis=1)
                expl[:, b] = (u_par[:, ix] ** 2).sum(axis=1)
                lev[:, b] = lev_change[:, ix].max(axis=1)
                sens[:, b] = sens_change[:, ix].max(axis=1)
        bus_ratio = np.sqrt(unexp) / (np.sqrt(expl) + 1e-9)
        bus = np.stack([dth, dv, unexp, expl, bus_ratio, lev, sens, weak_move], axis=2)
        return {"bus": bus, "global": glob, "dx_hat": dx, "r_perp": r_perp}
