"""The estimator method classes. Each changes exactly one thing about SEBase, so a difference
between two arms is a difference between estimators rather than between implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..formulas.estimation import gate_weights, huber_weights, normal_matrix, whitened_svd_basis
from ..formulas.linalg import condition_number
from .base import SEBase, require_physical

if TYPE_CHECKING:
    from ..dataset import FdiaGraph


class WLS(SEBase):
    """The audited baseline: least squares weighted by accuracy-class meter error, full state."""


class AdaptiveWeighting(SEBase):
    """Iteratively reweighted least squares (the Huber M-estimator).

    Each pass down-weights measurement i by min(1, c / |r_i|_normalized) and re-solves until the
    weights settle. c is the one hyperparameter; the paper's validation-selected values are 1.5,
    2.5 and 6.0 on IEEE 14, 118 and 300.
    """

    def __init__(self, c: float = 1.5, npass: int = 40, iters: int = 8, tol: float = 1e-4) -> None:
        super().__init__(npass=npass, iters=iters)
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        self.c = c
        self.tol = tol  # stop the reweighting passes once no weight moves by more than this

    def _solve(self, z: np.ndarray, thsl: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        x = super()._solve(z, thsl, w)
        return self._huber_passes(x, z, self.Wk if w is None else w, thsl, self.c, self.tol)


class ResidualRemoval(SEBase):
    """Largest-normalized-residual removal [HAN75] with an observability guard.

    Each pass removes, per record, the ONE measurement with the largest normalized residual when
    it exceeds the threshold, then re-solves; it stops when no residual exceeds the threshold or
    after npass removals. One at a time matters: a single gross error smears large residuals onto
    its honest neighbours, which fall back below the threshold once the bad meter is out.
    Critical measurements (residual structurally zero) are never removed, and a removal that would
    degrade conditioning beyond cond_mult times the full system's is refused (that meter is kept
    and the next largest is considered on the next pass). Paper thresholds: 4.0 (14) and 5.0
    (118); on IEEE 300 no threshold on the grid helped.
    """

    def __init__(
        self, threshold: float = 4.0, cond_mult: float = 100.0, npass: int = 40, iters: int = 8
    ) -> None:
        super().__init__(npass=npass, iters=iters)
        if threshold <= 0 or cond_mult < 1:
            raise ValueError(f"need threshold > 0 and cond_mult >= 1, got {threshold}, {cond_mult}")
        self.threshold = threshold
        self.cond_mult = cond_mult

    def _post_fit(self) -> None:
        self._cond_full = condition_number(normal_matrix(self.H, self.Wk))

    def _observable(self, w: np.ndarray) -> bool:
        # Runs once per removal candidate, so it uses the Cholesky/power-iteration condition
        # estimate rather than a full eigen-decomposition (formulas.linalg.condition_number).
        return condition_number(normal_matrix(self.H, w)) <= self.cond_mult * self._cond_full

    def _solve(self, z: np.ndarray, thsl: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        base = np.broadcast_to(self.Wk if w is None else w, z.shape)
        keep = np.ones_like(z)
        held = np.zeros(z.shape, bool)  # removal refused by the guard: never proposed again
        rows = np.arange(z.shape[0])
        x = self._w_solve(z, base * keep, thsl)
        for _ in range(self.npass):
            open_ = (keep > 0) & ~held & ~self.critical[None, :]
            rN = np.where(open_, self._nres(x, z, thsl), -np.inf)
            j = rN.argmax(axis=1)
            bad = rN[rows, j] > self.threshold
            if not bad.any():
                break
            for i in np.flatnonzero(bad):
                self._remove_one(i, j[i], keep, held, base)
            x = self._w_solve(z, base * keep, thsl)
        return x

    def _remove_one(self, i: int, j: int, keep: np.ndarray, held: np.ndarray, base: np.ndarray) -> None:
        """Remove meter j of record i when the system stays observable without it, else hold it."""
        trial = keep[i].copy()
        trial[j] = 0.0
        if self._observable(base[i] * trial):
            keep[i] = trial
        else:
            held[i, j] = True


class SubspacePrior(SEBase):
    """The learned operating-point prior, optionally composed with Huber reweighting.

    fit() whitens the benign training states per coordinate, takes the SVD, keeps the leading
    rank_frac fraction of directions and re-orthonormalizes (the whitened basis is otherwise
    catastrophically ill conditioned near full rank). The estimate is restricted to
    x = mean + VK c, a K-dim solve. reweight="huber" is the paper's proposed pair; the
    validation-selected rank fractions are 0.20, 0.50 and 0.50 on IEEE 14, 118 and 300.
    """

    def __init__(
        self,
        rank_frac: float = 0.5,
        reweight: Optional[str] = None,
        c: float = 1.5,
        npass: int = 40,
        iters: int = 8,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(npass=npass, iters=iters)
        self.tol = tol  # stop the Huber passes once no weight moves by more than this
        if reweight not in (None, "huber"):
            raise ValueError("reweight must be None or 'huber'")
        if not 0.0 < rank_frac <= 1.0:
            raise ValueError(f"rank_frac must be in (0, 1], got {rank_frac}")
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        self.rank_frac = rank_frac
        self.reweight = reweight
        self.c = c

    def _fit_states(self, x_benign: np.ndarray) -> None:
        self.K, self.VK = whitened_svd_basis(x_benign, self.rank_frac)

    def _subspace(self) -> np.ndarray:
        return self.VK

    def _solve(self, z: np.ndarray, thsl: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        x = super()._solve(z, thsl, w)
        if self.reweight is None:
            return x
        return self._huber_passes(x, z, self.Wk if w is None else w, thsl, self.c, self.tol)


class JacobianWeighting(SEBase):
    """Jacobian-informed reweighting: down-weight meters whose scan-to-scan change is physically
    unexplained (Abdulin & Narimani's r_perp), then solve once with those weights.

    r_perp = (I - P_H) dz is the part of the measurement change since the previous frame's estimate
    (dz = z_t - h(x_hat_{t-1}), `se.jacobian.JacobianFeatures`, so a timeline is required) that no
    state change can produce. An in-place corruption leaves a large r_perp on the tampered
    meters; a stealthy re-solve leaves none, so this arm expects to help on Ad/As/Ar and to match
    WLS on Aq/At/Al. The weight is Huber's, min(1, c / |r_perp_i / sigma_i|), computed from the
    temporal residual rather than from the estimate's own residual, so it needs no reweighting
    passes. reweight="huber" then runs the classical Huber passes on the estimate's own residual
    starting from those weights, so the temporal and the static evidence are both used.
    """

    def __init__(
        self,
        c: float = 3.0,
        reweight: Optional[str] = None,
        huber_c: float = 1.5,
        npass: int = 40,
        iters: int = 8,
        tol: float = 1e-4,
    ) -> None:
        super().__init__(npass=npass, iters=iters)
        if c <= 0 or huber_c <= 0:
            raise ValueError(f"c and huber_c must be > 0, got {c}, {huber_c}")
        if reweight not in (None, "huber"):
            raise ValueError("reweight must be None or 'huber'")
        self.c = c
        self.reweight = reweight  # "huber": Huber passes on the estimate's residual, from these weights
        self.huber_c = huber_c
        self.tol = tol  # stop the Huber passes once no weight moves by more than this

    def fit(self, ds: FdiaGraph, n_calib: int = 600, calibrate: str = "truth") -> JacobianWeighting:
        from .jacobian import JacobianFeatures

        super().fit(ds, n_calib, calibrate)
        self._jf = JacobianFeatures(estimator=self).fit(ds)  # built once: SVD, leverage, pseudo-inverse
        return self

    def weights(self, ds: FdiaGraph) -> np.ndarray:
        """Per-record meter weights [n, m] from the unexplained temporal residual."""
        require_physical(ds)
        d = ds.export(["node_x", "edge_x", "prev_node_x", "prev_edge_x", "prev_timestep"])
        u = np.abs(self._jf.transform(d)["r_perp"]) * np.sqrt(self.Wk)[None, :]
        return self.Wk[None, :] * huber_weights(u, self.c)

    def _record_weights(self, ds: FdiaGraph) -> np.ndarray:
        return self.weights(ds)

    def _solve(self, z: np.ndarray, thsl: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        x = super()._solve(z, thsl, w)
        if self.reweight != "huber":
            return x
        # the temporal weights first, then the classical passes on top
        return self._huber_passes(x, z, self.Wk if w is None else w, thsl, self.huber_c, self.tol)


class GatedPrior(SubspacePrior):
    """The headline estimator (subspace prior + Huber) with localization-gated weights.

    A fitted fdia_graph.localization localizer flags the attacked buses of each record; every
    meter on a flagged bus and every flow on a branch incident to it is down-weighted by
    `gate_factor` before the solve, so the low-rank benign prior supplies the state there instead
    of the tampered measurements. This is the route by which temporal or Jacobian-informed
    detection (which sees the stealthy re-solve families the residual cannot) can reach the
    estimate. `gate="oracle"` uses the true per-bus labels and gives the ceiling for any gate.
    `secured` names meters (indices into the masked measurement vector, the layout of
    `trust.TrustSelector.order`) the gate never down-weights: a secured meter reads its true
    value whatever bus the gate flags, and pulling it out with the rest of the bus's meters throws
    away exactly what securing it bought (on a `trust.secured_copy` every gate was worse than no
    gate until this exemption).
    """

    def __init__(
        self, gate: Any = None, gate_factor: float = 1e-3, secured: Optional[Sequence[int]] = None, **kw: Any
    ) -> None:
        super().__init__(**kw)
        if gate is None:
            raise ValueError("pass gate=<fitted localizer> or gate='oracle'")
        if not 0.0 < gate_factor <= 1.0:
            raise ValueError(f"gate_factor must be in (0, 1], got {gate_factor}")
        self.gate = gate
        self.gate_factor = gate_factor
        self.secured = np.asarray([] if secured is None else secured, int)

    def gated_weights(self, ds: FdiaGraph) -> np.ndarray:
        """Per-record meter weights [n, m]: Wk, times gate_factor on meters incident to flagged
        buses, the secured meters kept at Wk."""
        from .jacobian import bus_incidence

        if isinstance(self.gate, str):
            if self.gate != "oracle":
                raise ValueError(f"gate must be a fitted localizer or 'oracle', got {self.gate!r}")
            flags = ds.export(["y"])["y"].astype(bool)  # the ceiling: true labels
        else:
            flags = np.asarray(self.gate.localize(ds), bool)
        w = gate_weights(self.Wk, flags, bus_incidence(self, ds.edge_index_np), self.gate_factor)
        if len(self.secured):
            w[:, self.secured] = self.Wk[self.secured]
        return w

    def _record_weights(self, ds: FdiaGraph) -> np.ndarray:
        return self.gated_weights(ds)
