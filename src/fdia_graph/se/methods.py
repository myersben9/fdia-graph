"""The estimator method classes. Each changes exactly one thing about SEBase, so a difference
between two arms is a difference between estimators rather than between implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .base import SEBase

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

    def _solve(self, z: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        x = self._solve_plain(z, thsl)
        prev = None
        for _ in range(self.npass):
            a = np.minimum(1.0, self.c / np.maximum(self._nres(x, z, thsl), 1e-9))
            if prev is not None and np.abs(a - prev).max() < self.tol:
                break  # weights settled: further passes reproduce the same estimate
            x = self._w_solve(z, self.Wk * a, thsl)
            prev = a
        return x


class ResidualRemoval(SEBase):
    """Largest-normalized-residual removal with a real observability guard.

    Measurements whose normalized residual exceeds the threshold are removed and the record
    re-solved. Critical measurements (residual structurally zero) are never removed, and a removal
    set that would degrade conditioning beyond cond_mult times the full system's is walked back,
    restoring the lowest-residual removals first. Paper thresholds: 4.0 (14) and 5.0 (118); on
    IEEE 300 no threshold on the grid helped.
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
        self._cond_full = self._cond(self.H.T @ (self.Wk[:, None] * self.H))

    def _observable(self, w: np.ndarray) -> bool:
        # The guard runs once per bad record per trial, so it uses the Cholesky/power-iteration
        # condition estimate rather than a full eigen-decomposition (see SEBase._cond).
        return self._cond(self.H.T @ (w[:, None] * self.H)) <= self.cond_mult * self._cond_full

    def _solve(self, z: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        keep = np.ones_like(z)
        x = self._w_solve(z, self.Wk * keep, thsl)
        for _ in range(self.npass):
            rN = self._nres(x, z, thsl)
            bad = (rN > self.threshold) & (keep > 0) & (~self.critical)[None, :]
            if not bad.any():
                break
            prop = keep * (~bad)
            for i in np.where(bad.any(axis=1))[0]:
                trial = prop[i].copy()
                for _ in range(6):
                    if self._observable(self.Wk * trial):
                        break
                    back = np.where((keep[i] > 0) & (trial == 0))[0]
                    if len(back) == 0:
                        break
                    order = back[np.argsort(rN[i][back])]  # smallest residual restored first
                    half = order[: max(1, len(order) // 2)]
                    trial[half] = keep[i][half]
                prop[i] = trial
            keep = prop
            x = self._w_solve(z, self.Wk * keep, thsl)
        return x


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
        mean = x_benign.mean(axis=0)
        std = np.maximum(x_benign.std(axis=0), 1e-9)  # whiten: angle and voltage differ ~10x in scale
        _, _, Vt = np.linalg.svd((x_benign - mean) / std, full_matrices=False)
        K = max(1, int(round(self.rank_frac * x_benign.shape[1])))
        self.K = K
        self.VK = np.linalg.qr((Vt[:K].T * std[:, None]))[0]  # un-whiten, re-orthonormalize

    def _basis(self) -> np.ndarray:
        return self.VK

    def _solve(self, z: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        x = self._solve_plain(z, thsl)
        if self.reweight is None:
            return x
        prev = None
        for _ in range(self.npass):
            a = np.minimum(1.0, self.c / np.maximum(self._nres(x, z, thsl), 1e-9))
            if prev is not None and np.abs(a - prev).max() < self.tol:
                break  # weights settled
            x = self._w_solve(z, self.Wk * a, thsl)
            prev = a
        return x


class JacobianWeighting(SEBase):
    """Jacobian-informed reweighting: down-weight meters whose scan-to-scan change is physically
    unexplained (Abdulin & Narimani's r_perp), then solve once with those weights.

    r_perp = (I - P_H) dz is the part of the measurement change since the previous clean state that
    no state change can produce. An in-place corruption leaves a large r_perp on the tampered
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
    ) -> None:
        super().__init__(npass=npass, iters=iters)
        if c <= 0 or huber_c <= 0:
            raise ValueError(f"c and huber_c must be > 0, got {c}, {huber_c}")
        if reweight not in (None, "huber"):
            raise ValueError("reweight must be None or 'huber'")
        self.c = c
        self.reweight = (
            reweight  # "huber": Huber passes on the estimate's residual, starting from the Jacobian weights
        )
        self.huber_c = huber_c

    def weights(self, ds: "FdiaGraph") -> np.ndarray:
        """Per-record meter weights [n, m] from the unexplained temporal residual."""
        from .jacobian import JacobianFeatures

        jf = JacobianFeatures(estimator=self).fit(ds)
        d = ds.to_numpy(["node_x", "edge_x", "timestep"])
        u = np.abs(jf.transform(d)["r_perp"]) * np.sqrt(self.Wk)[None, :]
        return self.Wk[None, :] * np.minimum(1.0, self.c / np.maximum(u, 1e-9))

    def estimate(self, ds: "FdiaGraph", chunk: int = 1000) -> np.ndarray:
        d = ds.to_numpy(["node_x", "edge_x", "clean"])
        tr = self._truth_of(d["clean"])
        z = self._z_of(d["node_x"], d["edge_x"])
        w = self.weights(ds)
        out = np.empty((z.shape[0], self.SD))
        for s in range(0, z.shape[0], chunk):
            e = slice(s, s + chunk)
            x = self._w_solve(z[e], w[e], tr["thsl"][e])
            if self.reweight == "huber":  # temporal weights first, then the classical passes on top
                for _ in range(self.npass):
                    a = np.minimum(1.0, self.huber_c / np.maximum(self._nres(x, z[e], tr["thsl"][e]), 1e-9))
                    x = self._w_solve(z[e], w[e] * a, tr["thsl"][e])
            out[e] = x
        return out


class GatedPrior(SubspacePrior):
    """The headline estimator (subspace prior + Huber) with localization-gated weights.

    A fitted fdia_graph.localization localizer flags the attacked buses of each record; every
    meter on a flagged bus and every flow on a branch incident to it is down-weighted by
    `gate_factor` before the solve, so the low-rank benign prior supplies the state there instead
    of the tampered measurements. This is the route by which temporal or Jacobian-informed
    detection (which sees the stealthy re-solve families the residual cannot) can reach the
    estimate. `gate="oracle"` uses the true per-bus labels and gives the ceiling for any gate.
    """

    def __init__(self, gate: Any = None, gate_factor: float = 1e-3, **kw: Any) -> None:
        super().__init__(**kw)
        if gate is None:
            raise ValueError("pass gate=<fitted localizer> or gate='oracle'")
        if not 0.0 < gate_factor <= 1.0:
            raise ValueError(f"gate_factor must be in (0, 1], got {gate_factor}")
        self.gate = gate
        self.gate_factor = gate_factor

    def gated_weights(self, ds: "FdiaGraph") -> np.ndarray:
        """Per-record meter weights [n, m]: Wk, times gate_factor on meters incident to flagged buses."""
        from .jacobian import bus_incidence

        if isinstance(self.gate, str):
            if self.gate != "oracle":
                raise ValueError(f"gate must be a fitted localizer or 'oracle', got {self.gate!r}")
            flags = ds.to_numpy(["y"])["y"].astype(bool)  # the ceiling: true labels
        else:
            flags = np.asarray(self.gate.localize(ds), bool)
        inc = bus_incidence(self, ds.edge_index_np)
        w = np.repeat(self.Wk[None, :], flags.shape[0], axis=0)
        for b, ix in enumerate(inc):
            if len(ix):
                hit = flags[:, b]
                w[np.ix_(hit, ix)] *= self.gate_factor
        return w

    def estimate(self, ds: "FdiaGraph", chunk: int = 1000) -> np.ndarray:
        d = ds.to_numpy(["node_x", "edge_x", "clean"])
        tr = self._truth_of(d["clean"])
        z = self._z_of(d["node_x"], d["edge_x"])
        w = self.gated_weights(ds)
        out = np.empty((z.shape[0], self.SD))
        for s in range(0, z.shape[0], chunk):
            e = slice(s, s + chunk)
            x = self._w_solve(z[e], w[e], tr["thsl"][e])
            if self.reweight == "huber":
                prev = None
                for _ in range(self.npass):
                    a = np.minimum(1.0, self.c / np.maximum(self._nres(x, z[e], tr["thsl"][e]), 1e-9))
                    if prev is not None and np.abs(a - prev).max() < self.tol:
                        break
                    x = self._w_solve(z[e], w[e] * a, tr["thsl"][e])
                    prev = a
            out[e] = x
        return out
