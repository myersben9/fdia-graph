"""The estimator method classes. Each changes exactly one thing about SEBase, so a difference
between two arms is a difference between estimators rather than between implementations."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import SEBase


class WLS(SEBase):
    """The audited baseline: least squares weighted by accuracy-class meter error, full state."""


class AdaptiveWeighting(SEBase):
    """Iteratively reweighted least squares (the Huber M-estimator).

    Each pass down-weights measurement i by min(1, c / |r_i|_normalized) and re-solves until the
    weights settle. c is the one hyperparameter; the paper's validation-selected values are 1.5,
    2.5 and 6.0 on IEEE 14, 118 and 300.
    """

    def __init__(self, c: float = 1.5, npass: int = 40, iters: int = 8) -> None:
        super().__init__(npass=npass, iters=iters)
        if c <= 0:
            raise ValueError(f"c must be > 0, got {c}")
        self.c = c

    def _solve(self, z: np.ndarray, vsl: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        x = self._solve_plain(z, vsl, thsl)
        for _ in range(self.npass):
            a = np.minimum(1.0, self.c / np.maximum(self._nres(x, z, vsl, thsl), 1e-9))
            x = self._w_solve(z, self.Wk * a, vsl, thsl)
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
        A = self.H.T @ (self.Wk[:, None] * self.H)
        ev = np.linalg.eigvalsh(0.5 * (A + A.T))
        self._cond_full = ev.max() / max(ev.min(), 1e-300)

    def _observable(self, w: np.ndarray) -> bool:
        A = self.H.T @ (w[:, None] * self.H)
        ev = np.linalg.eigvalsh(0.5 * (A + A.T))
        if ev.min() <= 0:
            return False
        return ev.max() / ev.min() <= self.cond_mult * self._cond_full

    def _solve(self, z: np.ndarray, vsl: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        keep = np.ones_like(z)
        x = self._w_solve(z, self.Wk * keep, vsl, thsl)
        for _ in range(self.npass):
            rN = self._nres(x, z, vsl, thsl)
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
            x = self._w_solve(z, self.Wk * keep, vsl, thsl)
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
    ) -> None:
        super().__init__(npass=npass, iters=iters)
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

    def _solve(self, z: np.ndarray, vsl: np.ndarray, thsl: np.ndarray) -> np.ndarray:
        x = self._solve_plain(z, vsl, thsl)
        if self.reweight is None:
            return x
        for _ in range(self.npass):
            a = np.minimum(1.0, self.c / np.maximum(self._nres(x, z, vsl, thsl), 1e-9))
            x = self._w_solve(z, self.Wk * a, vsl, thsl)
        return x
