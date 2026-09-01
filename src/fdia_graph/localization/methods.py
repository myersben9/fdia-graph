"""The localizer method classes. Each changes exactly one thing about LocalizerBase — the per-bus
score — so a difference between two arms is a difference between detection signals, not between
calibration or metrics code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from .base import LocalizerBase

if TYPE_CHECKING:
    from ..dataset import FdiaGraph
    from ..se.base import SEBase


class SwingThreshold(LocalizerBase):
    """The temporal-spike detector: score = the bus's windowed relative-swing magnitude.

    The shard's swing feature is each scan's injection change as a z-score of the bus's typical
    recent change, so any attack edit that exceeds the noise floor appears as a spike the moment it
    starts — including the BDD-stealthy re-solve families. The one family built to defeat it is the
    slow ramp At, which stays inside typical per-scan change by construction.
    """

    def _fields(self) -> List[str]:
        return ["swing"]

    def _score(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        return np.abs(d["swing"]).max(axis=2)  # worst channel (dP or dQ) per bus


class DeltaThreshold(LocalizerBase):
    """Ablation arm: the raw one-scan injection change, scaled by the bus's benign RMS change.

    The same signal as SwingThreshold without the windowed typical-change normalization — one
    global scale per bus and channel instead of a running local one — so the gap between the two
    is exactly what the windowing buys.
    """

    def _fields(self) -> List[str]:
        return ["temporal_delta"]

    def _fit_stats(self, d: Dict[str, np.ndarray], ben: np.ndarray, ds: "FdiaGraph") -> None:
        td = d["temporal_delta"][ben]
        self.sd = np.maximum(np.sqrt((td**2).mean(axis=0)), 1e-9)  # [N, 2] benign RMS per channel

    def _score(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        return np.abs(d["temporal_delta"] / self.sd[None]).max(axis=2)


class ResidualLocalizer(LocalizerBase):
    """The classical arm: largest normalized residual from a state-estimation solve, per bus.

    Runs the composed estimator (default WLS from fdia_graph.se, any SEBase works), computes each
    measurement's normalized residual at the estimate, and scores a bus by the largest residual on
    the bus's own meters and its incident branch flows. This is textbook bad-data identification:
    it catches the in-place corruption families (Ad/As/Ar) and, by construction, misses the
    stealthy re-solve families (Aq/At/Al) whose measurements stay physics-consistent. Metering is
    sparse, so a bus with no metered channel and no metered incident flow scores zero and can never
    be localized by residuals. Needs the [se] extra (torch + pandapower).
    """

    def __init__(self, estimator: Optional["SEBase"] = None, fa_target: float = 0.01) -> None:
        super().__init__(fa_target=fa_target)
        self.estimator = estimator  # None -> a fresh WLS is fitted in fit()

    def _fields(self) -> List[str]:
        return ["node_x", "edge_x", "clean"]

    def _fit_stats(self, d: Dict[str, np.ndarray], ben: np.ndarray, ds: "FdiaGraph") -> None:
        from ..se import WLS

        self.est = self.estimator if self.estimator is not None else WLS()
        self.est.fit(ds)
        # Bus <- measurement incidence in the estimator's masked layout. Unmasked slot order is
        # [V(N), P(N), Q(N), theta(N), Pf(E), Qf(E)]; node channels touch their own bus, a flow
        # meter touches BOTH endpoints of its line (an injection edit perturbs every incident flow).
        N, E = self.est.N, self.est.E
        ei = d["edge_index"]
        inc = np.zeros((N, 4 * N + 2 * E), bool)
        for c in range(4):
            inc[np.arange(N), c * N + np.arange(N)] = True
        for c in range(2):
            cols = 4 * N + c * E + np.arange(E)
            inc[ei[0], cols] = True
            inc[ei[1], cols] = True
        incm = inc[:, self.est.mask]  # restrict to measurements that actually exist
        self._inc = [np.where(incm[b])[0] for b in range(N)]

    def _score(self, d: Dict[str, np.ndarray]) -> np.ndarray:
        # Same-package composition: the estimator's conversion/solve/residual internals are the
        # protocol being scored, so they are used directly rather than re-implemented here.
        chunk = 1000  # solve in blocks; matches SEBase.estimate's chunking
        est = self.est
        z = est._z_of(d["node_x"], d["edge_x"])
        thsl = est._truth_of(d["clean"])["thsl"]  # slack angle reference only; truth never read
        s = np.zeros((z.shape[0], est.N))
        for a in range(0, z.shape[0], chunk):
            e = slice(a, a + chunk)
            x = est._solve(z[e], thsl[e])
            rN = est._nres(x, z[e], thsl[e])
            for b, ix in enumerate(self._inc):
                if len(ix):
                    s[e, b] = rN[:, ix].max(axis=1)
        return s
