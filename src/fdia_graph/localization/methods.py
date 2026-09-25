"""The localizer method classes. Each changes exactly one thing about LocalizerBase — the per-bus
score — so a difference between two arms is a difference between detection signals, not between
calibration or metrics code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..formulas.projection import bus_incidence, meters_to_buses
from .base import LocalizerBase

if TYPE_CHECKING:
    from ..dataset import FdiaGraph
    from ..se.base import SEBase


class SwingThreshold(LocalizerBase):
    """The temporal-spike detector: score = the bus's windowed relative-swing magnitude.

    The dataset's swing feature is each scan's injection change as a z-score of the bus's typical
    recent change, so any attack edit that exceeds the noise floor appears as a spike the moment it
    starts — including the BDD-stealthy re-solve families. The one family built to defeat it is the
    slow ramp At, which stays inside typical per-scan change by construction.
    """

    def _fields(self) -> list[str]:
        return ["swing"]

    def _score(self, d: dict[str, np.ndarray], ds: FdiaGraph) -> np.ndarray:
        return np.abs(d["swing"]).max(axis=2)  # worst channel (dP or dQ) per bus


class DeltaThreshold(LocalizerBase):
    """Ablation arm: the raw one-scan injection change, scaled by the bus's benign RMS change.

    The same signal as SwingThreshold without the windowed typical-change normalization — one
    global scale per bus and channel instead of a running local one — so the gap between the two
    is exactly what the windowing buys.
    """

    def _fields(self) -> list[str]:
        return ["temporal_delta"]

    def _fit_stats(self, d: dict[str, np.ndarray], ben: np.ndarray, ds: FdiaGraph) -> None:
        td = d["temporal_delta"][ben]
        self.sd = np.maximum(np.sqrt((td**2).mean(axis=0)), 1e-9)  # [N, 2] benign RMS per channel

    def _score(self, d: dict[str, np.ndarray], ds: FdiaGraph) -> np.ndarray:
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

    def __init__(self, estimator: Optional[SEBase] = None, fa_target: float = 0.01) -> None:
        super().__init__(fa_target=fa_target)
        self.estimator = estimator  # None -> a fresh WLS; an unfitted one is fitted on the train split

    def _fields(self) -> list[str]:
        return ["node_x", "edge_x"]

    def _fit_stats(self, d: dict[str, np.ndarray], ben: np.ndarray, ds: FdiaGraph) -> None:
        from ..se import WLS

        self.est = self.estimator if self.estimator is not None else WLS()
        if not self.est.is_fitted:  # an estimator fitted elsewhere (another split, a gate) is kept as is
            self.est.fit(ds, calibrate="measured")  # a detector: measurements only
        # bus <- measurement incidence in the estimator's masked layout: a node channel touches its
        # own bus, a flow meter BOTH endpoints of its line (an injection edit perturbs every incident flow)
        self._inc = bus_incidence(self.est.N, self.est.E, ds.edge_index_np, self.est.mask)

    def _score(self, d: dict[str, np.ndarray], ds: FdiaGraph) -> np.ndarray:
        # Same-package composition: the estimator's conversion, solve and residual internals are the
        # protocol being scored, so they are used directly rather than re-implemented here. The
        # per-record weights come from the estimator's own hook, so a gated or Jacobian-weighted
        # estimator is scored as the estimator it is, not as its ungated parent.
        from ..se.base import require_physical

        require_physical(ds)
        est, chunk = self.est, 1000
        z = est._z_of(d["node_x"], d["edge_x"])
        thsl = est.ref_angles(len(z))
        x = est._estimate_arrays(z, thsl, est._record_weights(ds), chunk)
        s = np.empty((z.shape[0], est.N))
        for a in range(0, z.shape[0], chunk):
            e = slice(a, a + chunk)
            s[e] = meters_to_buses(est._nres(x[e], z[e], thsl[e]), self._inc, "max")
        return s
