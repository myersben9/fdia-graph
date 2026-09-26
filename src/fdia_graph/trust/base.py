"""The shared machinery of the trusted-meter selectors: the measurement Jacobian at the benign
mean (from a WLS estimator of `fdia_graph.se`), the attack-cost kernel, and the scoring that
pins the secured meters and asks the residual test whether the attack shows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..errors import NoBenignRecords
from ..formulas.trust import attack_cost, greedy_trusted_meters
from ..models.config import TrustConfig
from ..models.scores import TrustScores  # noqa: F401  re-exported: defined here before the models package

if TYPE_CHECKING:
    from ..dataset import FdiaGraph


class TrustSelector:
    """Base of the selectors: fit builds the estimator and the Jacobian and calls `_select`, which
    fills `order` (the meters to secure, masked measurement indices in the estimator's layout
    [V, P, Q, theta over the buses, then Pf, Qf over the branches, metered channels only]) and
    `cost` (the attack cost after each)."""

    def __init__(self, k: int, fa_target: float = 0.01) -> None:
        cfg = TrustConfig(k, fa_target)
        self.k, self.fa_target = cfg.k, cfg.fa_target
        self.order: list[int] = []
        self.cost: list[float] = []

    def fit(self, ds: FdiaGraph) -> TrustSelector:
        from ..se import WLS

        est = WLS()
        est.fit(ds, calibrate="measured")  # the residual test is a detector: measurements only
        self.est = est
        self.H = np.asarray(self.est.H, np.float64)  # [m, 2N-1] at the benign mean state
        self.m = self.H.shape[0]
        self.level = self._alarm_level(ds)
        self._select()
        return self

    def _alarm_level(self, ds: FdiaGraph, n_calib: int = 2000) -> float:
        """The residual alarm level: the (1 - fa_target) quantile of the largest normalized
        residual over benign training records, set before any attack is scored (the protocol of
        `fdia_graph.localization`)."""
        d = ds.export(["node_x", "edge_x", "family"])
        ben = np.flatnonzero(d["family"] == 0)
        if not len(ben):
            raise NoBenignRecords("fit needs benign records; pass the train split unfiltered")
        # spread over the whole benign set, not the first records (one early load regime on a timeline)
        ben = ben[np.linspace(0, len(ben) - 1, min(n_calib, len(ben))).round().astype(int)]
        est = self.est
        z = est._z_of(d["node_x"][ben], d["edge_x"][ben])
        return float(np.quantile(self._max_residual(z), 1.0 - self.fa_target))

    def _select(self) -> None:
        raise NotImplementedError

    def select(self, k: Optional[int] = None) -> np.ndarray:
        """The first k secured meters (default: all that were selected)."""
        return np.array(self.order[: self.k if k is None else k], int)

    # ---- scoring ------------------------------------------------------------------------------
    def _max_residual(self, z: np.ndarray) -> np.ndarray:
        """The largest normalized residual of every record after a WLS solve [HAN75]."""
        est, out = self.est, []
        x = est._estimate_arrays(z, None)
        for a in range(0, len(z), 1000):
            e = slice(a, a + 1000)
            out.append(est._nres(x[e], z[e]).max(axis=1))  # already non-negative
        return np.concatenate(out)

    def score(self, ds: FdiaGraph) -> TrustScores:
        """Residual detection per family with and without the secured meters, on a timeline view
        (the benign layer is what a secured meter reads), at the alarm level `fit` calibrated on
        benign training records; `false_alarm` is the benign rate of this view at that level."""
        from ..dataset import FAMILIES

        ds.require("physical_units", "benign_layer", by="the secured-meter score")
        d = ds.export(["node_x", "edge_x", "benign", "edge_benign", "family"])
        est = self.est
        z = est._z_of(d["node_x"], d["edge_x"])
        zb = est._z_of(d["benign"], d["edge_benign"])
        secured = self.select()
        zs = z.copy()
        zs[:, secured] = zb[:, secured]  # the attacker cannot write a secured meter
        before, after = self._max_residual(z), self._max_residual(zs)
        fam = d["family"]
        ben = fam == 0
        level = self.level
        det_b: dict[str, float] = {}
        det_a: dict[str, float] = {}
        for fid, name in FAMILIES.items():
            rows = fam == fid
            if fid and rows.any():
                det_b[name] = float((before[rows] > level).mean())
                det_a[name] = float((after[rows] > level).mean())
        return TrustScores(
            order=[int(i) for i in secured],
            cost=[float(c) for c in self.cost[: len(secured)]],
            detected_before=det_b,
            detected_after=det_a,
            false_alarm=float((before[ben] > level).mean()),
        )

    def attack_cost(self, secured: Optional[Any] = None) -> float:
        """The attack cost with these meters secured (default: the selection)."""
        s = self.select() if secured is None else np.asarray(secured, int)
        return attack_cost(self.H, s)[0]

    def secured_copy(
        self, ds: FdiaGraph, out: str, name: Optional[str] = None, k: Optional[int] = None
    ) -> str:
        """A copy of the timeline behind `ds` with the first `k` selected meters (default: all)
        reading their benign value on every frame, the attacker locked out of them, written to
        `out` and registered under `name` when given; the estimators and localizers read it like
        any timeline (`trust.secured_copy`)."""
        from .secured import secured_copy

        return secured_copy(self, ds, out, name, k)


class TrustedMeters(TrustSelector):
    """The greedy row-reduction selection [WU26]: secure, one at a time, the meter on the cheapest
    open attack whose protection leaves the attacker the costliest cheapest attack."""

    def _select(self) -> None:
        self.order, self.cost = greedy_trusted_meters(self.H, self.k)
