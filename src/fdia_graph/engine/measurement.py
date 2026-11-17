"""Measurement emission: the generator's meter readings of a stored state, as a file's layers.

The readings are `Measurement.h` of the state on the generator's `MeterPlan`, with `NoiseModel`'s bias
and jitter; this mixin lays them out per bus and per branch, as the attacks and the writer take them.
"""

from __future__ import annotations

import numpy as np

from ..constants.channels import NodeChannel
from ..constants.streams import JITTER_STREAM
from ..formulas.network import branch_flows, complex_voltages
from ..models.frames import Frame, MeterLayers, Scan
from .base import GridBase


class MeasurementMixin(GridBase):
    """Emit measurement graphs from a stored state. Mixed into FdiaGenerator."""

    def _jitter_stream(self) -> np.random.Generator:
        """The stream an emission draws its jitter from. With a `scan_key` t it is seeded by (seed, t), so
        frame t's noise does not depend on earlier draws: an attacked frame's true scan matches its benign
        twin, whatever attacks the timeline places. Without a key, the generator's own stream."""
        if self.scan_key is None:
            return self.rng
        return np.random.default_rng([self.seed, JITTER_STREAM, self.scan_key])

    def exact(self, X: np.ndarray) -> np.ndarray:
        """The exact readings z [..., M] of stored states X [..., N, 4] = |V|, Pinj, Qinj, angle, with no
        re-solve: the injections are the state's own, the flows and currents follow from its voltages."""
        X, N = np.asarray(X, float), NodeChannel
        return self.meas.h(X[..., N.V], X[..., N.THETA], (X[..., N.P_INJ], X[..., N.Q_INJ]))

    def layers(self, z: np.ndarray) -> MeterLayers:
        """One value per meter z [..., M] as a file's layers: node [..., N, 4], flow [..., E, 2] (the
        from end, the only one an IEEE case meters) or [..., E, 4] (both ends, `spec.both_ends`) and
        PMU current [..., E, 4]; zero where no meter sits."""
        full = self.plan.scatter(z, self.C, self.E)
        return full._replace(edge=full.edge[..., : self.spec.edge_width])

    def emit_from_state(self, X: np.ndarray) -> Scan:
        """The scan of one stored state X [N, 4]: each reading is the exact one with its meter's constant
        bias and this scan's jitter, drawn bus by bus (|V|, angle, P, Q), then the flows, then the currents."""
        noisy = self.noise.sample(self.exact(X), self._jitter_stream())
        x, m = self.layers(noisy.astype(np.float32)), self.masks
        return Scan(x.node, m.node, x.edge, m.edge, x.current, m.current)

    def benign_frame(self, X: np.ndarray) -> Frame:
        """The benign frame of one stored state X [N, 4]: its scan, with no label and no attack. One
        emission per scan, so a file is reproduced bit for bit from its seed."""
        scan = self.emit_from_state(X)
        seen = MeterLayers(scan.node_x, scan.edge_x, scan.i_x)
        return Frame.from_scan(scan, seen, np.zeros(self.C, np.uint8))

    def clean_flows_from_states(self, X: np.ndarray) -> np.ndarray:
        """The exact from-end flows [T, E, 2] (P MW, Q MVAr) of a stack of states [T, N, 4], with no noise:
        the clean edge layer an estimator is scored against. Unmetered branches are zero."""
        return self.layers(self.exact(X)).edge.astype(np.float32)

    def all_flows_from_states(self, X: np.ndarray) -> np.ndarray:
        """The exact from-end complex flow [T, E] (MW + j MVAr) of every branch, metered or not, for a
        stack of states [T, N, 4] (`clean_flows_from_states` has the metered branches only)."""
        X = np.asarray(X, float)
        V = complex_voltages(X[:, :, NodeChannel.V], X[:, :, NodeChannel.THETA])
        return branch_flows(V, self.net.yf, self.case.f, self.case.base_mva)

    def currents_from_states(self, X: np.ndarray) -> np.ndarray:
        """The exact PMU branch-current channels [T, E, 4] (`CurrentChannel` columns, per unit) of a stack of
        states [T, N, 4], zero where no PMU reads that end."""
        return self.layers(self.exact(X)).current
