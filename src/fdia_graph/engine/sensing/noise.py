"""Meter noise: the accuracy class of each meter split into a constant bias and a per-scan jitter."""

from __future__ import annotations

import numpy as np

from fdia_graph.engine.sensing.meters import JITTER_FRAC, MeterPlan
from fdia_graph.formulas.estimation import accuracy_class_sigma


class NoiseModel:
    """Holds each meter's accuracy class (`sigma_rel`, `sigma_abs`) and constant bias from a MeterPlan.
    A reading is the exact value with its bias (a gain on relative meters, an offset on absolute ones)
    plus a fresh Gaussian jitter, drawn once per scan, of `JITTER_FRAC` times the class; the floor of a
    relative meter is not scaled, so a near-zero reading keeps its noise. Readings are heteroscedastic:
    a relative meter's noise grows with its reading."""

    def __init__(self, plan: MeterPlan):
        self.sigma_rel, self.sigma_abs, self.bias = plan.sigma_rel, plan.sigma_abs, plan.bias
        self.relative = plan.sigma_rel > 0
        self._class = np.where(self.relative, plan.sigma_rel, plan.sigma_abs)  # the class of [ASP14]

    def sigmas(self, z: np.ndarray) -> np.ndarray:
        """The accuracy-class standard deviation of each meter at exact readings z [..., M], the total of
        bias and jitter (what an estimator weights by and what a change must exceed to count)."""
        return accuracy_class_sigma(np.abs(z), self._class, self.relative, self.sigma_abs)

    def sample(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One noisy scan of exact readings z [..., M] (a scan per row on a stack of frames)."""
        jitter = JITTER_FRAC * self.sigma_rel * np.abs(z) + np.where(
            self.relative, self.sigma_abs, JITTER_FRAC * self.sigma_abs
        )
        biased = np.where(self.relative, z * (1.0 + self.bias), z + self.bias)
        return biased + rng.normal(0.0, 1.0, np.shape(z)) * jitter
