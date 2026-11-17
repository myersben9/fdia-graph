"""Meter noise: the accuracy class of each meter split into a constant bias and a per-scan jitter."""

from __future__ import annotations

import numpy as np

from fdia_graph.constants.meters import JITTER_FRAC, MeterKind
from fdia_graph.formulas.noise import accuracy_class_sigma
from fdia_graph.sensing.meters import MeterPlan


class NoiseModel:
    """Each meter's accuracy class (`sigma_rel`, `sigma_abs`) and constant bias, from a MeterPlan.
    A reading is the exact value with its bias (a gain on relative meters, an offset on absolute ones)
    plus a Gaussian jitter, drawn once per scan, of `JITTER_FRAC` times the class. The floor of a
    relative meter is not scaled, so a near-zero reading keeps its noise. The real or imaginary part
    of a current phasor errs by its class times the phasor's magnitude |I|, added to it [C37118]."""

    def __init__(self, plan: MeterPlan):
        self.sigma_rel, self.sigma_abs, self.bias = plan.sigma_rel, plan.sigma_abs, plan.bias
        self.relative = plan.sigma_rel > 0
        self._class = np.where(self.relative, plan.sigma_rel, plan.sigma_abs)  # the class of [ASP14]
        self._other = plan.phasor_partner()  # the other part of a phasor's row, -1 on any other meter
        self._is_real = plan.kind == MeterKind.IR

    def sigmas(self, z: np.ndarray) -> np.ndarray:
        """The accuracy-class standard deviation of each meter at exact readings z [..., M], the total of
        bias and jitter (what an estimator weights by and what a change must exceed to count)."""
        return accuracy_class_sigma(self._size(z), self._class, self.relative, self.sigma_abs)

    def _size(self, z: np.ndarray) -> np.ndarray:
        """What a relative class is taken of: |z|, or the phasor's magnitude |I| on a part of one."""
        size = np.abs(z)
        part = self._other >= 0
        if not part.any():
            return size
        other = z[..., np.maximum(self._other, 0)]
        real, imag = np.where(self._is_real, z, other), np.where(self._is_real, other, z)
        return np.where(part, np.hypot(real, imag), size)

    def jitter(self, z: np.ndarray) -> np.ndarray:
        """The standard deviation of each meter's per-scan jitter at exact readings z [..., M], in the
        reading's units: `JITTER_FRAC` of the class, and the unscaled floor of a relative meter."""
        floor = np.where(self.relative, self.sigma_abs, JITTER_FRAC * self.sigma_abs)
        size = self._size(z)
        return JITTER_FRAC * self.sigma_rel * size + floor

    def sample(self, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One noisy scan of exact readings z [..., M] (a scan per row on a stack of frames)."""
        size = self._size(z)
        jitter = self.jitter(z)
        biased = np.where(self.relative, z * (1.0 + self.bias), z + self.bias)
        biased = np.where(self._other >= 0, z + self.bias * size, biased)
        return biased + rng.normal(0.0, 1.0, np.shape(z)) * jitter
