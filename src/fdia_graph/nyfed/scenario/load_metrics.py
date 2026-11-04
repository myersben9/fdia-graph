"""How a load series is compared with the real one (`LoadModel.validate`): the load factor, the distribution of
the load and of its five-minute steps, the hourly autocorrelation and the spectrum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fdia_graph.nyfed.scenario.load_sources import HOURS


def _step_distribution(x: np.ndarray) -> np.ndarray:
    """Histogram of five-minute steps relative to the mean load, |dx| / mean, on log bins from 1e-4 to 1."""
    d = np.abs(np.diff(x)) / np.nanmean(x)
    d = d[np.isfinite(d)]
    hist, _ = np.histogram(np.clip(d, 1e-4, 0.999), bins=np.logspace(-4, 0, 25))
    return hist / len(d)


def _hourly_mean(time: np.ndarray, x: np.ndarray) -> np.ndarray:
    """[8760, k] hourly means of 5-minute columns."""
    return (
        pd.DataFrame(x, index=pd.DatetimeIndex(time)).resample("1h").mean().interpolate().to_numpy()[:HOURS]
    )


_MONTH = pd.date_range("2019-01-01", periods=HOURS, freq="1h").month.to_numpy() - 1


def _load_factor(x):
    return np.array([x[_MONTH == m].mean() / x[_MONTH == m].max() for m in range(12)])


def _distribution(x):
    hist, _ = np.histogram(x / x.mean(), bins=np.linspace(0.4, 1.8, 29))
    return hist / len(x)


def _dlog(x):
    d = np.diff(np.log(x))
    return d - d.mean()


def _autocorrelation(x):
    d = _dlog(x)
    return np.array([np.corrcoef(d[:-k], d[k:])[0, 1] for k in range(1, 49)])


def _spectrum(x):
    d = _dlog(x)
    f = np.fft.rfftfreq(len(d))
    psd = np.abs(np.fft.rfft(d)) ** 2 / len(d)
    return psd[(f >= 1 / HOURS) & (f <= 1 / 10)]


_METRICS = {
    "load_factor": _load_factor,
    "distribution": _distribution,
    "autocorrelation": _autocorrelation,
    "spectrum": _spectrum,
}
