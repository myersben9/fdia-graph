"""How the within-hour terms of `LoadModel` were fitted (N31): the fluctuation's spread and autocorrelation on
the real zones of a held-out year (`fluctuation_fit`), and industrial switching on zone D's large steps
(`switching_fit`). Their results are the FLUCT_* and SWITCH_* constants of `loads.py`; run these again to refit
them on other data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genpareto

from fdia_graph.generation.nygrid.inputs import ZONES, ZoneLoad
from fdia_graph.generation.nygrid.loads import FRAME_MIN


def _kirby_hirst(x: np.ndarray) -> np.ndarray:
    """The regulation component of 5-minute loads `x [T, k]` relative to their load-following part, the
    rolling mean of L(t-3) .. L(t+2) (Kirby and Hirst, ORNL/TM-2001/13): x / mean - 1, NaN at the ends."""
    m = pd.DataFrame(x).rolling(6, min_periods=6).mean().shift(-2).to_numpy()
    return x / m - 1


def _robust_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _lag1(x: np.ndarray) -> float:
    ok = np.isfinite(x[1:]) & np.isfinite(x[:-1])
    return float(np.corrcoef(x[:-1][ok], x[1:][ok])[0, 1])


def _filtered_ar1(phi: float) -> tuple[float, float]:
    """Variance and lag-1 autocorrelation of a unit AR(1) of coefficient `phi` after `_kirby_hirst`'s filter."""
    h = np.array([0, 0, 0, 1, 0, 0]) - 1 / 6  # weights on x(t-3) .. x(t+2)
    i = np.arange(6)
    var = h @ phi ** np.abs(i[:, None] - i[None, :]) @ h
    return float(var), float(h @ phi ** np.abs(i[:, None] - i[None, :] - 1) @ h / var)


def fluctuation_fit(zone: ZoneLoad, exclude: tuple[str, ...] = ("H",)) -> dict:
    """FLUCT_C, FLUCT_BETA and FLUCT_PHI from the real zones of `zone` (module docstring), with the per-zone
    numbers behind them."""
    r = _kirby_hirst(zone.load)
    mean = np.nanmean(zone.load, axis=0)
    spread = np.array([_robust_std(r[:, k]) for k in range(len(ZONES))])
    lag1 = np.array([_lag1(np.clip(r[:, k], -5 * s, 5 * s)) for k, s in enumerate(spread)])
    keep = ~np.isin(np.array(ZONES), exclude)
    slope, icpt = np.polyfit(np.log(mean[keep]), np.log(spread[keep]), 1)
    fit = icpt + slope * np.log(mean[keep])
    r2 = 1 - np.var(np.log(spread[keep]) - fit) / np.var(np.log(spread[keep]))
    phis = np.linspace(-0.9, 0.99, 3800)
    phi = float(np.interp(np.median(lag1[keep]), [_filtered_ar1(p)[1] for p in phis], phis))
    var, _ = _filtered_ar1(phi)
    return {
        "c": float(np.exp(icpt) / np.sqrt(var)),
        "beta": float(-slope),
        "phi": phi,
        "c_residual": float(np.exp(icpt)),
        "r2": float(r2),
        "zones": list(ZONES),
        "mean_mw": mean.tolist(),
        "spread": spread.tolist(),
        "lag1": lag1.tolist(),
        "exclude": list(exclude),
    }


def switching_fit(zone: ZoneLoad, industrial_mw: float, name: str = "D", threshold: float = 0.05) -> dict:
    """SWITCH_* from one zone's real five-minute steps above `threshold` of its mean load (module
    docstring); `industrial_mw` is the model's mean industrial load in that zone."""
    x = zone.load[:, ZONES.index(name)]
    mean = float(np.nanmean(x))
    d = np.diff(x)
    big = np.flatnonzero(np.abs(np.nan_to_num(d)) > threshold * mean)
    # A switch spread over consecutive frames is one event.
    cut = np.flatnonzero((np.diff(big) != 1) | (np.sign(d[big][1:]) != np.sign(d[big][:-1]))) + 1
    runs = np.split(big, cut)
    when = np.array([r[0] for r in runs])
    step = np.array([d[r].sum() for r in runs])
    duration, first_down = [], []
    for i in range(len(runs)):
        j = np.flatnonzero(np.sign(step[i + 1 :]) != np.sign(step[i]))
        if len(j):
            duration.append(FRAME_MIN * (when[i + 1 + j[0]] - when[i]))
            first_down.append(step[i] < 0)
    size = np.abs(step) / industrial_mw
    u = threshold * mean / industrial_mw
    xi, _, scale = genpareto.fit(size - u, floc=0)
    logd = np.log(duration)
    hours = np.isfinite(d).sum() * FRAME_MIN / 60
    return {
        "rate": len(runs) / 2 / industrial_mw / hours,
        "u": u,
        "xi": float(xi),
        "scale": float(scale),
        "max": float(size.max()),
        "dur_mu": float(logd.mean()),
        "dur_sigma": float(logd.std()),
        "off": float(np.mean(first_down)),
        "steps": int(len(big)),
        "events": len(runs),
        "max_mw": float(np.abs(step).max()),
        "zone_mean_mw": mean,
        "industrial_mw": industrial_mw,
        "size_mw_q": np.percentile(np.abs(step), [10, 50, 90, 99]).tolist(),
        "duration_min_q": np.percentile(duration, [10, 25, 50, 75, 90]).tolist(),
    }
