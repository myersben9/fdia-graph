"""Per-bus load profiles by the bottom-up method of [LI21], scaled to the real zone load.

Each NY load bus aggregates prototype buildings and plants until each of its residential, commercial and
industrial components reaches its peak (Sec. II-D). The bus mix comes from its utility's sector sales
(II-B). The prototypes come from NREL ResStock and ComStock county profiles (II-C1), kept at their native
15 minutes, and from IAC plants on ORNL-style daily curves (II-C2), which are hourly rules held flat within
the hour. The zone's real 5-minute load is split over its buses in proportion to their 15-minute profiles.

Within the hour (N31) each bus then gets two terms of its own, so its five-minute steps are not its zone's:

- Fluctuation. An AR(1) factor on the bus's load, `p (1 + e)`, with `e` of standard deviation
  `FLUCT_C size^-FLUCT_BETA` (size the bus's mean load, MW) and lag-1 coefficient `FLUCT_PHI`. The constants
  were fitted on the real zones of a held-out year (2018, so 2019 is not fitted to itself) by a function
  removed from the package after release 0.21.0, where it can still be read and run (`_fluctuation_fit` in
  `fdia_graph/nyfed/scenario/loads.py`). A zone's fluctuation is its regulation component in the sense of Kirby and Hirst
  (ORNL/TM-2001/13): the 5-minute load less its 30-minute rolling mean of L(t-3) .. L(t+2), relative to
  that mean. Its spread is the robust (1.4826 MAD) standard deviation, so large industrial steps, modelled
  next, do not inflate it. The spread is fitted against zone mean load as `c size^-beta` across the zones;
  zone H (Millwood, 320 MW) is left out, its residual being four times what the other ten zones' fit
  predicts at its size (a metering artifact rather than load, as its near-white steps suggest). One `phi`
  is taken from the median lag-1 autocorrelation of the zones' residuals, read through the filter: white
  noise already gives -0.23 after the rolling mean, so the residual's autocorrelation is mapped back to the
  AR(1) coefficient that produces it, and `c` is scaled to the AR(1)'s own spread the same way. With beta
  0.5 independent bus terms add up to the zone's own fluctuation, so the rescaling below leaves each bus
  with its own spread and the zone with its real load.
- Industrial switching. Zone D's real five-minute steps above 5% of its mean load (2,799 in 2019, up to
  314 MW) are taken as large plants switching. The fit (`_switching_fit`, in release 0.21.0 beside the other)
  merged same-sign steps in consecutive frames into one event and modelled switch-off and switch-on pulses: a pulse starts at an event and ends at the
  next event of opposite sign. Pulses arrive as a Poisson process at `SWITCH_RATE` per MW of the bus's
  mean industrial load and hour (zone D's pulses, half its events, over the model's industrial load of zone
  D). A pulse's size relative to the bus's mean industrial load is the 5% threshold plus a generalized
  Pareto excess (peaks over threshold, shape `SWITCH_XI`, scale `SWITCH_SCALE`), truncated at the largest
  step seen; its duration is lognormal (`SWITCH_DUR_MU`, `SWITCH_DUR_SIGMA`, minutes) from the times
  between opposite-sign events; a share `SWITCH_OFF` of pulses start with a drop. A drop never takes more
  than the bus's industrial load at that frame.

Last, each zone's buses are rescaled to sum exactly to the zone's real 5-minute load.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import genpareto

from fdia_graph.constants import newyork
from fdia_graph.constants.time import FRAME_MINUTES, HOURS_PER_YEAR, QUARTERS_PER_YEAR
from fdia_graph.generation.nygrid.case import Case
from fdia_graph.generation.nygrid.inputs import ZoneLoad, _bus_weights
from fdia_graph.generation.nygrid.load_sources import (
    _building_stock,
    _county_centroids,
    _operating_rank,
    _plants,
    _sector_mix,
    _work_hours,
)
from fdia_graph.generation.nygrid.ownership import Ownership
from fdia_graph.generation.nygrid.tables import read_rows


@dataclass(frozen=True)
class LoadSeries:
    """Bus loads at every frame, `p [T, N]` MW and `q [T, N]` MVAr over all buses of the case."""

    time: np.ndarray
    p: np.ndarray
    q: np.ndarray


class LoadModel:
    """The bus mix and prototype pools of every NY load bus, and the aggregation of [2] over them."""

    def __init__(
        self,
        case: Case,
        own: Ownership,
        nygrid_data: str | Path,
        locations: str | Path,
        cache_dir: str | Path = "data/loads",
        year: int = 2019,
    ):
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        self.case, self.year = case, year
        self.weights = _bus_weights(Path(nygrid_data))  # [N, 11], NYgrid's share of each zone at each bus
        self.load_bus = np.flatnonzero(self.weights.sum(axis=1) > 0)
        self.zone_of = self.weights[self.load_bus].argmax(axis=1)

        mix = _sector_mix(cache)
        self.utility = own.names[own.bus_client[self.load_bus]]
        self.mix = np.stack([mix[u] for u in self.utility])  # [B, 3] residential, commercial, industrial

        stock = _building_stock(cache)
        # 2018 profiles stand in for 2019: shift one day (96 quarter hours) so weekdays line up (2018 starts
        # on a Monday, 2019 on a Tuesday) and repeat the last day.
        prof = stock["profile"].astype(float)
        self.proto = np.concatenate([prof[:, 96:], prof[:, -96:]], axis=1)
        self.proto_sector, self.proto_county = stock["sector"], stock["county"]
        com = self.proto_sector == "com"
        n_ft2 = np.array([newyork.REFERENCE_FT2[k] for k in stock["kind"][com]], dtype=float)
        self.proto[com] *= n_ft2[:, None]  # one reference building per commercial pick
        self.count = stock["size"].astype(float)  # buildings each prototype stands for in the county
        self.count[com] /= n_ft2

        # II-D pools: the five counties nearest each bus. Buses without a located substation keep the
        # NPCC coordinates.
        rows = {int(r["bus_id"]): r for r in read_rows(locations)}
        centroids = _county_centroids(cache)
        names = sorted(set(self.proto_county) & set(centroids))
        cxy = np.array([centroids[c] for c in names])
        self.pool_counties = []
        for b in self.load_bus:
            r = rows.get(int(case.bus_id[b]), {})
            lat, lon = (float(r["lat"]), float(r["lon"])) if r.get("lat") else (case.lat[b], case.lon[b])
            d = np.hypot(cxy[:, 0] - lat, (cxy[:, 1] - lon) * np.cos(np.deg2rad(lat)))
            self.pool_counties.append([names[k] for k in np.argsort(d)[: newyork.POOL_COUNTIES]])

        self.plants = _plants(cache)
        lf = self.plants["sic_lf"]
        self.plant_level = np.clip(
            (168 * lf - 35) / 133, 0.0, 1.0
        )  # flat off-hours level giving the weekly LF
        self.plant_days = np.minimum(np.round(self.plants["hours"] / 24), 365).astype(int)
        self.rank = _operating_rank(year)
        self.work = _work_hours()
        # A plant's yearly curve in p.u. sums to n days of the weekday curve plus the rest at the low level,
        # and is scaled so its integral is the plant's yearly kWh.
        pu_sum = 24 * 365 * self.plant_level + self.plant_days * (1 - self.plant_level) * self.work.sum()
        self.plant_scale = self.plants["kwh"] / pu_sum  # kW at 1 p.u., the plant's peak

    def _buildings(
        self, target_kw: float, sector: str, counties: list[str], pairs: int, rng: np.random.Generator
    ) -> np.ndarray:
        """II-D for residential or commercial load: pick prototype buildings of the bus's five counties,
        in proportion to how many each stands for, each with `pairs` random hour pairs swapped, until the
        sum peaks at `target_kw`. Returns the 15-minute sum [QUARTERS_PER_YEAR]."""
        idx = np.flatnonzero((self.proto_sector == sector) & np.isin(self.proto_county, counties))
        protos, prob = self.proto[idx], self.count[idx] / self.count[idx].sum()
        return self._aggregate(target_kw, protos, prob, pairs, rng)

    def _aggregate(self, target_kw, protos, prob, pairs, rng):
        """The loop of Fig. 5 on 15-minute prototypes. The peak is the hourly peak, as [2]'s prototypes
        are hourly, and a swap exchanges two whole hours (four rows each), so the method stays [2]'s."""
        total = np.zeros(QUARTERS_PER_YEAR)
        blocks = protos.reshape(len(protos), HOURS_PER_YEAR, 4)
        mean_peak = float(prob @ blocks.mean(axis=2).max(axis=1))
        quarter = np.arange(4)
        while _hourly_peak(total) < target_kw:
            # Many picks at once while far from the peak, one at a time near it, so the sum stops within
            # one building of the target as the one-by-one loop of Fig. 5 does.
            k = max(1, int(0.9 * (target_kw - _hourly_peak(total)) / mean_peak))
            picks = rng.choice(len(protos), size=k, p=prob)
            total += np.bincount(picks, minlength=len(protos)) @ protos
            for start in range(0, k, 25_000):
                who = np.repeat(picks[start : start + 25_000], pairs)
                a, b = rng.integers(0, HOURS_PER_YEAR, size=(2, len(who)))
                delta = (blocks[who, b] - blocks[who, a]).ravel()
                rows_a = (4 * a[:, None] + quarter).ravel()
                rows_b = (4 * b[:, None] + quarter).ravel()
                total += np.bincount(rows_a, delta, QUARTERS_PER_YEAR) - np.bincount(
                    rows_b, delta, QUARTERS_PER_YEAR
                )
        return total

    def _industry(self, target_kw: float, rng: np.random.Generator) -> np.ndarray:
        """II-C2 and II-D for industrial load: any IAC plant smaller than the target, each starting its
        operating days on a random day and with 50 hour pairs swapped, until the sum peaks at the target."""
        pool = np.flatnonzero(self.plant_scale < target_kw)
        total = np.zeros(HOURS_PER_YEAR)
        if target_kw <= 0 or len(pool) == 0:
            return total
        mean_peak = float(self.plant_scale[pool].mean())
        hour = np.tile(np.arange(24), 365)
        work = np.tile(self.work, 365) > 0
        while total.max() < target_kw:
            k = 2 * int((target_kw - total.max()) / mean_peak) + 1
            f = rng.choice(pool, size=k)
            s = rng.integers(0, 365, size=k)
            on = self.rank[s] < self.plant_days[f, None]  # [k, 365]
            lo, scale = self.plant_level[f], self.plant_scale[f]
            # Keep the picks up to the first one whose addition reaches the target, as the one-by-one loop
            # of Fig. 5 would. A plant adds a constant plus a per-day amount over the working hours, so the
            # peak after each pick follows from running sums over plants and days.
            const = np.cumsum(scale * lo)
            daily = np.cumsum((scale * (1 - lo))[:, None] * on, axis=0)  # [k, 365]
            day_work_max = np.where(work, total, -np.inf).reshape(365, 24).max(axis=1)
            peak = const + np.maximum(total[~work].max(), (daily + day_work_max).max(axis=1))
            k = int(np.argmax(peak >= target_kw)) + 1 if (peak >= target_kw).any() else k
            f, on, lo, scale = f[:k], on[:k], lo[:k], scale[:k]
            total += const[k - 1] + np.repeat(daily[k - 1], 24) * np.tile(self.work, 365)
            a, b = rng.integers(0, HOURS_PER_YEAR, size=(2, k, newyork.SWAP_PAIRS[2]))
            rows = np.arange(k)[:, None]

            def value(h):
                return scale[:, None] * (
                    lo[:, None] + (1 - lo[:, None]) * on[rows, h // 24] * self.work[hour[h]]
                )

            delta = (value(b) - value(a)).ravel()
            total += np.bincount(a.ravel(), delta, HOURS_PER_YEAR) - np.bincount(
                b.ravel(), delta, HOURS_PER_YEAR
            )
        return total

    def _quarterly(self, peak_mw: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """[QUARTERS_PER_YEAR, B] aggregated kW of every load bus, and of its industrial part alone, 15-minute rows of
        `year` in EST, for zone peaks `peak_mw [11]`. II-A: a bus's size is its NYgrid share of its zone's
        peak, split by its utility's mix. The hourly plant curves are held flat within each hour."""
        out = np.zeros((QUARTERS_PER_YEAR, len(self.load_bus)))
        ind = np.zeros((QUARTERS_PER_YEAR, len(self.load_bus)))
        for j, b in enumerate(self.load_bus):
            size_kw = 1000 * self.weights[b, self.zone_of[j]] * peak_mw[self.zone_of[j]]
            res, com, ind_kw = size_kw * self.mix[j]
            ind[:, j] = np.repeat(self._industry(ind_kw, rng), 4)
            out[:, j] = (
                self._buildings(res, "res", self.pool_counties[j], newyork.SWAP_PAIRS[0], rng)
                + self._buildings(com, "com", self.pool_counties[j], newyork.SWAP_PAIRS[1], rng)
                + ind[:, j]
            )
        return out, ind

    def profiles(self, zone_load: ZoneLoad, rng: np.random.Generator) -> LoadSeries:
        """Bus P and Q at every frame of `zone_load`. Each zone's real load is split over its load buses in
        proportion to their 15-minute aggregated profiles, interpolated to the frame time; each bus then
        gets its AR(1) fluctuation and industrial switching pulses (module docstring), and the zone's buses
        are rescaled so they sum to the zone exactly. Q follows at POWER_FACTOR. External buses keep their
        base load."""
        quarterly, ind = self._quarterly(np.nanmax(zone_load.load, axis=0), rng)
        zone_sum = np.zeros((QUARTERS_PER_YEAR, len(newyork.ZONES)))
        np.add.at(zone_sum.T, self.zone_of, quarterly.T)
        share = quarterly / zone_sum[:, self.zone_of]
        ind_share = ind / zone_sum[:, self.zone_of]

        # Frames are on NYISO's local clock; the prototypes on EST, each 15-minute mean placed at its middle.
        t = pd.DatetimeIndex(zone_load.time).tz_localize(
            "America/New_York", ambiguous=np.ones(len(zone_load.time), bool), nonexistent="shift_forward"
        )
        est = t.tz_convert("Etc/GMT+5").tz_localize(None)
        x = (est - pd.Timestamp(f"{self.year}-01-01")).total_seconds().to_numpy() / 900
        mid = np.arange(QUARTERS_PER_YEAR) + 0.5
        zone = zone_load.load[:, self.zone_of]
        base = np.stack([np.interp(x, mid, share[:, j]) for j in range(share.shape[1])], 1) * zone
        ind_f = np.stack([np.interp(x, mid, ind_share[:, j]) for j in range(share.shape[1])], 1) * zone

        size = np.nanmean(base, axis=0)  # each bus's mean load, MW
        bus = base * (1 + _ar1(len(x), newyork.FLUCT_C * size**-newyork.FLUCT_BETA, newyork.FLUCT_PHI, rng))
        bus += _switching(ind_f, rng)
        total = np.zeros((len(x), len(newyork.ZONES)))
        np.add.at(total.T, self.zone_of, bus.T)
        bus *= zone / total[:, self.zone_of]

        p = np.tile(np.where(self.case.is_ny, 0.0, self.case.pd), (len(x), 1))
        q = np.tile(np.where(self.case.is_ny, 0.0, self.case.qd), (len(x), 1))
        p[:, self.load_bus] = bus
        q[:, self.load_bus] = p[:, self.load_bus] * np.tan(np.arccos(newyork.POWER_FACTOR))
        return LoadSeries(time=zone_load.time, p=p, q=q)


def _hourly_peak(x: np.ndarray) -> float:
    """The peak of the hourly means of a 15-minute series."""
    return float(x.reshape(HOURS_PER_YEAR, 4).mean(axis=1).max())


def _ar1(n: int, sigma: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """[n, B] stationary AR(1) series, one per column, of standard deviation `sigma [B]` and lag-1
    coefficient `phi`, started from its stationary distribution."""
    w = rng.standard_normal((n, len(sigma)))
    # e(t) = phi e(t-1) + sqrt(1 - phi^2) w(t), with e(0) = w(0); lfilter's state for the first step is phi e(0).
    e, _ = lfilter([np.sqrt(1 - phi**2)], [1.0, -phi], w[1:], axis=0, zi=phi * w[:1])
    return np.vstack([w[:1], e]) * sigma


def _switching(ind: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """[T, B] industrial switching pulses on the industrial load `ind [T, B]` (MW) of every bus (module
    docstring): Poisson onsets at SWITCH_RATE per MW of mean industrial load and hour, sizes relative to that
    mean from the threshold plus a generalized Pareto excess truncated at SWITCH_MAX, lognormal durations, a
    share SWITCH_OFF starting with a drop. Drops are clipped so the industrial load never goes below zero."""
    T, B = ind.shape
    out = np.zeros((T, B))
    hours = T * FRAME_MINUTES / 60
    for j in range(B):
        mean = float(np.nanmean(ind[:, j]))
        if not mean > 0:
            continue
        n = rng.poisson(newyork.SWITCH_RATE * mean * hours)
        onset = rng.integers(0, T, size=n)
        xi, scale = newyork.SWITCH_XI, newyork.SWITCH_SCALE
        size = newyork.SWITCH_U + genpareto.rvs(xi, scale=scale, size=n, random_state=rng)
        # truncate by redrawing, so no mass piles at the maximum
        while (big := size > newyork.SWITCH_MAX).any():
            size[big] = newyork.SWITCH_U + genpareto.rvs(xi, scale=scale, size=big.sum(), random_state=rng)
        minutes = rng.lognormal(newyork.SWITCH_DUR_MU, newyork.SWITCH_DUR_SIGMA, size=n)
        length = np.maximum(1, np.round(minutes / FRAME_MINUTES)).astype(int)
        sign = np.where(rng.random(n) < newyork.SWITCH_OFF, -1.0, 1.0)
        delta = np.zeros(T + 1)
        np.add.at(delta, onset, sign * size * mean)
        np.add.at(delta, np.minimum(onset + length, T), -sign * size * mean)
        out[:, j] = np.cumsum(delta[:T])
    return np.maximum(out, -np.nan_to_num(ind))
