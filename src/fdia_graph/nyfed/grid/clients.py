"""Federated clients: each NY bus belongs to the electric utility whose service territory holds it."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fdia_graph.nyfed.grid.case import Case


@dataclass(frozen=True)
class Ownership:
    """`names [K]` utility names, `bus_client [N]` index into `names` (-1 outside New York),
    `tie_lines [L]` True where a branch joins two different owners."""

    names: np.ndarray
    bus_client: np.ndarray
    tie_lines: np.ndarray


def _rings(geometry: dict) -> list[np.ndarray]:
    """Outer and hole rings of a GeoJSON Polygon or MultiPolygon as [k, 2] lon/lat arrays."""
    polys = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    return [np.asarray(ring, dtype=float) for poly in polys for ring in poly]


def _inside(lon: float, lat: float, rings: list[np.ndarray]) -> bool:
    """Even-odd ray casting over all rings, so holes count as outside."""
    hits = 0
    for r in rings:
        x0, y0, x1, y1 = r[:-1, 0], r[:-1, 1], r[1:, 0], r[1:, 1]
        cross = (y0 > lat) != (y1 > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = x0 + (lat - y0) * (x1 - x0) / (y1 - y0)
        hits += int(np.count_nonzero(cross & (lon < xs)))
    return hits % 2 == 1


def _distance(lon: float, lat: float, rings: list[np.ndarray]) -> float:
    """Distance in degrees (longitude scaled by cos latitude) from a point to the nearest ring vertex."""
    k = np.cos(np.deg2rad(lat))
    return min(float(np.min(np.hypot((r[:, 0] - lon) * k, r[:, 1] - lat))) for r in rings)


def assign_clients(case: Case, territories: dict, locations: str | Path) -> Ownership:
    """Own each NY bus by the utility territory (NYS DPS, data.ny.gov awza-4vgu) that contains its
    substation. NPCC bus coordinates are jittered, so the substation location comes from `locations`
    (`data/territories/ny_bus_locations.csv`): a row with `lat`, `lon` is placed in the territories,
    a row without them names its `utility` directly (chosen by NYISO zone, see its `note`).
    Municipal systems are not clients, so a bus inside one goes to the utility territory that also
    covers the point, or the nearest one. Every utility is its own client, however few buses it
    has. External buses stay -1."""
    utils = [f for f in territories["features"] if f["properties"]["comp_short"] != "Municipal"]
    names = sorted({f["properties"]["comp_short"] for f in utils})
    rings = {
        n: [r for f in utils if f["properties"]["comp_short"] == n for r in _rings(f["geometry"])]
        for n in names
    }
    rows = {int(r["bus_id"]): r for r in csv.DictReader(open(locations, encoding="utf-8"))}

    bus_client = np.full(len(case.bus_id), -1)
    for i in np.flatnonzero(case.is_ny):
        row = rows.get(int(case.bus_id[i]))
        if row is None:
            raise ValueError(f"bus {case.bus_id[i]} ({case.name[i]}) has no row in {locations}")
        if not row["lat"]:
            if row["utility"] not in names:
                raise ValueError(f"bus {case.bus_id[i]}: utility {row['utility']!r} is not a territory")
            bus_client[i] = names.index(row["utility"])
            continue
        lon, lat = float(row["lon"]), float(row["lat"])
        holding = [k for k, n in enumerate(names) if _inside(lon, lat, rings[n])]
        bus_client[i] = (
            holding[0] if holding else int(np.argmin([_distance(lon, lat, rings[n]) for n in names]))
        )
    return Ownership(
        names=np.array(names), bus_client=bus_client, tie_lines=bus_client[case.f] != bus_client[case.t]
    )
