"""The production build: the inputs fetched at their pinned versions, and the `TimelineBuilder` with every
decided bound (N8, N27 to N49) in one place."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from .attacks.attacker import Attacker, generator_band, generator_q_capability, generator_ramps, natural_steps
from .attacks.interface_attack import InterfaceFdia
from .attacks.msfdia import NEAR_LIMIT
from .attacks.opf_attack import OpfFdia
from .attacks.ramp import Ramp
from .config import Settings, rng
from .grid.clients import assign_clients
from .grid.fixes import restore_765kv, split_zone_j
from .grid.interfaces import hourly_interfaces
from .grid.network import Network
from .grid.nygrid import modify_mpc, nygrid_units, read_npcc
from .io import read_pool
from .scenario.inputs import fetch_nyiso
from .scenario.pool import SOLVED, peak_ratings
from .sensing.measure import Measurement
from .sensing.meters import plan_meters
from .sensing.noise import NoiseModel
from .timeline.builder import TimelineBuilder

NYGRID_COMMIT = "47698b6c7823ae7b1bc6935e5601d5b64b8f918e"
NYGRID_ZIP = f"https://codeload.github.com/AndersonEnergyLab-Cornell/NYgrid/zip/{NYGRID_COMMIT}"
TERRITORIES = "https://data.ny.gov/api/geospatial/awza-4vgu?method=export&format=GeoJSON"  # NYS DPS
GEOJSON = "nys_electric_territories.geojson"
LOCATIONS = "ny_bus_locations.csv"  # each NY bus located on the territories (N12), shipped with the package
RISE = {"Am": 16, "Ap": 96}  # rise scans per family; each window adds as many exit scans (N37)
AT_LENGTH = 16  # the slow ramp's window (scans)
ATTACKED_FRAC = 0.2


def _download(url: str, path: Path) -> Path:
    """`url` saved to `path` once; a file already there is kept."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        with (
            urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=300) as r,
            open(part, "wb") as f,
        ):
            shutil.copyfileobj(r, f)
        part.replace(path)
    return path


def _nygrid(data: Path) -> None:
    """NYgrid's `Data/` folder at the pinned commit, under `data/NYgrid/Data`."""
    dest = data / "NYgrid" / "Data"
    if dest.exists():
        return
    archive = _download(NYGRID_ZIP, data / f"NYgrid_{NYGRID_COMMIT[:7]}.zip")
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) > 2 and parts[1] == "Data" and not name.endswith("/"):
                out = dest.joinpath(*parts[2:])
                out.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(out, "wb") as f:
                    shutil.copyfileobj(src, f)


def fetch_inputs(root: str | Path) -> Path:
    """Download the build's inputs under `root/data` and return that folder: NYgrid's `Data/` at commit
    47698b6, NYISO's five-minute zonal load ('pal') for the year, the NYS DPS utility territories, and the
    bus locations shipped with the package. A file already present is kept. The building-stock load
    profiles, EIA-861 and the industrial database are fetched by `LoadModel` the first time the pool is built."""
    data = Path(root) / "data"
    _nygrid(data)
    terr = data / "territories"
    _download(TERRITORIES, terr / GEOJSON)
    if not (terr / LOCATIONS).exists():
        shutil.copyfile(Path(__file__).parent / "grid" / LOCATIONS, terr / LOCATIONS)
    fetch_nyiso(Settings().year, data / "nyiso")
    return data


def production_builder(
    pool_path: str | Path,
    families: tuple[str, ...] = ("Am", "At"),
    *,
    interfaces: tuple[str, ...] = ("SPR/DUN-SOUTH",),
    workers: int = 12,
    settings: Settings = Settings(),
    root: str | Path = ".",
    survey: int = 12,
    survey_cache: str | Path | None = None,
    hold_external: bool = False,
) -> TimelineBuilder:
    """The `TimelineBuilder` of the release: the case, clients, meters and the pool at `pool_path`
    (relative to `root`, whose `data/` holds `fetch_inputs`'s files), with the `families` named. Am targets
    the NYISO `interfaces` (N48, N49; empty targets the branch ratings instead), Ap is bounded by each bus's
    natural five-minute step, and both come from the OPF attacker with the units' output (N42). `survey`
    is the onset survey's grid in frames (0 off), saved by chunk under `survey_cache` when given."""
    root = Path(root)
    nygrid = root / settings.nygrid_dir / "Data"
    terr = root / "data" / "territories"
    case = restore_765kv(split_zone_j(nygrid_units(modify_mpc(read_npcc(nygrid / "npcc.mat")), nygrid)))
    with open(terr / GEOJSON, encoding="utf-8") as f:
        own = assign_clients(case, json.load(f), terr / LOCATIONS)
    pool = read_pool(root / pool_path)
    case = peak_ratings(case, pool)  # production ratings over the whole pool (N8)
    net = Network(case)
    plan = plan_meters(net, own, rng(settings, "meters"))
    meas = Measurement(net, plan)
    att = Attacker(net, meas, hold_external=hold_external)  # N62: True keeps the interchange true
    # Ap: each bus's natural five-minute step by hour over the whole pool (what the attacker knows),
    # generator buses bounded by their units' five-minute ramp (nuclear 0, hydro and imports inf)
    steps = natural_steps(pool.time, pool.p, pool.solve_kind == SOLVED)
    steps[:, att.gen_bus] = generator_ramps(case)[att.gen_bus]
    hour = (pool.time.astype(np.int64) // 3600) % 24
    q_lo, q_hi = generator_q_capability(case)
    assert pool.unit_p is not None, (
        "the pool must carry each unit's output (built by tools/nyfed/run_pool.py)"
    )
    gen_output = np.zeros((len(pool.time), len(case.bus_id)))  # the units' output at each bus (N42)
    np.add.at(gen_output.T, case.unit_bus, np.nan_to_num(pool.unit_p).T)
    opf: dict[str, Any] = dict(
        gen_ramp=generator_ramps(case),
        gen_band=generator_band(case),
        q_lo=q_lo,
        q_hi=q_hi,
        step_bound=steps,
        frame_hour=hour,
        wait_share=NEAR_LIMIT,
        gen_output=gen_output,
        loads_only=False,
    )

    def am():
        if interfaces:
            ifs = hourly_interfaces(case, pool.time, nygrid, list(interfaces))
            return InterfaceFdia(att, ifs, bound_rise=False, exit_scans=RISE["Am"], **opf)
        return OpfFdia(att, case.rating, bound_rise=False, exit_scans=RISE["Am"], family="Am", **opf)

    every = {
        "Am": am,
        "Ap": lambda: OpfFdia(att, case.rating, bound_rise=True, exit_scans=RISE["Ap"], family="Ap", **opf),
        "At": lambda: Ramp(att),
    }
    fams = {f: every[f]() for f in families}
    length = {"Am": 2 * RISE["Am"], "Ap": 2 * RISE["Ap"], "At": AT_LENGTH}
    return TimelineBuilder(
        pool,
        meas,
        NoiseModel(plan),
        att,
        fams,
        own,
        settings,
        length={f: length[f] for f in fams},
        attacked_frac=ATTACKED_FRAC,
        workers=workers,
        verbose=True,
        survey=survey,
        survey_cache=survey_cache,
    )
