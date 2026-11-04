"""Who bears each attack's damage and whose grid it uses: the damage study's days split by New York zone, by
utility and by neighboring area (`fdia_graph.nyfed.operator.areas`).

    python tools/nyfed/area_damage.py OUT.csv STUDY [STUDY ...] [--modes trusting accounting oracle] [--dispatch cost]
                                       [--workers 8]

Run from the folder whose `data/` holds the inputs. Each day of each damage study runs through each operator in
closed loop (as the paper's damage study), under the least-movement dispatch or with `--dispatch cost` the least-cost
one, whose prices give each area's load payments, and OUT.csv gets one row per day, operator and area (`table` = damage),
plus, once per day, the attack's surface per area (`table` = surface) and each zone's lever on each interface
(`table` = lever).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fdia_graph.nyfed.experiments import add_study_arguments, load_calib, run_parallel, study_days, variant
from fdia_graph.nyfed.operator.areas import area_damage, attack_surface, neighbor_prices, zone_levers
from fdia_graph.nyfed.operator.response import MODES, operator_day, respond


def _job(
    path: str, pool: str, data: str, calib: tuple[float, float], modes: list[str], dispatch: str = "movement"
) -> list[dict]:
    day = operator_day(Path(path), pool, data)
    if day.onset < 0:  # a day without an attack
        return []
    tag = {
        "file": Path(path).name,
        "study": Path(path).parents[1].name,
        "onset": int(
            Path(path).stem.split("_")[1]
        ),  # the pool onset the file is named by; day.onset counts in the day
        "variant": variant(path),
    }
    prices = neighbor_prices(data, day.time)
    rows = [{**tag, "table": "surface", **row} for row in attack_surface(day)]
    rows += [{**tag, "table": "lever", **row} for row in zone_levers(day)]
    for mode in modes:
        r = respond(day, mode=mode, calib=calib, closed_loop=True, dispatch=dispatch)
        rows += [
            {**tag, "table": "damage", "mode": mode, "dispatch": dispatch, **row}
            for row in area_damage(r, day, prices)
        ]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_study_arguments(ap)
    ap.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = ap.parse_args()
    files = study_days(args.studies, args.limit)
    if not files:
        raise SystemExit(f"no attack days (days/am_*_*.h5) under {args.studies}")
    calib = load_calib(args.calib)
    tasks = [(f, args.pool, args.data, calib, args.modes, args.dispatch) for f in files]
    run_parallel(_job, tasks, args.out, workers=args.workers, every=10)


if __name__ == "__main__":
    main()
