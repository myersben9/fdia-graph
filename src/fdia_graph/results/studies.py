"""Studies over attack windows: the parts every research tool under `tools/nyfed/` repeats, in one place.

A study is a folder of single-episode day files, `days/am_<onset>_<mw>.h5`,
one per attack window: at the limit (`mw` 0) or past it. A tool reads those days, runs the operator or a detector on
each in parallel and writes one CSV. `study_days` lists the days, `variant` names a day's form, `load_calib` reads the
interchange check's calibration, `add_study_arguments` declares the options every tool shares (`add_dispatch_argument`
the operator's dispatch, for a tool with its own parser), and `run_parallel` runs a job over the days with a process
pool, writing the CSV as results come in so a stopped run keeps what it did.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "CALIB",
    "DATA",
    "POOL",
    "add_dispatch_argument",
    "add_study_arguments",
    "load_calib",
    "run_parallel",
    "study_days",
    "variant",
]

POOL = "out/pool_2019_v3.h5"  # the 2019 operating-point pool, relative to the work folder
DATA = "data/NYgrid/Data"  # NYgrid's data tables
CALIB = "results/nygrid.tie_check.json"  # the interchange check's calibration, a file of the work folder
# one BLAS thread per worker: the jobs are many small dense solves, and a pool of workers each running a full
# thread team oversubscribes the machine
_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def study_days(studies: Iterable[str | Path], limit: int = 0) -> list[str]:
    """The attack-window day files of every study folder (`<study>/days/am_*_*.h5`), sorted and without repeats;
    `limit` keeps the first that many (0 all), for a quick check."""
    files = sorted({str(p) for s in studies for p in Path(s, "days").glob("am_*_*.h5")})
    return files[:limit] if limit else files


def variant(path: str | Path) -> Literal["at_limit", "damaging"]:
    """The form of a day file's attack: `am_<onset>_0.h5` is at the limit, any other overshoot damaging."""
    return "at_limit" if Path(path).stem.endswith("_0") else "damaging"


def load_calib(path: str | Path = CALIB) -> tuple[float, float]:
    """The interchange check's benign-noise mean and standard deviation (MW), as `respond(calib=...)` takes them,
    from a JSON file `{"mean_mw": ..., "sd_mw": ...}` measured on benign days of the user's own work folder; the
    default path is relative to that folder, as `POOL` and `DATA` are."""
    cal = json.loads(Path(path).read_text())
    return float(cal["mean_mw"]), float(cal["sd_mw"])


def add_dispatch_argument(ap: argparse.ArgumentParser) -> None:
    """`--dispatch`: the operator's objective (`Dispatch`), least movement unless asked otherwise."""
    from fdia_graph.operator.policy import Dispatch  # here, so importing this module stays light

    ap.add_argument(
        "--dispatch",
        default="movement",
        choices=tuple(Dispatch.values()),
        help="the operator's dispatch: least movement (the default) or least cost, which also prices "
        "congestion",
    )


def add_study_arguments(
    ap: argparse.ArgumentParser, *, studies: bool = True, calib: bool = True, dispatch: bool = True
) -> None:
    """The options every study tool shares: the output CSV, the study folders, the pool, NYgrid's data, the
    interchange calibration, the operator's dispatch (`dispatch=False` for a tool that does not run the operator), the
    worker count and a `--limit` for a quick check on the first days."""
    ap.add_argument("out", help="the CSV to write")
    if studies:
        ap.add_argument("studies", nargs="+", help="study folders of day files")
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--data", default=DATA)
    if calib:
        ap.add_argument("--calib", default=CALIB, help="the interchange check's calibration (JSON)")
    if dispatch:
        add_dispatch_argument(ap)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N attack days (a quick check)")


def run_parallel(
    job: Callable[..., Any],
    tasks: Sequence[tuple],
    out: str | Path,
    *,
    workers: int,
    initializer: Callable[..., None] | None = None,
    initargs: tuple = (),
    sort_by: str | None = "file",
    every: int = 1,
    describe: Callable[[dict], str] | None = None,
) -> list[dict]:
    """Run `job(*task)` for every task over `workers` processes and write the rows to the CSV `out`.

    A job returns one row (a dict), several (a list of dicts) or None (nothing to report, such as a day without an
    attack). The CSV is rewritten every `every` finished tasks and at the end, so a stopped run keeps what it did,
    and the final file is sorted by `sort_by` (when that column exists) so it does not depend on which worker
    finished first; rows that tie on it keep the order of their tasks. Progress prints as `k/n [seconds]`, followed
    by `describe(row)` when given. Returns the rows in that final order.
    """
    import pandas as pd  # the tools' dependency, not the package's

    for var in _THREADS:  # inherited by the workers the pool starts
        os.environ.setdefault(var, "1")
    done: dict[int, list[dict]] = {}  # each task's rows, by the task's position
    t0 = time.perf_counter()

    def in_task_order() -> list[dict]:
        return [row for i in sorted(done) for row in done[i]]

    def write(final: bool = False) -> None:
        frame = pd.DataFrame(in_task_order())
        if final and sort_by and sort_by in frame:
            frame = frame.sort_values(sort_by, kind="stable")  # stable: ties keep the task order
        frame.to_csv(out, index=False)

    with ProcessPoolExecutor(workers, initializer=initializer, initargs=initargs) as ex:
        futs = {ex.submit(job, *task): i for i, task in enumerate(tasks)}
        for k, fut in enumerate(as_completed(futs), 1):
            got = fut.result()
            new = got if isinstance(got, list) else [] if got is None else [got]
            done[futs[fut]] = new
            if k % every == 0:
                write()
            note = f" {describe(new[-1])}" if describe and new else ""
            print(f"{k}/{len(tasks)}{note} [{time.perf_counter() - t0:.0f}s]", flush=True)
    write(final=True)
    rows = in_task_order()
    if sort_by and rows and all(sort_by in r for r in rows):
        rows = sorted(rows, key=lambda r: r[sort_by])  # Python's sort is stable, as the CSV's is
    return rows
