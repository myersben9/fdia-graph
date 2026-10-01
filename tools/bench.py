"""Time the estimators and the generator on the tiny timeline, and store the timings as a run of the
`bench` experiment in the results store (`results/bench.csv`), where docs/reference/BENCHMARKS.md
renders them from (a results block).

    python tools/bench.py            # store this machine's run
    python tools/bench.py --check    # exit 1 if any timing is more than 3x slower than the last run

Timings are per record for the estimators (fit excluded) and per frame for timeline generation,
in milliseconds, on the tiny IEEE-14 timeline the test suite builds (1000 frames, At and Am,
20-frame episodes, the settings of tests/conftest.TIMELINE_KW). They are for spotting a regression on one machine, not for comparing
machines: every run carries the CPU and the torch state (its provenance note) for that reason, and
the fixture recipe (a tag, its hash) so a new recipe starts a new baseline.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import platform
import shutil
import sys
import tempfile
import time

_CACHE = tempfile.mkdtemp(prefix="fdia_bench_")
os.environ["FDIA_GRAPH_CACHE"] = _CACHE
atexit.register(shutil.rmtree, _CACHE, ignore_errors=True)

import numpy as np

import fdia_graph as fg

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tests"))
from conftest import TIMELINE_KW  # noqa: E402  the test suite's tiny timeline, one definition

SLOW_FACTOR = 3.0
# the fixture recipe a row was measured on: rows of another recipe time other frames and meters, so
# `--check` compares only rows of this one (a new recipe starts a new table and a new baseline)
RECIPE = ",".join(f"{k}={v}" for k, v in sorted(TIMELINE_KW.items())) + ",families=At+Am,meters=hybrid"
RECIPE_TAG = hashlib.sha256(RECIPE.encode()).hexdigest()[:8]  # a tag value carries no "=" or ";"


def _timings() -> dict[str, float]:
    from fdia_graph.se import WLS, AdaptiveWeighting, SubspacePrior

    out: dict[str, float] = {}
    path = os.path.join(_CACHE, "tiny.h5")
    t0 = time.perf_counter()
    fg.generate("ieee14", "tiny", out=path, **TIMELINE_KW)
    gen_s = time.perf_counter() - t0  # generation alone; the load below only supplies the frame count
    out["generate"] = 1e3 * gen_s / len(fg.load("tiny"))
    train, test = fg.load("tiny", split="train"), fg.load("tiny", split="test")
    for name, est in (
        ("wls", WLS()),
        ("huber", AdaptiveWeighting(c=1.5)),
        ("prior+huber", SubspacePrior(rank_frac=0.2, reweight="huber", c=1.5)),
    ):
        est.fit(train)
        est.estimate(test)  # warm-up: first call pays for imports and caches
        t0 = time.perf_counter()
        for _ in range(3):
            est.estimate(test)
        out[name] = 1e3 * (time.perf_counter() - t0) / (3 * len(test))
    return out


def _machine() -> str:
    try:
        import torch

        torch_state = f"torch {torch.__version__}"
    except ImportError:
        torch_state = "no torch"
    return f"{platform.machine()} {platform.processor() or platform.system()}, numpy {np.__version__}, {torch_state}"


def _last_run(machine: str) -> dict[str, float]:
    """The timings of the most recent stored run on `machine` with this fixture recipe; empty when
    there is none, since runs from different machines or recipes are not comparable."""
    from fdia_graph.results import Store

    store = Store(os.path.join(os.path.dirname(HERE), "results"))
    runs = sorted(
        (p for p in store.runs() if p.experiment == "bench" and p.note == machine), key=lambda p: p.timestamp
    )
    for prov in reversed(runs):
        recs = store.query("bench", run_id=prov.run_id, recipe=RECIPE_TAG)
        if recs:
            return {r.method: r.value for r in recs}
    return {}


def main(check: bool) -> int:
    from fdia_graph.results import Run, Store

    t = _timings()
    machine = _machine()
    last = _last_run(machine)
    slow = {k: (last[k], v) for k, v in t.items() if k in last and v > SLOW_FACTOR * last[k]}
    for k, v in t.items():
        print(f"{k:16} {v:8.3f} ms/record" + (f"   (last {last[k]:.3f})" if k in last else ""))
    if check:
        if not last:
            print(
                f"no earlier run from this machine ({machine}) on this fixture recipe; nothing to compare, "
                "run without --check first"
            )
            return 0
        if slow:
            print("slower than 3x the last run from this machine:", slow)
            return 1
        print("no timing regression")
        return 0
    store = Store(os.path.join(os.path.dirname(HERE), "results"))
    with Run("bench", system="ieee14", settings={"recipe": RECIPE}, note=machine, store=store) as run:
        for name, ms in t.items():
            run.add("ms_per_record", ms, method=name, recipe=RECIPE_TAG)
    print(
        "stored a run of `bench`; run tools/results_docs.py --write to refresh docs/reference/BENCHMARKS.md"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
