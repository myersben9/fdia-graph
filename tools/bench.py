"""Time the estimators and the generator on the tiny shard, and write the numbers where a reader
can compare them with the last run (docs/reference/BENCHMARKS.md).

    python tools/bench.py            # append this machine's row to the table
    python tools/bench.py --check    # exit 1 if any timing is more than 3x slower than the last row

Timings are per record for the estimators (fit excluded) and per record for shard generation,
in milliseconds, on the tiny IEEE-14 shard the test suite builds (seed 1, 80 benign, 12 per
family). They are for spotting a regression on one machine, not for comparing machines: the
table carries the CPU and the torch state with every row for that reason.
"""

from __future__ import annotations

import atexit
import datetime as dt
import os
import platform
import re
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
DOC = os.path.join(os.path.dirname(HERE), "docs", "reference", "BENCHMARKS.md")
SHARD_KW = dict(per_family=12, n_benign=80, seed=1)
SLOW_FACTOR = 3.0


def _timings() -> dict[str, float]:
    from fdia_graph.se import WLS, AdaptiveWeighting, SubspacePrior

    out: dict[str, float] = {}
    path = os.path.join(_CACHE, "tiny.h5")
    t0 = time.perf_counter()
    fg.generate("ieee14", "tiny", out=path, **SHARD_KW)
    gen_s = time.perf_counter() - t0  # generation alone; the load below only supplies the record count
    out["generate ms/record"] = 1e3 * gen_s / len(fg.load("tiny"))
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
        out[f"{name} ms/record"] = 1e3 * (time.perf_counter() - t0) / (3 * len(test))
    return out


def _machine() -> str:
    try:
        import torch

        torch_state = f"torch {torch.__version__}"
    except ImportError:
        torch_state = "no torch"
    return f"{platform.machine()} {platform.processor() or platform.system()}, numpy {np.__version__}, {torch_state}"


def _last_row(text: str, machine: str) -> dict[str, float]:
    """The timings of the most recent row measured on `machine`; empty when there is none, since
    rows from different machines are not comparable."""
    lines = text.splitlines()
    heads = [ln for ln in lines if ln.startswith("| date")]
    if not heads:
        return {}
    header = [c.strip() for c in heads[0].strip("|").split("|")]
    for ln in reversed([ln for ln in lines if ln.startswith("| 20")]):
        row = dict(zip(header, [c.strip() for c in ln.strip("|").split("|")]))
        if row.get("machine") == machine:
            return {
                h: float(v) for h, v in row.items() if h.endswith("ms/record") and re.fullmatch(r"[0-9.]+", v)
            }
    return {}


def main(check: bool) -> int:
    t = _timings()
    text = open(DOC, encoding="utf8").read() if os.path.exists(DOC) else ""
    machine = _machine()
    last = _last_row(text, machine)
    slow = {k: (last[k], v) for k, v in t.items() if k in last and v > SLOW_FACTOR * last[k]}
    for k, v in t.items():
        print(f"{k:24} {v:8.3f}" + (f"   (last {last[k]:.3f})" if k in last else ""))
    if check:
        if not last:
            print(
                f"no earlier row from this machine ({machine}); nothing to compare, run without --check first"
            )
            return 0
        if slow:
            print("slower than 3x the last row from this machine:", slow)
            return 1
        print("no timing regression")
        return 0
    cols = ["date", *t.keys(), "version", "machine"]
    if not text:
        text = (
            "# Benchmarks\n\nPer-record timings on the tiny IEEE-14 shard, appended by `python tools/bench.py`; "
            "`--check` fails when a timing is more than 3x slower than the last row. One machine's rows are "
            "comparable with each other, not with another machine's.\n\n"
            "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
        )
    row = [dt.date.today().isoformat(), *(f"{v:.3f}" for v in t.values()), fg.__version__, machine]
    text = text.rstrip("\n") + "\n| " + " | ".join(row) + " |\n"
    open(DOC, "w", encoding="utf8", newline="\n").write(text)
    print("appended to", DOC)
    return 0


if __name__ == "__main__":
    sys.exit(main("--check" in sys.argv))
