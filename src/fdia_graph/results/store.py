"""The results store: one tidy CSV per experiment plus one table of runs, in a folder kept in git.

    results/<experiment>.csv   one row per Record (run_id, the keys, value, sd, ci_lo, ci_hi)
    results/runs.csv           one row per Provenance

A tidy CSV diffs well in review and needs nothing beyond the standard library. Writing a run
replaces any rows that run wrote before (a re-run of the same run id is idempotent); every other
run's rows stay, so the store keeps history and `latest` picks the newest run per measurement.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable
from dataclasses import asdict, fields
from typing import Optional

from ..models.errors import NoSuchResult
from ..models.results import KEYS, Provenance, Record, RunRecords, as_values

_RECORD_COLS = ("run_id", "experiment", "system", "method", "family", "split", "metric", "tags")
_VALUE_COLS = ("value", "sd", "ci_lo", "ci_hi")
_RUN_COLS = tuple(f.name for f in fields(Provenance))


def _num(text: str) -> Optional[float]:
    return float(text) if text != "" else None


class Store:
    """A folder of experiment tables. `Store()` is the repository's `results/`."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or os.environ.get("FDIA_RESULTS", default_path())
        self._cache: dict[str, tuple[float, list[Record]]] = {}  # experiment -> (file mtime, records)

    # ---- writing
    def write(self, provenance: Provenance, records: Iterable[Record]) -> int:
        """Store a run's records (all of one experiment) and its provenance; returns the row count."""
        rows = list(RunRecords(provenance, tuple(records)).records)
        os.makedirs(self.path, exist_ok=True)
        kept = [r for r in self._read(provenance.experiment) if r.run_id != provenance.run_id]
        stamped = [Record(**{**asdict(r), "run_id": provenance.run_id}) for r in rows]
        self._write_table(provenance.experiment, kept + stamped)
        runs = [p for p in self.runs() if p.run_id != provenance.run_id] + [provenance]
        self._write_runs(runs)
        return len(stamped)

    def _write_table(self, experiment: str, rows: list[Record]) -> None:
        rows.sort(key=lambda r: (r.run_id, *r.key()))
        with open(self._table(experiment), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(_RECORD_COLS + _VALUE_COLS)
            for r in rows:
                keys = [
                    r.run_id,
                    r.experiment,
                    r.system,
                    r.method,
                    r.family,
                    r.split,
                    r.metric,
                    ";".join(r.tags),
                ]
                w.writerow(
                    keys + [repr(r.value)] + ["" if v is None else repr(v) for v in (r.sd, r.ci_lo, r.ci_hi)]
                )

    def _write_runs(self, runs: list[Provenance]) -> None:
        runs.sort(key=lambda p: (p.experiment, p.timestamp, p.run_id))
        with open(os.path.join(self.path, "runs.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(_RUN_COLS)
            for p in runs:
                w.writerow(["" if v is None else v for v in (getattr(p, c) for c in _RUN_COLS)])

    # ---- reading
    def experiments(self) -> list[str]:
        """Every experiment with a table, sorted."""
        if not os.path.isdir(self.path):
            return []
        return sorted(f[:-4] for f in os.listdir(self.path) if f.endswith(".csv") and f != "runs.csv")

    def runs(self) -> list[Provenance]:
        path = os.path.join(self.path, "runs.csv")
        if not os.path.exists(path):
            return []
        with open(path, newline="", encoding="utf-8") as f:
            out = []
            for row in csv.DictReader(f):
                seed = row.pop("seed")
                dirty = row.pop("dirty")
                out.append(Provenance(**row, seed=int(seed) if seed else None, dirty=dirty == "True"))
            return out

    def query(self, experiment: str, **filters: object) -> list[Record]:
        """Every record of `experiment` whose keys match `filters`. A filter on a key takes a value
        or a collection of values; `tags` filters as "name=value" strings, all of which must hold;
        a filter on a tag name (e.g. `k="1.2"`) is shorthand for that tag."""
        want = {k: as_values(v) for k, v in filters.items() if k in _KEY_FIELDS}
        tags = [str(v) for v in as_values(filters.get("tags", ()))]
        tags += [f"{k}={v}" for k, v in filters.items() if k not in KEYS and k != "run_id"]
        return [r for r in self._read(experiment) if _matches(r, want, tags)]

    def latest(self, experiment: str, **filters: object) -> list[Record]:
        """`query`, keeping for each measurement only the record of the newest run."""
        when = {p.run_id: p.timestamp for p in self.runs()}
        best: dict[tuple[str, ...], Record] = {}
        for r in self.query(experiment, **filters):
            k = r.key()
            if k not in best or when.get(r.run_id, "") > when.get(best[k].run_id, ""):
                best[k] = r
        return list(best.values())

    def one(self, experiment: str, **filters: object) -> Record:
        """The single newest record matching `filters`; `NoSuchResult` when none or several match."""
        found = self.latest(experiment, **filters)
        if len(found) != 1:
            raise NoSuchResult(f"{experiment} {filters}: {len(found)} records match, expected one")
        return found[0]

    def _table(self, experiment: str) -> str:
        return os.path.join(self.path, f"{experiment}.csv")

    def _read(self, experiment: str) -> list[Record]:
        path = self._table(experiment)
        if not os.path.exists(path):
            return []
        stamp = os.path.getmtime(path)
        if experiment in self._cache and self._cache[experiment][0] == stamp:
            return self._cache[experiment][1]
        with open(path, newline="", encoding="utf-8") as f:
            rows = [
                Record(
                    **{k: row[k] for k in _RECORD_COLS if k != "tags"},
                    tags=tuple(t for t in row["tags"].split(";") if t),
                    value=float(row["value"]),
                    sd=_num(row["sd"]),
                    ci_lo=_num(row["ci_lo"]),
                    ci_hi=_num(row["ci_hi"]),
                )
                for row in csv.DictReader(f)
            ]
        self._cache[experiment] = (stamp, rows)
        return rows


_KEY_FIELDS = (*(k for k in KEYS if k != "tags"), "run_id")


def _matches(record: Record, want: dict[str, tuple[object, ...]], tags: list[str]) -> bool:
    """Whether `record` has one of the wanted values of every key filtered and every tag asked."""
    return all(getattr(record, k) in v for k, v in want.items()) and all(t in record.tags for t in tags)


def default_path() -> str:
    """The repository's `results/` folder: the nearest ancestor of the working directory holding one,
    else `results/` in the working directory."""
    here = os.path.abspath(os.getcwd())
    while True:
        if os.path.isdir(os.path.join(here, "results")) and os.path.exists(
            os.path.join(here, "pyproject.toml")
        ):
            return os.path.join(here, "results")
        parent = os.path.dirname(here)
        if parent == here:
            return os.path.join(os.getcwd(), "results")
        here = parent
