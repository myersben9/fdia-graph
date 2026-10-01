"""Turning records into what a reader sees: a formatted cell, a markdown table, a pivot of records by
two keys, and the CSV sidecar a figure is drawn from. Nothing here measures or stores anything."""

from __future__ import annotations

import csv
import os
from collections.abc import Iterable, Sequence
from typing import Optional

from ..models.results import METRICS, Record


def number(value: float, spec: str) -> str:
    """`value` in format `spec` ("d" rounds to a whole number, "%" multiplies by 100)."""
    return format(int(round(value)), "d") if spec == "d" else format(value, spec)


def cell(record: Optional[Record], spec: Optional[str] = None, scale: float = 1.0, sd: bool = False) -> str:
    """One table cell: the record's value (times `scale`) in `spec` (the metric's default when None),
    with " ± sd" when asked and the record carries a spread; "" for a missing record."""
    if record is None:
        return ""
    spec = spec or METRICS[record.metric].fmt
    text = number(record.value * scale, spec)
    if sd and record.sd is not None:
        text += " ± " + number(record.sd * scale, spec)
    return text


def key_of(record: Record, name: str) -> str:
    """A record's value of key `name`: a key field, else a tag; "a+b" joins two keys with "_"
    (a figure's columns by metric and family, "metric+family")."""
    if "+" in name:
        return "_".join(key_of(record, part) for part in name.split("+"))
    return str(getattr(record, name)) if name in _FIELDS else record.tag(name)


_FIELDS = ("system", "method", "family", "split", "metric", "run_id")


def pivot(records: Iterable[Record], row: str, col: str) -> dict[tuple[str, str], Record]:
    """The records indexed by (their `row` key, their `col` key); each pair must be one record."""
    out: dict[tuple[str, str], Record] = {}
    for r in records:
        out[(key_of(r, row), key_of(r, col))] = r
    return out


def table(header: Sequence[str], rows: Iterable[Sequence[str]], numeric_from: int = 1) -> str:
    """A markdown table; columns from `numeric_from` on are right-aligned."""
    head = list(header)
    align = ["---" if i < numeric_from else "---:" for i in range(len(head))]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def figure_data(records: Iterable[Record], path: str, row: str, col: str, spec: str = ".6g") -> str:
    """Write the CSV a figure is drawn from: one line per `row` key, one column per `col` key, the
    values in `spec`. Returns the path; the plot-data rule's sidecar, sourced from the store."""
    grid = pivot(records, row, col)
    rows = list(dict.fromkeys(r for r, _ in grid))
    cols = list(dict.fromkeys(c for _, c in grid))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow([row, *cols])
        for r in rows:
            w.writerow([r, *("" if (r, c) not in grid else number(grid[(r, c)].value, spec) for c in cols)])
    return path
