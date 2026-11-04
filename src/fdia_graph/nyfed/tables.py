"""The CSV tables of the inputs, read whole and closed: the NYgrid checkout's bus, unit and renewable tables and
the bus locations. One reader, so no input file is left open (each was a ResourceWarning before)."""

from __future__ import annotations

import csv
from pathlib import Path

__all__ = ["read_rows"]


def read_rows(path: str | Path, encoding: str = "utf-8") -> list[dict[str, str]]:
    """Every row of the CSV at `path` as a dict keyed by its header (`utf-8-sig` drops a byte-order mark)."""
    with open(path, encoding=encoding, newline="") as f:
        return list(csv.DictReader(f))
