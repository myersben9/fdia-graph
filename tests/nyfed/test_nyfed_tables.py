"""The input tables are read whole and closed: no ResourceWarning, a byte-order mark dropped (no data needed)."""

from __future__ import annotations

import gc
import warnings

from fdia_graph.nyfed.tables import read_rows


def test_read_rows_closes_the_file_and_drops_the_bom(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("﻿bus_id,zone\n1,A\n2,J\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        rows = read_rows(p, "utf-8-sig")
        gc.collect()  # an unclosed file would warn when collected
    assert rows == [{"bus_id": "1", "zone": "A"}, {"bus_id": "2", "zone": "J"}]
