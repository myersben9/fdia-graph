"""The input tables are read whole and closed: no ResourceWarning, a byte-order mark dropped (no data needed)."""

from __future__ import annotations

import gc
import warnings

from fdia_graph.generation.nygrid.tables import read_rows


def test_read_rows_closes_the_file_and_drops_the_bom(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("﻿bus_id,zone\n1,A\n2,J\n", encoding="utf-8")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always", ResourceWarning)  # a warning at collection is recorded, not raised
        rows = read_rows(p, "utf-8-sig")
        gc.collect()  # an unclosed file would warn when collected
    assert not [w for w in seen if issubclass(w.category, ResourceWarning)]
    assert rows == [{"bus_id": "1", "zone": "A"}, {"bus_id": "2", "zone": "J"}]
