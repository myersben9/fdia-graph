"""The study helpers every research tool shares: the day files of a study, a day's variant, the interchange
calibration, and the parallel runner that writes a sorted CSV (no data needed)."""

from __future__ import annotations

import json

import pytest

from fdia_graph.nyfed.experiments import load_calib, run_parallel, study_days, variant

pd = pytest.importorskip("pandas")


def test_study_days_and_variants(tmp_path):
    for study, names in (("a", ["am_5_0.h5", "am_5_175.h5"]), ("b", ["am_5_0.h5", "am_9_0.h5", "notes.txt"])):
        (tmp_path / study / "days").mkdir(parents=True)
        for n in names:
            (tmp_path / study / "days" / n).write_text("")
    days = study_days([tmp_path / "a", tmp_path / "b"])
    assert [p.split("days")[-1][1:] for p in days] == ["am_5_0.h5", "am_5_175.h5", "am_5_0.h5", "am_9_0.h5"]
    assert len(study_days([tmp_path / "a", tmp_path / "b"], limit=2)) == 2
    assert [variant(p) for p in days] == ["at_limit", "damaging", "at_limit", "at_limit"]


def test_load_calib(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps({"mean_mw": 28.41, "sd_mw": 13.76, "threshold_mw": 55.0}))
    assert load_calib(p) == (28.41, 13.76)


def test_run_parallel_writes_one_sorted_csv(tmp_path):
    out = tmp_path / "rows.csv"
    tasks = [([("file", "c"), ("x", 3)],), ([("file", "a"), ("x", 1)],), ([("file", "b"), ("x", 2)],)]
    rows = run_parallel(dict, tasks, out, workers=2)  # dict(pairs) is a picklable job returning one row
    assert sorted(r["file"] for r in rows) == ["a", "b", "c"]
    assert pd.read_csv(out)["file"].tolist() == ["a", "b", "c"]  # sorted whatever finished first
