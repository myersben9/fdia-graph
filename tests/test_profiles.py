"""Load sources and operator feeds: each reads itself, no network needed."""

import datetime as dt

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from fdia_graph import profiles  # noqa: E402
from fdia_graph.profiles import CsvColumn, Iso, IsoFolder, RawSeries, load_profile  # noqa: E402


def _standardized(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / x.std()


def test_a_raw_series_is_standardized():
    assert np.allclose(load_profile(RawSeries([1.0, 2.0, 3.0, 6.0])), _standardized([1, 2, 3, 6]))
    assert np.array_equal(load_profile(RawSeries([5.0, 5.0])), [0.0, 0.0])  # flat stays at zero


def test_a_csv_column_reads_that_column(tmp_path):
    f = tmp_path / "loads.csv"
    pd.DataFrame({"t": range(4), "mw": [10.0, 12.0, None, 11.0]}).to_csv(f, index=False)
    assert np.allclose(load_profile(CsvColumn(str(f), "mw")), _standardized([10, 12, 11]))


def test_an_iso_folder_concatenates_its_exports_in_time_order(tmp_path):
    ts, load = IsoFolder(Iso.NYISO).columns
    for name, hours, mw in (("b.csv", (2, 3), (30.0, 40.0)), ("a.csv", (0, 1), (10.0, 20.0))):
        stamps = [dt.datetime(2024, 1, 1, h) for h in hours]
        pd.DataFrame({ts: stamps, load: mw}).to_csv(tmp_path / name, index=False)
    got = load_profile(IsoFolder("NYISO", str(tmp_path)))  # the name is matched without regard to case
    assert np.allclose(got, _standardized([10, 20, 30, 40]))


def test_an_operator_without_an_export_format_is_refused_when_described(tmp_path):
    with pytest.raises(ValueError, match="no CSV export format is known for ercot"):
        IsoFolder(Iso.ERCOT, str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no CSV files"):
        IsoFolder(Iso.CAISO, str(tmp_path)).loads()


def test_the_old_call_form_warns_and_means_the_same(tmp_path):
    with pytest.warns(DeprecationWarning, match="IsoFolder, CsvColumn or RawSeries"):
        old = load_profile([1.0, 2.0, 4.0])
    assert np.array_equal(old, load_profile(RawSeries([1.0, 2.0, 4.0])))
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="column"):
        load_profile(str(tmp_path / "x.csv"))


def test_a_source_of_your_own_needs_only_loads():
    class Constant:
        def loads(self):
            return np.array([1.0, 3.0])

    assert np.allclose(load_profile(Constant()), [-1.0, 1.0])


def test_fetch_profile_asks_the_operator_feed(monkeypatch):
    asked = []

    class Feed:
        def fetch(self, start, end):
            asked.append((start, end))
            return pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2024-01-01", periods=3, freq="5min"))

    monkeypatch.setitem(profiles._FEEDS, Iso.CAISO, Feed())
    got = profiles.fetch_profile("caiso", "2024-01-01", dt.datetime(2024, 1, 2, 7))
    assert asked == [(dt.date(2024, 1, 1), dt.date(2024, 1, 2))]  # a datetime reads as its date
    assert np.allclose(got, _standardized([1, 2, 3]))
    assert set(profiles._FEEDS) == set(Iso)  # every operator has a feed


def test_a_gridstatus_feed_names_the_install_when_the_package_is_missing(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_gridstatus(name, *a, **k):
        if name == "gridstatus":
            raise ImportError("no gridstatus")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_gridstatus)
    with pytest.raises(ImportError, match=r"fdia-graph\[iso\].*NYISO works without it"):
        profiles.GridstatusFeed("CAISO").fetch(dt.date(2024, 1, 1), dt.date(2024, 1, 1))
