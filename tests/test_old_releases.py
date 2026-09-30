"""Old data releases stay readable. The generator makes only At and Am, but a v0.8.3 timeline, a
v0.7.2 record shard and a v0.7.x stream file (all seven families, the v0.8.3 meters) load, split,
filter by family name and score per family as before. The fixtures under tests/data were written by
the legacy recipe before it was removed (tests/data/README.md), since it can no longer build them.

Also here: the graph fields a GNN user relies on (the series admittance and the clean flow on every
branch), checked on a timeline."""

import os

import numpy as np
import pytest

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TIMELINE_V083 = os.path.join(DATA, "tiny_timeline_v083.h5")
STREAM_V071 = os.path.join(DATA, "tiny_stream_v071.npz")
OLD_FAMILIES = {"Aq", "Ad", "As", "Ar", "At", "Al", "Am"}


def _family_names(ds):
    from fdia_graph.dataset import FAMILIES

    return {FAMILIES[int(f)] for f in np.unique(ds.export(["family"])["family"])} - {"benign"}


def test_a_v083_timeline_loads_every_split_and_family():
    from fdia_graph.dataset import FdiaGraph

    ds = FdiaGraph(TIMELINE_V083)
    assert ds.is_timeline and ds.has_benign and ds.has_clean and _family_names(ds) == OLD_FAMILIES
    sizes = [len(FdiaGraph(TIMELINE_V083, split=s)) for s in ("train", "val", "test")]
    assert all(n > 0 for n in sizes) and sum(sizes) == len(ds)
    only = FdiaGraph(TIMELINE_V083, families=["Aq", "Al"])
    assert _family_names(only) == {"Aq", "Al"}
    held = FdiaGraph(TIMELINE_V083, split="train", heldout=True)
    assert not {"As", "Ar"} & _family_names(held)
    rec = ds[0]
    assert tuple(rec["clean"].shape) == (14, 4) and tuple(rec["edge_clean_full"].shape) == (ds.E, 2)
    assert ds.edge_attr_np.shape == (ds.E, 8)
    assert len(ds.episodes.onset) > 0 and ds.windows(8)[0].shape[1] == 8


def test_a_v083_timeline_reads_as_pyg():
    pytest.importorskip("torch_geometric")
    from fdia_graph.dataset import FdiaGraph

    g = FdiaGraph(TIMELINE_V083, split="test", format="pyg")[0]
    assert g.edge_index.shape[0] == 2 and g.edge_phys.shape[1] == 8  # the static [E,8] physics
    assert (
        g.edge_attr.shape[1] == 2 and g.edge_clean_full.shape[1] == 2
    )  # flows, and the clean flow everywhere


def test_old_releases_score_per_old_family():
    """A per-family estimator and localizer score on a v0.8.3 timeline names the old families."""
    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.localization import SwingThreshold
    from fdia_graph.se import WLS

    train, test = FdiaGraph(TIMELINE_V083, split="train"), FdiaGraph(TIMELINE_V083, split="test")
    se = WLS().fit(train).score(test)
    loc = SwingThreshold().fit(train).score(test)
    present = _family_names(test)
    assert present <= OLD_FAMILIES and {"Aq", "Ad", "As", "Ar"} <= present
    for name in present:
        assert se[name] is not None and loc[name] is not None


def test_a_v072_record_shard_scores_per_family():
    from conftest import SHARD_V072

    from fdia_graph.dataset import FdiaGraph
    from fdia_graph.se import WLS

    ds = FdiaGraph(SHARD_V072)
    rep = WLS().fit(ds).score(ds)
    assert rep["benign"] is not None and rep["geo"].angle_mae_deg > 0
    assert _family_names(ds) <= OLD_FAMILIES


def test_a_v071_stream_file_loads_and_windows(monkeypatch):
    """`load_stream` at a pre-timeline release reads the stream file (download replaced by the
    checked-in one), and `streams.windows` slides over it."""
    import fdia_graph as fg
    import fdia_graph.download as download
    from fdia_graph.streams import windows

    monkeypatch.setattr(download, "ensure_local", lambda spec: STREAM_V071)
    with pytest.warns(DeprecationWarning):
        s = fg.load_stream("ieee14", release="v0.7.1")
    assert s.system == 14 and 0 < s.attacked_frac < 1 and len(s.episodes) > 0
    with pytest.warns(DeprecationWarning):
        X, y = windows(s, 8, stride=4)
    assert X.shape[1] == 8 and len(X) == len(y)


@pytest.mark.parametrize("release", ["v0.7.1", "v0.7.2", "v0.8.0", "v0.8.1", "v0.8.3"])
def test_every_published_release_resolves_to_its_asset(release):
    from fdia_graph import registry

    spec = registry.resolve("ieee14", release=release)
    want = "timeline_ieee14.h5" if registry.is_timeline_release(release) else "ml_only_ieee14.h5"
    assert spec.file == want


def test_the_series_admittance_is_one_over_the_series_impedance(timeline):
    """edge_gs + j edge_bs = 1 / (r + j x) on every branch of a timeline (edge_attr columns 4, 5)."""
    import fdia_graph as fg

    ds = fg.load(timeline)
    a = ds.edge_attr_np.astype(np.float64)
    z = a[:, 0] + 1j * a[:, 1]
    want = np.where(np.abs(z) > 1e-12, 1.0 / np.where(np.abs(z) > 1e-12, z, 1.0), 0.0)
    assert np.allclose(a[:, 4] + 1j * a[:, 5], want, rtol=1e-5, atol=1e-6)


def test_the_clean_flow_on_every_branch_matches_the_metered_clean_flow(timeline):
    """edge_clean_full carries the clean flow on every branch and equals edge_clean wherever a flow
    meter reads it (edge_clean zeroes the unmetered ones)."""
    import fdia_graph as fg

    d = fg.load(timeline, split="test").export(["edge_clean", "edge_clean_full", "edge_m"])
    metered = d["edge_m"].astype(bool)
    assert np.allclose(d["edge_clean_full"][metered], d["edge_clean"][metered], rtol=1e-4, atol=1e-3)
    assert np.all(d["edge_clean"][~metered] == 0) and np.any(d["edge_clean_full"][~metered] != 0)
